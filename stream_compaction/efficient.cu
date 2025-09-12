#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define myblockSize 128

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void upSweep(int n, int* data, int stride) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            int idx = ((index + 1) << stride) - 1;
            if (idx > n) return;
            int left_son = idx - (1 << (stride - 1));
            data[idx] += data[left_son];
        }

        __global__ void downSweep(int n, int* data, int stride) {
            int index = blockDim.x * blockIdx.x + threadIdx.x;
            int idx = ((index + 1) << stride) - 1;
            if (idx >= n) return;
			int left_son = idx - (1 << (stride - 1));
			int temp = data[left_son];
			data[left_son] = data[idx];
			data[idx] += temp;
        }

        __global__ void scanBlocks(int n, int* odata, const int* idata, int* block_sum) {
            __shared__ int cache[myblockSize];

            // move in
            int tid = threadIdx.x;
            int global_idx = blockIdx.x * blockDim.x + tid;
            if (global_idx < n) {
                cache[tid] = idata[blockIdx.x * blockDim.x + tid];
            }
            else {
                cache[tid] = 0;
            }

            __syncthreads();

            // up sweep
            int iter = 0, temp_n = blockDim.x;
			while (temp_n >>= 1) {
                iter++;
			}

            for (int d = 1; d <= iter; d++) {
				int idx = ((tid + 1) << d) - 1;
                if (idx < myblockSize) {
					int left_son = idx - (1 << (d - 1));
					cache[idx] += cache[left_son];
                }
				__syncthreads();
            }

            //
            if (tid == 0) {
                block_sum[blockIdx.x] = cache[myblockSize - 1];
                cache[myblockSize - 1] = 0;
            }
			__syncthreads();

            // down sweep
            for (int d = iter; d >= 1; d--) {
				int idx = ((tid + 1) << d) - 1;
                if (idx < myblockSize) {
					int left_son = idx - (1 << (d - 1));
					int temp = cache[left_son];
					cache[left_son] = cache[idx];
					cache[idx] += temp;
                }
				__syncthreads();
            }

            if(global_idx < n) odata[blockIdx.x * blockDim.x + tid] = cache[tid];
        }

        __global__ void addOffsets(int n, int* data, const int* offsets) {
            int tid = threadIdx.x + blockDim.x * blockIdx.x;
            if (tid >= n) return;
            if (blockIdx.x > 0) {
            int offset = offsets[blockIdx.x];
            data[tid] += offset;
            }
        }

        // you can assume the n has already been ceiled to power of 2
        void exclusiveScan(int n, int* dev_odata, const int* dev_idata) {
            int blockCnt = (n + myblockSize - 1) / myblockSize;

            int* dev_block_sum_scan, * dev_block_sum;
            cudaMalloc((void**)&dev_block_sum, blockCnt * sizeof(int));
            cudaMalloc((void**)&dev_block_sum_scan, blockCnt * sizeof(int));

            scanBlocks << <blockCnt, myblockSize >> > (n, dev_odata, dev_idata, dev_block_sum);

            if (blockCnt > 1) {
                exclusiveScan(blockCnt, dev_block_sum_scan, dev_block_sum);
                addOffsets << <blockCnt, myblockSize >> > (n, dev_odata, dev_block_sum_scan);
            }
        }


        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            // TODO

            int iter = ilog2ceil(n);
            int n_ceil = 1 << iter;

            int* dev_idata, * dev_odata;
            //int* dev_input;
            cudaMalloc((void**)&dev_idata, n_ceil * sizeof(int));
            cudaMalloc((void**)&dev_odata, n_ceil * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();
            exclusiveScan(n_ceil, dev_odata, dev_idata);
            /*for (int d = 1; d <= iter; d++) {
                int gridSize = n_ceil >> d;
                upSweep << <gridSize, blockSize >> > (n_ceil, dev_data, d);
            }

            cudaMemset(dev_data + (n_ceil - 1), 0, sizeof(int));

            for (int d = iter; d >= 1; d--) {
                int gridSize = n_ceil >> d;
                downSweep << <gridSize, blockSize >> > (n_ceil, dev_data, d);
            }*/

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {

            // TODO
			int iter = ilog2ceil(n);
            int n_ceil = 1 << iter;

            int gridSize = (n + myblockSize - 1) / myblockSize;
            int* dev_indices, * dev_idata, * dev_odata, *dev_bools;
            int* indices = (int*)malloc(n * sizeof(int));
            int* bools = (int*)malloc(n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n_ceil * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n_ceil * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            timer().startGpuTimer();

            Common::kernMapToBoolean<<<gridSize, myblockSize>>>(n, dev_bools, dev_idata);
            exclusiveScan(n_ceil, dev_indices, dev_bools);
            Common::kernScatter << <gridSize, myblockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);

            timer().endGpuTimer();

            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            int ans;
            cudaMemcpy(&ans, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            free(indices);
            free(bools);
            cudaFree(dev_indices);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);


            if (idata[n - 1] != 0) ans++;
            return ans;
        }
    }
}
