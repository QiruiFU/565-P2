#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

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

        __global__ void addSelf(int n, int* data, const int* self) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) return;
            data[idx] += self[idx];
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO

            int iter = ilog2ceil(n);
            int n_ceil = 1 << iter;

            int blockSize = 1024;

            int* dev_data;
            int* dev_input;
            cudaMalloc((void**)&dev_data, n_ceil * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            for (int d = 1; d <= iter; d++) {
                int gridSize = n_ceil >> d;
                upSweep << <gridSize, blockSize >> > (n_ceil, dev_data, d);
            }

            cudaMemset(dev_data + (n_ceil - 1), 0, sizeof(int));

            for (int d = iter; d >= 1; d--) {
                int gridSize = n_ceil >> d;
                downSweep << <gridSize, blockSize >> > (n_ceil, dev_data, d);
            }

            cudaMalloc((void**)&dev_input, n * sizeof(int));
            cudaMemcpy(dev_input, idata, n * sizeof(int), cudaMemcpyHostToDevice);
            int gridSize = (n + blockSize - 1) / blockSize;
			addSelf << <gridSize, blockSize >> > (n, dev_data, dev_input);

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
            cudaFree(dev_input);

            timer().endGpuTimer();
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
            timer().startGpuTimer();

            // TODO
			int iter = ilog2ceil(n);
            int n_ceil = 1 << iter;

            int blockSize = 1024;
            int gridSize = (n + blockSize - 1) / blockSize;
            int* dev_indices, * dev_idata, * dev_odata, *dev_bools;
            int* indices = (int*)malloc(n * sizeof(int));
            int* bools = (int*)malloc(n * sizeof(int));
            cudaMalloc((void**)&dev_indices, n_ceil * sizeof(int));
            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            Common::kernMapToBoolean<<<gridSize, blockSize>>>(n, dev_bools, dev_idata);
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);
            
            // scan
            for (int d = 1; d <= iter; d++) {
                int ceilGridSize = n_ceil >> d;
                upSweep << <ceilGridSize, blockSize >> > (n_ceil, dev_indices, d);
            }

            cudaMemset(dev_indices + (n_ceil - 1), 0, sizeof(int));

            for (int d = iter; d >= 1; d--) {
                int ceilGridSize = n_ceil >> d;
                downSweep << <ceilGridSize, blockSize >> > (n_ceil, dev_indices, d);
            }

            Common::kernScatter << <gridSize, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            cudaMemcpy(odata, dev_odata, n * sizeof(int), cudaMemcpyDeviceToHost);

            int ans;
            cudaMemcpy(&ans, dev_indices + (n - 1), sizeof(int), cudaMemcpyDeviceToHost);

            free(indices);
            free(bools);
            cudaFree(dev_indices);
            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);

            timer().endGpuTimer();

            if (idata[n - 1] != 0) ans++;
            return ans;
        }
    }
}
