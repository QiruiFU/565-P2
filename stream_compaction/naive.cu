#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        // TODO: __global__
        __global__ void computePresumIter(int n, int* odata, const int* idata, const int stride) {
            int idx = blockDim.x * blockIdx.x + threadIdx.x;
            if (idx >= n) return;
            odata[idx] = idata[idx];

            __syncthreads();

            int target = (1 << (stride - 1)) + idx;
            if (target < n) {
				odata[target] = idata[idx] + idata[target];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            int blockSize = 1024;
            int gridSize = (n + blockSize - 1) / blockSize;

            int* dev_odata_1, * dev_odata_2;
            cudaMalloc((void**)&dev_odata_1, n * sizeof(int));
            cudaMalloc((void**)&dev_odata_2, n * sizeof(int));
            cudaMemcpy(dev_odata_1, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            // Kernel
            int iter = ilog2ceil(n);
            for (int d = 1; d <= iter; d++) {
                computePresumIter<<<gridSize, blockSize>>>(n, dev_odata_2, dev_odata_1, d);
                int* temp = dev_odata_1;
                dev_odata_1 = dev_odata_2;
                dev_odata_2 = temp;
            }

            cudaMemcpy(odata, dev_odata_1, n * sizeof(int), cudaMemcpyDeviceToHost);
            cudaFree(dev_odata_1);
            cudaFree(dev_odata_2);

            timer().endGpuTimer();
        }
    }
}
