/*
This program will numerically compute the integral of
                  4/(1+x*x)
from 0 to 1.  The value of this integral is pi -- which
is great since it gives us an easy way to check the answer.
The is the original sequential program.  It uses the timer
from the OpenMP runtime library
History: Written by Tim Mattson, 11/99.
*/

#include <stdio.h>
/*#include <omp.h>*/

#define BLOCK 13
#define THREAD 192
#define NUMSTEPS 1000000000
double step;
int tid;
float pi = 0;
__global__ void cal_pi(float *sum, int nbin, float step, int nthreads, int nblocks) {
	int i;
	float x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	for (i=idx; i< nbin; i+=nthreads*nblocks) {
		x = (i+0.5)*step;
		sum[idx] += 4.0/(1.0+x*x);
	}
}

int main ()
{
		dim3 dimGrid(BLOCK,1,1);
		dim3 dimBlock(THREAD,1,1);
	  int i;

	  double start_time, run_time;
		cudaEvent_t start, stop;
		float   elapsedTime;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);
	  float step = 1.0/NUMSTEPS;
		float *dev_x, *dev_pi, *dev_sum,*sum_host;
		size_t size = BLOCK*THREAD*sizeof(float);  //Taille mémoire
		sum_host = (float *)malloc(size);  //Allocation de la mémoire
		cudaMalloc((void **)&dev_sum,size); //Allocation de la mémoire au GPU
		cudaMemset(dev_sum, 0, size); //Initialisation du tableau à 0

		cal_pi <<<dimGrid, dimBlock >>> (dev_sum,NUMSTEPS,step,THREAD,BLOCK);

		cudaMemcpy(sum_host,dev_sum,size, cudaMemcpyDeviceToHost);
		// Copie du résultat du GPU vers le CPU
		for(tid=0; tid<THREAD*BLOCK; tid++)
			pi += sum_host[tid];
		pi *= step;

		cudaEventRecord(stop,0);
		cudaEventSynchronize( stop );
    cudaEventElapsedTime( &elapsedTime,start, stop );
    cudaEventDestroy( start );
    cudaEventDestroy( stop );
		free(sum_host);
		cudaFree(dev_sum);
	  printf("\n pi is %f in %f milliseconds\n ",pi,elapsedTime);
}
