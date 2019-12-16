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

#define BLOCK 65535
#define THREAD 32
#define NUMSTEPS 100000000
__global__ void cal_pi(double *sum, int nbin, double step, int nthreads, int nblocks) {

	__shared__ double s_blocksum[THREAD];

	int i;
	double x;
	int idx = blockIdx.x*blockDim.x+threadIdx.x;
	double t_sum=0.0;

	s_blocksum[threadIdx.x]=0.0;

	for (i=idx; i< nbin; i+=nthreads*nblocks) {
		x = (i+0.5)*step;
		t_sum += 4.0/(1.0+x*x);
	}


	
	s_blocksum[threadIdx.x]=t_sum;
	__syncthreads(); // 
	//Concat par block
	if(threadIdx.x == 0)
	{
		t_sum=0.0;
		for(i=0;i<THREAD;i++)
			t_sum+=s_blocksum[i];

		sum[blockIdx.x]=t_sum;
	}


	
}

int main ()
{
		dim3 dimGrid(BLOCK,1,1); // 	
		dim3 dimBlock(THREAD,1,1); //
	  	int i;
		int tid;

	  	double start_time, run_time;
		cudaEvent_t start, stop;
		float   elapsedTime;

		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start,0);

		double pi=0.0;
	  	double step = 1.0/NUMSTEPS;
		double sum_host[BLOCK];
		double *dev_sum;
		

		cudaMalloc((void **)&dev_sum,BLOCK*sizeof(double)); //Allocation de la mémoire au GPU
		//cudaMemset(dev_sum, 0, size); //Initialisation du tableau à 0

		cal_pi <<<dimGrid, dimBlock >>> (dev_sum,NUMSTEPS,step,THREAD,BLOCK);

		cudaMemcpy(sum_host,dev_sum,BLOCK*sizeof(double), cudaMemcpyDeviceToHost);// Copie du résultat du GPU vers le CPU
		
		for(tid=0; tid<BLOCK; tid++) // on rassemble toutes les sommes 
			pi += sum_host[tid];
		pi *= step;

		cudaEventRecord(stop,0);
		cudaEventSynchronize( stop );
    		cudaEventElapsedTime( &elapsedTime,start, stop );
   	 	cudaEventDestroy( start );
    		cudaEventDestroy( stop );
		//free(sum_host);
		cudaFree(dev_sum);
	  	printf("\n pi is %.15f in %f milliseconds\n ",pi,elapsedTime);
}
