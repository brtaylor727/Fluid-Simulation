#include <stdio.h>

#define MATRIX_SIZE 96;
#define NUMITER 100;

#define cudaErrorCheck(call) { cudaAssert(call,__FILE__,__LINE__); }

const float errortol = 1e-15;
const int numtimesteps = 7;
const float beta = 0.0;

const float rho = 60.;
const float nu = 1e-9;
const float g = 1e-6;


const float dt = 1e-6;
const float dx = 1e-3;

const float u0 = 1e-6;

void cudaAssert(const cudaError err, const char *file, const int line)
{ 
    if( cudaSuccess != err) {                                                
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        
                file, line, cudaGetErrorString(err) );
        exit(1);
    } 
}

__global__
void saxpy(int n, float a, float *x, float *y)
{
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

__global__
//void jacobi_iterator_GPU(float  U_new[MATRIX_SIZE][MATRIX_SIZE], float U[MATRIX_SIZE][MATRIX_SIZE], float  F [MATRIX_SIZE][MATRIX_SIZE])
void jacobi_iterator_GPU(float * U_new, float* U, float * F ,int num)
{

    int N = MATRIX_SIZE;
    int i = blockDim.x * blockIdx.x + threadIdx.x ;
    int j = blockDim.y * blockIdx.y + threadIdx.y ;

    int ROW = blockIdx.y*blockDim.y+threadIdx.y;
    int COL = blockIdx.x*blockDim.x+threadIdx.x;

    if (ROW < N && COL < N) {
        // each thread computes one element of the block sub-matrix
        for (int i = 0; i < N; i++) {
            //tmpSum += F[ROW * N + i] * F[i * N + COL];
	
	}

    }

	if( (j*N+i) %2 != num){

    if ((i < N -1) && (j < N -1) && (i > 0) && (j > 0))
    {
       U_new[j * N + i] = (U[j * N + (i - 1)] + U[j * N + (i + 1)] + U[(j - 1) * N + i] + U[(j + 1) * N + i] ) * 0.25 + F[j*N+i];
    }
    else
    {
        //U_new[j*N + i] = 0.;
	if (i == 0){
	    //modified a minux one to a plus one
	    U_new[j*N+i] = ( 2*U[j * N + (i + 1)] + U[(j ) * N + i] + U[(j ) * N + i] ) * 0.25 + F[j*N+i];
	}
    	if (i == N-1){
	   U_new[j*N+i] = (2*U[j * N + (i - 1)] + U[(j ) * N + i] + U[(j ) * N + i] ) * 0.25 + F[j*N+i];
	}
    	if (j == 0){

	    U_new[j*N+i] = (2*U[(j+1) * N + i ] + U[j * N + i] + U[j * N + i] ) * 0.25 + F[j*N+i];
	}
    	if (j == N-1){
	    U_new[j*N+i] = (2*U[(j-1) * N + i ] + U[j * N + i] + U[j * N + i] ) * 0.25 + F[j*N+i];
    	}

    }
	}else{
		U_new[j*N+i] = U[j*N+i];
    //direchlet boundary conditions
    
}
__global__
void difference_GPU(float * U_new, float* U, float* diff_U)
{

	//compute the elementwise subtraction of two matrices
	//to be used for finding the maximum error in the poisson calculation

    int N = MATRIX_SIZE;
    int i = blockDim.x * blockIdx.x + threadIdx.x ;
    int j = blockDim.y * blockIdx.y + threadIdx.y ;

    //only access the internal elements of the array for now
    if ((i < N -1) && (j < N -1) && (i > 0) && (j > 0))
    {
       diff_U[j*N+i] = U_new[j * N + i] -  U[j*N+i];
    }
    else
    {
        diff_U[j*N + i] = 0.;
    }
}

__global__
void max(float *a , float *c )
{
	const int N = MATRIX_SIZE;
        extern __shared__ int sdata[N*N];

        unsigned int tid = threadIdx.x;
        unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;

        sdata[tid] = a[i];

        __syncthreads();
        for(unsigned int s=blockDim.x/2; s>=1; s=s/2)
        {
        if(tid< s)
        {
        if(sdata[tid]>sdata[tid + s])
        {sdata[tid] = sdata[tid + s];}
        }
        //////////////////////////////
        __syncthreads();
        }
        if(tid == 0) c[blockIdx.x] = sdata[0];
}

float getmaxerror(float * diffmat){
	float maxerror = 0.;
	int N = MATRIX_SIZE;
	for (int i = 0; i < N;i++){
		for (int j = 0; j < N;j++){

			if (maxerror < abs(diffmat[i*N+j])){
				
		//		printf("maxerror: %f diffmat %f %i %i\n",maxerror,diffmat[i*N+j],i,j);
				maxerror = abs(diffmat[i*N+j]);
			}
		}
	}
	return maxerror;

}

void savemat(float * matrix,const char * name,int it,int N){

	char fn[100+1];
    
	snprintf(fn, 100, "./data/%s%04d.dat", name,it);

	FILE *f = fopen(fn, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	/* print integers and floats */
	for (int ix = 0;ix < N;ix++){
		for (int iy = 0; iy < N;iy++){
			fprintf(f, "%i %i %f\n", ix,iy,matrix[ix*N+iy]);
		}
	}
	
	fclose(f);
}


int main(void)
{

  int N = MATRIX_SIZE;




  //initialize velocity
  float *ux,*uy,*uintx,*uinty;
  ux = (float*)malloc(N*N*sizeof(float));
  uy = (float*)malloc(N*N*sizeof(float));
  uintx = (float*)malloc(N*N*sizeof(float));
  uinty = (float*)malloc(N*N*sizeof(float));

  for (int i = 0; i < N*N;i++){
	  ux[i] = 0.0;
	  uy[i] = 0e-1;
  }

  //initialize pressure
  float *p;
  p = (float*)malloc(N*N*sizeof(float));
  for (int i = 0; i < N*N;i++){
	  p[i] = 0.0;
  }

  //initialize phi
  float *phi;
  phi = (float*)malloc(N*N*sizeof(float));
  for (int i = 0; i < N*N;i++){
	  phi[i] = 0.0;
  }



  	//allocate memory for the U_new
	float *U_new,*U,*F,*diff_U;
	float *d_U_new,*d_U,*d_F,*d_diff_U;

	U_new = (float*)malloc(N*N*sizeof(float));
	U = (float*)malloc(N*N*sizeof(float));
	F = (float*)malloc(N*N*sizeof(float));
	diff_U = (float*)malloc(N*N*sizeof(float));

  	cudaMalloc(&d_U_new,N*N*sizeof(float));
  	cudaMalloc(&d_U,N*N*sizeof(float));
  	cudaMalloc(&d_F,N*N*sizeof(float));
  	cudaMalloc(&d_diff_U,N*N*sizeof(float));


  	//TODO: add some method to add some numbers to my matrix

  	for (int i = 0; i < N*N;i++) {
		F[i] = -.1;
		U[i] = 4.0;
		U_new[i] = 5.0;
		diff_U[i] = 10.0;
	}




  //begin time iteration
  for(int it = 0;it < numtimesteps; it++){ //begin time iteration

	  //first step: calculate the intermediate velocity

	for (int ix = 1; ix < N-1;ix++){
		for (int iy = 1; iy < N-1;iy++){



			float ax = -dt*ux[ix*N+iy] * (ux[(ix+1)*N+iy]-ux[(ix-1)*N+iy])/(2*dx); 
			float bx = -beta*dt*(p[(ix+1)*N+iy]-p[(ix-1)*N+iy])/(2*dx*rho); 
			float cx = dt*nu*(ux[(ix-1)*N+iy] - 2*ux[ix*N+iy] + ux[(ix+1)*N+iy])/(dx*dx);
			float ddx = dt*0.0;
	

			
			uintx[ix*N+iy] = ux[ix*N+iy] + ax + bx + cx + ddx;


			float ay = -dt*uy[ix*N+iy] * (uy[ix*N+1]-uy[ix*N+iy-1])/(2*dx); 
			float by = -beta*dt*(p[ix*N+iy+1]-p[ix*N+iy-1])/(2*dx*rho); 
			float cy = dt*nu*(uy[ix*N+iy-1] - 2*uy[ix*N+iy] + uy[ix*N+iy+1])/(dx*dx);
			float dy = -dt*g;

			printf("ay: %.5e by: %.5e cy: %.5e dy: %.5e, uy: %.5e %.5e\n",ay,by,cy,dy,uy[ix*N+iy+1],F[ix*N+iy]);
			uinty[ix*N+iy] = uy[ix*N+iy] + ay + by + cy + dy;

			}
	}

	for (int ix = 1; ix < N-1;ix++){
		for (int iy = 1; iy < N-1;iy++){


			//boundary conditions for u_intermediate
			if ( (ix == 0) | (iy == 0) | (ix == (N)) | (iy == (N))){
				uinty[ix*N+iy] = -u0;
				uintx[ix*N+iy] = 0.;
			}
			//boundary condition for outlet
			if (ix ==(N)){

				//uinty[(N)*N+iy] = uinty[(N-1)*N+iy];
				uinty[(N)*N+iy] = -u0;
			}
			//boundary condition for the inlet
			if(ix == 0){
				uinty[iy] = -u0;
			}

		  }
	  }


	//second step: calculate the poisson equation to solve for phi


	//define F
	for (int ix = 1; ix < N-1;ix++){
		for (int iy = 1; iy < N-1;iy++){
			F[ix*N+iy] = (.5*dx/dt)*rho * ( ux[(ix+1)*N+iy]-ux[(ix-1)*N+iy]  
				     + uy[ix*N+iy+1] - uy[ix*N+iy-1]	);
		}

	}
	

  	//send over the data to the gpu
	cudaMemcpy(d_U_new, U_new, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_U, phi, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, F, N*N*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_diff_U, diff_U, N*N*sizeof(float), cudaMemcpyHostToDevice);
  
	//check for error
//	cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
//	cudaErrorCheck( cudaThreadSynchronize() ); // Checks for execution error


	dim3 threadsPerBlock(4, 4);
	dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);
  

	//loop for jacobi iterator
	//run until convergence set by errortol
	int jac = 0;
	float maxer = 1000.;
	float maxer2 = 100.;
	
	int wait = 0.;
	while( maxer > errortol){
		jac++;
	   

		//run twice with the pointers swapped to avoid having to copy memory 
		jacobi_iterator_GPU<<<numBlocks, threadsPerBlock>>>( d_U_new, d_U, d_F ,0);
//		cudaDeviceSynchronize();
		jacobi_iterator_GPU<<<numBlocks, threadsPerBlock>>>( d_U, d_U_new, d_F ,1);
//		cudaDeviceSynchronize();

    		//every 20 iteration check if the convergence condition has been met
    		if ((jac%1000) == 0){
			maxer2 = maxer;
    			difference_GPU<<<numBlocks, threadsPerBlock>>>( d_U, d_U_new,d_diff_U);
	
//    			cudaDeviceSynchronize();

  			cudaMemcpy(diff_U, d_diff_U, N*N*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(U_new, d_U_new, N*N*sizeof(float), cudaMemcpyDeviceToHost);
			cudaMemcpy(U, d_U, N*N*sizeof(float), cudaMemcpyDeviceToHost);

//    			cudaDeviceSynchronize();

			
			for (int i = 0;i < N; i++){
				for (int j = 0; j < N;j++){
					printf("%d %d %f\n",i,j,U[i*N+j]);
				}
			}

  //      		cudaErrorCheck( cudaThreadSynchronize() ); // Checks for execution error
			maxer = getmaxerror(diff_U);

			printf("maximm error %.6E\n",maxer);
			if((maxer == maxer2) ){
				wait++;
				if (wait > 100.){
					break;
				}
			}
		}


	}

  	//check for errors with the kernal
//  	cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
//  	cudaErrorCheck( cudaThreadSynchronize() ); // Checks for execution error

	//send the memory back to the cpu
  	cudaMemcpy(phi, d_U_new, N*N*sizeof(float), cudaMemcpyDeviceToHost);
  	cudaMemcpy(phi, d_U, N*N*sizeof(float), cudaMemcpyDeviceToHost);


  	//third step: update the velocity
	for (int ix = 1;ix < N-1;ix++){
		for (int iy = 1; iy < N-1;iy++){
			ux[ix*N+iy] = uintx[ix*N+iy] - (dt/rho)*(phi[(ix+1)*N+iy] - phi[(ix-1)*N+iy])/(2*dx);
			uy[ix*N+iy] = uinty[ix*N+iy] - (dt/rho)*(phi[ix*N+iy+1] - phi[ix*N+iy-1])/(2*dx);
	//		printf("%f %f\n",uintx[ix*N+iy],uinty[ix*N+iy]);
		
	

			//boundary conditions for u_intermediate
			if ( (ix == 0) | (iy == 0) | (ix == (N-1)) | (iy == (N-1))){
				uy[ix*N+iy] = 0.;
				ux[ix*N+iy] = 0.;
			}
			//boundary condition for outlet
			if (ix ==(N-1)){

		//uinty[(N)*N+iy] = uinty[(N-1)*N+iy];
				uy[(N)*N+iy] = -u0;
			}
			//boundary condition for the inlet
			if(ix == 0){
				uy[iy] = -u0;
			}
		}
	}



  	//fouth step: update the pressure

	for (int ix = 0; ix < N;ix++){
		for (int iy = 0; iy < N;iy++){
			p[ix*N+iy] = phi[ix*N+iy] + beta*p[ix*N+iy];
		}
	}

	printf("Iteration: %i\n",it);
	
	savemat(ux,"ux",it,N);
	savemat(uy,"uy",it,N);
	savemat(phi,"phi",it,N);
	savemat(F,"F",it,N);

	  }// end time iteration



	free(U);
	free(U_new);
	free(F);
	cudaFree(d_U);
	cudaFree(d_U_new);
	cudaFree(d_F);

}





