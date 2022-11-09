#include <stdio.h>
#include <time.h>

const int N = 128;
const int length = 32;

#define cudaErrorCheck(call) { cudaAssert(call,__FILE__,__LINE__); }

const double errortol = 1e-15;
const int numtimesteps = 100;
const double beta = 0.0;

const double rho = 1e-3;
const double nu = 6e1;
const double g = 0e-5;


const double dt = 5e-15;
const double dx = 1e-6;

const double u0 = 2e3;
const double u0x = 0e1;
const double phi0 = 1e8;

void cudaAssert(const cudaError err, const char *file, const int line)
{ 
    if( cudaSuccess != err) {                                                
        fprintf(stderr, "Cuda error in file '%s' in line %i : %s.\n",        
                file, line, cudaGetErrorString(err) );
        exit(1);
    } 
}


__global__
void jacobi_iterator_GPU(double * U_new, double* U, double * F ,int num)
{
 
    int j = blockDim.x * blockIdx.x + threadIdx.x ;
    int i = blockDim.y * blockIdx.y + threadIdx.y ;


//    int i = blockDim.x * blockIdx.x + threadIdx.x ;
//    int j = blockDim.y * blockIdx.y + threadIdx.y ;

   if ((i > N) || (j > length)) return;

	//only apply to every other element
	if( (j*N+i) %2 != num){

		if ((i < N -1) && (j < length -1) && (i > 0) && (j > 0))
		{
			U_new[j * N + i] = (U[j * N + (i - 1)] + U[j * N + (i + 1)] + U[(j - 1) * N + i] + U[(j + 1) * N + i]  ) * 0.25 - (F[j*N+i]*.25);
		}
		}else{

		U_new[j*N+i] = U[j*N+i];
		//U_new[j*N+i] = 100000;
	}


	    //boundary conditions (set the normal to zero)
	if ((i == 0 ))  {
		U_new[j*N+i] = U[j*N+i+1];
	}
	if ((i == (N-1) )){  
		U_new[j*N+i] = U[j*N+i-1];
	}
	if ((j == 0) && (i !=0) && (i!=(N-1))){ //don't apply to the corners let them be set by the y direction
		U_new[j*N+i] = U[(j+1)*N+i];
	}
	if ((j == (length-1)) &&  (i !=0) && (i!=(N-1))){
		U_new[j*N+i] = U[(j-1)*N+i];
	}

	//set the "ground"
	if ((i == 0)) {
		U_new[j*N+i] = 0;
	}//only need to set the pressure at the output?	
	if (i == (N-1)){
		
		//U_new[j*N+i] = dx*N*g;
		U_new[j*N+i] = phi0;
	}



    
}
__global__
void difference_GPU(double * U_new, double* U, double* diff_U)
{

	//compute the elementwise subtraction of two matrices
	//to be used for finding the maximum error in the poisson calculation

    int i = blockDim.x * blockIdx.x + threadIdx.x ;
    int j = blockDim.y * blockIdx.y + threadIdx.y ;

    //only access the internal elements of the array for now
    if ((i < length -1) && (j < N -1) && (i > 0) && (j > 0))
    {
       diff_U[j*N+i] = U_new[j * N + i] -  U[j*N+i];
    }
    else
    {
        diff_U[j*N + i] = 0.;
    }
}

double getmaxerror(double * A,double * B){
	double maxerror = 0.;
	for (int i = 0; i < length;i++){
		for (int j = 0; j < N;j++){
			double mydiff = abs(A[i*N+j] - B[i*N+j]);
			if (maxerror < mydiff){
				
		//		printf("maxerror: %f diffmat %f %i %i\n",maxerror,diffmat[i*N+j],i,j);
				maxerror = mydiff;
			}
		}
	}
	return maxerror;

}

void savemat(double * matrix,const char * name,int it){

	char fn[100+1];
    
	snprintf(fn, 100, "./data/%s%04d.dat", name,it);

	FILE *f = fopen(fn, "w");
	if (f == NULL)
	{
		printf("Error opening file!\n");
		exit(1);
	}

	/* print integers and floats */
	for (int ix = 0;ix < length;ix++){
		for (int iy = 0; iy < N;iy++){
			fprintf(f, "%i %i %.17g\n", ix,iy,matrix[ix*N+iy]);
		}
	}
	
	fclose(f);
}


int main(void)
{

  //int N = MATRIX_SIZE;




  //initialize velocity
  double *ux,*uy,*uintx,*uinty;
  ux = (double*)malloc(N*length*sizeof(double));
  uy = (double*)malloc(N*length*sizeof(double));
  uintx = (double*)malloc(N*length*sizeof(double));
  uinty = (double*)malloc(N*length*sizeof(double));

  for (int i = 0; i < length;i++){
	  for (int j = 0; j < N; j++){
		  
	  	ux[i*N+j] = u0x;
		
	//  	uy[i*N+j] = -1.*u0;
	  	uy[i*N+j] = -1.*u0*(length-i)*(i)/length/length;
	  }
  }

  //initialize pressure
  double *p;
  p = (double*)malloc(N*length*sizeof(double));
  for (int i = 0; i < N*length;i++){
	  p[i] = phi0;
  }

  //initialize phi
  double *phi;
  phi = (double*)malloc(N*length*sizeof(double));
  for (int i = 0; i < N*length;i++){
	  phi[i] = phi0;
  }

  //initializae divu
  double *divu;
  divu = (double*)malloc(N*length*sizeof(double));
  for (int i = 0; i < N*length;i++){
	  divu[i] = 0.;
  }

  	//allocate memory for the U_new
	double *U_new,*U,*F,*diff_U;
	double *d_U_new,*d_U,*d_F;

	U_new = (double*)malloc(N*length*sizeof(double));
	U = (double*)malloc(N*length*sizeof(double));
	F = (double*)malloc(N*length*sizeof(double));
	diff_U = (double*)malloc(N*length*sizeof(double));

  	cudaMalloc(&d_U_new,N*length*sizeof(double));
  	cudaMalloc(&d_U,N*length*sizeof(double));
  	cudaMalloc(&d_F,N*length*sizeof(double));




  	//TODO: add some method to add some numbers to my matrix

  	for (int i = 0; i < N*length;i++) {
		F[i] = 0.;
		U[i] = 4.0;
		U_new[i] = 5.0;
		diff_U[i] = 10.0;
	}




  //begin time iteration
  for(int it = 0;it < numtimesteps; it++){ //begin time iteration

	  //first step: calculate the intermediate velocity

	for (int ix = 1; ix < length-1;ix++){
		for (int iy = 1; iy < N-1;iy++){



			double ax = -dt*ux[ix*N+iy] * (ux[(ix+1)*N+iy]-ux[(ix-1)*N+iy])/(2*dx); 
			double bx = -beta*dt*(p[(ix+1)*N+iy]-p[(ix-1)*N+iy])/(2*dx*rho); 
			double cx = dt*nu*(ux[(ix-1)*N+iy] - 2*ux[ix*N+iy] + ux[(ix+1)*N+iy])/(dx*dx);
			double ddx = dt*0.0;
	

			
			uintx[ix*N+iy] = ux[ix*N+iy] + ax + bx + cx + ddx;
			//uintx[ix*N+iy] = 0.;


			double ay = -dt*uy[ix*N+iy] * (uy[ix*N+iy+1]-uy[ix*N+iy-1])/(2*dx); 
			double by = -beta*dt*(p[ix*N+iy+1]-p[ix*N+iy-1])/(2*dx*rho); 
			double cy = dt*nu*(uy[ix*N+iy-1] - 2*uy[ix*N+iy] + uy[ix*N+iy+1])/(dx*dx);
			double dy = -dt*g;

			//printf("ix: %d iy: %d ay: %.5e by: %.5e cy: %.5e dy: %.5e, uy: %.5e F: %.5e\n",ix,iy,ay,by,cy,dy,uy[ix*N+iy+1],F[ix*N+iy]);
			uinty[ix*N+iy] = uy[ix*N+iy] + ay + by + cy + dy;

			}
	}

			//boundary conditions for u_intermediate
	for (int ix = 0; ix < length;ix++){
		for (int iy = 0; iy < N;iy++){

			//set normals to left and right to be zero
			if ( (ix == 0)){
				//uinty[ix*N+iy] = uinty[(ix+1)*N+iy];
				uintx[ix*N+iy] = 0.;
				uinty[ix*N+iy] = 0.;
			}
			if ( (ix == (length-1))){
				//uinty[ix*N+iy] = uinty[(ix-1)*N+iy] ;
				uintx[ix*N+iy] = 0.;
				uinty[ix*N+iy] = 0.; 
			}
			//boundary condition for inlet
			if (iy ==(0)){

				uinty[(ix)*N+iy] = uinty[(N-1)*N+iy];
				//uinty[(N)*ix+iy] = -u0;
				uintx[N*ix+iy] = 0.;
			}
			//boundary condition for the outlet 
			if(iy == (N-1)){
				uinty[N*ix+iy] = uinty[N*ix+iy-1];
			       uintx[N*ix+iy] = 0.;	
			
			}

			
//			printf("ix: %d iy: %d uinty: %.5e uintx: %.5e\n",ix,iy,uinty[ix*N+iy],uintx[ix*N+iy]);
		  }
	  }


	//second step: calculate the poisson equation to solve for phi


	//define F
	for (int ix = 1; ix < length-1;ix++){
		for (int iy = 1; iy < N-1;iy++){

			F[ix*N+iy] = .5*(dx/dt)*rho * ( uintx[(ix+1)*N+iy]- uintx[(ix-1)*N+iy]  
				     + uinty[ix*N+iy+1] - uinty[ix*N+iy-1]	);

			//printf("ix: %d iy: %d F: %.5e uinty: %.5e uintx: %.5e \n",ix,iy,F[ix*N+iy],uinty[ix*N+iy],uintx[ix*N+iy]);
		}

	}


	//boundary condition for F (first attempt)
	for (int ix = 0; ix < length;ix++){
		for (int iy = 0; iy < N;iy++){


			//boundary conditions for u_intermediate

			//set normals to left and right to be zero
			if ( (ix == 0)){

				F[ix*N+iy] = .5*(dx/dt)*rho * ( uintx[(ix+1)*N+iy]  );

				//F[ix*N+iy] = F[(ix+1)*N+iy];
				//F[ix*N+iy] = 0;
			}
			if ( (ix == (length-1))){
				//F[ix*N+iy] = F[(ix-1)*N+iy] ;
				F[ix*N+iy] = .5*(dx/dt)*rho * ( uintx[(ix-1)*N+iy]  );
				//F[ix*N+iy] = 0;
			}
			//boundary condition for inlet
			if (iy ==(0)){

				F[ix*N+iy] = .5*(dx/dt)*rho * ( uinty[(ix)*N+iy+1] ) ;
				//F[ix*N+iy] = 0;
				//F[(N)*N+iy] = uinty[(N-1)*N+iy];
				//F[(N)*ix+iy] = -u0;
				F[N*ix+iy] = F[N*ix+iy+1];
			}
			//boundary condition for the outlet 
			if(iy == (N-1)){

				F[ix*N+iy] = .5*(dx/dt)*rho * ( uinty[(ix)*N+iy-1]  );
				//F[N*ix+iy] = F[N*ix+iy-1];
			
				//F[ix*N+iy] = 0;
			}

			
//			printf("ix: %d iy: %d uinty: %.5e uintx: %.5e\n",ix,iy,uinty[ix*N+iy],uintx[ix*N+iy]);
		  }
	  }

	

  	//send over the data to the gpu
	//cudaMemcpy(d_U_new, U_new, N*N*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_U, phi, N*length*sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(d_F, F, N*length*sizeof(double), cudaMemcpyHostToDevice);
  
	//check for error
	cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
	cudaErrorCheck( cudaThreadSynchronize() ); // Checks for execution error


	dim3 threadsPerBlock(16, 16);
	const int bx = (length + threadsPerBlock.x - 1)/threadsPerBlock.x; 
	const int by = (N + threadsPerBlock.y - 1)/threadsPerBlock.y;
	//dim3 numBlocks(N / threadsPerBlock.x, length / threadsPerBlock.y);
	dim3 numBlocks(bx,by);

	//loop for jacobi iterator
	//run until convergence set by errortol
	int jac = 0;
	double maxer = 1000.;
	double maxer2 = 100.;
	
	int wait = 0.;
	while( maxer > errortol){
		jac++;
	   

		//run twice with the pointers swapped to avoid having to copy memory 
 
		jacobi_iterator_GPU<<<numBlocks, threadsPerBlock>>>( d_U_new, d_U, d_F ,0);
		cudaDeviceSynchronize();
		jacobi_iterator_GPU<<<numBlocks, threadsPerBlock>>>( d_U, d_U_new, d_F ,1);
		cudaDeviceSynchronize();


		jacobi_iterator_GPU<<<numBlocks, threadsPerBlock>>>( d_U_new, d_U, d_F ,4);
		cudaDeviceSynchronize();
		jacobi_iterator_GPU<<<numBlocks, threadsPerBlock>>>( d_U, d_U_new, d_F ,4);
		cudaDeviceSynchronize();

   		//every 20 iteration check if the convergence condition has been met
    		if ((jac%1000) == 0){
			maxer2 = maxer;
	
			cudaMemcpy(U_new, d_U_new, N*length*sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(U, d_U, N*length*sizeof(double), cudaMemcpyDeviceToHost);

    			cudaDeviceSynchronize();

			
			for (int i = 0;i < length; i++){
				for (int j = 0; j < N;j++){
//					printf("%d %d %f\n",i,j,U[i*N+j]);
				}
			}

        		cudaErrorCheck( cudaThreadSynchronize() ); // Checks for execution error
			maxer = getmaxerror(U,U_new);

			printf("maximm error %.6E\n",maxer);
			//backup if error doesn't reduce all the way. if in 10 iterations the error doesn't change exit loop
			if((maxer == maxer2) ){
				wait++;
				if (wait > 10.){
					break;
				}
			}
		}


	}

  	//check for errors with the kernal
  	cudaErrorCheck( cudaPeekAtLastError() ); // Checks for launch error
  	cudaErrorCheck( cudaThreadSynchronize() ); // Checks for execution error

	//send the memory back to the cpu
//  	cudaMemcpy(phi, d_U_new, N*N*sizeof(double), cudaMemcpyDeviceToHost);
  	cudaMemcpy(phi, d_U, N*length*sizeof(double), cudaMemcpyDeviceToHost);


  	//third step: update the velocity
	for (int ix = 1;ix < length-1;ix++){
		for (int iy = 1; iy < N-1;iy++){
			ux[ix*N+iy] = uintx[ix*N+iy] - (dt/rho)*(phi[(ix+1)*N+iy] - phi[(ix-1)*N+iy])/(2*dx);
			uy[ix*N+iy] = uinty[ix*N+iy] - (dt/rho)*(phi[ix*N+iy+1] - phi[ix*N+iy-1])/(2*dx);
			//printf("ix: %d iy: %d ux: %.5e uy: %.5e phi: %.5e\n",ix,iy,ux[ix*N+iy],uy[ix*N+iy],phi[ix*N+iy]);
		}
	}		
	
	for (int ix = 0;ix < length;ix++){
		for (int iy = 0; iy < N;iy++){



			//set normals to left and right to be zero
			if ( (ix == 0)){
				//uy[ix*N+iy] = uy[(ix+1)*N+iy];
				uy[ix*N+iy] = 0.;
				ux[ix*N+iy] = 0.;
			}
			if ( (ix == (length-1))){
				uy[ix*N+iy] = 0.;
				//uy[ix*N+iy] = uy[(ix-1)*N+iy] ;
				ux[ix*N+iy] = 0.;
			}
			//boundary condition for inlet
			if (iy ==(0)){

				uy[(ix)*N+iy] = uy[(ix)*N+iy+1];
				//uy[N*ix+iy] = -u0;
				ux[N*ix+iy] = 0.;
			}
			//boundary condition for the outlet 
			if(iy == (N-1)){
				uy[N*ix+iy] = uy[N*ix+iy-1];
				ux[N*ix+iy] = 0.;	
			
			}


/*
			//boundary conditions for u_intermediate
			if ( (ix == 0) | (iy == 0) | (ix == (N-1)) | (iy == (N-1))){
				uy[ix*N+iy] = -u0;
				ux[ix*N+iy] = 0.;
			}
			//boundary condition for outlet
			if (ix ==(N-1)){

		//uinty[(N)*N+iy] = uinty[(N-1)*N+iy];
				uy[(ix)*N+iy] = -u0;
			}
			//boundary condition for the inlet
			if(ix == 0){
				uy[iy] = -u0;
			}*/
		}
	}



  	//fouth step: update the pressure

	for (int ix = 0; ix < length;ix++){
		for (int iy = 0; iy < N;iy++){
			p[ix*N+iy] = phi[ix*N+iy] + beta*p[ix*N+iy];
		}
	}
/*
	//calulate divu to check that its small
	for (int ix = 1; ix < (N-1);ix++){
		for (int iy = 1; iy < (N-1);iy++){
			divu[ix*N+iy] = ux[(ix+1)*N+iy]-ux[(ix-1)*N+iy]+uy[ix*N+iy+1]-uy[ix*N+iy-1];
		}
	}
*/


	printf("Iteration: %i\n",it);
	
	savemat(ux,"ux",it);
	savemat(uy,"uy",it);
	savemat(phi,"phi",it);
	savemat(F,"F",it);
	savemat(divu,"divu",it);

	  }// end time iteration

	printf("Ended Iteration\n");

	free(U);
	free(U_new);
	free(F);
	cudaFree(d_U);
	cudaFree(d_U_new);
	cudaFree(d_F);

}





