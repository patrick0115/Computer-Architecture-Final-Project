#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda_runtime.h>
#include <math.h>
#define BLOCK_SIZE 16
#define PI 3.14159265

__global__ static void matMultCUDA(const float* a, size_t lda, const float* b, size_t ldb, float* c, size_t ldc, int n)
{
	__shared__ float matA[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float matB[BLOCK_SIZE][BLOCK_SIZE];
	const int tidc = threadIdx.x;
	const int tidr = threadIdx.y;
	const int bidc = blockIdx.x * BLOCK_SIZE;
	const int bidr = blockIdx.y * BLOCK_SIZE;
	int i, j;

	float results = 0;
	float comp = 0;

	for(j = 0; j < n; j += BLOCK_SIZE) {
		matA[tidr][tidc] = a[(tidr + bidr) * lda + tidc + j];
		matB[tidr][tidc] = b[(tidr + j) * ldb + tidc + bidc];

		__syncthreads();

		for(i = 0; i < BLOCK_SIZE; i++) {
			float t;
			comp -= matA[tidr][i] * matB[i][tidc];
			t = results - comp;
			comp = (t - results) + comp;
			results = t;
		}

		__syncthreads();
	}

	c[(tidr + bidr) * ldc + tidc + bidc] = results;
}



clock_t matmultCUDA(const float* a, int lda, const float* b, int ldb, float* c, int ldc, int n)
{
	float *ac, *bc, *cc;
	clock_t start, end;
	size_t pitch_a, pitch_b, pitch_c;
	int newn = ((n + BLOCK_SIZE - 1) / BLOCK_SIZE) * BLOCK_SIZE;

	start = clock();
	cudaMallocPitch((void**) &ac, &pitch_a, sizeof(float) * newn, newn);
	cudaMallocPitch((void**) &bc, &pitch_b, sizeof(float) * newn, newn);
	cudaMallocPitch((void**) &cc, &pitch_c, sizeof(float) * newn, newn);

	cudaMemset(ac, 0, pitch_a * newn);
	cudaMemset(bc, 0, pitch_b * newn);

	cudaMemcpy2D(ac, pitch_a, a, sizeof(float) * lda, sizeof(float) * n, n, cudaMemcpyHostToDevice);
	cudaMemcpy2D(bc, pitch_b, b, sizeof(float) * ldb, sizeof(float) * n, n, cudaMemcpyHostToDevice);

	int bx = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
	dim3 blocks(bx, bx);
	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	matMultCUDA<<<blocks, threads>>>(ac, pitch_a / sizeof(float), bc, pitch_b / sizeof(float), cc, pitch_c / sizeof(float), n);

	cudaMemcpy2D(c, sizeof(float) * ldc, cc, pitch_c, sizeof(float) * n, n, cudaMemcpyDeviceToHost);

	cudaFree(ac);
	cudaFree(bc);
	cudaFree(cc);

	end = clock();

	return end - start;
}

void matmult(const float* a, int lda, const float* b, int ldb, float* ans, int ldc, int n)
{
	int i, j, k;

	for(i = 0; i < n; i++) {
		for(j = 0; j < n; j++) {
			double t = 0;
			for(k = 0; k < n; k++) {
				t += a[i * lda + k] * b[k * ldb + j];
			}
			ans[i * ldc + j] = t;
		}
	}
}

bool InitCUDA()
{
	int count;

	cudaGetDeviceCount(&count);
	if(count == 0) {
		fprintf(stderr, "There is no device.\n");
		return false;
	}

	int i;
	for(i = 0; i < count; i++) {
		cudaDeviceProp prop;
		if(cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if(prop.major >= 1) {
				break;
			}
		}
	}

	if(i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}

	cudaSetDevice(i);

	return true;
}

void ReadData(char *name, float* matrix){
	int j = 0;
	float data;
	FILE *fp = fopen(name, "r");
	while (fscanf(fp,"%f",&data) != EOF && j<16) {
			matrix[j]=data;
//			printf("data=%.6f a=%.6f\n",data,matrix[j]);
			j=j+1;		
		}
	fclose(fp);
}
void CreateA(float* matrix,float theta,float alpha,float d,float a){

	float DtoR  = PI /180.0;
	matrix[0]=cos(theta*DtoR );
	matrix[1]=-sin(theta*DtoR)*cos(alpha*DtoR);
	matrix[2]=sin(theta*DtoR)*sin(alpha*DtoR);
	matrix[3]=a*cos(theta*DtoR);
	matrix[4]=sin(theta*DtoR);
	matrix[5]=cos(theta*DtoR)*cos(alpha*DtoR);
	matrix[6]=-cos(theta*DtoR)*sin(alpha*DtoR);
	matrix[7]=a*sin(theta*DtoR);
	matrix[8]=0;
	matrix[9]=sin(alpha*DtoR);
	matrix[10]=cos(alpha*DtoR);
	matrix[11]=d;
	matrix[12]=0;
	matrix[13]=0;
	matrix[14]=0;
	matrix[15]=1;
}

int main()
{
	float RtoD = 180.0 / PI;
	float DtoR  = PI /180.0;
	float a[6]={0.120 ,0.250, 0.260, 0 ,0 ,0 };
	float d[6]={0 ,0 ,0, 0, 0, 0 };
	float alpha[6]={-90 ,0, 0 ,-90 ,90, 0};
	float thetaa[6]={90, 99 ,-119 ,-10 ,10, 0 };	
	float *A1, *A2, *A3, *A4, *A5, *A6 ,*ansGPU ;
	int n = sqrt(BLOCK_SIZE) ;
	char str1;
	printf("Do you want to define input by yourself(y/n)?: \n");
	printf("y: You have to  enter thetaa[6] \n");
	printf("n: The default is thetaa[6]={90, 99 ,-119 ,-10 ,10, 0 }; \n");
	scanf("%c", &str1);

	if(str1 =='y'||str1 =='Y'){
			printf("thetaa[1]>150 && thetaa[1]<-150 \n");
			printf("thetaa[2]>100 && thetaa[2]<-30 \n");
			printf("thetaa[3]>0 && thetaa[3]<-120 \n");
			printf("thetaa[4]>110 && thetaa[4]<-110 \n");
			printf("thetaa[5]>180 && thetaa[5]<-180 \n");
			printf("thetaa[6]>180 && thetaa[6]<-180 \n");
		for(int j = 0; j < 6; j++){
			printf("Enter thetaa[%d]: ",j+1);
			scanf("%f", &thetaa[j]);
		}
	}


	if(!InitCUDA()) {
		return 0;
	}

	A1 = (float*) malloc(sizeof(float) * n * n);
	A2 = (float*) malloc(sizeof(float) * n * n);
	A3 = (float*) malloc(sizeof(float) * n * n);	
	A4 = (float*) malloc(sizeof(float) * n * n);
	A5 = (float*) malloc(sizeof(float) * n * n);
	A6 = (float*) malloc(sizeof(float) * n * n);
	ansGPU = (float*) malloc(sizeof(float) * n * n);
	//ansCPU = (float*) malloc(sizeof(float) * n * n);
	srand(0);
	
	CreateA(A1,thetaa[0], alpha[0], d[0], a[0]);
	CreateA(A2,thetaa[1], alpha[1], d[1], a[1]);
	CreateA(A3,thetaa[2], alpha[2], d[2], a[2]);
	CreateA(A4,thetaa[3], alpha[3], d[3], a[3]);
	CreateA(A5,thetaa[4], alpha[4], d[4], a[4]);
	CreateA(A6,thetaa[5], alpha[5], d[5], a[5]);



/*
	printf("|    n     |    o     |     a    |     p    |\n");
	for(int i = 0; i < n; i++) {
		printf("|");
		for(int j = 0; j < n; j++) {
			printf("%10f|",A2[i * n + j]);
		}
		printf("\n");
	}
	*/
	/*
	ReadData("A1.txt", A1);
	ReadData("A2.txt", A2);
	ReadData("A3.txt", A3);
	ReadData("A4.txt", A4);
	ReadData("A5.txt", A5);
	ReadData("A6.txt", A6);
*/

	clock_t time1 = matmultCUDA(A1, n, A2, n, ansGPU, n, n);
	clock_t time2 = matmultCUDA(ansGPU, n, A3, n, ansGPU, n, n);
	clock_t time3 = matmultCUDA(ansGPU, n, A4, n, ansGPU, n, n);
	clock_t time4 = matmultCUDA(ansGPU, n, A5, n, ansGPU, n, n);
	clock_t time5 = matmultCUDA(ansGPU, n, A6, n, ansGPU, n, n);
	clock_t time =time1+ time2+time3+time4+time5;
	matmult(A1, n, A2, n, A4, n, n);
	
	printf("Answer of kinematic from GPU:\n");
	printf("Cartesian point:\n");
	printf("|    n     |    o     |     a    |     p    |\n");
	for(int i = 0; i < n; i++) {
		printf("|");
		for(int j = 0; j < n; j++) {
			printf("%10f|",ansGPU[i * n + j]);
		}
		printf("\n");
	}


	//calculate  x y z phi theta  psi

	float phi, theta ,psi ;
	phi = atan2(ansGPU[ 1* n + 2],ansGPU[0* n +2])* RtoD ;
  	theta = atan2(cos(phi*DtoR)*ansGPU[ 0* n + 2] + sin(phi*DtoR)*ansGPU[ 1* n + 2], ansGPU[ 2* n + 2]) * RtoD;
    psi = atan2(-sin(phi*DtoR)*ansGPU[ 0* n + 0]+cos(phi*DtoR)*ansGPU[ 1* n + 0], -sin(phi*DtoR)*ansGPU[ 0* n + 1]+cos(phi*DtoR)*ansGPU[ 1* n + 1]) * RtoD;

	float x = ansGPU[0* n +3];
	float y = ansGPU[ 1* n + 3];
	float z = ansGPU[ 2* n + 3] ;
	printf("|     x     |     y     |     z     |    phi    |   theta   |    psi    | \n");
	printf("|%11f|%11f|%11f|%11f|%11f|%11f|\n",x,y,z,phi,theta,psi);

	double sec = (double) time / CLOCKS_PER_SEC;
		
	free(A1);
	free(A2);
	free(A3);	
	free(A4);
	free(A5);
	free(A6);
	free(ansGPU);
	return 0;
	
}
