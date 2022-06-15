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

int main()
{
	
	float *a, *b, *c, *d, *e, *f ,*ansGPU ;
	int n = sqrt(BLOCK_SIZE) ;

	if(!InitCUDA()) {
		return 0;
	}

	a = (float*) malloc(sizeof(float) * n * n);
	b = (float*) malloc(sizeof(float) * n * n);
	c = (float*) malloc(sizeof(float) * n * n);	
	d = (float*) malloc(sizeof(float) * n * n);
	e = (float*) malloc(sizeof(float) * n * n);
	f = (float*) malloc(sizeof(float) * n * n);
	ansGPU = (float*) malloc(sizeof(float) * n * n);
	//ansCPU = (float*) malloc(sizeof(float) * n * n);
	srand(0);    
	
	int i=0;
	int j=0;
	
	ReadData("A1.txt", a);
	ReadData("A2.txt", b);
	ReadData("A3.txt", c);
	ReadData("A4.txt", d);
	ReadData("A5.txt", e);
	ReadData("A6.txt", f);

	clock_t time1 = matmultCUDA(a, n, b, n, ansGPU, n, n);
	clock_t time2 = matmultCUDA(ansGPU, n, c, n, ansGPU, n, n);
	clock_t time3 = matmultCUDA(ansGPU, n, d, n, ansGPU, n, n);
	clock_t time4 = matmultCUDA(ansGPU, n, e, n, ansGPU, n, n);
	clock_t time5 = matmultCUDA(ansGPU, n, f, n, ansGPU, n, n);
	clock_t time =time1+ time2+time3+time4+time5;
	matmult(a, n, b, n, d, n, n);
	
	printf("Answer of kinematic from GPU:\n");
	printf("Cartesian point:\n");
	printf("|    n     |    o     |     a    |     p    |\n");
	for(i = 0; i < n; i++) {
		printf("|");
		for(j = 0; j < n; j++) {
			printf("%10f|",ansGPU[i * n + j]);
		}
		printf("\n");
	}

	free(a);
	free(b);
	free(c);	
	free(d);
	free(e);
	free(f);
	free(ansGPU);
	//calculate  x y z phi theta  psi
	float RtoD = 180.0 / PI;
	float DtoR  = PI /180.0;
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
	printf("Time used: %lf   (%lf GFLOPS)\n", sec, 2.0 * n * n * n / (sec * 1E9));
	return 0;
}
