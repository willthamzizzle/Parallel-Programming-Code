#include <stdio.h>
#include <omp.h>

#define N 4096
#define W 32 
#define M N+32
#define NUM_THREADS 16

/*	Name:	William Thammavong
/	Date:	9/29/2010
/	Code including all methods of speeding up code
*/

main (int argc, char **argv) {
  float A[M][M], B[W][W], C[N][N], Corig[N][N];
  int i,j,k,l, test;
  int ii, jj, kk, ll;

  int tileRow = 8;
  int tileColumn = 8;

  int temp;

  double start, stop;

  omp_set_num_threads(NUM_THREADS);

  // Image and template are randomly generated.  
  for(i=0;i<M;i++) {
    for(j=0;j<M;j++) {
      A[i][j] = (float) rand()/179583191.593;
    }
  }
  for (i=0;i<W;i++) {
    for(j=0;j<W;j++) {
      B[i][j] = (float) rand()/179583191.593;
    }
  }

  // Baseline computation for determining correctness.
  for (l=0; l<N; l++) {
     for (k=0; k<N; k++) {
       Corig[k][l] = 0.0;
       for (j=0; j<W; j++) {
          for (i=0; i<W; i++) {
             Corig[k][l] += A[k+i][l+j]*B[i][j];
          }
       }
    }
  }

  // Now we time the optimized version.
  // TODO: Optimize this code for SIMD, locality and parallelism.
  start = omp_get_wtime();

#pragma omp parallel for
  for (k=0; k<N; k+=tileColumn) 
  {
	  for (l=0; l<N; l+=tileRow) 
	  {
		  for (ll=l; ll<min(l+tileRow-1, N);ll++)
		  {
			  for (kk=k; kk<min(k+tileColumn-1, N); kk++)
			  {
				  for (i=0; i<W; i++) 
				  {
					  for (j=0; j<W; j++) 
					  {
						  C[kk][ll] += A[kk+i][ll+j]*B[i][j];
					  }
				  }
			  }
		  }
	  }
  }

  stop = omp_get_wtime();

  printf("Execution time=%lf\n", stop-start);

  // Did optimization change the answer?  Test for correctness.
  test = 1;
  for (i=0;i<N;i++) {
    for (j=0;j<N;j++) {
      if ((Corig[i][j] - C[i][j] > 0.5) || (C[i][j] - Corig[i][j] > 0.5)) {
	test = 0;
	printf("TEST FAILING at C[%d][%d] = %f, %f\n", i,j,Corig[i][j],C[i][j]);
        break;
      }
      if (!test) break;
    }
  }
  if (!test) {
    printf("TEST FAILED\n");
  }
  else {
      printf("TEST PASSED\n");
  }
}
