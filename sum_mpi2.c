/*
 *  sum_mpi2.c - Demonstrates parallelism via random fill and sum routines
 *               This program uses MPI.
 */

/*---------------------------------------------------------
 *  Parallel Summation 
 *   
 *  1. each processor generates numints random integers (in parallel)
 *  2. each processor sums his numints random integers (in parallel)
 *  3  MPI_Allreduce is used to combine the partial results
 *
 *  NOTE: steps 2-3 are repeated as many times as requested (numiterations)
 *---------------------------------------------------------*/
#include <stdlib.h>
#include <time.h>
#include <sys/resource.h>
#include <mpi.h>

/*==============================================================
 * p_generate_random_ints (processor-wise generation of random ints)
 *==============================================================*/
void p_generate_random_ints(int* memory, int n) {

  int i;

  /* generate & write this processor's random integers */
  for (i = 0; i < n; ++i) {
  
    memory[i] = rand();
  }

  return;
} 

/*==============================================================
 * p_summation (processor-wise summation of ints)
 *==============================================================*/
long p_summation(int* memory, int n) {

  int i;
  long sum;

  /* sum this processor's integers */
  sum = 0; /* init sum to zero */
  for (i = 0; i < n; ++i) {
  
    sum += memory[i];
  }
  return sum;
} 

/*==============================================================
 * print_elapsed (prints timing statistics)
 *==============================================================*/
 void print_elapsed(char* desc, struct timeval* start, struct timeval* end, int niters) {

  struct timeval elapsed;
  /* calculate elapsed time */
  if(start->tv_usec > end->tv_usec) {
  
    end->tv_usec += 1000000;
    end->tv_sec--;
  }
  elapsed.tv_usec = end->tv_usec - start->tv_usec;
  elapsed.tv_sec  = end->tv_sec  - start->tv_sec;

  printf("\n %s total elapsed time = %ld (usec)",
    desc, (elapsed.tv_sec*1000000 + elapsed.tv_usec) / niters);
}

/*==============================================================
 *  Main Program (Parallel Summation)
 *==============================================================*/
int main(int argc, char **argv) {

  int nprocs, numints, numiterations; /* command line args */
  
  int my_id, iteration;

  long sum;  
  long total_sum;       /* Total sum  */
  int* mymemory;        /* pointer to processes memory     */

  struct timeval gen_start, gen_end; /* gettimeofday stuff */
  struct timeval start, end;         /* gettimeofday stuff */
  struct timezone tzp;

  MPI_Status status;              /* Status variable for MPI operations */

  /*---------------------------------------------------------
   * Initializing the MPI environment
   * "nprocs" copies of this program will be initiated by MPI.
   * All the variables will be private, only the program owner could see its own variables
   * If there must be a inter-procesor communication, the communication must be 
   * done explicitly using MPI communication functions.
   *---------------------------------------------------------*/

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_id); /* Getting the ID for this process */

  /*---------------------------------------------------------
   *  Read Command Line
   *  - check usage and parse args
   *---------------------------------------------------------*/
   
  if(argc < 3) {
  
    if(my_id == 0)
      printf("Usage: %s [numints] [numiterations]\n\n", argv[0]);
      
    MPI_Finalize();
    exit(1);
  }

  numints       = atoi(argv[1]);
  numiterations = atoi(argv[2]);

  MPI_Comm_size(MPI_COMM_WORLD, &nprocs); /* Get number of processors */
  
  if(my_id == 0)
    printf("\nExecuting %s: nprocs=%d, numints=%d, numiterations=%d\n", 
            argv[0], nprocs, numints, numiterations);

  /*---------------------------------------------------------
   *  Initialization 
   *  - allocate memory for work area structures and work area
   *---------------------------------------------------------*/

  mymemory = (int *) malloc(sizeof(int) * numints);

  if(mymemory == NULL) {
  
    printf("Processor %d - unable to malloc()\n", my_id);
    MPI_Finalize();    
    exit(1);
  }

  /* get starting time */
  gettimeofday(&gen_start, &tzp);
  srand(my_id + time(NULL));                  /* Seed rand functions */
  p_generate_random_ints(mymemory, numints);  /* random parallel fill */
  gettimeofday(&gen_end, &tzp);
  
  if(my_id == 0) {
  
    print_elapsed("Input generated", &gen_start, &gen_end, 1);
  }

  MPI_Barrier(MPI_COMM_WORLD); /* Global barrier */

  gettimeofday(&start, &tzp);

  /* repeat for numiterations times */
  for (iteration = 0; iteration < numiterations; iteration++) { 

    sum = p_summation(mymemory, numints); /* Compute the local summation */

    MPI_Allreduce(&sum, &total_sum, 1, MPI_LONG, MPI_SUM, MPI_COMM_WORLD); 
  }
    
  if(my_id == 0) {
  
    gettimeofday(&end, &tzp);     
    print_elapsed("Summation", &start, &end, numiterations);
    printf("\n Total sum = %6ld\n", total_sum);
  }

  /*---------------------------------------------------------
   *  Cleanup 
   *  - free memory
   *---------------------------------------------------------*/

  /* free memory */

  free(mymemory);
  MPI_Finalize();
  return 0;
}
