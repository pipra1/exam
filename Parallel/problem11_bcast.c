/* Problem 11: Root reads parameters and broadcasts them with MPI_Bcast.
 * Compile: mpicc problem11_bcast.c -o p11
 * Run    : mpirun -np 4 ./p11                      (uses defaults 1000 50 256)
 *          mpirun -np 4 ./p11 1000 50 256          (or give values on the command line) */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int params[3] = {0, 0, 0};                 /* [0]=iterations [1]=threshold [2]=block size */
    if (rank == 0) {                           /* only the root "reads" the configuration */
        params[0] = (argc > 3) ? atoi(argv[1]) : 1000;
        params[1] = (argc > 3) ? atoi(argv[2]) : 50;
        params[2] = (argc > 3) ? atoi(argv[3]) : 256;
    }

    /* After this call EVERY process holds the root's three values. */
    MPI_Bcast(params, 3, MPI_INT, 0, MPI_COMM_WORLD);

    int iterations = params[0], threshold = params[1], block = params[2];

    /* Use the parameters: count iterations whose position inside a block >= threshold. */
    long count = 0;
    for (int i = 0; i < iterations; i++)
        if ((i % block) >= threshold) count++;

    for (int p = 0; p < size; p++) {           /* print in rank order */
        if (rank == p) {
            printf("P%d received: Iterations = %d, Threshold = %d, Block size = %d  -> result = %ld\n",
                   rank, iterations, threshold, block, count);
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
