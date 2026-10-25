/* Problem 4: Ping-pong between 2 processes: latency and bandwidth.
 * Compile: mpicc problem04_pingpong.c -o p04
 * Run    : mpirun -np 2 ./p04            */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#define ITER 5          /* sample uses 5; use 1000 for more stable numbers */

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) { if (rank == 0) printf("Run with exactly 2 processes\n"); MPI_Finalize(); return 1; }

    int sizes[] = {64, 256, 1024, 4096, 16384, 65536, 1048576};   /* bytes */
    int n = sizeof(sizes) / sizeof(sizes[0]);

    for (int s = 0; s < n; s++) {
        char *buf = calloc(sizes[s], 1);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int i = 0; i < ITER; i++) {
            if (rank == 0) {                  /* ping */
                MPI_Send(buf, sizes[s], MPI_CHAR, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(buf, sizes[s], MPI_CHAR, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {                          /* pong */
                MPI_Recv(buf, sizes[s], MPI_CHAR, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf, sizes[s], MPI_CHAR, 0, 0, MPI_COMM_WORLD);
            }
        }
        double total = MPI_Wtime() - t0;
        if (rank == 0) {
            double rtt = total / ITER;        /* average round-trip time */
            double one = rtt / 2.0;           /* one-way = half the round trip */
            printf("Message size       : %d bytes\n", sizes[s]);
            printf("Iterations         : %d\n", ITER);
            printf("Total round trips  : %d\n", ITER);
            printf("Average round-trip : %.9f seconds\n", rtt);
            printf("Average one-way    : %.9f seconds\n", one);
            printf("Bandwidth          : %.2f MB/s\n\n", sizes[s] / one / 1e6);
        }
        free(buf);
    }
    MPI_Finalize();
    return 0;
}
