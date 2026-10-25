/* Problem 3: Ring topology. Every process sends to (rank+1) and receives from (rank-1).
 * Part A: pass the value 100 around the ring.
 * Part B: time the ring for different message sizes.
 * Compile: mpicc problem03_ring.c -o p03
 * Run    : mpirun -np 4 ./p03            */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) { if (rank == 0) printf("Need >= 2 processes\n"); MPI_Finalize(); return 1; }

    int next = (rank + 1) % size;             /* wrap-around gives the "ring" */
    int prev = (rank - 1 + size) % size;

    /* ---------- Part A: pass the token 100 around ---------- */
    int value;
    if (rank == 0) {                          /* P0 starts: send first, receive last */
        value = 100;
        printf("P0 sends %d to P1\n", value); fflush(stdout);
        MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
        MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("P0 receives %d from P%d\n", value, prev);
        printf("Ring communication completed.\n"); fflush(stdout);
    } else {                                  /* others: receive first, then forward */
        MPI_Recv(&value, 1, MPI_INT, prev, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("P%d receives %d from P%d\n", rank, value, prev);
        printf("P%d sends %d to P%d\n", rank, value, next); fflush(stdout);
        MPI_Send(&value, 1, MPI_INT, next, 0, MPI_COMM_WORLD);
    }

    /* ---------- Part B: timing for different message sizes ---------- */
    int sizes[] = {8, 1024, 16384, 262144, 1048576};      /* bytes */
    int nsizes = sizeof(sizes) / sizeof(sizes[0]);
    const int ITER = 100;
    if (rank == 0) printf("\n%12s %22s\n", "Bytes", "Time per ring trip (s)");

    for (int s = 0; s < nsizes; s++) {
        char *buf = malloc(sizes[s]);
        MPI_Barrier(MPI_COMM_WORLD);
        double t0 = MPI_Wtime();
        for (int it = 0; it < ITER; it++) {
            if (rank == 0) {
                MPI_Send(buf, sizes[s], MPI_CHAR, next, 1, MPI_COMM_WORLD);
                MPI_Recv(buf, sizes[s], MPI_CHAR, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else {
                MPI_Recv(buf, sizes[s], MPI_CHAR, prev, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(buf, sizes[s], MPI_CHAR, next, 1, MPI_COMM_WORLD);
            }
        }
        double t = (MPI_Wtime() - t0) / ITER;
        if (rank == 0) printf("%12d %22.9f\n", sizes[s], t);
        free(buf);
    }
    MPI_Finalize();
    return 0;
}
