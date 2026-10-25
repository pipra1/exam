/* Problem 7: Two processes exchange arrays.
 * Version 1: MPI_Isend / MPI_Irecv / MPI_Waitall   (non-blocking)
 * Version 2: MPI_Send / MPI_Recv                   (blocking, ordered to avoid deadlock)
 * Both are timed and compared.
 * Compile: mpicc problem07_nonblocking.c -o p07
 * Run    : mpirun -np 2 ./p07            */
#include <mpi.h>
#include <stdio.h>
#define N 8
#define REPS 1000                             /* repeat to get measurable time */

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) { if (rank == 0) printf("Run with exactly 2 processes\n"); MPI_Finalize(); return 1; }

    int mine[N], other[N], result[N];
    int partner = 1 - rank;
    for (int i = 0; i < N; i++) mine[i] = (rank == 0) ? 10 * (i + 1) : i + 1;

    /* ---------- Non-blocking ---------- */
    MPI_Request req[2];
    MPI_Barrier(MPI_COMM_WORLD);
    double t0 = MPI_Wtime();
    for (int r = 0; r < REPS; r++) {
        MPI_Isend(mine,  N, MPI_INT, partner, 0, MPI_COMM_WORLD, &req[0]);  /* returns immediately */
        MPI_Irecv(other, N, MPI_INT, partner, 0, MPI_COMM_WORLD, &req[1]);  /* returns immediately */
        MPI_Waitall(2, req, MPI_STATUSES_IGNORE);                           /* wait for both */
    }
    double t_nb = (MPI_Wtime() - t0) / REPS;

    /* ---------- Blocking ---------- */
    MPI_Barrier(MPI_COMM_WORLD);
    t0 = MPI_Wtime();
    for (int r = 0; r < REPS; r++) {
        if (rank == 0) {                      /* P0: send then receive */
            MPI_Send(mine,  N, MPI_INT, partner, 1, MPI_COMM_WORLD);
            MPI_Recv(other, N, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        } else {                              /* P1: receive then send (order avoids deadlock) */
            MPI_Recv(other, N, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            MPI_Send(mine,  N, MPI_INT, partner, 1, MPI_COMM_WORLD);
        }
    }
    double t_b = (MPI_Wtime() - t0) / REPS;

    /* ---------- Use the received data: element-wise vector addition ---------- */
    for (int i = 0; i < N; i++) result[i] = mine[i] + other[i];

    for (int p = 0; p < 2; p++) {             /* print in rank order */
        if (rank == p) {
            printf("P%d received:", rank);
            for (int i = 0; i < N; i++) printf(" %d", other[i]);
            printf("\nP%d vector sum:", rank);
            for (int i = 0; i < N; i++) printf(" %d", result[i]);
            printf("\n"); fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }
    if (rank == 0) {
        printf("Nonblocking communication completed.\n");
        printf("Blocking communication time    = %.9f s (avg of %d runs)\n", t_b, REPS);
        printf("Nonblocking communication time = %.9f s (avg of %d runs)\n", t_nb, REPS);
    }
    MPI_Finalize();
    return 0;
}
