/* Problem 8: Deadlock demo and fix.
 * Compile: mpicc problem08_deadlock.c -o p08
 * Run safe version   : mpirun -np 2 ./p08
 * Run unsafe version : mpirun -np 2 ./p08 unsafe      (WILL HANG - press Ctrl+C)
 */
#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size != 2) { if (rank == 0) printf("Run with exactly 2 processes\n"); MPI_Finalize(); return 1; }

    int mine = (rank == 0) ? 100 : 200;
    int other, partner = 1 - rank;

    if (argc > 1 && strcmp(argv[1], "unsafe") == 0) {
        /* UNSAFE: both processes send first. MPI_Ssend only finishes when the
         * receiver has started its receive, but nobody reaches MPI_Recv -> deadlock.
         * (Plain MPI_Send may "accidentally" work for tiny messages because of
         *  internal buffering, so MPI_Ssend is used to make the problem certain.) */
        printf("P%d: Waiting to send %d\n", rank, mine); fflush(stdout);
        MPI_Ssend(&mine, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
        MPI_Recv(&other, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("P%d: never reached\n", rank);
    } else {
        /* FIX 1: break the symmetry - P0 sends then receives, P1 receives then sends. */
        if (rank == 0) {
            MPI_Send(&mine, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            printf("P0 sent %d\n", mine); fflush(stdout);
            MPI_Recv(&other, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("P0 received %d\n", other);
        } else {
            MPI_Recv(&other, 1, MPI_INT, partner, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            printf("P1 received %d\n", other); fflush(stdout);
            MPI_Send(&mine, 1, MPI_INT, partner, 0, MPI_COMM_WORLD);
            printf("P1 sent %d\n", mine);
        }
        MPI_Barrier(MPI_COMM_WORLD);
        if (rank == 0) printf("Communication completed without deadlock.\n");

        /* FIX 2 (alternative): MPI_Sendrecv does send+receive safely in one call. */
        int other2;
        MPI_Sendrecv(&mine, 1, MPI_INT, partner, 1,
                     &other2, 1, MPI_INT, partner, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (rank == 0) printf("(Also works with MPI_Sendrecv: P0 got %d)\n", other2);
    }
    MPI_Finalize();
    return 0;
}
