/* Problem 2: Root sends one matrix row to each worker with MPI_Send.
 * Each process sums its row and sends the sum back (worker MPI_Send -> root MPI_Recv).
 * Compile: mpicc problem02_row_sum.c -o p02
 * Run    : mpirun -np 4 ./p02            */
#include <mpi.h>
#include <stdio.h>
#define N 4                                   /* 4x4 matrix, 4 processes */

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size != N) {                          /* one row per process */
        if (rank == 0) printf("Please run with exactly %d processes.\n", N);
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        int M[N][N], sums[N], total = 0;
        for (int i = 0; i < N; i++)           /* fill matrix 1..16 */
            for (int j = 0; j < N; j++) M[i][j] = i * N + j + 1;

        for (int p = 1; p < N; p++)           /* send row p to process p */
            MPI_Send(M[p], N, MPI_INT, p, 0, MPI_COMM_WORLD);

        sums[0] = 0;                          /* root keeps row 0 for itself */
        for (int j = 0; j < N; j++) sums[0] += M[0][j];

        for (int p = 1; p < N; p++)           /* collect the other sums */
            MPI_Recv(&sums[p], 1, MPI_INT, p, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        for (int p = 0; p < N; p++) {
            printf("P%d: Row =", p);
            for (int j = 0; j < N; j++) printf(" %d", M[p][j]);
            printf("   Sum = %d\n", sums[p]);
            total += sums[p];
        }
        printf("Total matrix sum = %d\n", total);
    } else {
        int row[N], sum = 0;
        MPI_Recv(row, N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int j = 0; j < N; j++) sum += row[j];
        MPI_Send(&sum, 1, MPI_INT, 0, 1, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
