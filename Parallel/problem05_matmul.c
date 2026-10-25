/* Problem 5: Matrix multiplication C = A x B using MPI_Send / MPI_Recv.
 * Root sends B (whole) and a block of rows of A to every worker.
 * Each process computes its rows of C and sends them back.
 * N must be divisible by the number of processes.
 * Compile: mpicc problem05_matmul.c -o p05
 * Run    : mpirun -np 2 ./p05            */
#include <mpi.h>
#include <stdio.h>
#define N 2

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (N % size != 0) { if (rank == 0) printf("N must be divisible by #processes\n"); MPI_Finalize(); return 1; }

    int rows = N / size;                      /* rows handled by each process */
    int A[N][N], B[N][N], C[N][N];
    int myA[N][N], myC[N][N];                 /* only the first 'rows' rows are used */

    if (rank == 0) {
        int a[N][N] = {{1, 2}, {3, 4}};
        int b[N][N] = {{5, 6}, {7, 8}};
        for (int i = 0; i < N; i++) for (int j = 0; j < N; j++) { A[i][j] = a[i][j]; B[i][j] = b[i][j]; }

        for (int p = 1; p < size; p++) {      /* send B and p's rows of A */
            MPI_Send(B, N * N, MPI_INT, p, 0, MPI_COMM_WORLD);
            MPI_Send(A[p * rows], rows * N, MPI_INT, p, 1, MPI_COMM_WORLD);
        }
        for (int i = 0; i < rows; i++)        /* root computes its own rows (0..rows-1) */
            for (int j = 0; j < N; j++) {
                C[i][j] = 0;
                for (int k = 0; k < N; k++) C[i][j] += A[i][k] * B[k][j];
            }
        for (int p = 1; p < size; p++)        /* collect other rows of C */
            MPI_Recv(C[p * rows], rows * N, MPI_INT, p, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        printf("Matrix C = A x B\n");
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < N; j++) printf("%d ", C[i][j]);
            printf("\n");
        }
        for (int i = 0; i < N; i++)
            for (int j = 0; j < N; j++) {
                printf("C[%d][%d] =", i, j);
                for (int k = 0; k < N; k++) printf(" %s%d x %d", k ? "+ " : "", A[i][k], B[k][j]);
                printf(" = %d\n", C[i][j]);
            }
    } else {
        MPI_Recv(B, N * N, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        MPI_Recv(myA, rows * N, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < N; j++) {
                myC[i][j] = 0;
                for (int k = 0; k < N; k++) myC[i][j] += myA[i][k] * B[k][j];
            }
        MPI_Send(myC, rows * N, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
