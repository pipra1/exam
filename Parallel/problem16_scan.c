/* Problem 16: Prefix sum with MPI_Scan, compared with a manual point-to-point version.
 * Compile: mpicc problem16_scan.c -o p16
 * Run    : mpirun -np 4 ./p16            */
#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int value = 10 * (rank + 1);               /* P0=10, P1=20, P2=30, P3=40 */

    /* ---- Method 1: built-in MPI_Scan (inclusive prefix sum) ---- */
    int scan_result;
    double t0 = MPI_Wtime();
    MPI_Scan(&value, &scan_result, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    double t_scan = MPI_Wtime() - t0;

    /* ---- Method 2: manual chain. Receive running total from rank-1,
     *      add my value, pass it on to rank+1. ---- */
    int manual = value, prev_sum;
    t0 = MPI_Wtime();
    if (rank > 0) {
        MPI_Recv(&prev_sum, 1, MPI_INT, rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        manual += prev_sum;
    }
    if (rank < size - 1)
        MPI_Send(&manual, 1, MPI_INT, rank + 1, 0, MPI_COMM_WORLD);
    double t_manual = MPI_Wtime() - t0;

    for (int p = 0; p < size; p++) {           /* ordered printing */
        if (rank == p) {
            printf("P%d: Local value = %d, Prefix sum (MPI_Scan) = %d, Prefix sum (manual) = %d  %s\n",
                   rank, value, scan_result, manual, scan_result == manual ? "[match]" : "[MISMATCH]");
            fflush(stdout);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    int all[64];                               /* gather results to print the final line */
    MPI_Gather(&scan_result, 1, MPI_INT, all, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Prefix result:");
        for (int i = 0; i < size; i++) printf(" %d", all[i]);
        printf("\n");
    }
    double ts = t_scan, tm = t_manual, max_s, max_m;
    MPI_Reduce(&ts, &max_s, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    MPI_Reduce(&tm, &max_m, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Time MPI_Scan = %.9f s, manual = %.9f s\n", max_s, max_m);

    MPI_Finalize();
    return 0;
}
