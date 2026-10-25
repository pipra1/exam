/* Problem 15: Trapezoidal rule for the integral of f(x)=x^2 on [0,1].
 * Each process integrates its own sub-interval; MPI_Reduce adds the partial results.
 * Compile: mpicc problem15_trapezoid.c -o p15
 * Run    : mpirun -np 4 ./p15 1000           (last argument = number of intervals n) */
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

double f(double x) { return x * x; }

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a = 0.0, b = 1.0;
    int n = (argc > 1) ? atoi(argv[1]) : 1000;
    if (n % size != 0) { if (rank == 0) printf("n must be divisible by #processes\n"); MPI_Finalize(); return 1; }

    double h = (b - a) / n;                    /* width of one trapezoid */
    int local_n = n / size;                    /* trapezoids per process */
    double local_a = a + rank * local_n * h;   /* this process' interval start */
    double local_b = local_a + local_n * h;    /* ... and end */

    /* Trapezoid rule: h * [ f(a)/2 + f(x1) + ... + f(x_{n-1}) + f(b)/2 ] */
    double local = (f(local_a) + f(local_b)) / 2.0;
    for (int i = 1; i < local_n; i++) local += f(local_a + i * h);
    local *= h;

    printf("P%d: interval [%.4f, %.4f]  partial integral = %.9f\n", rank, local_a, local_b, local);
    fflush(stdout);

    double total = 0.0;
    MPI_Reduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);   /* add all partials */

    if (rank == 0) {
        double exact = 1.0 / 3.0;
        printf("n = %d\nNumerical integral ~ %.6f\nExact integral     = %.6f\nAbsolute error     ~ %.9f\n",
               n, total, exact, fabs(total - exact));
    }
    MPI_Finalize();
    return 0;
}
