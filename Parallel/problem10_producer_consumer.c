/* Problem 10: Producer-Consumer.
 * P0 = producer, all other ranks = consumers. Items are handed out round-robin.
 * A special tag (TAG_STOP) tells a consumer that there is no more work.
 * Compile: mpicc problem10_producer_consumer.c -o p10
 * Run    : mpirun -np 3 ./p10            */
#include <mpi.h>
#include <stdio.h>
#define NITEMS 8
#define TAG_DATA 0
#define TAG_STOP 1

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    if (size < 2) { if (rank == 0) printf("Need >= 2 processes\n"); MPI_Finalize(); return 1; }

    int consumers = size - 1;

    if (rank == 0) {                                   /* ---- PRODUCER ---- */
        for (int i = 0; i < NITEMS; i++) {
            int item = 10 * (i + 1);                   /* 10,20,...,80 */
            int dest = 1 + (i % consumers);            /* round-robin consumer */
            printf("P0 produced: %d (-> P%d)\n", item, dest); fflush(stdout);
            MPI_Send(&item, 1, MPI_INT, dest, TAG_DATA, MPI_COMM_WORLD);
        }
        int dummy = 0;
        for (int c = 1; c < size; c++)                 /* tell everyone to stop */
            MPI_Send(&dummy, 1, MPI_INT, c, TAG_STOP, MPI_COMM_WORLD);

        int processed = 0, cnt;                        /* collect counts */
        for (int c = 1; c < size; c++) {
            MPI_Recv(&cnt, 1, MPI_INT, c, 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            processed += cnt;
        }
        if (processed == NITEMS) printf("All %d items processed successfully.\n", NITEMS);
    } else {                                           /* ---- CONSUMER ---- */
        int item, count = 0;
        MPI_Status st;
        while (1) {
            MPI_Recv(&item, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
            if (st.MPI_TAG == TAG_STOP) break;         /* no more items */
            printf("P%d consumed: %d (processed result = %d)\n", rank, item, item * 2);
            fflush(stdout);
            count++;
        }
        MPI_Send(&count, 1, MPI_INT, 0, 2, MPI_COMM_WORLD);
    }
    MPI_Finalize();
    return 0;
}
