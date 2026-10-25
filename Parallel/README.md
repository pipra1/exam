# Parallel Processing Lab: Solutions (Problems 2, 3, 4, 5, 7, 8, 10, 11, 15, 16, 17, 18)
Excluded as requested: 1, 6, 9, 12, 13, 14.

## Setup (once)
    sudo apt install openmpi-bin libopenmpi-dev      # Linux / WSL
## Compile and run any problem
    mpicc problemXX_name.c -o pXX -lm
    mpirun -np <P> ./pXX            # add --oversubscribe if you have fewer cores than P

| Problem | File | Run with | Key idea |
|---|---|---|---|
| 2  | problem02_row_sum.c | -np 4 | Root MPI_Send one row to each worker; workers MPI_Send the row sum back |
| 3  | problem03_ring.c | -np 4 | Send to (rank+1)%size, receive from (rank-1+size)%size; P0 sends first to avoid deadlock; MPI_Wtime for sizes |
| 4  | problem04_pingpong.c | -np 2 | P0 send then recv, P1 recv then send; one-way = round-trip/2; bandwidth = bytes/one-way |
| 5  | problem05_matmul.c | -np 2 | Root sends B + row-block of A; each process computes its rows of C; results sent back |
| 7  | problem07_nonblocking.c | -np 2 | Isend + Irecv + Waitall vs ordered Send/Recv; timed over 1000 repetitions |
| 8  | problem08_deadlock.c | -np 2 (and `unsafe` arg) | Both send first with MPI_Ssend = deadlock; fix = reorder (or MPI_Sendrecv) |
| 10 | problem10_producer_consumer.c | -np 3 | P0 produces and sends round-robin; consumers loop until a STOP tag arrives |
| 11 | problem11_bcast.c | -np 4 | Root fills params[3]; MPI_Bcast gives every process the same copy |
| 15 | problem15_trapezoid.c | -np 4 ./p15 1000 | Each process integrates its sub-interval; MPI_Reduce(SUM) adds partial integrals |
| 16 | problem16_scan.c | -np 4 | MPI_Scan vs manual chain (recv from rank-1, add, send to rank+1) |
| 17 | problem17_alltoall.c | -np 2/4/8/16 | MPI_Alltoall transposes the data; timing for several message sizes |
| 18 | problem18_histogram.c | -np 4 | Scatter data; local histogram; MPI_Allreduce(SUM) gives everyone the global one |

## Notes
* Problem 8: run `mpirun -np 2 ./p08 unsafe` to see the deadlock (it hangs; press Ctrl+C).
  MPI_Ssend is used because plain MPI_Send can succeed on tiny messages due to internal buffering.
* Problem 18: the sample says "Total = 20", but the data has 16 values (4 processes x 4). The frequencies in the sample add up to 16, so 16 is correct.
* Timing results (3, 4, 7, 16, 17) vary by machine and run; change ITER in problem04 to 1000 for steadier numbers.
