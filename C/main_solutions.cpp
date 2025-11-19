// This include is required for tutorial to pass tests
#include "src.hpp"
#include <time.h>
#include <vector>
#include <omp.h>

void transpose_MPI(double* A, double* AT, int local_n, int global_n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double* send_buffer = new double[local_n*global_n];
    double* recv_buffer = new double[local_n*global_n];
    
    MPI_Request* send_requests = new MPI_Request[num_procs];
    MPI_Request* recv_requests = new MPI_Request[num_procs];

    int tag = 1024;

    int msg_size = local_n*local_n; // size of each msg

    // Post receives
    for (int i = 0; i < num_procs; i++)
    {
        MPI_Irecv(&(recv_buffer[i*msg_size]), msg_size, MPI_DOUBLE, i, tag, MPI_COMM_WORLD,
                &(recv_requests[i]));
    }

    // Pack and send data
    int ctr = 0;
    for (int i = 0; i < num_procs; i++)
    {
        for (int col = i*local_n; col < (i+1)*local_n; col++)
        {
            for (int row = 0; row < local_n; row++)
            {
                send_buffer[ctr++] = A[row*global_n+col];
            }
        }
        MPI_Isend(&(send_buffer[i*msg_size]), msg_size, MPI_DOUBLE, i, tag, MPI_COMM_WORLD,
                &(send_requests[i]));
    }
    
    // Wait for all communication to complete
    MPI_Waitall(num_procs, send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_procs, recv_requests, MPI_STATUSES_IGNORE);

    // Unpack data
    for (int row = 0; row < local_n; row++)
    {
        for (int i = 0; i < num_procs; i++)
        {
            for (int col = 0; col < local_n; col++)
            {
                AT[row*global_n + i*local_n + col] = recv_buffer[i*local_n*local_n + row*local_n + col];
            }
        }
    }

    // Clean up
    delete[] send_buffer;
    delete[] recv_buffer;
    delete[] send_requests;
    delete[] recv_requests;
}

// Transpose using MPI + OpenMP
void transpose_mpiOpenMP(double* A, double* AT, int local_n, int global_n)
{
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    double* send_buffer = new double[local_n*global_n];
    double* recv_buffer = new double[local_n*global_n];
    
    MPI_Request* send_requests = new MPI_Request[num_procs];
    MPI_Request* recv_requests = new MPI_Request[num_procs];

    int tag = 1024;

    int msg_size = local_n*local_n; // size of each msg

    // Post receives - mpi only
    for (int i = 0; i < num_procs; i++)
    {
        MPI_Irecv(&(recv_buffer[i*msg_size]), msg_size, MPI_DOUBLE, i, tag, MPI_COMM_WORLD,
                &(recv_requests[i]));
    }

    // Pack data - openmp
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < num_procs; i++)
    {
        for (int col = i*local_n; col < (i+1)*local_n; col++)
        {
            for (int row = 0; row < local_n; row++)
            {
                int ctr = i * msg_size + (col - i*local_n)*local_n + local_n;
                send_buffer[ctr++] = A[row*global_n+col];
            }
        }
    }

    // Send data - mpi only
    for (int i = 0; i < num_procs; i++)
    {
        MPI_Isend(&(send_buffer[i*msg_size]), msg_size, MPI_DOUBLE, i, tag, MPI_COMM_WORLD,
                &(send_requests[i]));
    }   
    
    // Wait for all communication to complete
    MPI_Waitall(num_procs, send_requests, MPI_STATUSES_IGNORE);
    MPI_Waitall(num_procs, recv_requests, MPI_STATUSES_IGNORE);

    // Unpack data - openmp
    #pragma omp parallel for collapse(3)
    for (int row = 0; row < local_n; row++)
    {
        for (int i = 0; i < num_procs; i++)
        {
            for (int col = 0; col < local_n; col++)
            {
                AT[row*global_n + i*local_n + col] = recv_buffer[i*local_n*local_n + row*local_n + col];
            }
        }
    }

    // Clean up
    delete[] send_buffer;
    delete[] recv_buffer;
    delete[] send_requests;
    delete[] recv_requests;
}


// Initialize, create random, finalize, and return
int tutorial_main(int argc, char* argv[])
{
    // 1. Initialize
    MPI_Init(&argc, &argv);
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    // 2. Perform transpose using MPI and MPI + OpenMP

        // Create random matrix A
    int global_n = 1024;
    int local_n = global_n / num_procs;
    int first_n = local_n * rank;

    double* A = new double[local_n * global_n];
    double* AT = new double[local_n * global_n];
    double* AT_new = new double[local_n * global_n];

    srand(time(NULL) + rank);
        // Fill matrix A with random values
    for (int i = 0; i < local_n; i++)
    {
        for (int j = 0; j < global_n; j++)
        {
            A[i*global_n+j] = (double)(rand()) / RAND_MAX;
        }
    }

        // Transpose using both methods
    transpose_MPI(A, AT, local_n, global_n);
    transpose_mpiOpenMP(A, AT_new, local_n, global_n);

    // 3. Compare your methods

    for (int i = 0; i < local_n; i++)
    {
        for (int j = 0; j < global_n; j++)
        {
            if (fabs(AT[i*global_n+j] - AT_new[i*global_n+j]) > 1e-06)
            {
                fprintf(stderr, "Rank %d got incorrect transpose at position (%d, %d)\n", rank, i, j);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
    }
    
    delete[] A;
    delete[] AT;
    delete[] AT_new;

    // 4. Finalize
    MPI_Finalize();

    return 0;
}
