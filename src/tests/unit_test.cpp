#include "src.hpp"
#include <mpi.h>
#include <unistd.h>   // for sleep
#include <cstdio>
#include <cmath>
#include <vector>
#include <time.h>
int main(int argc, char* argv[])
{
    PMPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    int global_n = 1024;
    int local_n = global_n / num_procs;

    double* A = new double[local_n * global_n];
    double* AT = new double[local_n * global_n];
    double* AT_new = new double[local_n * global_n];

    srand(time(NULL) + rank);
    for (int i = 0; i < local_n; i++)
    {
        for (int j = 0; j < global_n; j++)
        {
            A[i*global_n+j] = (double)(rand()) / RAND_MAX;
        }
    }

    transpose_MPI(A, AT, local_n, global_n);

    transpose_mpiOpenMP(A, AT_new, local_n, global_n);

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

    PMPI_Finalize();
    return 0;





}
