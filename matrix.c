#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

#define ROOT 0

void fill_matrix(double *mat, int rows, int cols) {
    for (int i = 0; i < rows * cols; i++)
        mat[i] = rand() % 10;
}

void print_matrix(double *mat, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++)
            printf("%6.2f ", mat[i * cols + j]);
        printf("\n");
    }
}

// Matrix multiplication (partial rows)
void local_matrix_multiply(double *A, double *B, double *C,
                           int local_rows, int N, int M) {
    for (int i = 0; i < local_rows; i++) {
        for (int j = 0; j < M; j++) {
            C[i * M + j] = 0;
            for (int k = 0; k < N; k++) {
                C[i * M + j] += A[i * N + k] * B[k * M + j];
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int rank, size;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rows = 4, cols = 4;
    int choice = 2; // 1=add, 2=multiply, 3=scalar, 4=transpose

    double *A = NULL, *B = NULL, *C = NULL;

    if (rank == ROOT) {
        A = malloc(rows * cols * sizeof(double));
        B = malloc(rows * cols * sizeof(double));
        C = malloc(rows * cols * sizeof(double));

        fill_matrix(A, rows, cols);
        fill_matrix(B, rows, cols);

        printf("Matrix A:\n");
        print_matrix(A, rows, cols);
        printf("\nMatrix B:\n");
        print_matrix(B, rows, cols);
    }

    MPI_Bcast(&choice, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&rows, 1, MPI_INT, ROOT, MPI_COMM_WORLD);
    MPI_Bcast(&cols, 1, MPI_INT, ROOT, MPI_COMM_WORLD);

    int local_rows = rows / size;
    double *local_A = malloc(local_rows * cols * sizeof(double));
    double *local_C = malloc(local_rows * cols * sizeof(double));

    if (choice == 2) {
        if (rank != ROOT) {
            B = malloc(rows * cols * sizeof(double));
        }
        MPI_Bcast(B, rows * cols, MPI_DOUBLE, ROOT, MPI_COMM_WORLD);
    }

    MPI_Scatter(A, local_rows * cols, MPI_DOUBLE,
                local_A, local_rows * cols, MPI_DOUBLE,
                ROOT, MPI_COMM_WORLD);

    // Perform selected operation
    if (choice == 1) {
        // Addition
        double *local_B = malloc(local_rows * cols * sizeof(double));
        MPI_Scatter(B, local_rows * cols, MPI_DOUBLE,
                    local_B, local_rows * cols, MPI_DOUBLE,
                    ROOT, MPI_COMM_WORLD);

        for (int i = 0; i < local_rows * cols; i++)
            local_C[i] = local_A[i] + local_B[i];

        free(local_B);
    }
    else if (choice == 2) {
        // Multiplication
        local_matrix_multiply(local_A, B, local_C, local_rows, cols, cols);
    }
    else if (choice == 3) {
        // Scalar multiply
        double scalar = 2.0;
        for (int i = 0; i < local_rows * cols; i++)
            local_C[i] = local_A[i] * scalar;
    }
    else if (choice == 4) {
        // Transpose (local only for simplicity)
        for (int i = 0; i < local_rows; i++)
            for (int j = 0; j < cols; j++)
                local_C[j * local_rows + i] = local_A[i * cols + j];
    }

    MPI_Gather(local_C, local_rows * cols, MPI_DOUBLE,
               C, local_rows * cols, MPI_DOUBLE,
               ROOT, MPI_COMM_WORLD);

    if (rank == ROOT) {
        printf("\nResult Matrix:\n");
        print_matrix(C, rows, cols);
    }

    free(local_A);
    free(local_C);
    if (rank == ROOT) {
        free(A); free(B); free(C);
    } else {
        free(B);
    }

    MPI_Finalize();
    return 0;
}
