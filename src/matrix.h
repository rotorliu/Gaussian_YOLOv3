#ifndef MATRIX_H
#define MATRIX_H
#include "darknet.h"

model do_kmeans(matrix data, int k);
matrix copy_matrix(matrix m);
void print_matrix(matrix m);

matrix hold_out_matrix(matrix *m, int n);
matrix resize_matrix(matrix m, int size);

float *pop_column(matrix *m, int c);

#endif
