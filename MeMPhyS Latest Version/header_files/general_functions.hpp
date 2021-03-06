// Author: Dr. Shantanu Shahane
#ifndef general_functions_H_ /* Include guard */
#define general_functions_H_
#include <ftw.h>
#include <sys/stat.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <random>
#include "metis.h"
#include "mpi.h"
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>
#include <Eigen/Eigenvalues>
#include <Eigen/Core>
#include <Eigen/SparseCore>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/SparseGenMatProd.h>
#include <Spectra/GenEigsRealShiftSolver.h>
#include <Spectra/MatOp/SparseGenRealShiftSolve.h>
#include <unsupported/Eigen/FFT>
using namespace std;

Eigen::VectorXcd calc_largest_magnitude_eigenvalue(Eigen::SparseMatrix<double, Eigen::RowMajor> &matrix);

void does_file_exist(const char *, const char *);

void check_mpi();

void print_to_terminal(vector<bool> &a, const char *text);

void print_to_terminal(vector<double> &a, int n_row, int n_col, const char *text);

void print_to_terminal(vector<double> &a, const char *text);

void print_to_terminal(vector<int> &a, const char *text);

void print_to_terminal(Eigen::VectorXd &a, const char *text);

void print_to_terminal(vector<vector<int>> &a, const char *text);

void print_to_terminal(vector<pair<int, int>> &a, const char *text);

void print_to_terminal(vector<int> &a, int n_row, int n_col, const char *text);

void print_to_terminal(idx_t *elem_vert_row, idx_t *elem_vert_col, int n_row, const char *text);

void print_to_terminal(vector<bool> &a, int n_row, int n_col, const char *text);

void print_to_terminal(vector<int> &sp_row, vector<int> &sp_col, const char *text);

void print_to_terminal(Eigen::MatrixXd &A, const char *text);

void print_to_terminal(Eigen::MatrixXcd &A, const char *text);

void print_to_terminal(Eigen::MatrixXi &A, const char *text);

void print_to_terminal(Eigen::SparseMatrix<double, Eigen::RowMajor> &A, const char *text);

void print_to_terminal(Eigen::SparseMatrix<int> &A, const char *text);

void write_csv(vector<double> &vect, int nr, int nc, const char *file_name);

void write_csv(vector<bool> &vect, int nr, int nc, const char *file_name);

void write_csv(double *vect, int nr, int nc, const char *file_name);

void write_csv(Eigen::MatrixXd &A, const char *file_name);

void write_csv(Eigen::MatrixXi &A, const char *file_name);

void write_csv(Eigen::MatrixXcd &A, const char *file_name);

void write_csv(Eigen::VectorXd &A, const char *file_name);

void write_csv_benchmark(vector<double> &xyz_interp, Eigen::VectorXd &reference, Eigen::VectorXd &simulation, const char *file_name, int dimension);

void write_csv_benchmark(vector<double> &xyz_interp, Eigen::VectorXd &simulation, const char *file_name, int dimension);

void write_csv(vector<double> &xyz, vector<bool> &boundary_flag, int dim, const char *file_name);

void write_csv(vector<double> &xyz, vector<bool> &boundary_flag, vector<bool> &periodic_bc_flag, int dim, const char *file_name);

void write_csv(vector<double> &xyz, vector<bool> &boundary_flag, int dimension, vector<string> &variable_names, vector<Eigen::VectorXd *> &variable_pointers, const char *file_name);

void write_csv(Eigen::MatrixXd &x_3d, Eigen::MatrixXd &y_3d, Eigen::MatrixXd &z_3d, vector<bool> &boundary_flag, vector<string> &variable_names, vector<Eigen::MatrixXd *> &variable_pointers, const char *file_name);

void write_csv(vector<double> &xyz, vector<bool> &boundary_flag, vector<bool> &periodic_bc_flag, vector<int> &periodic_bc_section, int dim, const char *file_name);

void write_csv(vector<double> &xyz, vector<bool> &boundary_flag, Eigen::VectorXd &A_ana, Eigen::VectorXd &A_num, int dim, const char *file_name);

void write_csv(Eigen::SparseMatrix<double, Eigen::RowMajor> &A, const char *file_name);

void write_csv(Eigen::SparseMatrix<int> &A, const char *file_name);

void write_csv(vector<int> &sp_row, vector<int> &sp_col, const char *file_name);

void write_csv(vector<int> &sp_row, vector<double> &sp_val, const char *file_name);

void write_csv(vector<vector<int>> &a, const char *file_name);

void write_csv(vector<vector<double>> &a, const char *file_name);

void write_csv(vector<tuple<int, int, double>> &a, const char *file_name);

void write_csv(vector<int> &vect, int nr, int nc, const char *file_name);

void k_smallest_elements(vector<double> &k_min_a, vector<int> &k_min_a_indices, vector<double> &a, int k);

vector<int> argsort(const vector<double> &v);

vector<double> calc_crowd_distance(vector<double> &x, vector<double> &y, vector<double> &z, int dim);

double max_abs(Eigen::VectorXd &a);

double round_digits(double a, int num_digits);

string to_string_precision(double a, int num_digits);

void cuthill_mckee_ordering(vector<vector<int>> &adjacency, vector<int> &order);

void reverse_cuthill_mckee_ordering(vector<vector<int>> &adjacency, vector<int> &order);

double vector_norm(vector<double> &a, int norm_type);

double vector_norm(double *a, int size, int norm_type);

void cross_product(double *result, double *u, double *v);

void calc_max_l1_error(vector<double> &a1, vector<double> &a2, double &max_err, double &l1_err);

void calc_max_l1_error(Eigen::VectorXd &a1, Eigen::VectorXd &a2, double &max_err, double &l1_err);

void calc_max_l1_error(Eigen::VectorXd &a1, Eigen::VectorXd &a2, double &max_err_boundary, double &l1_err_boundary, double &max_err_internal, double &l1_err_internal, vector<bool> &boundary_flag);

void calc_max_l1_relative_error(vector<double> &ana_val, vector<double> &num_val, double &max_err, double &l1_err);

void calc_max_l1_relative_error(Eigen::VectorXd &ana_val, Eigen::VectorXd &num_val, double &max_err, double &l1_err);

void calc_max_l1_relative_error(Eigen::VectorXd &ana_val, Eigen::VectorXd &num_val, double &max_err_boundary, double &l1_err_boundary, double &max_err_internal, double &l1_err_internal, vector<bool> &boundary_flag);

void gauss_siedel_eigen(Eigen::SparseMatrix<double, Eigen::RowMajor> &matrix, Eigen::VectorXd &source, Eigen::VectorXd &field_old, int num_iter, double omega);

Eigen::SparseMatrix<double, Eigen::RowMajor> convert_csc_to_csr_eigen(Eigen::SparseMatrix<double, Eigen::ColMajor> &matrix);

Eigen::SparseMatrix<double, Eigen::ColMajor> convert_csr_to_csc_eigen(Eigen::SparseMatrix<double, Eigen::RowMajor> &matrix);

void eigen_calc_fft(Eigen::MatrixXcd &phi_fft, Eigen::MatrixXd &phi, Eigen::FFT<double> &fft, Eigen::VectorXcd &eigen_temp_vec_complex);

void eigen_calc_ifft(Eigen::MatrixXd &phi, Eigen::MatrixXcd &phi_fft, Eigen::FFT<double> &fft, Eigen::VectorXcd &eigen_temp_vec_complex);

Eigen::MatrixXcd get_dft_matrix(int nw, double period);

Eigen::MatrixXcd get_idft_matrix(int nw, double period);

void calc_dft(Eigen::MatrixXcd &phi_dft, Eigen::MatrixXd &phi, Eigen::MatrixXcd &dft_matrix);

void calc_idft(Eigen::MatrixXd &phi, Eigen::MatrixXcd &phi_dft, Eigen::MatrixXcd &idft_matrix);

static int rmFiles(const char *pathname, const struct stat *sbuf, int type, struct FTW *ftwb);

void remove_full_directory(const char *path);

#endif
