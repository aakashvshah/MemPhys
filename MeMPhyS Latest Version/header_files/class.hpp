// Author: Dr. Shantanu Shahane
#ifndef class_H_ /* Include guard */
#define class_H_
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <unsupported/Eigen/SparseExtra>
#include <Eigen/SparseLU>
#include <Eigen/OrderingMethods>
#include <Eigen/Core>
#include "_hypre_utilities.h"
#include "HYPRE_krylov.h"
#include "HYPRE.h"
#include "HYPRE_parcsr_ls.h"
#include "general_functions.hpp"
using namespace std;

const int INTERIOR_TAG = -1;
// const int INLET_BC_TAG = 1;
// const int OUTLET_BC_TAG = 2;
// const int WALL_BC_TAG = 3;
// const int SYMMETRY_BC_TAG = 4;

template <typename T>
struct PointCloud
{ // taken from Nanoflann library: "utils.h": https://github.com/jlblancoc/nanoflann/blob/master/examples/utils.h
    struct Point
    {
        T x, y, z;
    };
    std::vector<Point> pts;
    inline size_t kdtree_get_point_count() const { return pts.size(); } // Must return the number of data points
    // Returns the dim'th component of the idx'th point in the class:
    // Since this is inlined and the "dim" argument is typically an immediate value, the
    //  "if/else's" are actually solved at compile time.
    inline T kdtree_get_pt(const size_t idx, const size_t dim) const
    {
        if (dim == 0)
            return pts[idx].x;
        else if (dim == 1)
            return pts[idx].y;
        else
            return pts[idx].z;
    }
    // Optional bounding-box computation: return false to default to a standard bbox computation loop.
    //   Return true if the BBOX was already computed by the class and returned in "bb" so it can be avoided to redo it again.
    //   Look at bb.size() to find out the expected dimensionality (e.g. 2 or 3 for point clouds)
    template <class BBOX>
    bool kdtree_get_bbox(BBOX & /* bb */) const { return false; }
};

class PARAMETERS
{
public:
    int cloud_size;                            // cloud size
    double cloud_size_multiplier;              // cloud size: cloud_size_multiplier * num_poly_terms
    int phs_deg;                               // degree of the polyharmonic spline: r^(phs_deg)
    int poly_deg;                              // degree of the polynomial to be appended
    int num_poly_terms;                        // number of the polynomial terms to be appended
    int dimension;                             // problem dimension (can be either 2 or 3 only)
    Eigen::MatrixXi polynomial_term_exponents; // exponents of the polynomial terms
    string meshfile;                           // meshfile name
    string output_file_prefix;                 // used to write all output files
    string restart_tecplot_file;               // tecplot file to restart solution with this initial condition
    double max_dx, min_dx, avg_dx;             // characteristic length scales of mesh
    vector<int> periodic_bc_index;             // periodic BC wrt 0:'x', 1:'y', 2:'z'; : size [no. of periodic axes]; ex. ['z','x']:[2,0]; ['y']:[1]
    double dt;
    double rho = -10.0, mu = rho; // should be set in the main file
    double steady_tolerance, solver_tolerance, Courant, precond_droptol;
    int nt, euclid_precond_level_hypre, gmres_kdim, n_iter;
    string solver_type; // hypre_ilu_gmres, eigen_direct, eigen_ilu_bicgstab

    double grad_x_eigval_real, grad_x_eigval_imag, grad_y_eigval_real, grad_y_eigval_imag, grad_z_eigval_real, grad_z_eigval_imag, laplace_eigval_real, laplace_eigval_imag;
    double cloud_id_timer, rcm_timer, cloud_misc_timer, points_timer, grad_laplace_coeff_timer, factoring_timer, solve_timer, total_timer;
    int nt_actual = 0;

    vector<double> rel_res_log, abs_res_log, regul_alpha_log, steady_error_log; // logs of size nt_actual
    vector<int> n_iter_actual;                                                  // logs of size nt_actual

public:
    PARAMETERS(string parameter_file, string gmsh_file);
    void read_calc_parameters(string parameter_file);
    void verify_parameters();
    void calc_cloud_num_points();
    void get_problem_dimension_msh();
    void calc_polynomial_term_exponents();
    void calc_dt(Eigen::SparseMatrix<double, Eigen::RowMajor> &grad_x, Eigen::SparseMatrix<double, Eigen::RowMajor> &grad_y, Eigen::SparseMatrix<double, Eigen::RowMajor> &grad_z, Eigen::SparseMatrix<double, Eigen::RowMajor> &laplacian, double u0, double v0, double w0, double alpha);
};

class POINTS
{ // uses Compressed Row Format of Sparse Matrices
public:
    vector<bool> corner_edge_vertices;                                   // true at corner (for 2D, 3D problems) and edge (for 3D problems only): size [nv]
    int nv;                                                              // Number of vertices in original msh file
    int nv_original;                                                     // Number of vertices
    vector<double> xyz_min, xyz_max, xyz_length;                         // each of size:[dim]
    vector<double> xyz;                                                  // vertex co-ordinates: size [nv X dim]
    vector<double> xyz_original;                                         // all vertex co-ordinates in original msh file: size [nv_original X dim]; nv_original can be greater than nv
    int nelem_original;                                                  // no. of elements in the original file
    vector<bool> elem_boundary_flag_original;                            // boundary_flag=1 if elem is on boundary; else boundary_flag=0: size [nelem_original]
    vector<vector<int>> elem_vert_original;                              // connectivity in original msh file
    vector<double> boundary_face_area_original;                          // area or length of boundary elements for 3D or 2D (used to compute fluxes at boundaries): size [nelem_original]
    vector<int> elem_bc_tag_original;                                    // bc_tag associated with each element (helps to identify various boundary areas and internal region): size [nelem_original]
    vector<int> iv_original_nearest_vert;                                // nearest vertex no for vertices in xyz_original: size [nv_original]; nv_original can be greater than nv
    vector<double> normal;                                               // vertex co-ordinates (relavant only for boundary vertices): size [nv X dim]
    vector<bool> boundary_flag;                                          // boundary_flag=1 if it is on boundary; else boundary_flag=0: size [nv]
    vector<vector<bool>> periodic_bc_flag;                               // periodic_bc_flag=1 if it is on periodic boundary; else periodic_bc_flag=0: size [nv][no. of periodic axes]
    vector<vector<int>> periodic_bc_section;                             // takes values [-1,0,1] for [near_min,middle,near_max] sections respectively: size [nv][no. of periodic axes]
    vector<int> bc_tag;                                                  // bc_tag associated with each vertex (helps to identify various boundary areas and internal region): size [nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> grad_x_matrix_EIGEN;    // used for convection source term size [points.nv X points.nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> grad_y_matrix_EIGEN;    // used for convection source term size [points.nv X points.nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> grad_z_matrix_EIGEN;    // used for convection source term size [points.nv X points.nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> laplacian_matrix_EIGEN; // used for diffusion source term size [points.nv X points.nv]

    Eigen::SparseMatrix<double, Eigen::RowMajor> grad_x_matrix_EIGEN_boundary, grad_x_matrix_EIGEN_internal;       //[points.nv X points.nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> grad_y_matrix_EIGEN_boundary, grad_y_matrix_EIGEN_internal;       //[points.nv X points.nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> grad_z_matrix_EIGEN_boundary, grad_z_matrix_EIGEN_internal;       //[points.nv X points.nv]
    Eigen::SparseMatrix<double, Eigen::RowMajor> laplacian_matrix_EIGEN_boundary, laplacian_matrix_EIGEN_internal; //[points.nv X points.nv]

public:
    POINTS(PARAMETERS &parameters);
    void read_points_xyz_msh(PARAMETERS &parameters);
    void read_points_flag_msh(PARAMETERS &parameters);
    void calc_vert_normal(PARAMETERS &parameters);
    void calc_boundary_face_area(PARAMETERS &parameters);
    void calc_elem_bc_tag(PARAMETERS &parameters);
    void read_elem_vert_complete_msh(PARAMETERS &parameters, vector<vector<int>> &elem_vert, vector<bool> &elem_boundary_flag);
    void calc_vert_nb_cv(PARAMETERS &parameters, vector<vector<int>> &vert_nb_cv, vector<vector<int>> &elem_vert);
    void calc_elem_normal_2D(vector<double> &elem_normal, vector<vector<int>> &vert_nb_cv, vector<vector<int>> &elem_vert, vector<bool> &elem_boundary_flag);
    void calc_elem_normal_3D(vector<double> &elem_normal, vector<vector<int>> &vert_nb_cv, vector<vector<int>> &elem_vert, vector<bool> &elem_boundary_flag);
    void delete_corner_edge_vertices(PARAMETERS &parameters);
    void delete_periodic_bc_vertices(PARAMETERS &parameters);
    void set_periodic_bc(PARAMETERS &parameters, vector<string> periodic_axis);
};

class CLOUD
{
public:
    vector<int> nb_points_row;      // neighboring points of each point: size: nv+1
    vector<int> nb_points_col;      // neighboring points of each point: size: nb_points_row[nv]
    vector<double> grad_x_coeff;    // coefficient for grad_x at each point: base: (CLOUD.nb_points_row, CLOUD.nb_points_col)
    vector<double> grad_y_coeff;    // coefficient for grad_y at each point: base: (CLOUD.nb_points_row, CLOUD.nb_points_col)
    vector<double> grad_z_coeff;    // coefficient for grad_z at each point: base: (CLOUD.nb_points_row, CLOUD.nb_points_col)
    vector<double> laplacian_coeff; // coefficient for laplacian at each point: base: (CLOUD.nb_points_row, CLOUD.nb_points_col)
    vector<double> cond_num_RBF;    // condition number of RBF A matrix for each point: size [nv]
    double cond_num_RBF_max;        // statistics of cond_num_RBF
    double cond_num_RBF_min;        // statistics of cond_num_RBF
    double cond_num_RBF_avg;        // statistics of cond_num_RBF
    vector<int> rcm_points_order;   // reordering list obtained by RCM algorithm
public:
    CLOUD(POINTS &points, PARAMETERS &parameters);
    void calc_cloud_points_slow(POINTS &points, PARAMETERS &parameters);
    void calc_cloud_points_fast(POINTS &points, PARAMETERS &parameters);
    void calc_cloud_points_fast_periodic_bc(POINTS &points, PARAMETERS &parameters);
    void calc_cloud_points_fast_periodic_bc_shifted(POINTS &points, PARAMETERS &parameters, vector<double> &xyz_shifted, vector<int> &periodic_bc_section_value);
    void re_order_points_reverse_cuthill_mckee(POINTS &points, PARAMETERS &parameters);
    void re_order_points(POINTS &points, PARAMETERS &parameters);
    void calc_iv_original_nearest_vert(POINTS &points, PARAMETERS &parameters);
    void calc_charac_dx(POINTS &points, PARAMETERS &parameters);
    void calc_grad_laplace_coeffs(POINTS &points, PARAMETERS &parameters);
    void EIGEN_set_grad_laplace_matrix(POINTS &points, PARAMETERS &parameters);
    void EIGEN_set_grad_laplace_matrix_separate(POINTS &points, PARAMETERS &parameters);
};

class SOLVER
{
public:
    // Functions
    void init(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &dirichlet_flag, double unsteady_term_coeff, double conv_term_coeff, double diff_term_coeff, bool log_flag);
    void HYPRE_set_coeff(PARAMETERS &parameters);
    void EIGEN_set_coeff();
    void general_solve(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &field_new, Eigen::VectorXd &field_old, Eigen::VectorXd &rhs);
    void set_solve_parameters();
    void calc_coeff_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);
    void SOR(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &field, Eigen::VectorXd &rhs);
    void scale_coeff(POINTS &points);

    // Variables
    HYPRE_IJMatrix coeff_HYPRE;
    HYPRE_ParCSRMatrix parcsr_coeff_HYPRE;
    HYPRE_IJVector source_HYPRE;
    HYPRE_ParVector par_source_HYPRE;
    HYPRE_IJVector X_HYPRE;
    HYPRE_ParVector par_X_HYPRE;
    HYPRE_Solver solver_X_HYPRE, precond_X_HYPRE;
    Eigen::SparseMatrix<double, Eigen::RowMajor> coeff_EIGEN;
    Eigen::SparseLU<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::COLAMDOrdering<int>> solver_eigen_direct;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::IncompleteLUT<double>> solver_eigen_ilu_bicgstab;

    vector<tuple<int, int, double>> coeff_matrix; // store coefficient matrix with sparsity structure of (row, col, value)
    vector<double> scale;

    int *rows_HYPRE; // array of size ncv going from 0 to ncv-1
    double *X;       // unknown vector
    double *source;  // source term

    string solver_type; // hypre_ilu_gmres, eigen_direct, eigen_ilu_bicgstab
    double unsteady_term_coeff_1, conv_term_coeff_1, diff_term_coeff_1;
    int print_flag = 0;             // decide how much to print during solution
    int n_iter;                     // max solver iterations
    int euclid_precond_level_hypre; // level setting in Euclid pre-conditioner
    double precond_droptol;         // setting in Euclid pre-conditioner
    int gmres_kdim;                 // GMRES max size of Krylov subspace
    double solver_tolerance;        // solver tolerance
    double l2_norm;
    double SOR_omega = 1.4;
    vector<double> SOR_rel_res, SOR_abs_res; // used only for SOR solvers; size:[n_iter]
    vector<bool> dirichlet_flag_1;           // true: use Dirichlet BC, else use Neumann BC
    int system_size;                         // no. of unknowns
    bool log_flag_1 = true;
};

class MULTIGRID
{
public:
    MULTIGRID(string &meshfolder1, vector<string> &meshfile_list1);
    void assemble_coeff();
    void single_grid_sor(vector<Eigen::VectorXd> &phi, vector<Eigen::VectorXd> &source, vector<vector<double>> &residual, int ig);

    string meshfolder;
    vector<string> meshfile_list;
    vector<PARAMETERS> parameters_list;
    vector<POINTS> points_list;
    vector<CLOUD> cloud_list;
    vector<vector<bool>> dirichlet_flag_list;
    vector<bool> full_neumann_flag_list;  // size: [ng]: true for full Neumann BC
    int ng;                               // no. of grid levels
    int n_iter = 10000;                   // no of single grid iterations
    double sor_omega = 1.4;               // over-relaxation factor for SOR
    vector<vector<int>> coeff_mat_row;    // coeff matrix CSR format row: size: [ng][nv+1]
    vector<vector<int>> coeff_mat_col;    // coeff matrix CSR format columns: size: [ng][nb_points_row[nv]]
    vector<vector<double>> coeff_mat_val; // coeff matrix CSR format values: size: [ng][nb_points_row[nv]]

    // vector<int> system_size;                   //nv or nv+1 based for regularization: size: [ng]
    // vector<Eigen::VectorXd> coeff_mat_col_sum; //column sum of coeff matrix: size: [ng][nv]
    // vector<Eigen::VectorXd> phi, source;       //unknown field and source: size: [ng][nv]
    // vector<vector<double>> regul_alpha; //regularization alpha: size: [ng][n_iter_actual]
};

class FRACTIONAL_STEP_1
{ // hat velocity formulation
public:
    // Functions
    FRACTIONAL_STEP_1(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &u_dirichlet_flag1, vector<bool> &v_dirichlet_flag1, vector<bool> &p_dirichlet_flag1, int temporal_order1);
    void check_bc(POINTS &points, PARAMETERS &parameters);
    void single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, int it1);
    void single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y, int it1);
    void calc_vel_hat(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_vel_hat(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y);
    void calc_pressure(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_vel_corr(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old);
    void extras(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);

    // Variables
    SOLVER solver_p;
    Eigen::VectorXd zero_vector, zero_vector_1;
    Eigen::VectorXd uh, vh;
    Eigen::VectorXd p_source;
    Eigen::VectorXd u_source_old, v_source_old;
    Eigen::VectorXd u_source_old_old, v_source_old_old; // used only for multi-step method
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    bool p_bc_full_neumann;
    int temporal_order = -1, it;
};

class FRACTIONAL_STEP_FOURIER
{ // hat velocity formulation with meshless in (X,Y) and fourier in (Z)
public:
    // Functions
    FRACTIONAL_STEP_FOURIER(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &u_dirichlet_flag1, vector<bool> &v_dirichlet_flag1, vector<bool> &w_dirichlet_flag1, vector<bool> &p_dirichlet_flag1, double period1, int nw1, int temporal_order1);
    FRACTIONAL_STEP_FOURIER(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &u_dirichlet_flag1, vector<bool> &v_dirichlet_flag1, vector<bool> &w_dirichlet_flag1, vector<bool> &p_dirichlet_flag1, vector<bool> &T_dirichlet_flag1, double thermal_k1, double thermal_Cp1, double period1, int nw1, int temporal_order1);
    void set_p_coeff_eigen(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);
    void set_p_solver_eigen(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);
    void check_settings(POINTS &points, PARAMETERS &parameters);
    void single_timestep(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &p_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old, int it1);
    void single_timestep(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &p_new, Eigen::MatrixXd &T_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old, Eigen::MatrixXd &T_old, int it1);
    void single_timestep(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &p_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old, Eigen::MatrixXd &body_force_x, Eigen::MatrixXd &body_force_y, Eigen::MatrixXd &body_force_z, int it1);
    void single_timestep(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &p_new, Eigen::MatrixXd &T_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old, Eigen::MatrixXd &T_old, Eigen::MatrixXd &body_force_x, Eigen::MatrixXd &body_force_y, Eigen::MatrixXd &body_force_z, int it1);
    void calc_vel_hat(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old);
    void calc_T(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &T_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &T_old);
    void calc_vorticity(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &vort_x, Eigen::MatrixXd &vort_y, Eigen::MatrixXd &vort_z);
    void calc_q_criterion(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &q_criterion);
    void calc_vel_hat(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old, Eigen::MatrixXd &body_force_x, Eigen::MatrixXd &body_force_y, Eigen::MatrixXd &body_force_z);
    void calc_vel_source(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXcd &vel_source, Eigen::MatrixXd &u, Eigen::MatrixXd &v, Eigen::MatrixXd &w, Eigen::MatrixXcd &phi_ft, Eigen::MatrixXd &phi);
    void calc_pressure(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &p_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old);
    void calc_vel_corr(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &p_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old);

    // Variables
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::IncompleteLUT<double>> *p_solver_eigen_ilu_bicgstab;
    vector<Eigen::SparseMatrix<double, Eigen::RowMajor>> p_coeff_eigen;
    Eigen::MatrixXcd uh_ft, vh_ft, wh_ft;
    Eigen::MatrixXcd vort_x_ft, vort_y_ft, vort_z_ft, q_criterion_ft;
    Eigen::MatrixXcd u_old_ft, v_old_ft, w_old_ft, T_old_ft;
    Eigen::MatrixXcd u_new_ft, v_new_ft, w_new_ft, T_new_ft;
    Eigen::MatrixXcd body_force_x_ft, body_force_y_ft, body_force_z_ft;
    Eigen::MatrixXcd p_source_ft, p_new_ft, p_old_ft;
    Eigen::VectorXd p_temp_vec_real, wave_nos_square, wave_nos;
    Eigen::VectorXcd p_temp_vec_comp;
    Eigen::MatrixXd vel_temp_mat_real_xy, vel_temp_mat_real_z, x, y, z;
    Eigen::MatrixXcd u_source_old_ft, v_source_old_ft, w_source_old_ft, T_source_old_ft, vel_temp_mat_comp_xy, vel_temp_mat_comp_z;
    Eigen::MatrixXcd u_source_old_old_ft, v_source_old_old_ft, w_source_old_old_ft, T_source_old_old_ft; // used only for multi-step method
    Eigen::MatrixXcd dft_matrix, idft_matrix;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, w_dirichlet_flag, p_dirichlet_flag, T_dirichlet_flag;
    bool p_bc_full_neumann;
    int temporal_order = -1, it, nw, dim;
    double period, thermal_k, thermal_Cp;
    double dft_idft_timer = 0.0, ppe_timer = 0.0, vel_timer = 0.0, total_time_marching_timer = 0.0;
    vector<int> ppe_num_iter;
    // vector<double> ppe_rel_res;
};
class PARTICLE_TRANSPORT_FOURIER
{ // track particles with flow in periodic fourier flow in Z direction
public:
    // Functions
    PARTICLE_TRANSPORT_FOURIER(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, FRACTIONAL_STEP_FOURIER &fractional_step_fourier, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, Eigen::VectorXd &stokes_num, Eigen::VectorXd &density, double flow_charac_time);
    void single_timestep(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, FRACTIONAL_STEP_FOURIER &fractional_step_fourier, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, double &physical_time);
    void calc_neighbors_xy_plane(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position);
    void calc_triangles_xy_plane(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);
    void calc_dt(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &velocity);
    void update_nearest_vert_xy_plane(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, FRACTIONAL_STEP_FOURIER &fractional_step_fourier, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity);
    void calc_z_plane_range(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, FRACTIONAL_STEP_FOURIER &fractional_step_fourier, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity);
    void update_particle_nb_triangle_shape_function(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity);
    double calc_triangle_area_double(double x1, double y1, double x2, double y2, double x3, double y3);
    void apply_boundary_conditions(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, FRACTIONAL_STEP_FOURIER &fractional_step_fourier, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, double time_offset);
    void interp_flow_velocities(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, FRACTIONAL_STEP_FOURIER &fractional_step_fourier, Eigen::MatrixXd &position, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, double time_offset);
    void calc_acceleration(PARAMETERS &parameters, Eigen::MatrixXd &velocity);

    // Variables
    int np;                                 // no. of particles
    Eigen::MatrixXd RK_k1;                  // stores position and velocity of particles (row: [x,y,x_dot,y_dot]) size: [np X (2*dim)]
    Eigen::MatrixXd RK_k2;                  // stores position and velocity of particles (row: [x,y,x_dot,y_dot]) size: [np X (2*dim)]
    Eigen::MatrixXd vel_temp, pos_temp;     // stores position and velocity temporarily (row: [x,y,x_dot,y_dot]) size: [np X dim]
    PointCloud<double> points_xyz_nf;       // nanoflann structure of point coordinates (size: [points.nv X dim])
    Eigen::MatrixXi particle_nb_triangle;   // nearest point numbers forming a triangle for each particle (size: [np X (dim+1)])
    Eigen::VectorXi particle_z_plane_range; // particle lies between z[*this] and z[*this+1] (size: [np])
    Eigen::MatrixXd shape_function;         // shape function for interpolation at each particle (size: [np X (dim+1)])
    Eigen::MatrixXi point_nb_point;         // nearest points for each point (size: [points.nv X (small number)])
    Eigen::MatrixXi point_triangles;        // point-pairs which form edges of triangles incident on the every point (size: [points.nv X (dim X small number)])
    Eigen::MatrixXd point_triangles_2_area; // areas of triangles incident on the every point (size: [points.nv X (small number)])
    Eigen::VectorXd u_flow, v_flow, w_flow; // interpolated values of flow velocities (size: [np])
    Eigen::VectorXd diameter;               // diameter of particles (size: [np])
    Eigen::VectorXd mass;                   // mass of particles (size: [np])
    Eigen::MatrixXd acceleration;           // acceleration of particles (size: [np X dim])
    vector<bool> boundary_flag;             // true: particle lies on boundary else false (size: [np])
    double dt;                              // timestep for particle transport
    double z_min, z_max;                    // min and max values of Z coordinate
    int it, nt;                             // timesteps of particle transport per timestep of fluid flow
};

class PARTICLE_TRANSPORT
{ // track particles with flow
public:
    // Functions
    PARTICLE_TRANSPORT(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, Eigen::VectorXd &stokes_num, Eigen::VectorXd &density, double flow_charac_time);
    void single_timestep(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, double &physical_time);
    void calc_neighbors(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position);
    void calc_triangles(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);
    // void update_neighbors(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity);
    void update_particle_nb_triangle_shape_function(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity);
    void update_nearest_vert(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity);
    // void calc_shape_function(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity);
    void apply_boundary_conditions(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, double time_offset);
    void interp_flow_velocities(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, double time_offset);
    double calc_triangle_area_double(double x1, double y1, double x2, double y2, double x3, double y3);
    void calc_dt(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::MatrixXd &velocity);
    void calc_acceleration(PARAMETERS &parameters, Eigen::MatrixXd &velocity);

    // Variables
    int np;                               // no. of particles
    Eigen::MatrixXd RK_k1;                // stores position and velocity of particles (row: [x,y,x_dot,y_dot]) size: [np X (2*dim)]
    Eigen::MatrixXd RK_k2;                // stores position and velocity of particles (row: [x,y,x_dot,y_dot]) size: [np X (2*dim)]
    Eigen::MatrixXd vel_temp, pos_temp;   // stores position and velocity temporarily (row: [x,y,x_dot,y_dot]) size: [np X dim]
    PointCloud<double> points_xyz_nf;     // nanoflann structure of point coordinates (size: [points.nv X dim])
    Eigen::MatrixXi particle_nb_triangle; // nearest point numbers forming a triangle for each particle (size: [np X (dim+1)])
    // Eigen::MatrixXd particle_nb_point_dist_sq; //square of distances of nearest points for each particle (size: [np X (dim+1)])
    Eigen::MatrixXd shape_function;         // shape function for interpolation at each particle (size: [np X (dim+1)])
    Eigen::MatrixXi point_nb_point;         // nearest points for each point (size: [points.nv X (small number)])
    Eigen::MatrixXi point_triangles;        // point-pairs which form edges of triangles incident on the every point (size: [points.nv X (dim X small number)])
    Eigen::MatrixXd point_triangles_2_area; // areas of triangles incident on the every point (size: [points.nv X (small number)])
    Eigen::VectorXd u_flow, v_flow;         // interpolated values of flow velocities (size: [np])
    Eigen::VectorXd diameter;               // diameter of particles (size: [np])
    Eigen::VectorXd mass;                   // mass of particles (size: [np])
    Eigen::MatrixXd acceleration;           // acceleration of particles (size: [np X dim])
    vector<bool> boundary_flag;             // true: particle lies on boundary else false (size: [np])
    double dt;                              // timestep for particle transport
    int it, nt;                             // timesteps of particle transport per timestep of fluid flow
};

class IMPLICIT_SCALAR_TRANSPORT_SOLVER
{ // scalar transport with spatially varying velocity field: BDF2
public:
    // Functions
    IMPLICIT_SCALAR_TRANSPORT_SOLVER(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &dirichlet_flag1, int precond_freq_it1, double unsteady_coeff1, double conv_coeff1, double diff_coeff1, bool solver_log_flag1);
    void single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &phi_new, Eigen::VectorXd &phi_old, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, int it1);
    void set_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new);
    void modify_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new);
    void calc_nb_points_col_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);

    // Variables
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::IncompleteLUT<double>> solver_eigen;
    Eigen::VectorXd zero_vector, source, phi_old_old;
    vector<bool> dirichlet_flag;
    vector<int> nb_points_col_matrix;
    bool bc_full_neumann, solver_log_flag;
    int it, precond_freq_it;
    double unsteady_coeff, conv_coeff, diff_coeff;
    double bdf2_alpha_1 = 1.5, bdf2_alpha_2 = -2.0, bdf2_alpha_3 = 0.5; // https://en.wikipedia.org/wiki/Backward_differentiation_formula
};

class SEMI_IMPLICIT_SPLIT_SOLVER
{ // iterative split solver for Navier-Stokes equations
public:
    // Functions
    SEMI_IMPLICIT_SPLIT_SOLVER(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &u_dirichlet_flag1, vector<bool> &v_dirichlet_flag1, vector<bool> &p_dirichlet_flag1, int n_outer_iter1, double iterative_tolerance1, int precond_freq_it1);
    void check_bc(POINTS &points, PARAMETERS &parameters);
    void single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, int it1, vector<int> &n_outer_iter_log, vector<double> &iterative_l1_err_log, vector<double> &iterative_max_err_log);
    void single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y, int it1, vector<int> &n_outer_iter_log, vector<double> &iterative_l1_err_log, vector<double> &iterative_max_err_log);
    void set_vel_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new);
    void calc_nb_points_col_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);
    void modify_vel_matrix(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new);
    void calc_vel(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_vel(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y);
    void calc_pressure(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_vel_corr(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new);
    void extras(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);

    // Variables
    SOLVER solver_p;
    Eigen::SparseMatrix<double, Eigen::RowMajor> matrix_u, matrix_v;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::IncompleteLUT<double>> solver_eigen_u, solver_eigen_v;
    Eigen::VectorXd zero_vector, zero_vector_1;
    Eigen::VectorXd normal_mom_x, normal_mom_y;
    Eigen::VectorXd p_prime, p_source, u_source, v_source, u_prime, v_prime, u_iter_old, v_iter_old, u_old_old, v_old_old;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    vector<int> nb_points_col_matrix_u, nb_points_col_matrix_v;
    bool p_bc_full_neumann;
    double iterative_tolerance, iterative_l1_err, iterative_max_err;
    int it, n_outer_iter, outer_iter, precond_freq_it;
    double bdf2_alpha_1 = 1.5, bdf2_alpha_2 = -2.0, bdf2_alpha_3 = 0.5; // https://en.wikipedia.org/wiki/Backward_differentiation_formula
};

class FRACTIONAL_STEP_FSI
{ // hat velocity formulation for fluid structure interaction
public:
    // Functions
    FRACTIONAL_STEP_FSI(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &u_dirichlet_flag1, vector<bool> &v_dirichlet_flag1, vector<bool> &p_dirichlet_flag1, int temporal_order1);
    void check_bc(POINTS &points, PARAMETERS &parameters);
    double single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, int it1);
    void calc_flags(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);
    void update_cloud(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);
    void update_RBF_coeff(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);
    void calc_vel_hat(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_pressure(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_vel_corr(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old);
    double extras(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void EIGEN_set_ppe_coeff(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);
    void EIGEN_update_ppe_coeff(POINTS &points, CLOUD &cloud, PARAMETERS &parameters);

    // Variables
    // SOLVER solver_p;
    Eigen::SparseMatrix<double, Eigen::RowMajor> ppe_coeff;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::IncompleteLUT<double>> solver_eigen_ilu_bicgstab;
    // Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>> solver_eigen_bicgstab;
    Eigen::VectorXd zero_vector, zero_vector_1;
    Eigen::VectorXd uh, vh;
    Eigen::VectorXd p_source;
    Eigen::VectorXd u_source_old, v_source_old;
    Eigen::VectorXd u_source_old_old, v_source_old_old; // used only for multi-step method
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag;
    vector<bool> iv_active_flag_new, iv_active_flag_old, iv_update_flag;
    bool p_bc_full_neumann;
    int temporal_order = -1, it;
};

class COUPLED_NEWTON
{ // coupled solver for Navier-Stokes: Vanka SP, Leaf GK. Fully-coupled solution of pressure-linked fluid flow equations. Argonne National Lab., IL (USA); 1983 Aug 1.
public:
    // Functions
    COUPLED_NEWTON(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &p_dirichlet_flag1, int n_outer_iter1, int n_precond1, int n_linear_iter1);
    void iterate(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_source(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_jacobian(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_jacobian_triplet_xmom(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_jacobian_triplet_ymom(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);
    void calc_jacobian_triplet_cont(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old);

    // Variables
    Eigen::SparseMatrix<double, Eigen::RowMajor> jacobian;
    vector<Eigen::Triplet<double>> triplet;
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::IncompleteLUT<double>> solver_eigen_ilu_bicgstab;
    Eigen::VectorXd X_old, X_new, delta_X, source, source_xmom, source_ymom, source_cont;
    vector<bool> p_dirichlet_flag;
    bool p_bc_full_neumann;
    int n_outer_iter, n_precond, n_linear_iter;
};

class SOLIDIFICATION
{
public:
    // Functions
    SOLIDIFICATION(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, int temporal_order1);
    void single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &T_new, Eigen::VectorXd &T_old, Eigen::VectorXd &fs_new, Eigen::VectorXd &fs_old, int it);

    // Variables
    Eigen::VectorXd dfs_dT, T_source;
    Eigen::VectorXd T_old_old, fs_old_old; // used only for multi-step method
    double Tliq = 915.0;
    double Tsol = 850.0;
    double Tf = 935.2;
    double Teps = 2.0;
    double k_partition = 0.13;
    double Lf = 390000.0;
    double Cp = 1006.0;
    double alpha = 5E-5;
    int temporal_order = -1;
};

#endif