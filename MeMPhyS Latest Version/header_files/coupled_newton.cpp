//Author: Dr. Shantanu Shahane
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
#include "class.hpp"
using namespace std;

COUPLED_NEWTON::COUPLED_NEWTON(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &p_dirichlet_flag1, int n_outer_iter1, int n_precond1, int n_linear_iter1)
{
    p_dirichlet_flag = p_dirichlet_flag1;
    p_bc_full_neumann = true;
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv] && p_dirichlet_flag[iv])
        { //boundary point found with dirichlet BC
            p_bc_full_neumann = false;
            break;
        }
    if (p_bc_full_neumann)
    {
        cout << "\n\nCOUPLED_NEWTON::COUPLED_NEWTON Full Neumann BC of pressure\n\n";
        // throw bad_exception();
    }
    if (strcmp(parameters.solver_type.c_str(), "eigen_ilu_bicgstab") != 0)
    {
        cout << "\n\nERROR from COUPLED_NEWTON::COUPLED_NEWTON solver_type should be eigen_ilu_bicgstab; current value: " << parameters.solver_type << "\n\n";
        throw bad_exception();
    }
    if (p_bc_full_neumann)
    {
        X_new = Eigen::VectorXd::Zero((parameters.dimension + 1) * points.nv + 1);
        X_old = Eigen::VectorXd::Zero((parameters.dimension + 1) * points.nv + 1);
        source = Eigen::VectorXd::Zero((parameters.dimension + 1) * points.nv + 1);
    }
    else
    {
        delta_X = Eigen::VectorXd::Zero((parameters.dimension + 1) * points.nv);
        X_new = Eigen::VectorXd::Zero((parameters.dimension + 1) * points.nv);
        X_old = Eigen::VectorXd::Zero((parameters.dimension + 1) * points.nv);
        source = Eigen::VectorXd::Zero((parameters.dimension + 1) * points.nv);
    }
    n_outer_iter = n_outer_iter1, n_precond = n_precond1, n_linear_iter = n_linear_iter1;
    source_xmom = Eigen::VectorXd::Zero(points.nv);
    source_ymom = Eigen::VectorXd::Zero(points.nv);
    source_cont = Eigen::VectorXd::Zero(points.nv);
    solver_eigen_ilu_bicgstab.setTolerance(parameters.solver_tolerance); //default is machine precision (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#ac160a444af8998f93da9aa30e858470d)
    solver_eigen_ilu_bicgstab.setMaxIterations(n_linear_iter);           //default is twice number of columns (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#af83de7a7d31d9d4bd1fef6222b07335b)
    solver_eigen_ilu_bicgstab.preconditioner().setDroptol(parameters.precond_droptol);
    parameters.factoring_timer = 0.0;
}

void COUPLED_NEWTON::iterate(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    for (int iv = 0; iv < points.nv; iv++)
        X_old[3 * iv] = u_old[iv], X_old[3 * iv + 1] = v_old[iv], X_old[3 * iv + 2] = p_old[iv];
    clock_t t1 = clock(), t2;
    double max_err, l1_err;
    cout << "\n\n";
    for (int iter = 0; iter < n_outer_iter; iter++)
    {
        t2 = clock();
        calc_jacobian(points, cloud, parameters, u_old, v_old, p_old);
        solver_eigen_ilu_bicgstab.compute(jacobian); //preconditioning
        parameters.factoring_timer = parameters.factoring_timer + ((double)(clock() - t2)) / CLOCKS_PER_SEC;
        calc_source(points, cloud, parameters, u_old, v_old, p_old);
        delta_X = solver_eigen_ilu_bicgstab.solve(source);
        X_new = X_old - delta_X;
        double abs_res = (jacobian * delta_X - source).norm();
        parameters.rel_res_log.push_back(abs_res / source.norm());
        parameters.abs_res_log.push_back(abs_res);
        parameters.n_iter_actual.push_back(solver_eigen_ilu_bicgstab.iterations());
        calc_max_l1_error(X_old, X_new, max_err, l1_err);
        if (p_bc_full_neumann)
            printf("    COUPLED_NEWTON::iterate regularization constant: %g, outer_iter: %i\n", delta_X[(parameters.dimension + 1) * points.nv], iter);
        printf("    COUPLED_NEWTON::iterate outer_iter: %i at time: %g seconds\n", iter, ((double)(clock() - t1)) / CLOCKS_PER_SEC);
        printf("    COUPLED_NEWTON::iterate steady state max_err: %g, l1_err: %g, steady_tolerance: %g\n", max_err, l1_err, parameters.steady_tolerance);
        printf("    linear_iter: %i, linear_abs_res: %g, linear_rel_res: %g, linear_solver_tol: %g\n\n", parameters.n_iter_actual[iter], parameters.abs_res_log[iter], parameters.rel_res_log[iter], parameters.solver_tolerance);
        X_old = X_new;
        for (int iv = 0; iv < points.nv; iv++)
            u_old[iv] = X_old[3 * iv], v_old[iv] = X_old[3 * iv + 1], p_old[iv] = X_old[3 * iv + 2];

        if (l1_err < parameters.steady_tolerance && iter > 1)
            break;
    }
    for (int iv = 0; iv < points.nv; iv++)
        u_new[iv] = X_new[3 * iv], v_new[iv] = X_new[3 * iv + 1], p_new[iv] = X_new[3 * iv + 2];
}

void COUPLED_NEWTON::calc_source(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    source_xmom = points.grad_x_matrix_EIGEN * (u_old.cwiseProduct(u_old));
    source_xmom = source_xmom + points.grad_y_matrix_EIGEN * (u_old.cwiseProduct(v_old));
    source_xmom = source_xmom + points.grad_x_matrix_EIGEN * (p_old / parameters.rho);
    source_xmom = source_xmom - points.laplacian_matrix_EIGEN * (u_old * parameters.mu / parameters.rho);

    source_ymom = points.grad_x_matrix_EIGEN * (u_old.cwiseProduct(v_old));
    source_ymom = source_ymom + points.grad_y_matrix_EIGEN * (v_old.cwiseProduct(v_old));
    source_ymom = source_ymom + points.grad_y_matrix_EIGEN * (p_old / parameters.rho);
    source_ymom = source_ymom - points.laplacian_matrix_EIGEN * (v_old * parameters.mu / parameters.rho);

    source_cont = points.grad_x_matrix_EIGEN * u_old + points.grad_y_matrix_EIGEN * v_old;

    int dim = parameters.dimension;
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (points.boundary_flag[iv])
        {                                                   //should be zero at boundaries: dirichlet
            source[3 * iv] = 0.0, source[3 * iv + 1] = 0.0; //u and v always dirichlet
            if (p_dirichlet_flag[iv])
                source[3 * iv + 2] = 0.0; //dirichlet for pressure
            else
            { //neumann for pressure: normal momentum eq
                source[3 * iv + 2] = (points.normal[dim * iv] * source_xmom[iv]) + (points.normal[dim * iv + 1] * source_ymom[iv]);
            }
        }
        else
        {
            source[3 * iv] = source_xmom[iv];
            source[3 * iv + 1] = source_ymom[iv];
            source[3 * iv + 2] = source_cont[iv];
        }
    }
    source[points.nv] = 0.0;
}

void COUPLED_NEWTON::calc_jacobian_triplet_xmom(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    int ivnb;
    double coeff;
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (points.boundary_flag[iv]) //dirichlet
            triplet.push_back(Eigen::Triplet<double>(3 * iv, 3 * iv, 1.0));
        else
        {
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                ivnb = cloud.nb_points_col[i1];
                coeff = (2.0 * cloud.grad_x_coeff[i1] * u_old[ivnb]);
                coeff = coeff + (cloud.grad_y_coeff[i1] * v_old[ivnb]);
                coeff = coeff - (cloud.laplacian_coeff[i1] * parameters.mu / parameters.rho);
                triplet.push_back(Eigen::Triplet<double>(3 * iv, 3 * ivnb, coeff)); // (dF[3*iv] / du[nb])
                coeff = (cloud.grad_y_coeff[i1] * u_old[ivnb]);
                triplet.push_back(Eigen::Triplet<double>(3 * iv, 3 * ivnb + 1, coeff)); // (dF[3*iv] / dv[nb])
                coeff = (cloud.grad_x_coeff[i1] / parameters.rho);
                triplet.push_back(Eigen::Triplet<double>(3 * iv, 3 * ivnb + 2, coeff)); // (dF[3*iv] / dp[nb])
            }
        }
    }
}

void COUPLED_NEWTON::calc_jacobian_triplet_ymom(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    int ivnb;
    double coeff;
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (points.boundary_flag[iv]) //dirichlet
            triplet.push_back(Eigen::Triplet<double>(3 * iv + 1, 3 * iv + 1, 1.0));
        else
        {
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                ivnb = cloud.nb_points_col[i1];
                coeff = (cloud.grad_x_coeff[i1] * v_old[ivnb]);
                triplet.push_back(Eigen::Triplet<double>(3 * iv + 1, 3 * ivnb, coeff)); // (dF[3*iv+1] / du[nb])
                coeff = (cloud.grad_x_coeff[i1] * u_old[ivnb]);
                coeff = coeff + (2.0 * cloud.grad_y_coeff[i1] * v_old[ivnb]);
                coeff = coeff - (cloud.laplacian_coeff[i1] * parameters.mu / parameters.rho);
                triplet.push_back(Eigen::Triplet<double>(3 * iv + 1, 3 * ivnb + 1, coeff)); // (dF[3*iv+1] / dv[nb])
                coeff = (cloud.grad_y_coeff[i1] / parameters.rho);
                triplet.push_back(Eigen::Triplet<double>(3 * iv + 1, 3 * ivnb + 2, coeff)); // (dF[3*iv+1] / dp[nb])
            }
        }
    }
}

void COUPLED_NEWTON::calc_jacobian_triplet_cont(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    int ivnb, dim = parameters.dimension;
    double coeff, coeff_x, coeff_y;
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (p_bc_full_neumann)
        {
            triplet.push_back(Eigen::Triplet<double>(3 * iv + 2, 3 * points.nv, 1.0)); //derivative wrt pressure
            triplet.push_back(Eigen::Triplet<double>(3 * points.nv, 3 * iv + 2, 1.0)); //derivative wrt regularization parameter
        }
        if (points.boundary_flag[iv])
        {
            if (p_dirichlet_flag[iv]) //dirichlet for pressure
                triplet.push_back(Eigen::Triplet<double>(3 * iv + 2, 3 * iv + 2, 1.0));
            else
            { //neumann for pressure: normal momentum eq
                for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                {
                    ivnb = cloud.nb_points_col[i1];
                    coeff_x = (2.0 * cloud.grad_x_coeff[i1] * u_old[ivnb]);
                    coeff_x = coeff_x + (cloud.grad_y_coeff[i1] * v_old[ivnb]);
                    coeff_x = coeff_x - (cloud.laplacian_coeff[i1] * parameters.mu / parameters.rho);
                    coeff_y = (cloud.grad_x_coeff[i1] * v_old[ivnb]);
                    coeff = (points.normal[dim * iv] * coeff_x) + (points.normal[dim * iv + 1] * coeff_y);
                    triplet.push_back(Eigen::Triplet<double>(3 * iv + 2, 3 * ivnb, coeff)); // (dF[3*iv] / du[nb])
                    coeff_x = (cloud.grad_y_coeff[i1] * u_old[ivnb]);
                    coeff_y = (cloud.grad_x_coeff[i1] * u_old[ivnb]);
                    coeff_y = coeff_y + (2.0 * cloud.grad_y_coeff[i1] * v_old[ivnb]);
                    coeff_y = coeff_y - (cloud.laplacian_coeff[i1] * parameters.mu / parameters.rho);
                    coeff = (points.normal[dim * iv] * coeff_x) + (points.normal[dim * iv + 1] * coeff_y);
                    triplet.push_back(Eigen::Triplet<double>(3 * iv + 2, 3 * ivnb + 1, coeff)); // (dF[3*iv] / dv[nb])
                    coeff_x = (cloud.grad_x_coeff[i1] / parameters.rho);
                    coeff_y = (cloud.grad_y_coeff[i1] / parameters.rho);
                    coeff = (points.normal[dim * iv] * coeff_x) + (points.normal[dim * iv + 1] * coeff_y);
                    triplet.push_back(Eigen::Triplet<double>(3 * iv + 2, 3 * ivnb + 2, coeff)); // (dF[3*iv] / dp[nb])
                }
            }
        }
        else
        {
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                ivnb = cloud.nb_points_col[i1];
                coeff = (cloud.grad_x_coeff[i1]);
                triplet.push_back(Eigen::Triplet<double>(3 * iv + 2, 3 * ivnb, coeff)); // (dF[3*iv+2] / du[nb])
                coeff = (cloud.grad_y_coeff[i1]);
                triplet.push_back(Eigen::Triplet<double>(3 * iv + 2, 3 * ivnb + 1, coeff)); // (dF[3*iv+2] / dv[nb])
            }
        }
    }
}

void COUPLED_NEWTON::calc_jacobian(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    calc_jacobian_triplet_xmom(points, cloud, parameters, u_old, v_old, p_old);
    calc_jacobian_triplet_ymom(points, cloud, parameters, u_old, v_old, p_old);
    calc_jacobian_triplet_cont(points, cloud, parameters, u_old, v_old, p_old);
    jacobian.resize(X_old.size(), X_old.size());
    jacobian.setFromTriplets(triplet.begin(), triplet.end());
    jacobian.makeCompressed();
    triplet.clear();
}