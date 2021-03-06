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

FRACTIONAL_STEP_1::FRACTIONAL_STEP_1(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &u_dirichlet_flag1, vector<bool> &v_dirichlet_flag1, vector<bool> &p_dirichlet_flag1, int temporal_order1)
{
    temporal_order = temporal_order1;
    u_dirichlet_flag = u_dirichlet_flag1, v_dirichlet_flag = v_dirichlet_flag1, p_dirichlet_flag = p_dirichlet_flag1;
    check_bc(points, parameters);
    clock_t clock_t1 = clock(), clock_t2 = clock();
    solver_p.init(points, cloud, parameters, p_dirichlet_flag, 0.0, 0.0, 1.0, true);
    parameters.factoring_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;

    zero_vector = Eigen::VectorXd::Zero(points.nv);
    zero_vector_1 = Eigen::VectorXd::Zero(points.nv + 1);
    uh = zero_vector, vh = zero_vector;

    p_bc_full_neumann = true;
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv] && p_dirichlet_flag[iv])
        { //boundary point found with dirichlet BC
            p_bc_full_neumann = false;
            break;
        }

    if (p_bc_full_neumann)
        p_source = zero_vector_1; //this is full Neumann for pressure
    else
        p_source = zero_vector;
    u_source_old = zero_vector, v_source_old = zero_vector;
    if (temporal_order == 2)
        u_source_old_old = zero_vector, v_source_old_old = zero_vector;
}

void FRACTIONAL_STEP_1::single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, int it1)
{
    it = it1;
    if (p_source.size() != p_new.size())
        p_new = Eigen::VectorXd::Zero(p_source.size());
    if (p_source.size() != p_old.size())
        p_old = Eigen::VectorXd::Zero(p_source.size());
    calc_vel_hat(points, parameters, u_old, v_old, p_old);
    calc_pressure(points, parameters, u_new, v_new, p_new, u_old, v_old, p_old);
    calc_vel_corr(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old);
    extras(points, parameters, u_new, v_new, p_new, u_old, v_old, p_old);
}

void FRACTIONAL_STEP_1::single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y, int it1)
{
    it = it1;
    if (p_source.size() != p_new.size())
        p_new = Eigen::VectorXd::Zero(p_source.size());
    if (p_source.size() != p_old.size())
        p_old = Eigen::VectorXd::Zero(p_source.size());
    calc_vel_hat(points, parameters, u_old, v_old, p_old, body_force_x, body_force_y);
    calc_pressure(points, parameters, u_new, v_new, p_new, u_old, v_old, p_old);
    calc_vel_corr(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old);
    extras(points, parameters, u_new, v_new, p_new, u_old, v_old, p_old);
}

void FRACTIONAL_STEP_1::check_bc(POINTS &points, PARAMETERS &parameters)
{
    int u_dirichlet_flag_sum = accumulate(u_dirichlet_flag.begin(), u_dirichlet_flag.end(), 0);
    if (u_dirichlet_flag_sum == 0)
    {
        printf("\n\nERROR from FRACTIONAL_STEP_1::check_bc Setting u_dirichlet_flag to full Neumann BC is not permitted; sum of u_dirichlet_flag: %i\n\n", u_dirichlet_flag_sum);
        throw bad_exception();
    }
    int v_dirichlet_flag_sum = accumulate(v_dirichlet_flag.begin(), v_dirichlet_flag.end(), 0);
    if (v_dirichlet_flag_sum == 0)
    {
        printf("\n\nERROR from FRACTIONAL_STEP_1::check_bc Setting v_dirichlet_flag to full Neumann BC is not permitted; sum of v_dirichlet_flag: %i\n\n", v_dirichlet_flag_sum);
        throw bad_exception();
    }
    if (parameters.rho < 0 || parameters.mu < 0)
    {
        printf("\n\nERROR from FRACTIONAL_STEP_1::check_bc Some parameters are not set; parameters.rho: %g, parameters.mu: %g\n\n", parameters.rho, parameters.mu);
        throw bad_exception();
    }
    if (temporal_order != 1 && temporal_order != 2)
    {
        printf("\n\nERROR from FRACTIONAL_STEP_1::check_bc temporal_order should be either '1' or '2'; current value: %i\n\n", temporal_order);
        throw bad_exception();
    }
}

void FRACTIONAL_STEP_1::calc_vel_hat(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y)
{
    u_source_old = (parameters.rho * (-u_old.cwiseProduct(points.grad_x_matrix_EIGEN * u_old) - v_old.cwiseProduct(points.grad_y_matrix_EIGEN * u_old))) + (parameters.mu * points.laplacian_matrix_EIGEN * u_old) + body_force_x;
    v_source_old = (parameters.rho * (-u_old.cwiseProduct(points.grad_x_matrix_EIGEN * v_old) - v_old.cwiseProduct(points.grad_y_matrix_EIGEN * v_old))) + (parameters.mu * points.laplacian_matrix_EIGEN * v_old) + body_force_y;
    if (temporal_order == 1 || it == 0)
    { //Euler method for first timestep of multistep method
        uh = u_old + ((parameters.dt / parameters.rho) * u_source_old);
        vh = v_old + ((parameters.dt / parameters.rho) * v_source_old);
    }
    else
    { //Second order Adam-Bashforth for subsequent timesteps
        uh = u_old + ((parameters.dt / parameters.rho) * (1.5 * u_source_old - 0.5 * u_source_old_old));
        vh = v_old + ((parameters.dt / parameters.rho) * (1.5 * v_source_old - 0.5 * v_source_old_old));
    }
}

void FRACTIONAL_STEP_1::calc_vel_hat(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    u_source_old = (parameters.rho * (-u_old.cwiseProduct(points.grad_x_matrix_EIGEN * u_old) - v_old.cwiseProduct(points.grad_y_matrix_EIGEN * u_old))) + (parameters.mu * points.laplacian_matrix_EIGEN * u_old);
    v_source_old = (parameters.rho * (-u_old.cwiseProduct(points.grad_x_matrix_EIGEN * v_old) - v_old.cwiseProduct(points.grad_y_matrix_EIGEN * v_old))) + (parameters.mu * points.laplacian_matrix_EIGEN * v_old);
    if (temporal_order == 1 || it == 0)
    { //Euler method for first timestep of multistep method
        uh = u_old + ((parameters.dt / parameters.rho) * u_source_old);
        vh = v_old + ((parameters.dt / parameters.rho) * v_source_old);
    }
    else
    { //Second order Adam-Bashforth
        uh = u_old + ((parameters.dt / parameters.rho) * (1.5 * u_source_old - 0.5 * u_source_old_old));
        vh = v_old + ((parameters.dt / parameters.rho) * (1.5 * v_source_old - 0.5 * v_source_old_old));
    }
}

void FRACTIONAL_STEP_1::calc_pressure(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    int dim = parameters.dimension;
    p_source.head(points.nv) = ((points.grad_x_matrix_EIGEN_internal * uh) + (points.grad_y_matrix_EIGEN_internal * vh)) * parameters.rho / parameters.dt;
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
        {
            if (p_dirichlet_flag[iv]) //dirichlet BC
                p_source[iv] = p_old[iv];
            else //normal momentum
                p_source[iv] = -parameters.rho * ((u_old[iv] - uh[iv]) * points.normal[dim * iv] + (v_old[iv] - vh[iv]) * points.normal[dim * iv + 1]) / parameters.dt;
        }
    if (p_source.rows() == points.nv + 1)
        p_source[points.nv] = 0.0;
    solver_p.general_solve(points, parameters, p_new, p_old, p_source);
}

void FRACTIONAL_STEP_1::calc_vel_corr(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old)
{
    u_new = uh - (parameters.dt * (points.grad_x_matrix_EIGEN * p_new.head(points.nv)) / parameters.rho);
    v_new = vh - (parameters.dt * (points.grad_y_matrix_EIGEN * p_new.head(points.nv)) / parameters.rho);
    double diag_coeff, rhs, off_diag_coeff;
    int ivnb, dim = parameters.dimension;
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
        {
            if (!p_dirichlet_flag[iv])
            { //for dirichlet pressure, u_new comes from velocity correction above
                if (u_dirichlet_flag[iv])
                    u_new[iv] = u_old[iv]; //dirichlet BC
                else
                { //neumann BC
                    rhs = 0.0;
                    for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                    {
                        ivnb = cloud.nb_points_col[i1];
                        if (iv == ivnb)
                            diag_coeff = (cloud.grad_x_coeff[i1] * points.normal[dim * iv]) + (cloud.grad_y_coeff[i1] * points.normal[dim * iv + 1]);
                        else
                        {
                            off_diag_coeff = (cloud.grad_x_coeff[i1] * points.normal[dim * iv]) + (cloud.grad_y_coeff[i1] * points.normal[dim * iv + 1]);
                            rhs = rhs - (off_diag_coeff * u_new[ivnb]);
                        }
                    }
                    u_new[iv] = (rhs / diag_coeff);
                }
            }
        }
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
        {
            if (!p_dirichlet_flag[iv])
            { //for dirichlet pressure, v_new comes from velocity correction above
                if (v_dirichlet_flag[iv])
                    v_new[iv] = v_old[iv]; //dirichlet BC
                else
                { //neumann BC
                    rhs = 0.0;
                    for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                    {
                        ivnb = cloud.nb_points_col[i1];
                        if (iv == ivnb)
                            diag_coeff = (cloud.grad_x_coeff[i1] * points.normal[dim * iv]) + (cloud.grad_y_coeff[i1] * points.normal[dim * iv + 1]);
                        else
                        {
                            off_diag_coeff = (cloud.grad_x_coeff[i1] * points.normal[dim * iv]) + (cloud.grad_y_coeff[i1] * points.normal[dim * iv + 1]);
                            rhs = rhs - (off_diag_coeff * v_new[ivnb]);
                        }
                    }
                    v_new[iv] = (rhs / diag_coeff);
                }
            }
        }
}

void FRACTIONAL_STEP_1::extras(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    // double total_steady_err, max_err, l1_err;
    // calc_max_l1_error(u_old, u_new, max_err, l1_err);
    // total_steady_err = l1_err / parameters.dt;
    // calc_max_l1_error(v_old, v_new, max_err, l1_err);
    // total_steady_err += l1_err / parameters.dt;
    if (temporal_order == 2)
        u_source_old_old = u_source_old, v_source_old_old = v_source_old;
    // parameters.steady_error_log.push_back(total_steady_err);
    // return total_steady_err;
}