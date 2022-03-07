// Author: Dr. Shantanu Shahane
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

FRACTIONAL_STEP_FOURIER::FRACTIONAL_STEP_FOURIER(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &u_dirichlet_flag1, vector<bool> &v_dirichlet_flag1, vector<bool> &w_dirichlet_flag1, vector<bool> &p_dirichlet_flag1, double period1, int nw1, int temporal_order1)
{
    temporal_order = temporal_order1, period = period1, nw = nw1, dim = parameters.dimension;
    u_dirichlet_flag = u_dirichlet_flag1, v_dirichlet_flag = v_dirichlet_flag1, w_dirichlet_flag = w_dirichlet_flag1, p_dirichlet_flag = p_dirichlet_flag1;
    check_settings(points, parameters);

    p_bc_full_neumann = true;
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv] && p_dirichlet_flag[iv])
        { // boundary point found with dirichlet BC
            p_bc_full_neumann = false;
            break;
        }

    dft_matrix = get_dft_matrix(nw, period), idft_matrix = get_idft_matrix(nw, period);
    wave_nos = Eigen::VectorXd::Zero(2 * nw + 1);
    for (int iw = 0; iw < 2 * nw + 1; iw++)
        if (iw <= nw)
            wave_nos[iw] = iw * 2 * M_PI / period;
        else
            wave_nos[iw] = -wave_nos[2 * nw + 1 - iw];
    wave_nos_square = wave_nos.cwiseProduct(wave_nos);
    x = Eigen::MatrixXd::Zero(points.nv, 2 * nw + 1), y = x, z = x;
    double dz = period / (2.0 * nw + 1.0);
    for (int iw = 0; iw < 2 * nw + 1; iw++)
        for (int iv = 0; iv < points.nv; iv++)
        {
            x(iv, iw) = points.xyz[dim * iv];
            y(iv, iw) = points.xyz[dim * iv + 1];
            z(iv, iw) = dz * iw;
        }

    uh_ft = Eigen::MatrixXcd::Zero(points.nv, 2 * nw + 1), vh_ft = uh_ft, wh_ft = uh_ft;
    body_force_x_ft = uh_ft, body_force_y_ft = uh_ft, body_force_z_ft = uh_ft;
    vort_x_ft = uh_ft, vort_y_ft = uh_ft, vort_z_ft = uh_ft, q_criterion_ft = uh_ft;
    u_new_ft = uh_ft, v_new_ft = uh_ft, w_new_ft = uh_ft, p_new_ft = uh_ft;
    u_old_ft = uh_ft, v_old_ft = uh_ft, w_old_ft = uh_ft, p_old_ft = uh_ft;
    u_source_old_ft = uh_ft, v_source_old_ft = uh_ft, w_source_old_ft = uh_ft, vel_temp_mat_comp_xy = uh_ft, vel_temp_mat_comp_z = uh_ft;
    if (temporal_order == 2)
        u_source_old_old_ft = uh_ft, v_source_old_old_ft = uh_ft, w_source_old_old_ft = uh_ft;
    vel_temp_mat_real_xy = Eigen::MatrixXd::Zero(points.nv, 2 * nw + 1), vel_temp_mat_real_z = vel_temp_mat_real_xy;
    p_source_ft = Eigen::MatrixXcd::Zero(points.nv + 1, 2 * nw + 1); // defined with an extra entry for regularization but not always used
    p_temp_vec_real = Eigen::VectorXd::Zero(points.nv + 1);          // defined with an extra entry for regularization but not always used
    p_temp_vec_comp = Eigen::VectorXcd::Zero(points.nv + 1);         // defined with an extra entry for regularization but not always used
    set_p_coeff_eigen(points, cloud, parameters);
    set_p_solver_eigen(points, cloud, parameters);
}

FRACTIONAL_STEP_FOURIER::FRACTIONAL_STEP_FOURIER(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &u_dirichlet_flag1, vector<bool> &v_dirichlet_flag1, vector<bool> &w_dirichlet_flag1, vector<bool> &p_dirichlet_flag1, vector<bool> &T_dirichlet_flag1, double thermal_k1, double thermal_Cp1, double period1, int nw1, int temporal_order1)
{
    temporal_order = temporal_order1, period = period1, nw = nw1, dim = parameters.dimension;
    u_dirichlet_flag = u_dirichlet_flag1, v_dirichlet_flag = v_dirichlet_flag1, w_dirichlet_flag = w_dirichlet_flag1, p_dirichlet_flag = p_dirichlet_flag1, T_dirichlet_flag = T_dirichlet_flag1;
    thermal_k = thermal_k1, thermal_Cp = thermal_Cp1;
    check_settings(points, parameters);

    p_bc_full_neumann = true;
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv] && p_dirichlet_flag[iv])
        { // boundary point found with dirichlet BC
            p_bc_full_neumann = false;
            break;
        }

    dft_matrix = get_dft_matrix(nw, period), idft_matrix = get_idft_matrix(nw, period);
    wave_nos = Eigen::VectorXd::Zero(2 * nw + 1);
    for (int iw = 0; iw < 2 * nw + 1; iw++)
        if (iw <= nw)
            wave_nos[iw] = iw * 2 * M_PI / period;
        else
            wave_nos[iw] = -wave_nos[2 * nw + 1 - iw];
    wave_nos_square = wave_nos.cwiseProduct(wave_nos);
    x = Eigen::MatrixXd::Zero(points.nv, 2 * nw + 1), y = x, z = x;
    double dz = period / (2.0 * nw + 1.0);
    for (int iw = 0; iw < 2 * nw + 1; iw++)
        for (int iv = 0; iv < points.nv; iv++)
        {
            x(iv, iw) = points.xyz[dim * iv];
            y(iv, iw) = points.xyz[dim * iv + 1];
            z(iv, iw) = dz * iw;
        }

    uh_ft = Eigen::MatrixXcd::Zero(points.nv, 2 * nw + 1), vh_ft = uh_ft, wh_ft = uh_ft;
    body_force_x_ft = uh_ft, body_force_y_ft = uh_ft, body_force_z_ft = uh_ft;
    vort_x_ft = uh_ft, vort_y_ft = uh_ft, vort_z_ft = uh_ft, q_criterion_ft = uh_ft;
    u_new_ft = uh_ft, v_new_ft = uh_ft, w_new_ft = uh_ft, p_new_ft = uh_ft, T_new_ft = uh_ft;
    u_old_ft = uh_ft, v_old_ft = uh_ft, w_old_ft = uh_ft, p_old_ft = uh_ft, T_old_ft = uh_ft;
    u_source_old_ft = uh_ft, v_source_old_ft = uh_ft, w_source_old_ft = uh_ft, T_source_old_ft = uh_ft, vel_temp_mat_comp_xy = uh_ft, vel_temp_mat_comp_z = uh_ft;
    if (temporal_order == 2)
        u_source_old_old_ft = uh_ft, v_source_old_old_ft = uh_ft, w_source_old_old_ft = uh_ft, T_source_old_old_ft = uh_ft;
    vel_temp_mat_real_xy = Eigen::MatrixXd::Zero(points.nv, 2 * nw + 1), vel_temp_mat_real_z = vel_temp_mat_real_xy;
    p_source_ft = Eigen::MatrixXcd::Zero(points.nv + 1, 2 * nw + 1); // defined with an extra entry for regularization but not always used
    p_temp_vec_real = Eigen::VectorXd::Zero(points.nv + 1);          // defined with an extra entry for regularization but not always used
    p_temp_vec_comp = Eigen::VectorXcd::Zero(points.nv + 1);         // defined with an extra entry for regularization but not always used
    set_p_coeff_eigen(points, cloud, parameters);
    set_p_solver_eigen(points, cloud, parameters);
}

void FRACTIONAL_STEP_FOURIER::set_p_coeff_eigen(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
{
    Eigen::SparseMatrix<double, Eigen::RowMajor> empty_matrix;
    for (int iw = 0; iw <= nw; iw++) //[nw+1 : 2*nw+1] are complex conjugates of [nw : 1]
        p_coeff_eigen.push_back(empty_matrix);
    vector<Eigen::Triplet<double>> triplet;
    int iv_nb, system_size;
    double val;
    for (int iw = 0; iw <= nw; iw++)
    {
        for (int iv = 0; iv < points.nv; iv++)
        {
            if (points.boundary_flag[iv])
            {                             // lies on boundary
                if (p_dirichlet_flag[iv]) // apply dirichlet BC
                    triplet.push_back(Eigen::Triplet<double>(iv, iv, 1.0));
                else // apply neumann BC
                    for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                    {
                        iv_nb = cloud.nb_points_col[i1];
                        val = points.normal[dim * iv] * cloud.grad_x_coeff[i1] + points.normal[dim * iv + 1] * cloud.grad_y_coeff[i1];
                        triplet.push_back(Eigen::Triplet<double>(iv, iv_nb, val));
                    }
            }
            else
            {
                for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                {
                    iv_nb = cloud.nb_points_col[i1];
                    val = cloud.laplacian_coeff[i1];
                    if (iv == iv_nb)
                        val = val - wave_nos_square[iw];
                    triplet.push_back(Eigen::Triplet<double>(iv, iv_nb, val));
                }
            }
        }
        if (iw == 0 && p_bc_full_neumann)
            system_size = points.nv + 1;
        else // other waves have a -wave_nos[iw]**2 added to diagonal: thus matrix is invertible
            system_size = points.nv;
        if (system_size == points.nv + 1)
        { // extra constraint (regularization: http://www-e6.ijs.si/medusa/wiki/index.php/Poisson%27s_equation) for neumann BCs
            for (int iv = 0; iv < points.nv; iv++)
            {
                triplet.push_back(Eigen::Triplet<double>(iv, points.nv, 1.0)); // last column
                triplet.push_back(Eigen::Triplet<double>(points.nv, iv, 1.0)); // last row
            }
        }
        p_coeff_eigen[iw].resize(system_size, system_size);
        p_coeff_eigen[iw].setFromTriplets(triplet.begin(), triplet.end());
        p_coeff_eigen[iw].makeCompressed();
        triplet.clear();
    }
}

void FRACTIONAL_STEP_FOURIER::set_p_solver_eigen(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
{
    p_solver_eigen_ilu_bicgstab = new Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::IncompleteLUT<double>>[nw + 1];
    for (int iw = 0; iw <= nw; iw++)
    {
        p_solver_eigen_ilu_bicgstab[iw].setTolerance(parameters.solver_tolerance);
        p_solver_eigen_ilu_bicgstab[iw].setMaxIterations(parameters.n_iter);
        p_solver_eigen_ilu_bicgstab[iw].preconditioner().setDroptol(parameters.precond_droptol);
        p_solver_eigen_ilu_bicgstab[iw].compute(p_coeff_eigen[iw]);
    }
}

void FRACTIONAL_STEP_FOURIER::check_settings(POINTS &points, PARAMETERS &parameters)
{
    if (parameters.rho < 0 || parameters.mu < 0)
    {
        printf("\n\nERROR from FRACTIONAL_STEP_FOURIER::check_settings Some parameters are not set; parameters.rho: %g, parameters.mu: %g\n\n", parameters.rho, parameters.mu);
        throw bad_exception();
    }
    if (temporal_order != 1 && temporal_order != 2)
    {
        printf("\n\nERROR from FRACTIONAL_STEP_FOURIER::check_settings temporal_order should be either '1' or '2'; current value: %i\n\n", temporal_order);
        throw bad_exception();
    }
    if (dim != 2)
    {
        printf("\n\nERROR from FRACTIONAL_STEP_FOURIER::check_settings defined only for 2D problems parameters.dimension: %i\n\n", dim);
        throw bad_exception();
    }
}

void FRACTIONAL_STEP_FOURIER::calc_T(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &T_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &T_old)
{
    // -d(u_ft*T_ft)/dx - d(v_ft*T_ft)/dy:
    // T_source_old_ft = -points.grad_x_matrix_EIGEN * (u_old_ft.cwiseProduct(T_old_ft)) - points.grad_y_matrix_EIGEN * (v_old_ft.cwiseProduct(T_old_ft)); //this diverges

    clock_t clock_t1 = clock();
    // d(u*T)/dx + d(v*T)/dy:
    vel_temp_mat_real_xy = points.grad_x_matrix_EIGEN * (u_old.cwiseProduct(T_old)) + points.grad_y_matrix_EIGEN * (v_old.cwiseProduct(T_old));
    vel_temp_mat_real_z = w_old.cwiseProduct(T_old); // w*phi
    vel_timer = vel_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    clock_t1 = clock();
    calc_dft(vel_temp_mat_comp_xy, vel_temp_mat_real_xy, dft_matrix); // ft(d(u*T)/dx + d(v*T)/dy)
    calc_dft(vel_temp_mat_comp_z, vel_temp_mat_real_z, dft_matrix);   // ft(w*phi)
    dft_idft_timer = dft_idft_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    clock_t1 = clock();
    vel_temp_mat_comp_z = vel_temp_mat_comp_z * (1i * wave_nos).asDiagonal(); // 1i*k*ft(w*phi)
    T_source_old_ft = -(vel_temp_mat_comp_xy + vel_temp_mat_comp_z);

    double thermal_diff = thermal_k / (parameters.rho * thermal_Cp);
    // diffusion terms: alpha * ([Lap_2d]*T_ft):
    T_source_old_ft = T_source_old_ft + (thermal_diff * (points.laplacian_matrix_EIGEN * T_old_ft));
    // diffusion terms: alpha * (-T_ft*k*k)
    T_source_old_ft = T_source_old_ft - (thermal_diff * (T_old_ft * wave_nos_square.asDiagonal()));
    if (temporal_order == 1 || it == 0) // Euler method for first timestep of multistep method
        T_new_ft = T_old_ft + ((parameters.dt) * T_source_old_ft);
    else // Second order Adam-Bashforth
        T_new_ft = T_old_ft + ((0.5 * parameters.dt) * (3.0 * T_source_old_ft - T_source_old_old_ft));
    for (int iw = 0; iw < 2 * nw + 1; iw++)
        for (int iv = 0; iv < points.nv; iv++)
            if (points.boundary_flag[iv])
            { // apply BC
                if (T_dirichlet_flag[iv])
                    T_new_ft(iv, iw) = T_old_ft(iv, iw);
                else
                {
                    printf("\n\nERROR from FRACTIONAL_STEP_FOURIER::calc_T Neumann BC not defined for energy equation at iv: %i\n\n", iv);
                    throw bad_exception();
                }
            }
    vel_timer = vel_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    clock_t1 = clock();
    calc_idft(T_new, T_new_ft, idft_matrix);
    dft_idft_timer = dft_idft_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
}

void FRACTIONAL_STEP_FOURIER::single_timestep(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &p_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old, int it1)
{
    it = it1;
    clock_t clock_t1 = clock();
    if (it1 == 0)
    { // initialize ft matrices
        calc_dft(u_old_ft, u_old, dft_matrix), calc_dft(v_old_ft, v_old, dft_matrix), calc_dft(w_old_ft, w_old, dft_matrix), calc_dft(p_old_ft, p_old, dft_matrix);
        dft_idft_timer = dft_idft_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    }
    calc_vel_hat(points, parameters, u_old, v_old, w_old, p_old);
    calc_pressure(points, parameters, u_new, v_new, w_new, p_new, u_old, v_old, w_old, p_old);
    calc_vel_corr(points, cloud, parameters, u_new, v_new, w_new, p_new, u_old, v_old, w_old);
    u_old_ft = u_new_ft, v_old_ft = v_new_ft, w_old_ft = w_new_ft, p_old_ft = p_new_ft;
    if (temporal_order == 2)
        u_source_old_old_ft = u_source_old_ft, v_source_old_old_ft = v_source_old_ft, w_source_old_old_ft = w_source_old_ft;
    total_time_marching_timer = total_time_marching_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
}

void FRACTIONAL_STEP_FOURIER::single_timestep(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &p_new, Eigen::MatrixXd &T_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old, Eigen::MatrixXd &T_old, int it1)
{
    it = it1;
    clock_t clock_t1 = clock();
    if (it1 == 0)
    { // initialize ft matrices
        calc_dft(u_old_ft, u_old, dft_matrix), calc_dft(v_old_ft, v_old, dft_matrix), calc_dft(w_old_ft, w_old, dft_matrix), calc_dft(p_old_ft, p_old, dft_matrix), calc_dft(T_old_ft, T_old, dft_matrix);
        dft_idft_timer = dft_idft_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    }
    calc_vel_hat(points, parameters, u_old, v_old, w_old, p_old);
    calc_pressure(points, parameters, u_new, v_new, w_new, p_new, u_old, v_old, w_old, p_old);
    calc_vel_corr(points, cloud, parameters, u_new, v_new, w_new, p_new, u_old, v_old, w_old);
    calc_T(points, parameters, T_new, u_old, v_old, w_old, T_old);
    u_old_ft = u_new_ft, v_old_ft = v_new_ft, w_old_ft = w_new_ft, p_old_ft = p_new_ft, T_old_ft = T_new_ft;
    if (temporal_order == 2)
        u_source_old_old_ft = u_source_old_ft, v_source_old_old_ft = v_source_old_ft, w_source_old_old_ft = w_source_old_ft, T_source_old_old_ft = T_source_old_ft;
    total_time_marching_timer = total_time_marching_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
}

void FRACTIONAL_STEP_FOURIER::single_timestep(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &p_new, Eigen::MatrixXd &T_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old, Eigen::MatrixXd &T_old, Eigen::MatrixXd &body_force_x, Eigen::MatrixXd &body_force_y, Eigen::MatrixXd &body_force_z, int it1)
{
    it = it1;
    clock_t clock_t1 = clock();
    if (it1 == 0)
    { // initialize ft matrices
        calc_dft(u_old_ft, u_old, dft_matrix), calc_dft(v_old_ft, v_old, dft_matrix), calc_dft(w_old_ft, w_old, dft_matrix), calc_dft(p_old_ft, p_old, dft_matrix), calc_dft(T_old_ft, T_old, dft_matrix);
    }
    calc_dft(body_force_x_ft, body_force_x, dft_matrix), calc_dft(body_force_y_ft, body_force_y, dft_matrix), calc_dft(body_force_z_ft, body_force_z, dft_matrix);
    dft_idft_timer = dft_idft_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    calc_vel_hat(points, parameters, u_old, v_old, w_old, p_old, body_force_x, body_force_y, body_force_z);
    calc_pressure(points, parameters, u_new, v_new, w_new, p_new, u_old, v_old, w_old, p_old);
    calc_vel_corr(points, cloud, parameters, u_new, v_new, w_new, p_new, u_old, v_old, w_old);
    calc_T(points, parameters, T_new, u_old, v_old, w_old, T_old);
    u_old_ft = u_new_ft, v_old_ft = v_new_ft, w_old_ft = w_new_ft, p_old_ft = p_new_ft, T_old_ft = T_new_ft;
    if (temporal_order == 2)
        u_source_old_old_ft = u_source_old_ft, v_source_old_old_ft = v_source_old_ft, w_source_old_old_ft = w_source_old_ft, T_source_old_old_ft = T_source_old_ft;
    total_time_marching_timer = total_time_marching_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
}

void FRACTIONAL_STEP_FOURIER::single_timestep(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &p_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old, Eigen::MatrixXd &body_force_x, Eigen::MatrixXd &body_force_y, Eigen::MatrixXd &body_force_z, int it1)
{
    it = it1;
    clock_t clock_t1 = clock();
    if (it1 == 0)
    { // initialize ft matrices
        calc_dft(u_old_ft, u_old, dft_matrix), calc_dft(v_old_ft, v_old, dft_matrix), calc_dft(w_old_ft, w_old, dft_matrix), calc_dft(p_old_ft, p_old, dft_matrix);
    }
    calc_dft(body_force_x_ft, body_force_x, dft_matrix), calc_dft(body_force_y_ft, body_force_y, dft_matrix), calc_dft(body_force_z_ft, body_force_z, dft_matrix);
    dft_idft_timer = dft_idft_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    calc_vel_hat(points, parameters, u_old, v_old, w_old, p_old, body_force_x, body_force_y, body_force_z);
    calc_pressure(points, parameters, u_new, v_new, w_new, p_new, u_old, v_old, w_old, p_old);
    calc_vel_corr(points, cloud, parameters, u_new, v_new, w_new, p_new, u_old, v_old, w_old);
    u_old_ft = u_new_ft, v_old_ft = v_new_ft, w_old_ft = w_new_ft, p_old_ft = p_new_ft;
    if (temporal_order == 2)
        u_source_old_old_ft = u_source_old_ft, v_source_old_old_ft = v_source_old_ft, w_source_old_old_ft = w_source_old_ft;
    total_time_marching_timer = total_time_marching_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
}

void FRACTIONAL_STEP_FOURIER::calc_vel_hat(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old)
{
    calc_vel_source(points, parameters, u_source_old_ft, u_old, v_old, w_old, u_old_ft, u_old);
    calc_vel_source(points, parameters, v_source_old_ft, u_old, v_old, w_old, v_old_ft, v_old);
    calc_vel_source(points, parameters, w_source_old_ft, u_old, v_old, w_old, w_old_ft, w_old);
    clock_t clock_t1 = clock();
    if (temporal_order == 1 || it == 0)
    { // Euler method for first timestep of multistep method
        uh_ft = u_old_ft + ((parameters.dt / parameters.rho) * u_source_old_ft);
        vh_ft = v_old_ft + ((parameters.dt / parameters.rho) * v_source_old_ft);
        wh_ft = w_old_ft + ((parameters.dt / parameters.rho) * w_source_old_ft);
    }
    else
    { // Second order Adam-Bashforth
        uh_ft = u_old_ft + ((0.5 * parameters.dt / parameters.rho) * (3.0 * u_source_old_ft - u_source_old_old_ft));
        vh_ft = v_old_ft + ((0.5 * parameters.dt / parameters.rho) * (3.0 * v_source_old_ft - v_source_old_old_ft));
        wh_ft = w_old_ft + ((0.5 * parameters.dt / parameters.rho) * (3.0 * w_source_old_ft - w_source_old_old_ft));
    }
    vel_timer = vel_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
}

void FRACTIONAL_STEP_FOURIER::calc_vel_hat(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old, Eigen::MatrixXd &body_force_x, Eigen::MatrixXd &body_force_y, Eigen::MatrixXd &body_force_z)
{
    calc_vel_source(points, parameters, u_source_old_ft, u_old, v_old, w_old, u_old_ft, u_old);
    clock_t clock_t1 = clock();
    u_source_old_ft = u_source_old_ft + body_force_x_ft;
    vel_timer = vel_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    calc_vel_source(points, parameters, v_source_old_ft, u_old, v_old, w_old, v_old_ft, v_old);
    clock_t1 = clock();
    v_source_old_ft = v_source_old_ft + body_force_y_ft;
    vel_timer = vel_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    calc_vel_source(points, parameters, w_source_old_ft, u_old, v_old, w_old, w_old_ft, w_old);
    clock_t1 = clock();
    w_source_old_ft = w_source_old_ft + body_force_z_ft;
    if (temporal_order == 1 || it == 0)
    { // Euler method for first timestep of multistep method
        uh_ft = u_old_ft + ((parameters.dt / parameters.rho) * u_source_old_ft);
        vh_ft = v_old_ft + ((parameters.dt / parameters.rho) * v_source_old_ft);
        wh_ft = w_old_ft + ((parameters.dt / parameters.rho) * w_source_old_ft);
    }
    else
    { // Second order Adam-Bashforth
        uh_ft = u_old_ft + ((0.5 * parameters.dt / parameters.rho) * (3.0 * u_source_old_ft - u_source_old_old_ft));
        vh_ft = v_old_ft + ((0.5 * parameters.dt / parameters.rho) * (3.0 * v_source_old_ft - v_source_old_old_ft));
        wh_ft = w_old_ft + ((0.5 * parameters.dt / parameters.rho) * (3.0 * w_source_old_ft - w_source_old_old_ft));
    }
    vel_timer = vel_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
}

void FRACTIONAL_STEP_FOURIER::calc_vel_source(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXcd &vel_source, Eigen::MatrixXd &u, Eigen::MatrixXd &v, Eigen::MatrixXd &w, Eigen::MatrixXcd &phi_ft, Eigen::MatrixXd &phi)
{ // phi: field to be convected and diffused
    clock_t clock_t1 = clock();
    // d(u*phi)/dx + d(v*phi)/dy:
    //  vel_temp_mat_real_xy = u.cwiseProduct(points.grad_x_matrix_EIGEN * phi) + v.cwiseProduct(points.grad_y_matrix_EIGEN * phi); //conservative formulation
    vel_temp_mat_real_xy = points.grad_x_matrix_EIGEN * (u.cwiseProduct(phi)) + points.grad_y_matrix_EIGEN * (v.cwiseProduct(phi)); // non-conservative formulation
    vel_timer = vel_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    // ft(d(u*phi)/dx + d(v*phi)/dy):
    clock_t1 = clock();
    calc_dft(vel_temp_mat_comp_xy, vel_temp_mat_real_xy, dft_matrix);
    dft_idft_timer = dft_idft_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    clock_t1 = clock();
    vel_temp_mat_real_z = w.cwiseProduct(phi); // w*phi
    vel_timer = vel_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    clock_t1 = clock();
    calc_dft(vel_temp_mat_comp_z, vel_temp_mat_real_z, dft_matrix); // ft(w*phi)
    dft_idft_timer = dft_idft_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    clock_t1 = clock();
    vel_temp_mat_comp_z = vel_temp_mat_comp_z * (1i * wave_nos).asDiagonal(); // 1i*k*ft(w*phi)
    // convection terms: -rho * (ft(//d(u*phi)/dx + d(v*phi)/dy) + 1i*k*ft(w*phi))
    vel_source = -parameters.rho * (vel_temp_mat_comp_xy + vel_temp_mat_comp_z);
    vel_source = vel_source + parameters.mu * ((points.laplacian_matrix_EIGEN * phi_ft) - (phi_ft * wave_nos_square.asDiagonal())); // diffusion terms: mu * ([Lap_2d]*phi_ft - phi_ft*k*k)
    vel_timer = vel_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
}

void FRACTIONAL_STEP_FOURIER::calc_pressure(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &p_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &p_old)
{
    clock_t clock_t1 = clock();
    // d(ft(uh))/dx + d(ft(vh))/dy:
    p_source_ft.topRows(points.nv) = ((points.grad_x_matrix_EIGEN_internal * uh_ft) + (points.grad_y_matrix_EIGEN_internal * vh_ft));
    p_source_ft.topRows(points.nv) = p_source_ft.topRows(points.nv) + (wh_ft * (1i * wave_nos).asDiagonal()); // 1i*k*ft(wh)
    p_source_ft = p_source_ft * parameters.rho / parameters.dt;
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
        {
            if (p_dirichlet_flag[iv]) // dirichlet BC
                p_source_ft.row(iv) = p_old_ft.row(iv);
            else // normal momentum
                p_source_ft.row(iv) = -parameters.rho * ((u_old_ft.row(iv) - uh_ft.row(iv)) * points.normal[dim * iv] + (v_old_ft.row(iv) - vh_ft.row(iv)) * points.normal[dim * iv + 1]) / parameters.dt;
        }

    int system_size;
    for (int iw = 0; iw <= nw; iw++)
    {
        system_size = p_solver_eigen_ilu_bicgstab[iw].rows();
        if (system_size == points.nv + 1)
        { // has regularization
            p_temp_vec_comp.head(points.nv) = p_old_ft.col(iw);
            // cout << "\nFRACTIONAL_STEP_FOURIER::calc_pressure 3 iw: " << iw << "\n";
            p_temp_vec_comp.real() = p_solver_eigen_ilu_bicgstab[iw].solveWithGuess(p_source_ft.col(iw).real(), p_temp_vec_comp.real());
            ppe_num_iter.push_back(p_solver_eigen_ilu_bicgstab[iw].iterations());
            p_new_ft.col(iw).real() = p_temp_vec_comp.head(points.nv).real();
        }
        else
        { // no regularization
            p_new_ft.col(iw).real() = p_solver_eigen_ilu_bicgstab[iw].solveWithGuess(p_source_ft.col(iw).real().head(points.nv), p_old_ft.col(iw).real());
            ppe_num_iter.push_back(p_solver_eigen_ilu_bicgstab[iw].iterations());
            p_new_ft.col(iw).imag() = p_solver_eigen_ilu_bicgstab[iw].solveWithGuess(p_source_ft.col(iw).imag().head(points.nv), p_old_ft.col(iw).imag());
            ppe_num_iter.push_back(p_solver_eigen_ilu_bicgstab[iw].iterations());
        }
        if (iw > 0)
            p_new_ft.col(2 * nw + 1 - iw) = p_new_ft.col(iw).conjugate();
    }
    ppe_timer = ppe_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    clock_t1 = clock();
    calc_idft(p_new, p_new_ft, idft_matrix);
    dft_idft_timer = dft_idft_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
}

void FRACTIONAL_STEP_FOURIER::calc_vel_corr(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &p_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old)
{
    clock_t clock_t1 = clock();
    u_new_ft = uh_ft - ((points.grad_x_matrix_EIGEN * p_new_ft) * (parameters.dt / parameters.rho));
    v_new_ft = vh_ft - ((points.grad_y_matrix_EIGEN * p_new_ft) * (parameters.dt / parameters.rho));
    w_new_ft = wh_ft - ((p_new_ft * (1i * wave_nos).asDiagonal()) * (parameters.dt / parameters.rho));
    vel_timer = vel_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    // write_csv(w_new_ft, "w_new_ft.csv");
    // clock_t1 = clock();
    // calc_idft(u_new, u_new_ft, idft_matrix), calc_idft(v_new, v_new_ft, idft_matrix), calc_idft(w_new, w_new_ft, idft_matrix);
    // dft_idft_timer = dft_idft_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    // write_csv(u_new, "u_new_3d_before_BC.csv");
    double diag_coeff, rhs, off_diag_coeff;
    complex<double> rhs_ft;
    int ivnb, dim = parameters.dimension;
    clock_t1 = clock();
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
        {
            if (!p_dirichlet_flag[iv])
            { // for dirichlet pressure, u_new comes from velocity correction above
                if (u_dirichlet_flag[iv])
                    u_new_ft.row(iv) = u_old_ft.row(iv); // dirichlet BC
                else
                { // neumann BC
                    for (int iw = 0; iw < 2 * nw + 1; iw++)
                    {
                        rhs_ft = 0.0 + (0.0 * 1i);
                        for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                        {
                            ivnb = cloud.nb_points_col[i1];
                            if (iv == ivnb)
                                diag_coeff = (cloud.grad_x_coeff[i1] * points.normal[dim * iv]) + (cloud.grad_y_coeff[i1] * points.normal[dim * iv + 1]);
                            else
                            {
                                off_diag_coeff = (cloud.grad_x_coeff[i1] * points.normal[dim * iv]) + (cloud.grad_y_coeff[i1] * points.normal[dim * iv + 1]);
                                rhs_ft = rhs_ft - (off_diag_coeff * u_new_ft(ivnb, iw));
                            }
                        }
                        u_new_ft(iv, iw) = (rhs_ft / diag_coeff);
                    }
                }
            }
        }

    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
        {
            if (!p_dirichlet_flag[iv])
            { // for dirichlet pressure, v_new comes from velocity correction above
                if (v_dirichlet_flag[iv])
                    v_new_ft.row(iv) = v_old_ft.row(iv); // dirichlet BC
                else
                { // neumann BC
                    for (int iw = 0; iw < 2 * nw + 1; iw++)
                    {
                        rhs_ft = 0.0 + (0.0 * 1i);
                        for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                        {
                            ivnb = cloud.nb_points_col[i1];
                            if (iv == ivnb)
                                diag_coeff = (cloud.grad_x_coeff[i1] * points.normal[dim * iv]) + (cloud.grad_y_coeff[i1] * points.normal[dim * iv + 1]);
                            else
                            {
                                off_diag_coeff = (cloud.grad_x_coeff[i1] * points.normal[dim * iv]) + (cloud.grad_y_coeff[i1] * points.normal[dim * iv + 1]);
                                rhs_ft = rhs_ft - (off_diag_coeff * v_new_ft(ivnb, iw));
                            }
                        }
                        v_new_ft(iv, iw) = (rhs_ft / diag_coeff);
                    }
                }
            }
        }

    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
        {
            if (!p_dirichlet_flag[iv])
            { // for dirichlet pressure, w_new comes from velocity correction above
                if (w_dirichlet_flag[iv])
                    w_new_ft.row(iv) = w_old_ft.row(iv); // dirichlet BC
                else
                { // neumann BC
                    for (int iw = 0; iw < 2 * nw + 1; iw++)
                    {
                        rhs_ft = 0.0 + (0.0 * 1i);
                        for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                        {
                            ivnb = cloud.nb_points_col[i1];
                            if (iv == ivnb)
                                diag_coeff = (cloud.grad_x_coeff[i1] * points.normal[dim * iv]) + (cloud.grad_y_coeff[i1] * points.normal[dim * iv + 1]);
                            else
                            {
                                off_diag_coeff = (cloud.grad_x_coeff[i1] * points.normal[dim * iv]) + (cloud.grad_y_coeff[i1] * points.normal[dim * iv + 1]);
                                rhs_ft = rhs_ft - (off_diag_coeff * w_new_ft(ivnb, iw));
                            }
                        }
                        w_new_ft(iv, iw) = (rhs_ft / diag_coeff);
                    }
                }
            }
        }

    vel_timer = vel_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;

    clock_t1 = clock();
    calc_idft(u_new, u_new_ft, idft_matrix), calc_idft(v_new, v_new_ft, idft_matrix), calc_idft(w_new, w_new_ft, idft_matrix);
    dft_idft_timer = dft_idft_timer + ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
}

void FRACTIONAL_STEP_FOURIER::calc_vorticity(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &vort_x, Eigen::MatrixXd &vort_y, Eigen::MatrixXd &vort_z)
{
    vort_x_ft = (points.grad_y_matrix_EIGEN * w_new_ft) - (v_new_ft * (1i * wave_nos).asDiagonal());
    vort_y_ft = (u_new_ft * (1i * wave_nos).asDiagonal()) - (points.grad_x_matrix_EIGEN * w_new_ft);
    vort_z_ft = (points.grad_x_matrix_EIGEN * v_new_ft) - (points.grad_y_matrix_EIGEN * u_new_ft);
    calc_idft(vort_x, vort_x_ft, idft_matrix);
    calc_idft(vort_y, vort_y_ft, idft_matrix);
    calc_idft(vort_z, vort_z_ft, idft_matrix);
}

// https://www.cfd-online.com/Forums/blogs/sbaffini/1931-q-criterion-isosurfaces-matlab.html
void FRACTIONAL_STEP_FOURIER::calc_q_criterion(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &q_criterion)
{
    vel_temp_mat_comp_xy = points.grad_x_matrix_EIGEN * u_new_ft; // du_dx
    q_criterion_ft = -0.5 * (vel_temp_mat_comp_xy.cwiseProduct(vel_temp_mat_comp_xy));
    vel_temp_mat_comp_xy = points.grad_y_matrix_EIGEN * v_new_ft; // dv_dy
    q_criterion_ft = q_criterion_ft - (0.5 * (vel_temp_mat_comp_xy.cwiseProduct(vel_temp_mat_comp_xy)));
    vel_temp_mat_comp_xy = w_new_ft * (1i * wave_nos).asDiagonal(); // dw_dz
    q_criterion_ft = q_criterion_ft - (0.5 * (vel_temp_mat_comp_xy.cwiseProduct(vel_temp_mat_comp_xy)));

    // du_dy * dv_dx:
    q_criterion_ft = q_criterion_ft - ((points.grad_y_matrix_EIGEN * u_new_ft).cwiseProduct(points.grad_x_matrix_EIGEN * v_new_ft));
    // du_dz * dw_dx:
    q_criterion_ft = q_criterion_ft - ((u_new_ft * (1i * wave_nos).asDiagonal()).cwiseProduct(points.grad_x_matrix_EIGEN * w_new_ft));
    // dv_dz * dw_dy:
    q_criterion_ft = q_criterion_ft - ((v_new_ft * (1i * wave_nos).asDiagonal()).cwiseProduct(points.grad_y_matrix_EIGEN * w_new_ft));
    calc_idft(q_criterion, q_criterion_ft, idft_matrix);
}