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
#include "postprocessing_functions.hpp"
#include "nanoflann.hpp"
#include "coefficient_computations.hpp"
using namespace std;

FRACTIONAL_STEP_FSI::FRACTIONAL_STEP_FSI(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, vector<bool> &u_dirichlet_flag1, vector<bool> &v_dirichlet_flag1, vector<bool> &p_dirichlet_flag1, int temporal_order1)
{
    temporal_order = temporal_order1;
    u_dirichlet_flag = u_dirichlet_flag1, v_dirichlet_flag = v_dirichlet_flag1, p_dirichlet_flag = p_dirichlet_flag1;
    check_bc(points, parameters);
    for (int iv = 0; iv < points.nv; iv++)
        iv_active_flag_new.push_back(true), iv_active_flag_old.push_back(true), iv_update_flag.push_back(false);

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
    // clock_t t1 = clock();
    // solver_p.init(points, cloud, parameters, p_dirichlet_flag, 0.0, 0.0, 1.0, true);
    // EIGEN_set_ppe_coeff(points, cloud, parameters);
    // solver_eigen_ilu_bicgstab.compute(ppe_coeff); //precondition and initialize solver
    // parameters.factoring_timer = ((double)(clock() - t1)) / CLOCKS_PER_SEC;
    if (p_bc_full_neumann)
        p_source = zero_vector_1; //this is full Neumann for pressure
    else
        p_source = zero_vector;
    u_source_old = zero_vector, v_source_old = zero_vector;
    if (temporal_order == 2)
        u_source_old_old = zero_vector, v_source_old_old = zero_vector;

    if (strcmp(parameters.solver_type.c_str(), "eigen_ilu_bicgstab") != 0)
    {
        cout << "\n\nERROR from FRACTIONAL_STEP_FSI::FRACTIONAL_STEP_FSI solver_type should be eigen_ilu_bicgstab; current value: " << parameters.solver_type << "\n\n";
        throw bad_exception();
    }
}

void FRACTIONAL_STEP_FSI::check_bc(POINTS &points, PARAMETERS &parameters)
{
    int u_dirichlet_flag_sum = accumulate(u_dirichlet_flag.begin(), u_dirichlet_flag.end(), 0);
    if (u_dirichlet_flag_sum == 0)
    {
        printf("\n\nERROR from FRACTIONAL_STEP_FSI::check_bc Setting u_dirichlet_flag to full Neumann BC is not permitted; sum of u_dirichlet_flag: %i\n\n", u_dirichlet_flag_sum);
        throw bad_exception();
    }
    int v_dirichlet_flag_sum = accumulate(v_dirichlet_flag.begin(), v_dirichlet_flag.end(), 0);
    if (v_dirichlet_flag_sum == 0)
    {
        printf("\n\nERROR from FRACTIONAL_STEP_FSI::check_bc Setting v_dirichlet_flag to full Neumann BC is not permitted; sum of v_dirichlet_flag: %i\n\n", v_dirichlet_flag_sum);
        throw bad_exception();
    }
    if (parameters.rho < 0 || parameters.mu < 0)
    {
        printf("\n\nERROR from FRACTIONAL_STEP_FSI::check_bc Some parameters are not set; parameters.rho: %g, parameters.mu: %g\n\n", parameters.rho, parameters.mu);
        throw bad_exception();
    }
    if (temporal_order != 1 && temporal_order != 2)
    {
        printf("\n\nERROR from FRACTIONAL_STEP_FSI::check_bc temporal_order should be either '1' or '2'; current value: %i\n\n", temporal_order);
        throw bad_exception();
    }
}

void FRACTIONAL_STEP_FSI::calc_flags(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
{
    double x, y, r, radius = 1.0;
    int dim = parameters.dimension, ivnb;
    for (int iv = 0; iv < points.nv; iv++)
    {
        x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1], r = sqrt(x * x + y * y);
        if ((r < (radius + parameters.min_dx)) && !points.boundary_flag[iv])
            iv_active_flag_new[iv] = false;
        else
            iv_active_flag_new[iv] = true;
    }

    for (int iv = 0; iv < points.nv; iv++)
    {
        iv_update_flag[iv] = false;
        if (iv_active_flag_new[iv])
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                ivnb = cloud.nb_points_col[i1];
                if (iv_active_flag_new[ivnb] != iv_active_flag_old[ivnb])
                { //iv is active but some ivnb has changed status; hence coeff of iv have to be recomputed
                    iv_update_flag[iv] = true;
                    break;
                }
            }
    }

    // cout << "\n";
    // for (int iv = 0; iv < points.nv; iv++)
    //     if (!iv_active_flag_new[iv])
    //     {
    //         x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1], r = sqrt(x * x + y * y);
    //         printf("\nFRACTIONAL_STEP_FSI::calc_flags iv: %i, x: %g, y: %g, r: %g, boundary_flag: %i; nb_points:\n", iv, x, y, r, (int(points.boundary_flag[iv])));
    //     }
    // cout << "\n\nFRACTIONAL_STEP_FSI::calc_flags it: " << it << " update ivs: ";
    // int update_count = 0;
    // for (int iv = 0; iv < points.nv; iv++)
    //     if (iv_update_flag[iv])
    //     {
    //         // cout << "(" << iv << " " << points.boundary_flag[iv] << "), ";
    //         update_count++;
    //         // x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1], r = sqrt(x * x + y * y);
    //         // printf("\nFRACTIONAL_STEP_FSI::calc_flags iv: %i, x: %g, y: %g, r: %g, boundary_flag: %i; nb_points:\n", iv, x, y, r, (int(points.boundary_flag[iv])));
    //         // for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
    //         //     cout << cloud.nb_points_col[i1] << ", ";
    //         // cout << "\n";
    //     }
    // cout << "\nFRACTIONAL_STEP_FSI::calc_flags update_count: " << update_count << "\n\n";
}

void FRACTIONAL_STEP_FSI::update_cloud(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
{
    PointCloud<double> cloud_nf_for_interior, cloud_nf_for_boundary;
    int dim = parameters.dimension;
    cloud_nf_for_interior.pts.resize(points.nv);
    cloud_nf_for_boundary.pts.resize(points.nv);
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (iv_active_flag_new[iv])
        {
            cloud_nf_for_interior.pts[iv].x = points.xyz[dim * iv];
            cloud_nf_for_interior.pts[iv].y = points.xyz[dim * iv + 1];
            if (dim == 3)
                cloud_nf_for_interior.pts[iv].z = points.xyz[dim * iv + 2];
            else
                cloud_nf_for_interior.pts[iv].z = 0.0; //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
        }
        else
        { //de-activated points should not be coupled to any points
            cloud_nf_for_boundary.pts[iv].x = numeric_limits<double>::infinity();
            cloud_nf_for_boundary.pts[iv].y = numeric_limits<double>::infinity();
            cloud_nf_for_boundary.pts[iv].z = numeric_limits<double>::infinity();
        }
    }
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (iv_active_flag_new[iv])
        {
            if (!points.boundary_flag[iv])
            { //internal points are coupled with boundary points
                cloud_nf_for_boundary.pts[iv].x = points.xyz[dim * iv];
                cloud_nf_for_boundary.pts[iv].y = points.xyz[dim * iv + 1];
                if (dim == 3)
                    cloud_nf_for_boundary.pts[iv].z = points.xyz[dim * iv + 2];
                else
                    cloud_nf_for_boundary.pts[iv].z = 0.0; //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
            }
            else
            { //all boundary co-ordinates set to infinity so that they are never coupled with any boundary point
                cloud_nf_for_boundary.pts[iv].x = numeric_limits<double>::infinity();
                cloud_nf_for_boundary.pts[iv].y = numeric_limits<double>::infinity();
                cloud_nf_for_boundary.pts[iv].z = numeric_limits<double>::infinity(); //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
            }
        }
        else
        { //de-activated points should not be coupled to any points
            cloud_nf_for_boundary.pts[iv].x = numeric_limits<double>::infinity();
            cloud_nf_for_boundary.pts[iv].y = numeric_limits<double>::infinity();
            cloud_nf_for_boundary.pts[iv].z = numeric_limits<double>::infinity();
        }
    }

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> nanoflann_kd_tree_for_interior;
    nanoflann_kd_tree_for_interior index_for_interior(3, cloud_nf_for_interior, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index_for_interior.buildIndex();

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> nanoflann_kd_tree_for_boundary;
    nanoflann_kd_tree_for_boundary index_for_boundary(3, cloud_nf_for_boundary, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    index_for_boundary.buildIndex();

    vector<size_t> nb_vert(parameters.cloud_size);
    vector<double> nb_dist(parameters.cloud_size);
    double query_pt[3];
    int iv_nb;
    for (int iv = 0; iv < points.nv; iv++)
        if (iv_update_flag[iv])
        {
            query_pt[0] = points.xyz[dim * iv], query_pt[1] = points.xyz[dim * iv + 1];
            if (dim == 3)
                query_pt[2] = points.xyz[dim * iv + 2];
            else
                query_pt[2] = 0.0; //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
            if (points.boundary_flag[iv])
            {
                index_for_boundary.knnSearch(&query_pt[0], parameters.cloud_size, &nb_vert[0], &nb_dist[0]);
                cloud.nb_points_col[cloud.nb_points_row[iv]] = iv;
                for (int i1 = 0; i1 < nb_vert.size() - 1; i1++)
                { //first entry is "iv": hence "nb_vert.size() - 1"
                    iv_nb = nb_vert[i1];
                    if (points.boundary_flag[iv_nb])
                    {
                        cout << "\n\nERROR from CLOUD::calc_cloud_points_fast boundary iv: " << iv << " (boundary_flag[iv]: " << points.boundary_flag[iv] << ") tried to couple to a boundary vertex: " << iv_nb << " (boundary_flag[iv_nb]: " << points.boundary_flag[iv_nb] << ") \n\n";
                        throw bad_exception();
                    }
                    else
                        cloud.nb_points_col[cloud.nb_points_row[iv] + i1 + 1] = iv_nb;
                }
            }
            else
            { //internal points
                index_for_interior.knnSearch(&query_pt[0], parameters.cloud_size, &nb_vert[0], &nb_dist[0]);
                for (int i1 = 0; i1 < nb_vert.size(); i1++)
                    cloud.nb_points_col[cloud.nb_points_row[iv] + i1] = nb_vert[i1];
            }
        }
}

void FRACTIONAL_STEP_FSI::update_RBF_coeff(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
{
    vector<double> vert;
    vector<int> central_vert_list;
    Eigen::MatrixXd laplacian, grad_x, grad_y, grad_z;
    int dim = parameters.dimension, iv_nb, i1;
    central_vert_list.push_back(0);
    double scale[3], cond_num;
    for (int iv = 0; iv < points.nv; iv++)
        if (iv_update_flag[iv])
        {
            central_vert_list[0] = 0; //coefficient for first vertex needed
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                iv_nb = cloud.nb_points_col[i1];
                for (int i = 0; i < dim; i++)
                    vert.push_back(points.xyz[dim * iv_nb + i]);
            }
            shifting_scaling(vert, scale, dim);
            cond_num = calc_PHS_RBF_grad_laplace_single_vert(vert, parameters, laplacian, grad_x, grad_y, grad_z, scale, central_vert_list);
            vert.clear();
            for (int i1 = 0; i1 < laplacian.size(); i1++)
            { //(nb_points_row[iv + 1] - nb_points_row[iv]) = laplacian.size()
                cloud.grad_x_coeff[cloud.nb_points_row[iv] + i1] = grad_x(0, i1);
                cloud.grad_y_coeff[cloud.nb_points_row[iv] + i1] = grad_y(0, i1);
                if (dim == 3)
                    cloud.grad_z_coeff[cloud.nb_points_row[iv] + i1] = grad_z(0, i1);
                cloud.laplacian_coeff[cloud.nb_points_row[iv] + i1] = laplacian(0, i1);
            }
        }
    laplacian.resize(0, 0); //free memory
    grad_x.resize(0, 0);    //free memory
    grad_y.resize(0, 0);    //free memory
    grad_z.resize(0, 0);    //free memory
}

void FRACTIONAL_STEP_FSI::calc_vel_hat(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
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

void FRACTIONAL_STEP_FSI::calc_pressure(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
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
    // solver_p.general_solve(points, parameters, p_new, p_old, p_source);
    p_new = solver_eigen_ilu_bicgstab.solveWithGuess(p_source, p_old);
    parameters.rel_res_log.push_back((ppe_coeff * p_new - p_source).norm() / p_source.norm());
    parameters.abs_res_log.push_back((ppe_coeff * p_new - p_source).norm());
    parameters.n_iter_actual.push_back(solver_eigen_ilu_bicgstab.iterations());
    if (p_bc_full_neumann)
        parameters.regul_alpha_log.push_back(p_new[points.nv]);
    for (int iv = 0; iv < points.nv; iv++)
        if (!iv_active_flag_new[iv])
            p_new[iv] = 0.0; //deactivated points set to zero
}

void FRACTIONAL_STEP_FSI::calc_vel_corr(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old)
{
    u_new = uh - (parameters.dt * (points.grad_x_matrix_EIGEN_internal * p_new.head(points.nv)) / parameters.rho);
    v_new = vh - (parameters.dt * (points.grad_y_matrix_EIGEN_internal * p_new.head(points.nv)) / parameters.rho);
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
    for (int iv = 0; iv < points.nv; iv++)
        if (!iv_active_flag_new[iv])
            u_new[iv] = 0.0, v_new[iv] = 0.0; //deactivated points set to zero
}

double FRACTIONAL_STEP_FSI::extras(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old)
{
    double total_steady_err, max_err, l1_err;
    calc_max_l1_error(u_old, u_new, max_err, l1_err);
    total_steady_err = l1_err / parameters.dt;
    calc_max_l1_error(v_old, v_new, max_err, l1_err);
    total_steady_err += l1_err / parameters.dt;
    if (temporal_order == 2)
        u_source_old_old = u_source_old, v_source_old_old = v_source_old;
    iv_active_flag_old = iv_active_flag_new;
    parameters.steady_error_log.push_back(total_steady_err);
    return total_steady_err;
}

void FRACTIONAL_STEP_FSI::EIGEN_update_ppe_coeff(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
{
    int dim = parameters.dimension, iv_nb;
    double val;
    for (int iv = 0; iv < points.nv; iv++)
        if (iv_update_flag[iv])
        {
            // cout << "\nFRACTIONAL_STEP_FSI::EIGEN_update_ppe_coeff iv: " << iv << endl;
            for (Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator it(ppe_coeff, iv); it; ++it)
                it.valueRef() = 0.0; //set current value of entire row to zero

            if (points.boundary_flag[iv])
            { //lies on boundary
                if (p_dirichlet_flag[iv])
                { //apply dirichlet BC
                    ppe_coeff.coeffRef(iv, iv) = 1.0;
                }
                else
                { //apply neumann BC
                    for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                    {
                        iv_nb = cloud.nb_points_col[i1];
                        val = points.normal[dim * iv] * cloud.grad_x_coeff[i1] + points.normal[dim * iv + 1] * cloud.grad_y_coeff[i1];
                        if (dim == 3)
                            val += points.normal[dim * iv + 2] * cloud.grad_z_coeff[i1];
                        ppe_coeff.coeffRef(iv, iv_nb) = val;
                    }
                }
            }
            else
            {
                for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                {
                    iv_nb = cloud.nb_points_col[i1];
                    val = cloud.laplacian_coeff[i1]; //diffusion
                    ppe_coeff.coeffRef(iv, iv_nb) = val;
                }
            }
        }
    if (!ppe_coeff.isCompressed())
        ppe_coeff.makeCompressed();
}

void FRACTIONAL_STEP_FSI::EIGEN_set_ppe_coeff(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
{
    int dim = parameters.dimension, iv_nb;
    double val;
    vector<Eigen::Triplet<double>> triplet;
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (iv_active_flag_new[iv])
        {
            if (points.boundary_flag[iv])
            { //lies on boundary
                if (p_dirichlet_flag[iv])
                { //apply dirichlet BC
                    triplet.push_back(Eigen::Triplet<double>(iv, iv, 1.0));
                }
                else
                { //apply neumann BC
                    for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                    {
                        iv_nb = cloud.nb_points_col[i1];
                        val = points.normal[dim * iv] * cloud.grad_x_coeff[i1] + points.normal[dim * iv + 1] * cloud.grad_y_coeff[i1];
                        if (dim == 3)
                            val += points.normal[dim * iv + 2] * cloud.grad_z_coeff[i1];
                        triplet.push_back(Eigen::Triplet<double>(iv, iv_nb, val));
                    }
                }
            }
            else
            {
                for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
                {
                    iv_nb = cloud.nb_points_col[i1];
                    val = cloud.laplacian_coeff[i1]; //diffusion
                    triplet.push_back(Eigen::Triplet<double>(iv, iv_nb, val));
                }
            }
        }
        else
            triplet.push_back(Eigen::Triplet<double>(iv, iv, 1.0));
    }
    if (p_bc_full_neumann)
    { //extra constraint (regularization: http://www-e6.ijs.si/medusa/wiki/index.php/Poisson%27s_equation) for neumann BCs
        for (int iv = 0; iv < points.nv; iv++)
        {
            triplet.push_back(Eigen::Triplet<double>(iv, points.nv, 1.0)); //last column
            triplet.push_back(Eigen::Triplet<double>(points.nv, iv, 1.0)); //last row
        }
        triplet.push_back(Eigen::Triplet<double>(points.nv, points.nv, 0.0)); //last entry
    }
    int ppe_system_size = points.nv;
    if (p_bc_full_neumann)
        ppe_system_size = points.nv + 1;
    ppe_coeff.resize(ppe_system_size, ppe_system_size);
    ppe_coeff.setFromTriplets(triplet.begin(), triplet.end());
    ppe_coeff.makeCompressed();
    triplet.clear();

    solver_eigen_ilu_bicgstab.setTolerance(parameters.solver_tolerance); //default is machine precision (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#ac160a444af8998f93da9aa30e858470d)
    solver_eigen_ilu_bicgstab.setMaxIterations(parameters.n_iter);       //default is twice number of columns (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#af83de7a7d31d9d4bd1fef6222b07335b)
    solver_eigen_ilu_bicgstab.preconditioner().setDroptol(parameters.precond_droptol);
}

double FRACTIONAL_STEP_FSI::single_timestep_2d(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &p_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::VectorXd &p_old, int it1)
{
    it = it1;
    if (it == 0)
        parameters.factoring_timer = 0.0;
    calc_flags(points, cloud, parameters);
    update_cloud(points, cloud, parameters);
    update_RBF_coeff(points, cloud, parameters);
    cloud.EIGEN_set_grad_laplace_matrix(points, parameters);
    cloud.EIGEN_set_grad_laplace_matrix_separate(points, parameters);
    // if (it == 0)
    // {
    clock_t t1 = clock();
    // solver_p.init(points, cloud, parameters, p_dirichlet_flag, 0.0, 0.0, 1.0, true);
    EIGEN_set_ppe_coeff(points, cloud, parameters);
    // EIGEN_update_ppe_coeff(points, cloud, parameters);
    // if (it == 0)
    solver_eigen_ilu_bicgstab.compute(ppe_coeff); //preconditioning
    parameters.factoring_timer = parameters.factoring_timer + ((double)(clock() - t1)) / CLOCKS_PER_SEC;
    // }
    if (p_source.size() != p_new.size())
        p_new = Eigen::VectorXd::Zero(p_source.size());
    if (p_source.size() != p_old.size())
        p_old = Eigen::VectorXd::Zero(p_source.size());
    calc_vel_hat(points, parameters, u_old, v_old, p_old);
    calc_pressure(points, parameters, u_new, v_new, p_new, u_old, v_old, p_old);
    calc_vel_corr(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old);
    double total_steady_err = extras(points, parameters, u_new, v_new, p_new, u_old, v_old, p_old);
    return total_steady_err;
}