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

MULTIGRID::MULTIGRID(string &meshfolder1, vector<string> &meshfile_list1)
{
    meshfolder = meshfolder1, meshfile_list = meshfile_list1;
    ng = meshfile_list.size();
    for (int ig = 0; ig < ng; ig++)
    {
        PARAMETERS parameters("parameters_file.csv", meshfolder + meshfile_list[ig]);
        POINTS points(parameters);
        CLOUD cloud(points, parameters);
        parameters_list.push_back(parameters);
        points_list.push_back(points);
        cloud_list.push_back(cloud);
    }

    cout << "\n\nMULTIGRID::MULTIGRID ng: " << ng << endl;
    for (int ig = 0; ig < ng; ig++)
    {
        cout << "MULTIGRID::MULTIGRID ig: " << ig << ", nv: " << points_list[ig].nv << ", avg_dx: " << parameters_list[ig].avg_dx << endl;
        cout << "MULTIGRID::MULTIGRID ig: " << ig << ", RBF condition number max: " << cloud_list[ig].cond_num_RBF_max << ", avg: " << cloud_list[ig].cond_num_RBF_avg << ", min: " << cloud_list[ig].cond_num_RBF_min << endl;
    }
    cout << "\n\n";
}

void MULTIGRID::assemble_coeff()
{
    if (dirichlet_flag_list.size() == 0)
    {
        printf("\n\nERROR from MULTIGRID::assemble_coeff dirichlet_flag_list is empty\n\n");
        throw bad_exception();
    }
    for (int ig = 0; ig < ng; ig++)
        if (dirichlet_flag_list[ig].size() != points_list[ig].nv)
        {
            printf("\n\nERROR from MULTIGRID::assemble_coeff dirichlet_flag_list[%i].size()=%li does not match with points_list[%i].nv=%i\n\n", ig, dirichlet_flag_list[ig].size(), ig, points_list[ig].nv);
            throw bad_exception();
        }

    for (int ig = 0; ig < ng; ig++)
    {
        full_neumann_flag_list.push_back(true); //initialize
        for (int iv = 0; iv < points_list[ig].nv; iv++)
            if (points_list[ig].boundary_flag[iv] && dirichlet_flag_list[ig][iv])
            { //dirichlet boundary point spotted
                full_neumann_flag_list[ig] = false;
                break;
            }
    }
    print_to_terminal(full_neumann_flag_list, "MULTIGRID::assemble_coeff full_neumann_flag_list");
    // cout << "MULTIGRID::assemble_coeff full_neumann_flag_list: " << full_neumann_flag_list << "\n\n";
    // for (int ig = 0; ig < ng; ig++)
    // {
    //     if (full_neumann_flag_list[ig]) //with regularization
    //         system_size.push_back(points_list[ig].nv + 1);
    //     else //without regularization
    //         system_size.push_back(points_list[ig].nv);
    //     phi.push_back(Eigen::VectorXd::Zero(system_size[ig]));
    //     source.push_back(Eigen::VectorXd::Zero(system_size[ig]));
    // }
    int dim = parameters_list[0].dimension, ivnb;
    double val;

    vector<int> empty_int_vector;
    vector<double> empty_double_vector;
    for (int ig = 0; ig < ng; ig++)
    {
        coeff_mat_row.push_back(empty_int_vector), coeff_mat_col.push_back(empty_int_vector);
        coeff_mat_val.push_back(empty_double_vector);
        coeff_mat_row[ig].push_back(0);
        for (int iv = 0; iv < points_list[ig].nv; iv++)
        {
            if (points_list[ig].boundary_flag[iv])
            { //boundary point
                if (dirichlet_flag_list[ig][iv])
                { //dirichlet BC
                    coeff_mat_row[ig].push_back(coeff_mat_row[ig][iv] + 1);
                    coeff_mat_col[ig].push_back(iv), coeff_mat_val[ig].push_back(1.0);
                }
                else
                { //neumann BC
                    coeff_mat_row[ig].push_back(coeff_mat_row[ig][iv] + (cloud_list[ig].nb_points_row[iv + 1] - cloud_list[ig].nb_points_row[iv]));
                    // if (full_neumann_flag_list[ig]) //with regularization
                    //     coeff_mat_row[ig].push_back(coeff_mat_row[ig][iv] + (cloud_list[ig].nb_points_row[iv + 1] - cloud_list[ig].nb_points_row[iv]) + 1);
                    // else //without regularization
                    //     coeff_mat_row[ig].push_back(coeff_mat_row[ig][iv] + (cloud_list[ig].nb_points_row[iv + 1] - cloud_list[ig].nb_points_row[iv]));
                    for (int i1 = cloud_list[ig].nb_points_row[iv]; i1 < cloud_list[ig].nb_points_row[iv + 1]; i1++)
                    {
                        ivnb = cloud_list[ig].nb_points_col[i1];
                        val = points_list[ig].normal[dim * iv] * cloud_list[ig].grad_x_coeff[i1] + points_list[ig].normal[dim * iv + 1] * cloud_list[ig].grad_y_coeff[i1];
                        if (dim == 3)
                            val += points_list[ig].normal[dim * iv + 2] * cloud_list[ig].grad_z_coeff[i1];

                        coeff_mat_col[ig].push_back(ivnb);
                        coeff_mat_val[ig].push_back(val);
                    }
                    // if (full_neumann_flag_list[ig])
                    // { //with regularization: last column with unit coefficient
                    //     coeff_mat_col[ig].push_back(points_list[ig].nv);
                    //     coeff_mat_val[ig].push_back(1.0);
                    // }
                }
            }
            else
            { //interior point
                // if (ig == 0 && iv == 4)
                //     cout << "   MULTIGRID::assemble_coeff ig: " << ig << ", iv: " << iv << ", cloud_list[ig].nb_points_row[iv]: " << cloud_list[ig].nb_points_row[iv] << ", cloud_list[ig].nb_points_row[iv + 1]: " << cloud_list[ig].nb_points_row[iv + 1] << endl;
                coeff_mat_row[ig].push_back(coeff_mat_row[ig][iv] + (cloud_list[ig].nb_points_row[iv + 1] - cloud_list[ig].nb_points_row[iv]));
                // if (full_neumann_flag_list[ig]) //with regularization
                //     coeff_mat_row[ig].push_back(coeff_mat_row[ig][iv] + (cloud_list[ig].nb_points_row[iv + 1] - cloud_list[ig].nb_points_row[iv]) + 1);
                // else //without regularization
                //     coeff_mat_row[ig].push_back(coeff_mat_row[ig][iv] + (cloud_list[ig].nb_points_row[iv + 1] - cloud_list[ig].nb_points_row[iv]));
                for (int i1 = cloud_list[ig].nb_points_row[iv]; i1 < cloud_list[ig].nb_points_row[iv + 1]; i1++)
                {
                    // if (ig == 0 && iv == 4)
                    //     cout << "   MULTIGRID::assemble_coeff ig: " << ig << ", iv: " << iv << ", cloud_list[ig].nb_points_col[i1]: " << cloud_list[ig].nb_points_col[i1] << ", cloud_list[ig].laplacian_coeff[i1]: " << cloud_list[ig].laplacian_coeff[i1] << endl;
                    coeff_mat_col[ig].push_back(cloud_list[ig].nb_points_col[i1]);
                    coeff_mat_val[ig].push_back(cloud_list[ig].laplacian_coeff[i1]); //diffusion
                }
                // if (full_neumann_flag_list[ig])
                // { //with regularization: last column with unit coefficient
                //     coeff_mat_col[ig].push_back(points_list[ig].nv);
                //     coeff_mat_val[ig].push_back(1.0);
                // }
            }
        }
        //all actual columns and unit last column (for regularization if required) added so far
        // if (full_neumann_flag_list[ig])
        // { //add regularization at row[nv-1] and shift current row[nv-1] to row[nv]
        //     int iv = points_list[ig].nv - 1;
        //     for (int i1 = coeff_mat_row[ig][iv]; i1 < coeff_mat_row[ig][iv + 1]; i1++)
        //     { //store row[nv-1] into temp locations
        //         empty_int_vector.push_back(coeff_mat_col[ig][i1]);
        //         empty_double_vector.push_back(coeff_mat_val[ig][i1]);
        //     }
        //     for (int i1 = coeff_mat_row[ig][iv + 1] - 1; i1 >= coeff_mat_row[ig][iv]; i1--)
        //     { //remove row[nv-1]
        //         coeff_mat_col[ig].pop_back();
        //         coeff_mat_val[ig].pop_back();
        //     }

        //     coeff_mat_row[ig][iv + 1] = coeff_mat_row[ig][iv] + points_list[ig].nv;
        //     coeff_mat_col[ig].push_back(points_list[ig].nv - 1);   //add diagonal entry first
        //     coeff_mat_val[ig].push_back(1.0);                      //add diagonal entry first
        //     for (int iv1 = 0; iv1 < points_list[ig].nv - 1; iv1++) //add regularization row
        //         coeff_mat_col[ig].push_back(iv1), coeff_mat_val[ig].push_back(1.0);

        //     if (ig == 0)
        //     {
        //         print_to_terminal(empty_int_vector, "MULTIGRID::assemble_coeff empty_int_vector");
        //         print_to_terminal(empty_double_vector, "MULTIGRID::assemble_coeff empty_double_vector");
        //     }
        //     coeff_mat_row[ig].push_back(coeff_mat_row[ig][iv + 1] + empty_int_vector.size());
        //     coeff_mat_col[ig].push_back(points_list[ig].nv); //add diagonal entry first
        //     coeff_mat_val[ig].push_back(1.0);                //add diagonal entry first
        //     for (int i1 = 0; i1 < empty_int_vector.size() - 1; i1++)
        //     { //add the previously removed row[nv-1] without diagonal entry
        //         coeff_mat_col[ig].push_back(empty_int_vector[i1]);
        //         coeff_mat_val[ig].push_back(empty_double_vector[i1]);
        //     }

        //     empty_int_vector.clear(), empty_double_vector.clear();
        // }
    }

    // for (int ig = 0; ig < ng; ig++)
    // {
    //     coeff_mat_col_sum.push_back(Eigen::VectorXd::Zero(points_list[ig].nv));
    //     for (int iv = 0; iv < points_list[ig].nv; iv++)
    //         for (int i1 = coeff_mat_row[ig][iv]; i1 < coeff_mat_row[ig][iv + 1]; i1++)
    //         {
    //             ivnb = coeff_mat_col[ig][i1]; //column number
    //             coeff_mat_col_sum[ig][ivnb] = coeff_mat_col_sum[ig][ivnb] + coeff_mat_val[ig][i1];
    //         }
    // }
}

void MULTIGRID::single_grid_sor(vector<Eigen::VectorXd> &phi, vector<Eigen::VectorXd> &source, vector<vector<double>> &residual, int ig)
{
    double diag_coeff, rhs, phi_new, res, res_init, alpha;
    // vector<double> empty_double_vector;
    // if (full_neumann_flag_list[ig]) //regularization for full Neumann case
    //     regul_alpha.push_back(empty_double_vector);
    // cout << "\n\n";
    // phi[ig].head(points_list[ig].nv) = phi1[ig];
    // source[ig].head(points_list[ig].nv) = source1[ig];
    // if (full_neumann_flag_list[ig])
    // {
    //     phi[ig][points_list[ig].nv] = 0.0;
    //     source[ig][points_list[ig].nv - 1] = 0.0;                            //regularization equation
    //     source[ig][points_list[ig].nv] = source[ig][points_list[ig].nv - 1]; //actual last equation
    // }
    for (int iter = 0; iter < n_iter; iter++)
    {
        for (int iv = 0; iv < points_list[ig].nv; iv++)
        {
            rhs = source[ig][iv];
            // if (full_neumann_flag_list[ig] && (iter > 1000))
            //     rhs = rhs - regul_alpha[ig].back();
            diag_coeff = 0.0;
            for (int i1 = coeff_mat_row[ig][iv]; i1 < coeff_mat_row[ig][iv + 1]; i1++)
            {
                if (coeff_mat_col[ig][i1] == iv)
                    diag_coeff = coeff_mat_val[ig][i1];
                else
                    rhs = rhs - (coeff_mat_val[ig][i1] * phi[ig][coeff_mat_col[ig][i1]]);
            }
            if (fabs(diag_coeff) < 1E-10)
            {
                printf("\n\nERROR from MULTIGRID::single_grid_sor ig: %i, iv: %i, iter: %i, diag_coeff: %g, boundary_flag: %i, dirichlet_flag: %i\n\n", ig, iv, iter, diag_coeff, int(points_list[ig].boundary_flag[iv]), int(dirichlet_flag_list[ig][iv]));
                throw bad_exception();
            }
            phi_new = rhs / diag_coeff;
            phi[ig][iv] = (sor_omega * phi_new) + ((1.0 - sor_omega) * phi[ig][iv]);
        }

        // if (full_neumann_flag_list[ig])
        // {                                                         //regularization for full Neumann case
        //     phi[ig].array() = phi[ig].array() - (phi[ig].mean()); //set sum to zero
        //     alpha = (coeff_mat_col_sum[ig].cwiseProduct(phi[ig])).sum();
        //     alpha = (source[ig].sum() - alpha) / points_list[ig].nv;
        //     regul_alpha[ig].push_back(alpha);
        // }

        // for (int iv = 0; iv < points_list[ig].nv; iv++)
        // {
        //     if (dirichlet_flag_list[ig][iv] || !points_list[ig].boundary_flag[iv])
        //     { //[boundary point with dirichlet] || [interior point]
        //         rhs = source[ig][iv];
        //         for (int i1 = coeff_mat_row[ig][iv]; i1 < coeff_mat_row[ig][iv + 1]; i1++)
        //         {
        //             if (coeff_mat_col[ig][i1] == iv)
        //                 diag_coeff = coeff_mat_val[ig][i1];
        //             else
        //                 rhs = rhs - (coeff_mat_val[ig][i1] * phi[ig][coeff_mat_col[ig][i1]]);
        //         }
        //         if (fabs(diag_coeff) < 1E-10)
        //         {
        //             printf("\n\nERROR from MULTIGRID::single_grid_sor ig: %i, iv: %i, iter: %i, diag_coeff: %g, boundary_flag: %i, dirichlet_flag: %i\n\n", ig, iv, iter, diag_coeff, int(points_list[ig].boundary_flag[iv]), int(dirichlet_flag_list[ig][iv]));
        //             throw bad_exception();
        //         }
        //         phi_new = rhs / diag_coeff;
        //         phi[ig][iv] = (sor_omega * phi_new) + ((1.0 - sor_omega) * phi[ig][iv]);
        //     }
        // }
        // for (int iv = 0; iv < points_list[ig].nv; iv++)
        // {
        //     if (!dirichlet_flag_list[ig][iv] && points_list[ig].boundary_flag[iv])
        //     { //boundary point with neumann
        //         rhs = source[ig][iv];
        //         for (int i1 = coeff_mat_row[ig][iv]; i1 < coeff_mat_row[ig][iv + 1]; i1++)
        //         {
        //             if (coeff_mat_col[ig][i1] == iv)
        //                 diag_coeff = coeff_mat_val[ig][i1];
        //             else
        //                 rhs = rhs - (coeff_mat_val[ig][i1] * phi[ig][coeff_mat_col[ig][i1]]);
        //         }
        //         if (fabs(diag_coeff) < 1E-10)
        //         {
        //             printf("\n\nERROR from MULTIGRID::single_grid_sor ig: %i, iv: %i, iter: %i, diag_coeff: %g, boundary_flag: %i, dirichlet_flag: %i\n\n", ig, iv, iter, diag_coeff, int(points_list[ig].boundary_flag[iv]), int(dirichlet_flag_list[ig][iv]));
        //             throw bad_exception();
        //         }
        //         phi_new = rhs / diag_coeff;
        //         phi[ig][iv] = (sor_omega * phi_new) + ((1.0 - sor_omega) * phi[ig][iv]);
        //     }
        // }
        // phi[ig].array() = phi[ig].array() - phi[ig][0]; //subtract level
        // phi[ig].array() = phi[ig].array() - (phi[ig].mean()); //set sum to zero

        // phi1[ig] = phi[ig].head(points_list[ig].nv);
        // if (full_neumann_flag_list[ig])
        //     regul_alpha[ig].push_back(phi[ig][points_list[ig].nv]);

        residual[ig].push_back(0.0);
        for (int iv = 0; iv < points_list[ig].nv; iv++)
        {
            res = source[ig][iv];
            for (int i1 = coeff_mat_row[ig][iv]; i1 < coeff_mat_row[ig][iv + 1]; i1++)
                res = res - (coeff_mat_val[ig][i1] * phi[ig][coeff_mat_col[ig][i1]]);
            residual[ig][iter] = residual[ig][iter] + (res * res);
        }
        residual[ig][iter] = sqrt(residual[ig][iter] / points_list[ig].nv);
        if (iter == 0)
            res_init = residual[ig][0];
        residual[ig][iter] = residual[ig][iter] / res_init; //normalize by initial residual
        // printf("single_grid_sor ig: %i, nv: %i, iter: %i, res: %g\n", ig, points_list[ig].nv, iter, residual[ig][iter]);
    }
    // cout << "\n\n";
}