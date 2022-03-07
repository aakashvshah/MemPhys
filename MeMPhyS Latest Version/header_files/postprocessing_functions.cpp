// Author: Dr. Shantanu Shahane
#include "postprocessing_functions.hpp"

double calc_boundary_flux(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &phi_x, Eigen::VectorXd &phi_y, int bc_tag)
{ // sum of [(phi_x*normal_x + phi_y*normal_y) * elem_area] for all elements with given bc_tag
    double flux = 0.0, icv_flux;
    int iv_orig, iv_new, dim = parameters.dimension;
    for (int icv = 0; icv < points.nelem_original; icv++)
    {
        if (points.elem_boundary_flag_original[icv])
        {
            if (points.elem_bc_tag_original[icv] == bc_tag)
            {
                icv_flux = 0.0;
                for (int i1 = 0; i1 < points.elem_vert_original[icv].size(); i1++)
                {
                    iv_orig = points.elem_vert_original[icv][i1];
                    iv_new = points.iv_original_nearest_vert[iv_orig];
                    icv_flux = icv_flux + (phi_x[iv_new] * points.normal[dim * iv_new] + phi_y[iv_new] * points.normal[dim * iv_new + 1]);
                }
                flux = flux + (icv_flux * points.boundary_face_area_original[icv] / ((double)points.elem_vert_original[icv].size()));
            }
        }
    }
    return flux;
}

double calc_boundary_flux(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &phi_x, Eigen::VectorXd &phi_y, Eigen::VectorXd &phi_z, int bc_tag)
{ // sum of [(phi_x*normal_x + phi_y*normal_y + phi_z*normal_z) * elem_area] for all elements with given bc_tag
    double flux = 0.0, icv_flux;
    int iv_orig, iv_new, dim = parameters.dimension;
    for (int icv = 0; icv < points.nelem_original; icv++)
    {
        if (points.elem_boundary_flag_original[icv])
        {
            if (points.elem_bc_tag_original[icv] == bc_tag)
            {
                icv_flux = 0.0;
                for (int i1 = 0; i1 < points.elem_vert_original[icv].size(); i1++)
                {
                    iv_orig = points.elem_vert_original[icv][i1];
                    iv_new = points.iv_original_nearest_vert[iv_orig];
                    icv_flux = icv_flux + (phi_x[iv_new] * points.normal[dim * iv_new] + phi_y[iv_new] * points.normal[dim * iv_new + 1] + phi_z[iv_new] * points.normal[dim * iv_new + 2]);
                }
                flux = flux + (icv_flux * points.boundary_face_area_original[icv] / ((double)points.elem_vert_original[icv].size()));
            }
        }
    }
    return flux;
}

void write_iteration_details(PARAMETERS &parameters)
{
    FILE *file;
    string output_file = parameters.output_file_prefix + "_iteration_details.csv";
    file = fopen(output_file.c_str(), "w");
    if (parameters.regul_alpha_log.size() == 0)
    {
        fprintf(file, "it,n_iter_actual,rel_res_log,abs_res_log\n");
        for (int it = 0; it < parameters.n_iter_actual.size(); it++)
            fprintf(file, "%i,%i,%.16g,%.16g\n", it, parameters.n_iter_actual[it], parameters.rel_res_log[it], parameters.abs_res_log[it]);
    }
    else
    {
        fprintf(file, "it,n_iter_actual,rel_res_log,abs_res_log,regul_alpha_log\n");
        for (int it = 0; it < parameters.n_iter_actual.size(); it++)
            fprintf(file, "%i,%i,%.16g,%.16g,%.16g\n", it, parameters.n_iter_actual[it], parameters.rel_res_log[it], parameters.abs_res_log[it], parameters.regul_alpha_log[it]);
    }

    fclose(file);
    cout << "\nwrite_iteration_details wrote " << output_file << "\n\n";
}

void write_simulation_details(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
{
    char hostname[HOST_NAME_MAX];
    char username[LOGIN_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    getlogin_r(username, LOGIN_NAME_MAX);

    FILE *file;
    string output_file = parameters.output_file_prefix + "_simulation_details.csv";
    file = fopen(output_file.c_str(), "w");
    fprintf(file, "hostname,%s\n", hostname);
    fprintf(file, "username,%s\n", username);
    fprintf(file, "\n");

    fprintf(file, "Grid,Details\n");
    fprintf(file, "meshfile,name,,,,%s\n", parameters.meshfile.c_str());
    if (parameters.restart_tecplot_file.size() > 0)
        fprintf(file, "restart,file,,,,%s\n", parameters.restart_tecplot_file.c_str());
    fprintf(file, "Problem,dimension,%i\n", parameters.dimension);
    fprintf(file, "No. of,nodes,nv,%i\n", points.nv);
    int nv_boundary = 0;
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv])
            nv_boundary++;
    fprintf(file, "No. of,nodes,nv_boundary,%i\n", nv_boundary);
    fprintf(file, "No. of,nodes,nv_internal,%i\n", points.nv - nv_boundary);
    fprintf(file, "Characteristic,dx_max,%.16g\n", parameters.max_dx);
    fprintf(file, "Characteristic,dx_min,%.16g\n", parameters.min_dx);
    fprintf(file, "Characteristic,dx_avg,%.16g\n", parameters.avg_dx);
    fprintf(file, "\n");

    fprintf(file, "RBF,Details\n");
    fprintf(file, "phs_deg,%i\n", parameters.phs_deg);
    fprintf(file, "poly_deg,%i\n", parameters.poly_deg);
    fprintf(file, "cloud_size_multiplier,%.16g\n", parameters.cloud_size_multiplier);
    fprintf(file, "num_poly_terms,%i\n", parameters.num_poly_terms);
    fprintf(file, "cloud_size,%i\n", parameters.cloud_size);
    fprintf(file, "RBF_cond_num_max,%.16g\n", cloud.cond_num_RBF_max);
    fprintf(file, "RBF_cond_num_min,%.16g\n", cloud.cond_num_RBF_min);
    fprintf(file, "RBF_cond_num_avg,%.16g\n", cloud.cond_num_RBF_avg);
    fprintf(file, "grad_x_eigval,%.16g,%.16g\n", parameters.grad_x_eigval_real, parameters.grad_x_eigval_imag);
    fprintf(file, "grad_y_eigval,%.16g,%.16g\n", parameters.grad_y_eigval_real, parameters.grad_y_eigval_imag);
    if (parameters.dimension == 3)
        fprintf(file, "grad_z_eigval,%.16g,%.16g\n", parameters.grad_z_eigval_real, parameters.grad_z_eigval_imag);
    fprintf(file, "laplace_eigval,%.16g,%.16g\n", parameters.laplace_eigval_real, parameters.laplace_eigval_imag);
    fprintf(file, "\n");

    fprintf(file, "Solver,Settings\n");
    fprintf(file, "nt,%i\n", parameters.nt);
    fprintf(file, "nt_actual,%i\n", parameters.nt_actual);
    fprintf(file, "timestep_dt,%.16g,seconds\n", parameters.dt);
    fprintf(file, "steady_tolerance,%.16g\n", parameters.steady_tolerance);
    fprintf(file, "solver_tolerance,%.16g\n", parameters.solver_tolerance);
    fprintf(file, "Courant,%.16g\n", parameters.Courant);
    fprintf(file, "solver_type,%s\n", parameters.solver_type.c_str());
    fprintf(file, "gmres_kdim,%i\n", parameters.gmres_kdim);
    fprintf(file, "euclid_precond_level_hypre,%i\n", parameters.euclid_precond_level_hypre);
    fprintf(file, "precond_droptol,%.16g\n", parameters.precond_droptol);
    fprintf(file, "n_iter,%i\n", parameters.n_iter);
    fprintf(file, "\n");

    fprintf(file, "Fluid and Flow,Details\n");
    fprintf(file, "mu,%.16g\n", parameters.mu);
    fprintf(file, "rho,%.16g\n", parameters.rho);
    fprintf(file, "\n");

    fprintf(file, "Run Times,(seconds)\n");
    fprintf(file, "points_timer,%.16g\n", parameters.points_timer);
    fprintf(file, "cloud_id_timer,%.16g\n", parameters.cloud_id_timer);
    fprintf(file, "rcm_timer,%.16g\n", parameters.rcm_timer);
    fprintf(file, "cloud_misc_timer,%.16g\n", parameters.cloud_misc_timer);
    fprintf(file, "grad_laplace_coeff_timer,%.16g\n", parameters.grad_laplace_coeff_timer);
    fprintf(file, "factoring_timer,%.16g\n", parameters.factoring_timer);
    fprintf(file, "solve_timer,%.16g\n", parameters.solve_timer);
    fprintf(file, "total_timer,%.16g\n", parameters.total_timer);

    fclose(file);
    cout << "\nwrite_simulation_details wrote " << output_file << "\n\n";
}

void calc_navier_stokes_errors_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_ana, Eigen::VectorXd &v_ana, Eigen::VectorXd &p_ana, Eigen::VectorXd &u_num, Eigen::VectorXd &v_num, Eigen::VectorXd &p_num)
{
    cout << "\ncalc_navier_stokes_errors_2D started (max., avg.)\n";
    double max_err, l1_err, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal;
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(points.nv), residual = zero_vector;

    residual = (points.grad_x_matrix_EIGEN * u_num) + (points.grad_y_matrix_EIGEN * v_num);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    continuity: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    calc_max_l1_error(u_ana, u_num, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(u_ana, u_num, max_err, l1_err);
    printf("    U X-vel: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    calc_max_l1_error(v_ana, v_num, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(v_ana, v_num, max_err, l1_err);
    printf("    V Y-vel: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    p_num = p_num - (Eigen::VectorXd::Ones(p_num.size()) * (p_num[0] - p_ana[0])); // reset level to analytical solution
    if (p_ana.rows() == points.nv + 1)
        p_ana[points.nv] = 0.0, p_num[points.nv] = 0.0;
    calc_max_l1_error(p_ana, p_num, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(p_ana, p_num, max_err, l1_err);
    printf("    Pressure: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);
    cout << "calc_navier_stokes_errors_2D ended (max., avg.)\n";
}

void calc_navier_stokes_residuals_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p)
{
    if (parameters.rho < 0 || parameters.mu < 0)
    {
        printf("\n\nERROR from calc_navier_stokes_residuals_2D Some parameters are not set; parameters.rho: %g, parameters.mu: %g\n\n", parameters.rho, parameters.mu);
        throw bad_exception();
    }
    cout << "\ncalc_navier_stokes_residuals_2D started (max., avg.)\n";
    double max_err, l1_err, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal;
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(points.nv), residual = zero_vector;
    residual = (points.grad_x_matrix_EIGEN * u) + (points.grad_y_matrix_EIGEN * v);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    continuity: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    residual = u.cwiseProduct(points.grad_x_matrix_EIGEN * u) + v.cwiseProduct(points.grad_y_matrix_EIGEN * u) - (parameters.mu * points.laplacian_matrix_EIGEN * u / parameters.rho) + (points.grad_x_matrix_EIGEN * p.head(points.nv) / parameters.rho);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    X-mom: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    residual = u.cwiseProduct(points.grad_x_matrix_EIGEN * v) + v.cwiseProduct(points.grad_y_matrix_EIGEN * v) - (parameters.mu * points.laplacian_matrix_EIGEN * v / parameters.rho) + (points.grad_y_matrix_EIGEN * p.head(points.nv) / parameters.rho);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    Y-mom: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    cout << "calc_navier_stokes_residuals_2D ended (max., avg.)\n\n";
}

void calc_navier_stokes_residuals_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y)
{
    if (parameters.rho < 0 || parameters.mu < 0)
    {
        printf("\n\nERROR from calc_navier_stokes_residuals_2D Some parameters are not set; parameters.rho: %g, parameters.mu: %g\n\n", parameters.rho, parameters.mu);
        throw bad_exception();
    }
    cout << "\ncalc_navier_stokes_residuals_2D started (max., avg.)\n";
    double max_err, l1_err, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal;
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(points.nv), residual = zero_vector;
    residual = (points.grad_x_matrix_EIGEN * u) + (points.grad_y_matrix_EIGEN * v);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    continuity: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    residual = u.cwiseProduct(points.grad_x_matrix_EIGEN * u) + v.cwiseProduct(points.grad_y_matrix_EIGEN * u) - (parameters.mu * points.laplacian_matrix_EIGEN * u / parameters.rho) + (points.grad_x_matrix_EIGEN * p.head(points.nv) / parameters.rho) - (body_force_x / parameters.rho);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    X-mom: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    residual = u.cwiseProduct(points.grad_x_matrix_EIGEN * v) + v.cwiseProduct(points.grad_y_matrix_EIGEN * v) - (parameters.mu * points.laplacian_matrix_EIGEN * v / parameters.rho) + (points.grad_y_matrix_EIGEN * p.head(points.nv) / parameters.rho) - (body_force_y / parameters.rho);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    Y-mom: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    cout << "calc_navier_stokes_residuals_2D ended (max., avg.)\n\n";
}

void calc_navier_stokes_residuals_3D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &w, Eigen::VectorXd &p)
{
    if (parameters.rho < 0 || parameters.mu < 0)
    {
        printf("\n\nERROR from calc_navier_stokes_residuals_3D Some parameters are not set; parameters.rho: %g, parameters.mu: %g\n\n", parameters.rho, parameters.mu);
        throw bad_exception();
    }
    cout << "\ncalc_navier_stokes_residuals_3D started (max., avg.)\n";
    double max_err, l1_err, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal;
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(points.nv), residual = zero_vector;
    residual = (points.grad_x_matrix_EIGEN * u) + (points.grad_y_matrix_EIGEN * v) + (points.grad_z_matrix_EIGEN * w);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    continuity: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    residual = u.cwiseProduct(points.grad_x_matrix_EIGEN * u) + v.cwiseProduct(points.grad_y_matrix_EIGEN * u) + w.cwiseProduct(points.grad_z_matrix_EIGEN * u) - (parameters.mu * points.laplacian_matrix_EIGEN * u / parameters.rho) + (points.grad_x_matrix_EIGEN * p.head(points.nv) / parameters.rho);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    X-mom: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    residual = u.cwiseProduct(points.grad_x_matrix_EIGEN * v) + v.cwiseProduct(points.grad_y_matrix_EIGEN * v) + w.cwiseProduct(points.grad_z_matrix_EIGEN * v) - (parameters.mu * points.laplacian_matrix_EIGEN * v / parameters.rho) + (points.grad_y_matrix_EIGEN * p.head(points.nv) / parameters.rho);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    Y-mom: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    residual = u.cwiseProduct(points.grad_x_matrix_EIGEN * w) + v.cwiseProduct(points.grad_y_matrix_EIGEN * w) + w.cwiseProduct(points.grad_z_matrix_EIGEN * w) - (parameters.mu * points.laplacian_matrix_EIGEN * w / parameters.rho) + (points.grad_z_matrix_EIGEN * p.head(points.nv) / parameters.rho);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    Z-mom: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    cout << "calc_navier_stokes_residuals_3D ended (max., avg.)\n\n";
}

void calc_navier_stokes_residuals_3D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &w, Eigen::VectorXd &p, Eigen::VectorXd &body_force_x, Eigen::VectorXd &body_force_y, Eigen::VectorXd &body_force_z)
{
    if (parameters.rho < 0 || parameters.mu < 0)
    {
        printf("\n\nERROR from calc_navier_stokes_residuals_3D Some parameters are not set; parameters.rho: %g, parameters.mu: %g\n\n", parameters.rho, parameters.mu);
        throw bad_exception();
    }
    cout << "\ncalc_navier_stokes_residuals_3D started (max., avg.)\n";
    double max_err, l1_err, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal;
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(points.nv), residual = zero_vector;
    residual = (points.grad_x_matrix_EIGEN * u) + (points.grad_y_matrix_EIGEN * v) + (points.grad_z_matrix_EIGEN * w);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    continuity: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    residual = u.cwiseProduct(points.grad_x_matrix_EIGEN * u) + v.cwiseProduct(points.grad_y_matrix_EIGEN * u) + w.cwiseProduct(points.grad_z_matrix_EIGEN * u) - (parameters.mu * points.laplacian_matrix_EIGEN * u / parameters.rho) + (points.grad_x_matrix_EIGEN * p.head(points.nv) / parameters.rho) - (body_force_x / parameters.rho);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    X-mom: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    residual = u.cwiseProduct(points.grad_x_matrix_EIGEN * v) + v.cwiseProduct(points.grad_y_matrix_EIGEN * v) + w.cwiseProduct(points.grad_z_matrix_EIGEN * v) - (parameters.mu * points.laplacian_matrix_EIGEN * v / parameters.rho) + (points.grad_y_matrix_EIGEN * p.head(points.nv) / parameters.rho) - (body_force_y / parameters.rho);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    Y-mom: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    residual = u.cwiseProduct(points.grad_x_matrix_EIGEN * w) + v.cwiseProduct(points.grad_y_matrix_EIGEN * w) + w.cwiseProduct(points.grad_z_matrix_EIGEN * w) - (parameters.mu * points.laplacian_matrix_EIGEN * w / parameters.rho) + (points.grad_z_matrix_EIGEN * p.head(points.nv) / parameters.rho) - (body_force_z / parameters.rho);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    Z-mom: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    cout << "calc_navier_stokes_residuals_3D ended (max., avg.)\n\n";
}

void calc_navier_stokes_errors_3D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_ana, Eigen::VectorXd &v_ana, Eigen::VectorXd &w_ana, Eigen::VectorXd &p_ana, Eigen::VectorXd &u_num, Eigen::VectorXd &v_num, Eigen::VectorXd &w_num, Eigen::VectorXd &p_num)
{
    cout << "\ncalc_navier_stokes_errors_3D started (max., avg.)\n";
    double max_err, l1_err, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal;
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(points.nv), residual = zero_vector;

    residual = (points.grad_x_matrix_EIGEN * u_num) + (points.grad_y_matrix_EIGEN * v_num) + (points.grad_z_matrix_EIGEN * w_num);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    printf("    continuity: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    calc_max_l1_error(u_ana, u_num, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(u_ana, u_num, max_err, l1_err);
    printf("    U X-vel: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    calc_max_l1_error(v_ana, v_num, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(v_ana, v_num, max_err, l1_err);
    printf("    V Y-vel: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    calc_max_l1_error(w_ana, w_num, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(w_ana, w_num, max_err, l1_err);
    printf("    W Z-vel: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    p_num.head(points.nv) = p_num.head(points.nv) - (Eigen::VectorXd::Ones(points.nv) * (p_num[0] - p_ana[0])); // reset level to analytical solution
    if (p_ana.rows() == points.nv + 1)
        p_ana[points.nv] = 0.0, p_num[points.nv] = 0.0;
    calc_max_l1_error(p_ana, p_num, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(p_ana, p_num, max_err, l1_err);
    printf("    Pressure: internal: (%g, %g), boundary: (%g, %g), overall: (%g, %g)\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);
    cout << "calc_navier_stokes_errors_3D ended (max., avg.)\n";
}

void write_navier_stokes_errors_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u_ana, Eigen::VectorXd &v_ana, Eigen::VectorXd &p_ana, Eigen::VectorXd &u_num, Eigen::VectorXd &v_num, Eigen::VectorXd &p_num)
{
    FILE *file;
    string output_file = parameters.output_file_prefix + "_navier_stokes_errors.csv";
    file = fopen(output_file.c_str(), "w");
    fprintf(file, "Errors,wrt,Analytical,solution\n");
    fprintf(file, ",max_internal,avg_internal,max_boundary,avg_boundary,max_overall,avg_overall\n");

    double max_err, l1_err, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal;
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(points.nv), residual = zero_vector;

    residual = (points.grad_x_matrix_EIGEN * u_num) + (points.grad_y_matrix_EIGEN * v_num);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    fprintf(file, "continuity,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    calc_max_l1_error(u_ana, u_num, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(u_ana, u_num, max_err, l1_err);
    fprintf(file, "U X-vel,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    calc_max_l1_error(v_ana, v_num, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(v_ana, v_num, max_err, l1_err);
    fprintf(file, "V Y-vel,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    p_num = p_num - (Eigen::VectorXd::Ones(p_num.size()) * (p_num[0] - p_ana[0])); // reset level to analytical solution
    p_ana[points.nv] = 0.0, p_num[points.nv] = 0.0;
    calc_max_l1_error(p_ana, p_num, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(p_ana, p_num, max_err, l1_err);
    fprintf(file, "Pressure,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    fclose(file);
    cout << "\nwrite_navier_stokes_errors_2D wrote " << output_file << "\n\n";
}

void write_navier_stokes_residuals_2D(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p, string output_file_suffix)
{
    if (parameters.rho < 0 || parameters.mu < 0)
    {
        printf("\n\nERROR from write_navier_stokes_residuals_2D Some parameters are not set; parameters.rho: %g, parameters.mu: %g\n\n", parameters.rho, parameters.mu);
        throw bad_exception();
    }
    FILE *file;
    string output_file = parameters.output_file_prefix + output_file_suffix;
    file = fopen(output_file.c_str(), "w");
    fprintf(file, "Numerical,Residuals,wrt,Given,solution\n");
    fprintf(file, ",max_internal,avg_internal,max_boundary,avg_boundary,max_overall,avg_overall\n");
    double max_err, l1_err, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal;
    Eigen::VectorXd zero_vector = Eigen::VectorXd::Zero(points.nv), residual = zero_vector;
    residual = (points.grad_x_matrix_EIGEN * u) + (points.grad_y_matrix_EIGEN * v);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    fprintf(file, "continuity,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    residual = u.cwiseProduct(points.grad_x_matrix_EIGEN * u) + v.cwiseProduct(points.grad_y_matrix_EIGEN * u) - (parameters.mu * points.laplacian_matrix_EIGEN * u / parameters.rho) + (points.grad_x_matrix_EIGEN * p.head(points.nv) / parameters.rho);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    fprintf(file, "X-mom,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    residual = u.cwiseProduct(points.grad_x_matrix_EIGEN * v) + v.cwiseProduct(points.grad_y_matrix_EIGEN * v) - (parameters.mu * points.laplacian_matrix_EIGEN * v / parameters.rho) + (points.grad_y_matrix_EIGEN * p.head(points.nv) / parameters.rho);
    calc_max_l1_error(zero_vector, residual, max_err_boundary, l1_err_boundary, max_err_internal, l1_err_internal, points.boundary_flag);
    calc_max_l1_error(zero_vector, residual, max_err, l1_err);
    fprintf(file, "Y-mom,%.16g,%.16g,%.16g,%.16g,%.16g,%.16g\n", max_err_internal, l1_err_internal, max_err_boundary, l1_err_boundary, max_err, l1_err);

    fclose(file);
    cout << "\nwrite_navier_stokes_residuals_2D wrote " << output_file << "\n\n";
}

void write_tecplot_temporal_variables_header(POINTS &points, PARAMETERS &parameters, vector<string> &variable_names)
{
    FILE *file;
    string output_file = parameters.output_file_prefix + "_tecplot_temporal.plt";
    file = fopen(output_file.c_str(), "w");
    if (parameters.dimension == 2)
        fprintf(file, "VARIABLES=\"x\", \"y\"");
    else
        fprintf(file, "VARIABLES=\"x\", \"y\", \"z\"");
    for (int var = 0; var < variable_names.size(); var++)
        fprintf(file, ", \"%s\"", variable_names[var].c_str());
    fprintf(file, "\n");
    fclose(file);
}

void write_tecplot_temporal_particles_header(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &position, vector<string> &variable_names)
{
    FILE *file;
    string output_file = parameters.output_file_prefix + "_tecplot_particles_temporal.plt";
    file = fopen(output_file.c_str(), "w");
    if (position.cols() == 2)
        fprintf(file, "VARIABLES=\"x\", \"y\", \"u\", \"v\"");
    else
        fprintf(file, "VARIABLES=\"x\", \"y\", \"z\", \"u\", \"v\", \"w\"");
    for (int var = 0; var < variable_names.size(); var++)
        fprintf(file, ", \"%s\"", variable_names[var].c_str());
    fprintf(file, "\n");
    fclose(file);
}

void write_tecplot_temporal_variables_header(POINTS &points, PARAMETERS &parameters, vector<string> &variable_names, Eigen::MatrixXd &x_3d, Eigen::MatrixXd &y_3d, Eigen::MatrixXd &z_3d)
{
    if (parameters.dimension != 2)
    {
        printf("\n\nERROR from write_tecplot_temporal_variables_header defined only for 2D problems parameters.dimension: %i\n\n", parameters.dimension);
        throw bad_exception();
    }
    FILE *file;
    string output_file = parameters.output_file_prefix + "_tecplot_temporal.plt";
    file = fopen(output_file.c_str(), "w");
    if (x_3d.cols() == 1) // problem with a single wave (zeroth: mean wave)
        fprintf(file, "VARIABLES=\"x\", \"y\"");
    else
        fprintf(file, "VARIABLES=\"x\", \"y\", \"z\"");
    for (int var = 0; var < variable_names.size(); var++)
        fprintf(file, ", \"%s\"", variable_names[var].c_str());
    fprintf(file, "\n");
    fclose(file);
}

void write_tecplot_temporal_particles(POINTS &points, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, vector<string> &variable_names, vector<Eigen::VectorXd *> &variable_pointers, double physical_time)
{
    int dim = position.cols(), np = position.rows();
    string output_file = parameters.output_file_prefix + "_tecplot_particles_temporal.plt";
    FILE *file;
    file = fopen(output_file.c_str(), "a");
    if (dim == 2)
        fprintf(file, "ZONE T=\"%.16g\", I=%i, J=1, F=POINT\n", physical_time, np);
    else
        fprintf(file, "ZONE T=\"%.16g\", I=%i, J=1, K=1, F=POINT\n", physical_time, np);
    for (int ip = 0; ip < np; ip++)
    {
        for (int i1 = 0; i1 < dim; i1++)
            fprintf(file, "%.16g ", position(ip, i1)); // write x, y, z
        for (int i1 = 0; i1 < dim; i1++)
            fprintf(file, "%.16g ", velocity(ip, i1)); // write u, v, w
        for (int var = 0; var < variable_pointers.size(); var++)
            fprintf(file, "%.16g ", (*variable_pointers[var])[ip]); // write variables
        fprintf(file, "\n");
    }
    fclose(file);
}

void write_tecplot_temporal_variables(POINTS &points, PARAMETERS &parameters, vector<string> &variable_names, vector<Eigen::VectorXd *> &variable_pointers, int it)
{
    int dim = parameters.dimension, ncv = 0, iv1;
    for (int icv = 0; icv < points.elem_vert_original.size(); icv++)
        if (points.elem_vert_original[icv].size() == dim + 1)
            ncv++; // triangles for 2D and tetrahedrons for 3D
    string output_file = parameters.output_file_prefix + "_tecplot_temporal.plt";
    FILE *file;
    file = fopen(output_file.c_str(), "a");
    // fprintf(file, "# Fields it: %i\n", it);
    if (dim == 2)
        fprintf(file, "ZONE T=\"%.16g\"  DATAPACKING=POINT, NODES=%i, ELEMENTS=%i, ZONETYPE=FETRIANGLE\n", it * parameters.dt, points.nv_original, ncv);
    else
        fprintf(file, "ZONE T=\"%.16g\"  DATAPACKING=POINT, NODES=%i, ELEMENTS=%i, ZONETYPE=FETETRAHEDRON\n", it * parameters.dt, points.nv_original, ncv);

    for (int iv0 = 0; iv0 < points.nv_original; iv0++)
    {
        iv1 = points.iv_original_nearest_vert[iv0];
        for (int i1 = 0; i1 < dim; i1++) // write x, y, z
            fprintf(file, "%.16g ", points.xyz_original[dim * iv0 + i1]);
        for (int var = 0; var < variable_pointers.size(); var++) // write variables
            fprintf(file, "%.16g ", (*variable_pointers[var])[iv1]);
        fprintf(file, "\n");
    }

    for (int icv = 0; icv < points.elem_vert_original.size(); icv++)
        if (points.elem_vert_original[icv].size() == dim + 1)
        { // triangles for 2D, tetrahedrons for 3D
            for (int i1 = 0; i1 < points.elem_vert_original[icv].size(); i1++)
                fprintf(file, "%i ", points.elem_vert_original[icv][i1] + 1);
            fprintf(file, "\n");
        }
    fclose(file);
}

void write_tecplot_temporal_variables_fourier_zeroth_wave(POINTS &points, PARAMETERS &parameters, vector<string> &variable_names, vector<Eigen::MatrixXd *> &variable_pointers, Eigen::MatrixXd &x_3d, Eigen::MatrixXd &y_3d, Eigen::MatrixXd &z_3d, int it)
{ // problem with a single wave (zeroth: mean wave)
    int ncv = 0, iv1, dim = parameters.dimension, nz = x_3d.cols();
    if (nz != 1)
    {
        printf("\n\nERROR from write_tecplot_temporal_variables_fourier_zeroth_wave defined only for Fourier meshless problems with zeroth wave x_3d.cols(): %i (expected vaue: 1)\n\n", nz);
        throw bad_exception();
    }
    for (int icv = 0; icv < points.elem_vert_original.size(); icv++)
        if (points.elem_vert_original[icv].size() == dim + 1)
            ncv++; // triangles for 2D
    string output_file = parameters.output_file_prefix + "_tecplot_temporal.plt";
    FILE *file;
    file = fopen(output_file.c_str(), "a");
    // fprintf(file, "# Fields it: %i\n", it);
    fprintf(file, "ZONE T=\"%.16g\"  DATAPACKING=POINT, NODES=%i, ELEMENTS=%i, ZONETYPE=FETRIANGLE\n", it * parameters.dt, points.nv_original, ncv);
    for (int iv0 = 0; iv0 < points.nv_original; iv0++)
    {
        iv1 = points.iv_original_nearest_vert[iv0];
        for (int i1 = 0; i1 < dim; i1++) // write x, y, z
            fprintf(file, "%.16g ", points.xyz_original[dim * iv0 + i1]);
        for (int var = 0; var < variable_pointers.size(); var++) // write variables
            fprintf(file, "%.16g ", (*variable_pointers[var])(iv1, 0));
        fprintf(file, "\n");
    }

    for (int icv = 0; icv < points.elem_vert_original.size(); icv++)
        if (points.elem_vert_original[icv].size() == dim + 1)
        { // triangles for 2D, tetrahedrons for 3D
            for (int i1 = 0; i1 < points.elem_vert_original[icv].size(); i1++)
                fprintf(file, "%i ", points.elem_vert_original[icv][i1] + 1);
            fprintf(file, "\n");
        }
    fclose(file);
}

void write_tecplot_temporal_variables(POINTS &points, PARAMETERS &parameters, vector<string> &variable_names, vector<Eigen::MatrixXd *> &variable_pointers, Eigen::MatrixXd &x_3d, Eigen::MatrixXd &y_3d, Eigen::MatrixXd &z_3d, int it)
{
    if (parameters.dimension != 2)
    {
        printf("\n\nERROR from write_tecplot_temporal_variables defined only for 2D problems parameters.dimension: %i\n\n", parameters.dimension);
        throw bad_exception();
    }
    int ncv = 0, iv1, dim = parameters.dimension, nz = x_3d.cols();
    if (nz == 1)
    { // problem with a single wave (zeroth: mean wave)
        write_tecplot_temporal_variables_fourier_zeroth_wave(points, parameters, variable_names, variable_pointers, x_3d, y_3d, z_3d, it);
        return;
    }
    double dz = z_3d(0, 1) - z_3d(0, 0), z_min = z_3d(0, 0);
    for (int icv = 0; icv < points.elem_vert_original.size(); icv++)
        if (points.elem_vert_original[icv].size() == dim + 1)
            ncv++; // triangles for 2D

    string output_file = parameters.output_file_prefix + "_tecplot_temporal.plt";
    FILE *file;
    file = fopen(output_file.c_str(), "a");
    // fprintf(file, "# Fields it: %i\n", it);
    fprintf(file, "ZONE T=\"%.16g\"  DATAPACKING=POINT, NODES=%i, ELEMENTS=%i, ZONETYPE=FEBRICK\n", it * parameters.dt, points.nv_original * nz, ncv * (nz - 1));

    for (int iz = 0; iz < nz; iz++)
    {
        for (int iv0 = 0; iv0 < points.nv_original; iv0++)
        {
            iv1 = points.iv_original_nearest_vert[iv0];
            fprintf(file, "%.16g %.16g %.16g", points.xyz_original[dim * iv0 + 0], points.xyz_original[dim * iv0 + 1], z_min + (iz * dz));
            for (int var = 0; var < variable_pointers.size(); var++) // write variables
                fprintf(file, " %.16g", (*variable_pointers[var])(iv1, iz));
            fprintf(file, "\n");
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");

    for (int iz = 0; iz < nz - 1; iz++)
        for (int icv = 0; icv < points.elem_vert_original.size(); icv++)
            if (points.elem_vert_original[icv].size() == dim + 1)
            { // triangles for 2D
                // for (int i1 = 0; i1 < points.elem_vert_original[icv].size(); i1++)
                fprintf(file, "%i %i %i %i ", points.elem_vert_original[icv][0] + 1 + (iz * points.nv_original), points.elem_vert_original[icv][1] + 1 + (iz * points.nv_original), points.elem_vert_original[icv][2] + 1 + (iz * points.nv_original), points.elem_vert_original[icv][2] + 1 + (iz * points.nv_original));
                fprintf(file, "%i %i %i %i\n", points.elem_vert_original[icv][0] + 1 + ((iz + 1) * points.nv_original), points.elem_vert_original[icv][1] + 1 + ((iz + 1) * points.nv_original), points.elem_vert_original[icv][2] + 1 + ((iz + 1) * points.nv_original), points.elem_vert_original[icv][2] + 1 + ((iz + 1) * points.nv_original));
            }

    fclose(file);
}

void read_restart_tecplot_file(POINTS &points, PARAMETERS &parameters, string filename, vector<string> &tecplot_variable_names, vector<Eigen::VectorXd *> &variable_pointers)
{
    does_file_exist(filename.c_str(), "Called from read_restart_tecplot_file");
    parameters.restart_tecplot_file = filename;
    FILE *file;
    int itemp, count = 0, dim = parameters.dimension;
    vector<string>::iterator it_string;
    double dtemp;
    double *dtemp_array;
    char ctemp[10000];
    string line_string;
    file = fopen(filename.c_str(), "r");
    fgets(ctemp, 10000, file);
    line_string = ctemp;
    stringstream ss(line_string);
    vector<string> line_string_split{istream_iterator<string>{ss}, istream_iterator<string>{}};
    vector<int> tecplot_variable_names_indices;
    for (int i1 = 0; i1 < tecplot_variable_names.size(); i1++)
    {
        it_string = find(line_string_split.begin(), line_string_split.end(), "\"" + tecplot_variable_names[i1] + "\",");
        if (it_string == line_string_split.end())
        {
            cout << "\n\nERROR from read_restart_tecplot_file unable to find variable: " << tecplot_variable_names[i1] << " in first line of file: " << filename << "\n";
            cout << "First line: " << line_string << "\n\n";
            throw bad_exception();
        }
        tecplot_variable_names_indices.push_back(it_string - line_string_split.begin() - dim); // subtract dim for "X", "Y", "Z"
    }

    // print_to_terminal(tecplot_variable_names_indices, "read_restart_tecplot_file");

    fgets(ctemp, 10000, file);
    line_string = ctemp;
    itemp = line_string.find("NODES");
    if (line_string.find("NODES") == string::npos)
    {
        cout << "\n\nERROR from read_restart_tecplot_file unable to find NODES in second line of file: " << filename << "\n";
        cout << "Second line: " << line_string << "\n\n";
        throw bad_exception();
    }
    line_string.erase(0, itemp);
    string nv_string = regex_replace(line_string, regex("[^0-9]*([0-9]+).*"), string("$1"));
    int nv_tecplot = stoi(nv_string);
    // cout << "read_restart_tecplot_file nv_tecplot: " << nv_tecplot << "\n\n";

    PointCloud<double> xyz_tecplot;
    vector<Eigen::VectorXd> variable_tecplot;
    for (int i1 = 0; i1 < tecplot_variable_names.size(); i1++)
        variable_tecplot.push_back(Eigen::VectorXd::Zero(nv_tecplot));
    xyz_tecplot.pts.resize(nv_tecplot);
    int nvar_tecplot = line_string_split.size() - dim; // subtract dim for "X", "Y", "Z"
    dtemp_array = new double[nvar_tecplot];
    // cout << "read_restart_tecplot_file nvar_tecplot: " << nvar_tecplot << "\n\n";
    for (int iv = 0; iv < nv_tecplot; iv++)
    {
        fscanf(file, "%lf ", &dtemp), xyz_tecplot.pts[iv].x = dtemp; // x co-ordinate
        fscanf(file, "%lf ", &dtemp), xyz_tecplot.pts[iv].y = dtemp; // y co-ordinate
        if (dim == 3)
            fscanf(file, "%lf ", &dtemp), xyz_tecplot.pts[iv].z = dtemp; // z co-ordinate
        else
            xyz_tecplot.pts[iv].z = 0.0;
        for (int i1 = 0; i1 < nvar_tecplot; i1++)
            fscanf(file, "%lf ", &dtemp), dtemp_array[i1] = dtemp; //, cout << dtemp << " ";
        for (int i1 = 0; i1 < tecplot_variable_names.size(); i1++)
            variable_tecplot[i1][iv] = dtemp_array[tecplot_variable_names_indices[i1]];
    }
    fclose(file);

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> nanoflann_kd_tree;
    nanoflann_kd_tree nf_index(3, xyz_tecplot, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    nf_index.buildIndex();
    vector<size_t> nb_vert(1);
    vector<double> nb_dist(1);
    double query_pt[3];
    query_pt[0] = 0.0, query_pt[1] = 0.0, query_pt[2] = 0.0;
    for (int iv = 0; iv < points.nv; iv++)
    {
        for (int i1 = 0; i1 < dim; i1++)
            query_pt[i1] = points.xyz[dim * iv + i1];
        nf_index.knnSearch(&query_pt[0], 1, &nb_vert[0], &nb_dist[0]);
        for (int var = 0; var < variable_pointers.size(); var++)
            (*variable_pointers[var])[iv] = variable_tecplot[var][nb_vert[0]];
    }
    for (int i1 = 0; i1 < tecplot_variable_names.size(); i1++)
        variable_tecplot[i1].resize(0);
}

void read_restart_tecplot_file(POINTS &points, PARAMETERS &parameters, string filename, vector<string> &tecplot_variable_names, vector<Eigen::MatrixXd *> &variable_pointers, Eigen::MatrixXd &x_3d, Eigen::MatrixXd &y_3d, Eigen::MatrixXd &z_3d)
{ // Eigen::MatrixXd type of variable_pointers are used only for 3D periodic problems
    does_file_exist(filename.c_str(), "Called from read_restart_tecplot_file");
    parameters.restart_tecplot_file = filename;
    FILE *file;
    int itemp, count = 0;
    vector<string>::iterator it_string;
    double dtemp;
    double *dtemp_array;
    char ctemp[10000];
    string line_string;
    file = fopen(filename.c_str(), "r");
    fgets(ctemp, 10000, file);
    line_string = ctemp;
    replace(line_string.begin(), line_string.end(), ',', ' ');
    stringstream ss(line_string);
    vector<string> line_string_split{istream_iterator<string>{ss}, istream_iterator<string>{}};
    vector<int> tecplot_variable_names_indices;
    for (int i1 = 0; i1 < tecplot_variable_names.size(); i1++)
    {
        it_string = find(line_string_split.begin(), line_string_split.end(), "\"" + tecplot_variable_names[i1] + "\"");
        if (it_string == line_string_split.end())
        {
            cout << "\n\nERROR from read_restart_tecplot_file unable to find variable: " << tecplot_variable_names[i1] << " in first line of file: " << filename << "\n";
            cout << "First line: " << line_string << "\n\n";
            throw bad_exception();
        }
        tecplot_variable_names_indices.push_back(it_string - line_string_split.begin() - 3); // subtract 3 for "X", "Y", "Z"
    }

    fgets(ctemp, 10000, file);
    line_string = ctemp;
    itemp = line_string.find("NODES");
    if (line_string.find("NODES") == string::npos)
    {
        cout << "\n\nERROR from read_restart_tecplot_file unable to find NODES in second line of file: " << filename << "\n";
        cout << "Second line: " << line_string << "\n\n";
        throw bad_exception();
    }
    line_string.erase(0, itemp);
    string nv_string = regex_replace(line_string, regex("[^0-9]*([0-9]+).*"), string("$1"));
    int nv_tecplot = stoi(nv_string);

    PointCloud<double> xyz_tecplot;
    vector<Eigen::VectorXd> variable_tecplot;
    for (int i1 = 0; i1 < tecplot_variable_names.size(); i1++)
        variable_tecplot.push_back(Eigen::VectorXd::Zero(nv_tecplot));
    int dim = parameters.dimension;
    xyz_tecplot.pts.resize(nv_tecplot);
    int nvar_tecplot = line_string_split.size() - 3; // subtract 3 for "X", "Y", "Z"
    dtemp_array = new double[nvar_tecplot];
    for (int iv = 0; iv < nv_tecplot; iv++)
    {
        fscanf(file, "%lf ", &dtemp), xyz_tecplot.pts[iv].x = dtemp; // x co-ordinate
        fscanf(file, "%lf ", &dtemp), xyz_tecplot.pts[iv].y = dtemp; // y co-ordinate
        fscanf(file, "%lf ", &dtemp), xyz_tecplot.pts[iv].z = dtemp; // z co-ordinate
        for (int i1 = 0; i1 < nvar_tecplot; i1++)
            fscanf(file, "%lf ", &dtemp), dtemp_array[i1] = dtemp;
        for (int i1 = 0; i1 < tecplot_variable_names.size(); i1++)
            variable_tecplot[i1][iv] = dtemp_array[tecplot_variable_names_indices[i1]];
    }
    fclose(file);

    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> nanoflann_kd_tree;
    nanoflann_kd_tree nf_index(3, xyz_tecplot, nanoflann::KDTreeSingleIndexAdaptorParams(10 /* max leaf */));
    nf_index.buildIndex();
    vector<size_t> nb_vert(1);
    vector<double> nb_dist(1);
    double query_pt[3];
    for (int iv = 0; iv < x_3d.rows(); iv++)
        for (int i1 = 0; i1 < x_3d.cols(); i1++)
        {
            query_pt[0] = x_3d(iv, i1), query_pt[1] = y_3d(iv, i1), query_pt[2] = z_3d(iv, i1);
            nf_index.knnSearch(&query_pt[0], 1, &nb_vert[0], &nb_dist[0]);
            for (int var = 0; var < variable_pointers.size(); var++)
                (*variable_pointers[var])(iv, i1) = variable_tecplot[var][nb_vert[0]];
        }
    for (int i1 = 0; i1 < tecplot_variable_names.size(); i1++)
        variable_tecplot[i1].resize(0);
}

void write_tecplot_steady_variables(POINTS &points, PARAMETERS &parameters, vector<string> &variable_names, vector<Eigen::VectorXd *> &variable_pointers)
{
    int ncv = 0, iv1, dim = parameters.dimension;
    for (int icv = 0; icv < points.elem_vert_original.size(); icv++)
        if (points.elem_vert_original[icv].size() == dim + 1)
            ncv++; // triangles for 2D, tetrahedrons for 3D
    FILE *file;
    string output_file = parameters.output_file_prefix + "_tecplot.plt";
    file = fopen(output_file.c_str(), "w");
    if (dim == 2)
        fprintf(file, "VARIABLES=\"x\", \"y\"");
    else
        fprintf(file, "VARIABLES=\"x\", \"y\", \"z\"");
    for (int var = 0; var < variable_names.size(); var++)
        fprintf(file, ", \"%s\"", variable_names[var].c_str());
    fprintf(file, "\n");
    if (dim == 2)
        fprintf(file, "ZONE T=\"%.16g\"  DATAPACKING=POINT, NODES=%i, ELEMENTS=%i, ZONETYPE=FETRIANGLE\n", 1.0, points.nv_original, ncv);
    else
        fprintf(file, "ZONE T=\"%.16g\"  DATAPACKING=POINT, NODES=%i, ELEMENTS=%i, ZONETYPE=FETETRAHEDRON\n", 1.0, points.nv_original, ncv);

    for (int iv0 = 0; iv0 < points.nv_original; iv0++)
    {
        iv1 = points.iv_original_nearest_vert[iv0];
        for (int i1 = 0; i1 < dim; i1++) // write x, y, z
            fprintf(file, "%.16g ", points.xyz_original[dim * iv0 + i1]);
        for (int var = 0; var < variable_pointers.size(); var++) // write variables
            fprintf(file, "%.16g ", (*variable_pointers[var])[iv1]);
        fprintf(file, "\n");
    }

    for (int icv = 0; icv < points.elem_vert_original.size(); icv++)
        if (points.elem_vert_original[icv].size() == dim + 1)
        { // triangles for 2D, tetrahedrons for 3D
            for (int i1 = 0; i1 < points.elem_vert_original[icv].size(); i1++)
                fprintf(file, "%i ", points.elem_vert_original[icv][i1] + 1);
            fprintf(file, "\n");
        }
    fclose(file);
    cout << "\nwrite_tecplot_steady_variables wrote " << output_file << "\n\n";
}

void write_tecplot_steady_variables(POINTS &points, PARAMETERS &parameters, vector<string> &variable_names, vector<Eigen::MatrixXd *> &variable_pointers, Eigen::MatrixXd &x_3d, Eigen::MatrixXd &y_3d, Eigen::MatrixXd &z_3d)
{
    if (parameters.dimension != 2)
    {
        printf("\n\nERROR from write_tecplot_steady_variables defined only for 2D problems parameters.dimension: %i\n\n", parameters.dimension);
        throw bad_exception();
    }
    int ncv = 0, iv1, dim = parameters.dimension, nz = x_3d.cols();
    double dz = z_3d(0, 1) - z_3d(0, 0), z_min = z_3d(0, 0);
    for (int icv = 0; icv < points.elem_vert_original.size(); icv++)
        if (points.elem_vert_original[icv].size() == dim + 1)
            ncv++; // triangles for 2D
    FILE *file;
    string output_file = parameters.output_file_prefix + "_tecplot.plt";
    file = fopen(output_file.c_str(), "w");
    fprintf(file, "VARIABLES=\"x\", \"y\", \"z\"");
    for (int var = 0; var < variable_names.size(); var++)
        fprintf(file, ", \"%s\"", variable_names[var].c_str());
    fprintf(file, "\n");
    fprintf(file, "ZONE T=\"%.16g\"  DATAPACKING=POINT, NODES=%i, ELEMENTS=%i, ZONETYPE=FEBRICK\n", 1.0, points.nv_original * nz, ncv * (nz - 1));

    for (int iz = 0; iz < nz; iz++)
    {
        for (int iv0 = 0; iv0 < points.nv_original; iv0++)
        {
            iv1 = points.iv_original_nearest_vert[iv0];
            fprintf(file, "%.16g %.16g %.16g", points.xyz_original[dim * iv0 + 0], points.xyz_original[dim * iv0 + 1], z_min + (iz * dz));
            for (int var = 0; var < variable_pointers.size(); var++) // write variables
                fprintf(file, " %.16g", (*variable_pointers[var])(iv1, iz));
            fprintf(file, "\n");
        }
        fprintf(file, "\n");
    }
    fprintf(file, "\n");

    for (int iz = 0; iz < nz - 1; iz++)
        for (int icv = 0; icv < points.elem_vert_original.size(); icv++)
            if (points.elem_vert_original[icv].size() == dim + 1)
            { // triangles for 2D
                // for (int i1 = 0; i1 < points.elem_vert_original[icv].size(); i1++)
                fprintf(file, "%i %i %i %i ", points.elem_vert_original[icv][0] + 1 + (iz * points.nv_original), points.elem_vert_original[icv][1] + 1 + (iz * points.nv_original), points.elem_vert_original[icv][2] + 1 + (iz * points.nv_original), points.elem_vert_original[icv][2] + 1 + (iz * points.nv_original));
                fprintf(file, "%i %i %i %i\n", points.elem_vert_original[icv][0] + 1 + ((iz + 1) * points.nv_original), points.elem_vert_original[icv][1] + 1 + ((iz + 1) * points.nv_original), points.elem_vert_original[icv][2] + 1 + ((iz + 1) * points.nv_original), points.elem_vert_original[icv][2] + 1 + ((iz + 1) * points.nv_original));
            }

    fclose(file);
    cout << "\nwrite_tecplot_steady_variables wrote " << output_file << "\n\n";
}

void write_csv_xyz(vector<double> &xyz, PARAMETERS &parameters, const char *file_name)
{
    int dim = parameters.dimension;
    FILE *file;
    file = fopen(file_name, "w");
    if (dim == 2)
    {
        fprintf(file, "no,x,y\n");
        for (int i1 = 0; i1 < (int)(xyz.size() / dim); i1++)
            fprintf(file, "%i,%.16g,%.16g\n", i1, xyz[i1 * dim], xyz[i1 * dim + 1]);
    }
    else
    {
        fprintf(file, "no,x,y,z\n");
        for (int i1 = 0; i1 < (int)(xyz.size() / dim); i1++)
            fprintf(file, "%i,%.16g,%.16g,%.16g\n", i1, xyz[i1 * dim], xyz[i1 * dim + 1], xyz[i1 * dim + 2]);
    }
    fclose(file);
}

void write_csv_temporal_data_init(int size, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "w");
    fprintf(file, "time(s)");
    for (int i1 = 0; i1 < size; i1++)
        fprintf(file, ",%i", i1);
    fprintf(file, "\n");
    fclose(file);
}

void write_csv_temporal_data(Eigen::VectorXd &data, double time, const char *file_name)
{
    FILE *file;
    file = fopen(file_name, "a");
    fprintf(file, "%.16g", time);
    for (int i1 = 0; i1 < data.rows(); i1++)
        fprintf(file, ",%.16g", data[i1]);
    fprintf(file, "\n");
    fclose(file);
}