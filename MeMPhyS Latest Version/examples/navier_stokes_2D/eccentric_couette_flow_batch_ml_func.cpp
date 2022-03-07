// Author: Dr. Shantanu Shahane
// compile together with eccentric_couette_flow_batch_ml.cpp
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
#include "../../header_files/coefficient_computations.hpp"
using namespace std;

class INTERPOLATE
{ // This interpolation adds overshoots and undershoots at the small gap region
public:
    int n_xi, n_theta;
    Eigen::VectorXd theta_interp, xi_interp, u_interp, v_interp, p_interp, T_interp;
    vector<double> xy_interp;
    double delta_theta, delta_xi, r_i, r_o, ecc, omega_i, omega_o;
    Eigen::SparseMatrix<double, Eigen::RowMajor> xy_interp_mat;

    INTERPOLATE(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p, Eigen::VectorXd &T, double r_i1, double r_o1, double ecc1, double omega_i1, double omega_o1, int n_xi1, int n_theta1);
};

INTERPOLATE::INTERPOLATE(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p, Eigen::VectorXd &T, double r_i1, double r_o1, double ecc1, double omega_i1, double omega_o1, int n_xi1, int n_theta1)
{
    n_xi = n_xi1, n_theta = n_theta1;
    r_i = r_i1, r_o = r_o1, ecc = ecc1, omega_i = omega_i1, omega_o = omega_o1;
    delta_xi = 1.0 / (n_xi - 1.0);
    delta_theta = (360.0 / n_theta) * (M_PI / 180.0);
    theta_interp = Eigen::VectorXd::Zero(n_xi * n_theta), xi_interp = theta_interp;
    double l, x, y;
    for (int i_xi = 0; i_xi < n_xi; i_xi++)
        for (int i_th = 0; i_th < n_theta; i_th++)
        {
            xi_interp[i_xi * n_theta + i_th] = i_xi * delta_xi;
            theta_interp[i_xi * n_theta + i_th] = i_th * delta_theta;
        }
    for (int i1 = 0; i1 < xi_interp.size(); i1++)
    {
        l = sqrt(pow(r_o, 2) - pow(ecc * sin(theta_interp[i1]), 2) - ecc * cos(theta_interp[i1]));
        x = ecc + (r_i + ((l - r_i) * xi_interp[i1])) * cos(theta_interp[i1]);
        y = (r_i + ((l - r_i) * xi_interp[i1])) * sin(theta_interp[i1]);
        xy_interp.push_back(x), xy_interp.push_back(y);
    }
    xy_interp_mat = calc_interp_matrix(xy_interp, points, parameters);
    u_interp = xy_interp_mat * u, v_interp = xy_interp_mat * v, p_interp = xy_interp_mat * p.head(points.nv), T_interp = xy_interp_mat * T;

    string filename = parameters.output_file_prefix + "_fields_nxi_" + to_string(n_xi) + "_ntheta_" + to_string(n_theta) + ".csv";
    FILE *file;
    file = fopen(filename.c_str(), "w");
    fprintf(file, "u,v,p,T");
    for (int i1 = 0; i1 < u_interp.size(); i1++)
        fprintf(file, "\n%g,%g,%g,%g", u_interp[i1], v_interp[i1], p_interp[i1], T_interp[i1]);
    fclose(file);
}

void write_csv_gmsh(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p, Eigen::VectorXd &T)
{
    int ncv = 0, iv1, dim = parameters.dimension;
    for (int icv = 0; icv < points.elem_vert_original.size(); icv++)
        if (points.elem_vert_original[icv].size() == 3) // triangles
            ncv++;
    FILE *file;
    string filename = parameters.output_file_prefix + "_fields_gmsh.csv";
    file = fopen(filename.c_str(), "w");
    fprintf(file, "x,y,u,v,p,T");
    for (int iv0 = 0; iv0 < points.nv_original; iv0++)
    {
        iv1 = points.iv_original_nearest_vert[iv0];
        fprintf(file, "\n%g,%g,%g,%g,%g,%g", points.xyz_original[dim * iv0], points.xyz_original[dim * iv0 + 1], u[iv1], v[iv1], p[iv1], T[iv1]);
    }
    fclose(file);

    filename = parameters.output_file_prefix + "_triangles_gmsh.csv";
    file = fopen(filename.c_str(), "w");
    fprintf(file, "v1,v2,v3");
    for (int icv = 0; icv < points.elem_vert_original.size(); icv++)
        if (points.elem_vert_original[icv].size() == 3) // triangles
            fprintf(file, "\n%i,%i,%i", points.elem_vert_original[icv][0], points.elem_vert_original[icv][1], points.elem_vert_original[icv][2]);
    fclose(file);
}

class TORQUE_HEAT
{
public:
    double delta_theta, r_i, r_o, ecc, omega_i, omega_o, torque_charac, torque_i, torque_non_dim_i, torque_o, torque_non_dim_o, heat_transfer_i, heat_transfer_non_dim_i, heat_transfer_o, heat_transfer_non_dim_o, heat_transfer_charac;
    double k_thermal = 1.0; // dummy value: cancels in heat_transfer_non_dim
    int n_theta;
    vector<double> xy_interp_theta_i, xy_interp_theta_o;
    Eigen::SparseMatrix<double, Eigen::RowMajor> xy_interp_theta_mat_i, xy_interp_theta_mat_o;
    Eigen::VectorXd theta, sigma_xx, sigma_xy, sigma_yy, sigma_xx_interp_theta, sigma_xy_interp_theta, sigma_yy_interp_theta, grad_x_T, grad_y_T, sigma_rt_interp_theta, grad_x_T_interp_theta, grad_y_T_interp_theta, local_heat_flux_i, local_heat_flux_o;

    TORQUE_HEAT(POINTS &points, PARAMETERS &parameters, double r_i1, double r_o1, double ecc1, double omega_i1, double omega_o1, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p, Eigen::VectorXd &T, int n_theta1);
    void calc_torque(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p);
    void calc_heat(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &T);
    void write_csv(POINTS &points, PARAMETERS &parameters);
};

TORQUE_HEAT::TORQUE_HEAT(POINTS &points, PARAMETERS &parameters, double r_i1, double r_o1, double ecc1, double omega_i1, double omega_o1, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p, Eigen::VectorXd &T, int n_theta1)
{
    r_i = r_i1, r_o = r_o1, ecc = ecc1, omega_i = omega_i1, omega_o = omega_o1, n_theta = n_theta1;
    theta = Eigen::VectorXd::Zero(n_theta);
    delta_theta = (360.0 / n_theta) * (M_PI / 180.0);
    for (int i1 = 0; i1 < n_theta; i1++)
    {
        theta[i1] = i1 * delta_theta;
        xy_interp_theta_i.push_back(ecc + (r_i * cos(theta[i1]))), xy_interp_theta_i.push_back(r_i * sin(theta[i1]));
        xy_interp_theta_o.push_back(r_o * cos(theta[i1])), xy_interp_theta_o.push_back(r_o * sin(theta[i1]));
    }
    xy_interp_theta_mat_i = calc_interp_matrix(xy_interp_theta_i, points, parameters);
    xy_interp_theta_mat_o = calc_interp_matrix(xy_interp_theta_o, points, parameters);
    sigma_xx = Eigen::VectorXd::Zero(points.nv), sigma_xy = sigma_xx, sigma_yy = sigma_xx, grad_x_T = sigma_xx, grad_y_T = sigma_xx;
    sigma_rt_interp_theta = Eigen::VectorXd::Zero(n_theta), sigma_xx_interp_theta = sigma_rt_interp_theta, sigma_xy_interp_theta = sigma_rt_interp_theta, sigma_yy_interp_theta = sigma_rt_interp_theta;
    local_heat_flux_i = sigma_rt_interp_theta, local_heat_flux_o = sigma_rt_interp_theta, grad_x_T_interp_theta = sigma_rt_interp_theta, grad_y_T_interp_theta = sigma_rt_interp_theta;

    calc_torque(points, parameters, u, v, p);
    calc_heat(points, parameters, T);
    printf("TORQUE_HEAT::TORQUE_HEAT Inner Cylinder torque_charac: %g, torque_num: %g, torque_num_non_dim: %g\n", torque_charac, torque_i, torque_non_dim_i);
    printf("TORQUE_HEAT::TORQUE_HEAT Outer Cylinder torque_charac: %g, torque_num: %g, torque_num_non_dim: %g\n", torque_charac, torque_o, torque_non_dim_o);
    printf("TORQUE_HEAT::TORQUE_HEAT Inner Cylinder heat_transfer_charac: %g, heat_transfer_num: %g, heat_transfer_num_non_dim: %g\n", heat_transfer_charac, heat_transfer_i, heat_transfer_non_dim_i);
    printf("TORQUE_HEAT::TORQUE_HEAT Outer Cylinder heat_transfer_charac: %g, heat_transfer_num: %g, heat_transfer_num_non_dim: %g\n\n", heat_transfer_charac, heat_transfer_o, heat_transfer_non_dim_o);
    double ht_diff = 100.0 * fabs(heat_transfer_non_dim_i - heat_transfer_non_dim_o) / (0.5 * (fabs(heat_transfer_non_dim_i) + fabs(heat_transfer_non_dim_o)));
    // double torque_diff = 100.0 * fabs(torque_non_dim_i - torque_non_dim_o) / (0.5 * (fabs(torque_non_dim_i) + fabs(torque_non_dim_o)));
    printf("TORQUE_HEAT::TORQUE_HEAT Inner-Outer Cylinder Percent differences:  heat_transfer: %g\n\n", ht_diff);
    write_csv(points, parameters);
}

void TORQUE_HEAT::write_csv(POINTS &points, PARAMETERS &parameters)
{
    FILE *file;
    file = fopen((parameters.output_file_prefix + "_local_heat_flux.csv").c_str(), "w");
    fprintf(file, "theta,inner_cylinder,outer_cylinder");
    for (int i1 = 0; i1 < theta.size(); i1++)
        fprintf(file, "\n%g,%g,%g", theta[i1] * 180.0 / M_PI, local_heat_flux_i[i1], local_heat_flux_o[i1]);
    fclose(file);

    file = fopen((parameters.output_file_prefix + "_torque_heat.csv").c_str(), "w");
    fprintf(file, "torque_charac,torque_i,torque_non_dim_i,torque_o,torque_non_dim_o,heat_transfer_charac,heat_transfer_i,heat_transfer_non_dim_i,heat_transfer_o,heat_transfer_non_dim_o\n");
    fprintf(file, "%g,%g,%g,%g,%g,%g,%g,%g,%g,%g\n", torque_charac, torque_i, torque_non_dim_i, torque_o, torque_non_dim_o, heat_transfer_charac, heat_transfer_i, heat_transfer_non_dim_i, heat_transfer_o, heat_transfer_non_dim_o);
    fclose(file);
}

void TORQUE_HEAT::calc_heat(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &T)
{
    int dim = parameters.dimension;
    double del_T = 1.0;
    heat_transfer_charac = k_thermal * del_T; // per unit length: W/m
    grad_x_T = points.grad_x_matrix_EIGEN * T;
    grad_y_T = points.grad_y_matrix_EIGEN * T;
    double delta_theta = (360.0 / n_theta) * (M_PI / 180.0), x, y, r;

    grad_x_T_interp_theta = xy_interp_theta_mat_i * grad_x_T;
    grad_y_T_interp_theta = xy_interp_theta_mat_i * grad_y_T;
    for (int i1 = 0; i1 < grad_x_T_interp_theta.size(); i1++)
    {
        x = xy_interp_theta_i[dim * i1] - ecc, y = xy_interp_theta_i[dim * i1 + 1], r = sqrt(x * x + y * y);
        local_heat_flux_i[i1] = k_thermal * ((grad_x_T_interp_theta[i1] * x / r) + (grad_y_T_interp_theta[i1] * y / r));
    }
    heat_transfer_i = 0.0;
    for (int i1 = 0; i1 < local_heat_flux_i.size(); i1++)
        if (i1 % 2 == 0) // simpson's rule: multiplied by 2
            heat_transfer_i = heat_transfer_i + (2.0 * local_heat_flux_i[i1]);
        else // simpson's rule: multiplied by 4
            heat_transfer_i = heat_transfer_i + (4.0 * local_heat_flux_i[i1]);
    heat_transfer_i = fabs(heat_transfer_i) * r_i * delta_theta / 3.0;
    heat_transfer_non_dim_i = heat_transfer_i / heat_transfer_charac;

    grad_x_T_interp_theta = xy_interp_theta_mat_o * grad_x_T;
    grad_y_T_interp_theta = xy_interp_theta_mat_o * grad_y_T;
    for (int i1 = 0; i1 < grad_x_T_interp_theta.size(); i1++)
    {
        x = xy_interp_theta_o[dim * i1], y = xy_interp_theta_o[dim * i1 + 1], r = sqrt(x * x + y * y);
        local_heat_flux_o[i1] = k_thermal * ((grad_x_T_interp_theta[i1] * x / r) + (grad_y_T_interp_theta[i1] * y / r));
    }
    heat_transfer_o = 0.0;
    for (int i1 = 0; i1 < local_heat_flux_o.size(); i1++)
        if (i1 % 2 == 0) // simpson's rule: multiplied by 2
            heat_transfer_o = heat_transfer_o + (2.0 * local_heat_flux_o[i1]);
        else // simpson's rule: multiplied by 4
            heat_transfer_o = heat_transfer_o + (4.0 * local_heat_flux_o[i1]);
    heat_transfer_o = fabs(heat_transfer_o) * r_o * delta_theta / 3.0;
    heat_transfer_non_dim_o = heat_transfer_o / heat_transfer_charac;
}

void TORQUE_HEAT::calc_torque(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &p)
{
    double u_theta_i = r_i * omega_i, u_theta_o = r_o * omega_o;
    torque_charac = parameters.mu * (fabs(r_i * u_theta_i) + fabs(r_o * u_theta_o));
    if (n_theta % 2 != 0)
    {
        cout << "\n\nError from TORQUE_HEAT::calc_torque n_theta should be even, n_theta: " << n_theta << "\n\n";
        throw bad_exception();
    }
    sigma_xx = -p.head(points.nv) + 2 * parameters.mu * (points.grad_x_matrix_EIGEN * u);
    sigma_yy = -p.head(points.nv) + 2 * parameters.mu * (points.grad_y_matrix_EIGEN * v);
    sigma_xy = parameters.mu * ((points.grad_y_matrix_EIGEN * u) + (points.grad_x_matrix_EIGEN * v));
    double x, y;
    int dim = parameters.dimension;

    torque_i = 0.0;
    sigma_xx_interp_theta = xy_interp_theta_mat_i * sigma_xx;
    sigma_xy_interp_theta = xy_interp_theta_mat_i * sigma_xy;
    sigma_yy_interp_theta = xy_interp_theta_mat_i * sigma_yy;
    for (int iv = 0; iv < sigma_xx_interp_theta.size(); iv++)
    {
        x = xy_interp_theta_i[dim * iv] - ecc, y = xy_interp_theta_i[dim * iv + 1];
        sigma_rt_interp_theta[iv] = (sigma_yy_interp_theta[iv] - sigma_xx_interp_theta[iv]) * (x * y) / (x * x + y * y) + sigma_xy_interp_theta[iv] * (x * x - y * y) / (x * x + y * y);
    }
    for (int i1 = 0; i1 < sigma_rt_interp_theta.size(); i1++)
        if (i1 % 2 == 0) // simpson's rule: multiplied by 2
            torque_i = torque_i + (2.0 * sigma_rt_interp_theta[i1]);
        else // simpson's rule: multiplied by 4
            torque_i = torque_i + (4.0 * sigma_rt_interp_theta[i1]);
    torque_i = fabs(torque_i) * r_i * r_i * delta_theta / 3.0;
    torque_non_dim_i = torque_i / torque_charac;

    torque_o = 0.0;
    sigma_xx_interp_theta = xy_interp_theta_mat_o * sigma_xx;
    sigma_xy_interp_theta = xy_interp_theta_mat_o * sigma_xy;
    sigma_yy_interp_theta = xy_interp_theta_mat_o * sigma_yy;
    for (int iv = 0; iv < sigma_xx_interp_theta.size(); iv++)
    {
        x = xy_interp_theta_o[dim * iv], y = xy_interp_theta_o[dim * iv + 1];
        sigma_rt_interp_theta[iv] = (sigma_yy_interp_theta[iv] - sigma_xx_interp_theta[iv]) * (x * y) / (x * x + y * y) + sigma_xy_interp_theta[iv] * (x * x - y * y) / (x * x + y * y);
    }
    for (int i1 = 0; i1 < sigma_rt_interp_theta.size(); i1++)
        if (i1 % 2 == 0) // simpson's rule: multiplied by 2
            torque_o = torque_o + (2.0 * sigma_rt_interp_theta[i1]);
        else // simpson's rule: multiplied by 4
            torque_o = torque_o + (4.0 * sigma_rt_interp_theta[i1]);
    torque_o = fabs(torque_o) * r_o * r_o * delta_theta / 3.0;
    torque_non_dim_o = torque_o / torque_charac;
}

class CONC_ANALYTICAL
{
public:
    double delta_theta, r_i, r_o, omega_i, omega_o, torque_ana, torque_ana_non_dim, heat_transfer_ana, heat_transfer_ana_non_dim, torque_charac, heat_transfer_charac;
    double k_thermal = 1.0; // dummy value: cancels in heat_transfer_ana_non_dim
    Eigen::VectorXd u_ana, v_ana, T_ana, u_err, v_err, T_err;

    CONC_ANALYTICAL(POINTS &points, PARAMETERS &parameters, TORQUE_HEAT &torque_heat, double r_i1, double r_o1, double omega_i1, double omega_o1, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &T);
    void calc_ana_vel(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v);
    void calc_ana_T(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &T);
};

CONC_ANALYTICAL::CONC_ANALYTICAL(POINTS &points, PARAMETERS &parameters, TORQUE_HEAT &torque_heat, double r_i1, double r_o1, double omega_i1, double omega_o1, Eigen::VectorXd &u, Eigen::VectorXd &v, Eigen::VectorXd &T)
{
    r_i = r_i1, r_o = r_o1, omega_i = omega_i1, omega_o = omega_o1;
    calc_ana_vel(points, parameters, u, v);
    calc_ana_T(points, parameters, T);
    printf("\n\nCONC_ANALYTICAL::CONC_ANALYTICAL Errors u: %g, v: %g, T: %g\n", u_err.lpNorm<1>() / u_err.size(), v_err.lpNorm<1>() / v_err.size(), T_err.lpNorm<1>() / T_err.size());
    // printf("CONC_ANALYTICAL::CONC_ANALYTICAL torque_charac: %g, torque_ana: %g, torque_ana_non_dim: %g\n", torque_charac, torque_ana, torque_ana_non_dim);
    // printf("CONC_ANALYTICAL::CONC_ANALYTICAL heat_transfer_charac: %g, heat_transfer_ana: %g, heat_transfer_ana_non_dim: %g\n\n", heat_transfer_charac, heat_transfer_ana, heat_transfer_ana_non_dim);
    double ht_err_i = 100.0 * fabs(torque_heat.heat_transfer_non_dim_i - heat_transfer_ana_non_dim) / heat_transfer_ana_non_dim;
    double ht_err_o = 100.0 * fabs(torque_heat.heat_transfer_non_dim_o - heat_transfer_ana_non_dim) / heat_transfer_ana_non_dim;
    double torque_err_i = 100.0 * fabs(torque_heat.torque_non_dim_i - torque_ana_non_dim) / torque_ana_non_dim;
    double torque_err_o = 100.0 * fabs(torque_heat.torque_non_dim_o - torque_ana_non_dim) / torque_ana_non_dim;
    printf("CONC_ANALYTICAL::CONC_ANALYTICAL Percent Errors: heat_transfer_i: %g, heat_transfer_o: %g, torque_i: %g, torque_o: %g\n\n", ht_err_i, ht_err_o, torque_err_i, torque_err_o);
    vector<string> variable_names{"u_num", "v_num", "T_num", "u_ana", "v_ana", "T_ana", "u_err", "v_err", "T_err"};
    vector<Eigen::VectorXd *> variable_pointers{&u, &v, &T, &u_ana, &v_ana, &T_ana, &u_err, &v_err, &T_err};
    string temp_name = parameters.output_file_prefix;
    parameters.output_file_prefix = parameters.output_file_prefix + "_ana";
    write_tecplot_steady_variables(points, parameters, variable_names, variable_pointers);
    parameters.output_file_prefix = temp_name;
}

void CONC_ANALYTICAL::calc_ana_T(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &T)
{
    double C1 = -1.0 / log(r_o / r_i), x, y, r;
    double C2 = log(r_o) / log(r_o / r_i);
    int dim = parameters.dimension;
    T_ana = Eigen::VectorXd::Zero(points.nv);
    for (int iv = 0; iv < points.nv; iv++)
    {
        x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1], r = sqrt(x * x + y * y);
        T_ana[iv] = (C1 * log(r)) + C2;
    }
    T_err = (T_ana - T).cwiseAbs();
    double del_T = 1.0;
    heat_transfer_ana = fabs(2.0 * M_PI * k_thermal * C1); // per unit length: W/m
    heat_transfer_charac = k_thermal * del_T;              // per unit length: W/m
    heat_transfer_ana_non_dim = heat_transfer_ana / heat_transfer_charac;
    // double l_charac = r_o - r_i, del_T = 1.0;
    // Nusselt_ana_i = fabs(l_charac * C1 / (del_T * r_i));
    // Nusselt_ana_o = fabs(l_charac * C1 / (del_T * r_o));
}

void CONC_ANALYTICAL::calc_ana_vel(POINTS &points, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v)
{
    double eta = r_i / r_o, x, y, r, u_theta;
    double C1 = -((omega_i * eta * eta) - omega_o) / (1.0 - (eta * eta));
    double C2 = r_i * r_i * (omega_i - omega_o) / (1.0 - (eta * eta));
    int dim = parameters.dimension;
    u_ana = Eigen::VectorXd::Zero(points.nv), v_ana = u_ana;
    for (int iv = 0; iv < points.nv; iv++)
    {
        x = points.xyz[dim * iv], y = points.xyz[dim * iv + 1], r = sqrt(x * x + y * y);
        u_theta = (C1 * r) + (C2 / r);
        u_ana[iv] = (-u_theta * y / r), v_ana[iv] = (u_theta * x / r);
    }
    u_err = (u_ana - u).cwiseAbs(), v_err = (v_ana - v).cwiseAbs();
    torque_ana = fabs(4.0 * M_PI * parameters.mu * C2);
    double u_theta_i = r_i * omega_i, u_theta_o = r_o * omega_o;
    torque_charac = parameters.mu * (fabs(u_theta_i * r_i) + fabs(u_theta_o * r_o));
    torque_ana_non_dim = torque_ana / torque_charac;
}