// Author: Dr. Shantanu Shahane
// compile: time make eccentric_couette_flow_batch_ml
// execute: time ./out
#include "../../header_files/class.hpp"
#include "../../header_files/postprocessing_functions.hpp"
#include "eccentric_couette_flow_batch_ml_func.cpp"
using namespace std;

int main(int argc, char *argv[])
{
    MPI_Init(&argc, &argv);
    clock_t clock_t0 = clock();

    double ecc_ratio = 0.5;            // range: [0.0, 0.5]
    double r_i = 0.6;                  // range: [0.1, 0.6]
    double Re_o = 100;                 // range: [10.0, 100.0]
    double Re_i = -50;                 // range: [-100.0, -10.0] OR [10.0, 100.0]
    double Pr = 2.0;                   // range: [0.1, 2.0]
    double r_o = 1.0, u_theta_o = 1.0; // fixed
    int n_msh = 200;

    string msh_file = "/media/shantanu/Data/All Simulation Results/Meshless_Methods/URAP Eccentric Couette Flow/CAD_Mesh_files/";
    msh_file = msh_file + "ecc_ratio_" + to_string_precision(ecc_ratio, 1) + "_r_i_" + to_string_precision(r_i, 1) + "_n_" + to_string(n_msh) + ".msh";
    PARAMETERS parameters("parameters_file.csv", msh_file);
    double d = r_o - r_i, ecc = ecc_ratio * d;
    parameters.rho = 1.0, parameters.mu = parameters.rho * u_theta_o * d / Re_o;
    double u_theta_i = Re_i * parameters.mu / (parameters.rho * d);
    double nu = parameters.mu / parameters.rho, thermal_diff = nu / Pr;
    POINTS points(parameters);
    CLOUD cloud(points, parameters);
    Eigen::VectorXd p_new = Eigen::VectorXd::Zero(points.nv + 1);
    Eigen::VectorXd u_new = Eigen::VectorXd::Zero(points.nv), v_new = u_new, T_new = u_new;
    Eigen::VectorXd u_old = u_new, v_old = v_new, p_old = p_new, T_old = T_new;
    double x, y, z, r;
    vector<bool> u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, T_dirichlet_flag;
    for (int iv = 0; iv < points.nv; iv++)
    {
        u_dirichlet_flag.push_back(true);  // initialize to dirichlet
        v_dirichlet_flag.push_back(true);  // initialize to dirichlet
        T_dirichlet_flag.push_back(true);  // initialize to dirichlet
        p_dirichlet_flag.push_back(false); // initialize to neumann

        if (points.boundary_flag[iv])
        {
            x = points.xyz[parameters.dimension * iv] - ecc;
            y = points.xyz[parameters.dimension * iv + 1];
            r = sqrt(x * x + y * y);
            if (fabs(r - r_i) < 1E-5) // dirichlet BC: inner boundary
                u_new[iv] = (-u_theta_i * y / r), v_new[iv] = (u_theta_i * x / r), T_new[iv] = 1.0;
            x = points.xyz[parameters.dimension * iv];
            y = points.xyz[parameters.dimension * iv + 1];
            r = sqrt(x * x + y * y);
            if (points.boundary_flag[iv])
                if (fabs(r - r_o) < 1E-5) // dirichlet BC: outer boundary
                    u_new[iv] = (-u_theta_o * y / r), v_new[iv] = (u_theta_o * x / r), T_new[iv] = 0.0;
        }
    }
    u_old = u_new, v_old = v_new, T_old = T_new;
    parameters.calc_dt(points.grad_x_matrix_EIGEN, points.grad_y_matrix_EIGEN, points.grad_z_matrix_EIGEN, points.laplacian_matrix_EIGEN, max_abs(u_new), max_abs(v_new), 0.0, parameters.mu / parameters.rho);

    double iterative_tolerance = 1E-4;
    int precond_freq_it = 10000, n_outer_iter = 5;
    SEMI_IMPLICIT_SPLIT_SOLVER semi_implicit_split_solver(points, cloud, parameters, u_dirichlet_flag, v_dirichlet_flag, p_dirichlet_flag, n_outer_iter, iterative_tolerance, precond_freq_it);
    vector<int> n_outer_iter_log;
    vector<double> iterative_l1_err_log, iterative_max_err_log, total_steady_l1_err_log;
    double total_steady_l1_err = 1000.0;
    clock_t clock_t1 = clock(), clock_t2 = clock();
    cout << "\nTime marching started\n\n";
    for (int it = 0; it < parameters.nt; it++)
    {
        semi_implicit_split_solver.iterative_tolerance = total_steady_l1_err * parameters.dt;
        if (semi_implicit_split_solver.iterative_tolerance > iterative_tolerance)
            semi_implicit_split_solver.iterative_tolerance = iterative_tolerance;
        semi_implicit_split_solver.single_timestep_2d(points, cloud, parameters, u_new, v_new, p_new, u_old, v_old, p_old, it, n_outer_iter_log, iterative_l1_err_log, iterative_max_err_log);
        total_steady_l1_err = (u_new - u_old).lpNorm<1>();
        total_steady_l1_err += (v_new - v_old).lpNorm<1>();
        total_steady_l1_err = total_steady_l1_err / (parameters.dimension * parameters.dt * u_new.size());
        // total_steady_l1_err_log.push_back(total_steady_l1_err);

        double runtime = ((double)(clock() - clock_t2)) / CLOCKS_PER_SEC;
        if (runtime > 1.0 || it == 0 || it == 1 || it == parameters.nt - 1 || total_steady_l1_err < parameters.steady_tolerance) //|| true
        {
            printf("    pressure regularization alpha: %g\n", p_new[points.nv]);
            if (iterative_max_err_log.size() > 0)
                printf("    Outer iterations: l1_error: %g, max_error: %g, iter_num: %i, tolerance: %g\n", iterative_l1_err_log[it], iterative_max_err_log[it], n_outer_iter_log[it], semi_implicit_split_solver.iterative_tolerance);
            printf("    total steady state l1_error: %g, steady_tolerance: %g\n", total_steady_l1_err, parameters.steady_tolerance);
            printf("    Completed it: %i of nt: %i, dt: %g, in CPU time: %g seconds\n\n", it, parameters.nt, parameters.dt, ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC);
            clock_t2 = clock();
        }
        parameters.nt_actual = it + 1;
        u_old = u_new, v_old = v_new, p_old = p_new;
        if (total_steady_l1_err < parameters.steady_tolerance && it > 1)
            break;
    }

    IMPLICIT_SCALAR_TRANSPORT_SOLVER implicit_scalar_transport_solver(points, cloud, parameters, T_dirichlet_flag, precond_freq_it, 0.0, 1.0, -thermal_diff, false);
    implicit_scalar_transport_solver.single_timestep_2d(points, cloud, parameters, T_new, T_old, u_new, v_new, 0);

    parameters.solve_timer = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    printf("Time marching ended; factoring_timer: %g, solve_timer:%g seconds\n\n", parameters.factoring_timer, parameters.solve_timer);
    parameters.total_timer = ((double)(clock() - clock_t0)) / CLOCKS_PER_SEC;

    TORQUE_HEAT torque_heat(points, parameters, r_i, r_o, ecc, u_theta_i / r_i, u_theta_o / r_o, u_new, v_new, p_new, T_new, 360);
    if (fabs(ecc_ratio) < 1E-5)
        CONC_ANALYTICAL conc_analytical(points, parameters, torque_heat, r_i, r_o, u_theta_i / r_i, u_theta_o / r_o, u_new, v_new, T_new);
    write_csv_gmsh(points, parameters, u_new, v_new, p_new, T_new);

    write_simulation_details(points, cloud, parameters); // write_iteration_details(parameters);
    vector<string> variable_names{"u_new", "v_new", "p_new", "T_new"};
    vector<Eigen::VectorXd *> variable_pointers{&u_new, &v_new, &p_new, &T_new};
    write_tecplot_steady_variables(points, parameters, variable_names, variable_pointers);
}