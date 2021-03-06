//Author: Dr. Shantanu Shahane
#include "coefficient_computations.hpp"

void shifting_scaling(double *xyz_interp, vector<double> &vert, double *scale, int dim)
{
    int nv = (int)(vert.size() / dim);
    double min_val[3], max_val[3]; //min and max of X, Y and Z co-ordinates
    for (int i = 0; i < dim; i++)
    { //initialize as first node value
        min_val[i] = vert[i];
        max_val[i] = vert[i];
    }
    for (int iv = 0; iv < nv; iv++)
    { //find max and min
        for (int i = 0; i < dim; i++)
        { //update
            if (min_val[i] > vert[dim * iv + i])
                min_val[i] = vert[dim * iv + i];
            if (max_val[i] < vert[dim * iv + i])
                max_val[i] = vert[dim * iv + i];
        }
    }
    for (int i = 0; i < dim; i++)
        scale[i] = max_val[i] - min_val[i];
    for (int iv = 0; iv < nv; iv++) //shift to [0, 1] by subtracting min_val and dividing by scale
        for (int i = 0; i < dim; i++)
            vert[dim * iv + i] = (vert[dim * iv + i] - min_val[i]) / scale[i];

    for (int i = 0; i < dim; i++)
        xyz_interp[i] = (xyz_interp[i] - min_val[i]) / scale[i];
}

void shifting_scaling(vector<double> &vert, double *scale, int dim)
{ //shift and scale to range [0, 1]; scale: array of size 3 with values of scaling in X, Y and Z
    int nv = (int)(vert.size() / dim);
    double min_val[3], max_val[3]; //min and max of X, Y and Z co-ordinates
    for (int i = 0; i < dim; i++)
    { //initialize as first node value
        min_val[i] = vert[i];
        max_val[i] = vert[i];
    }
    for (int iv = 0; iv < nv; iv++)
    { //find max and min
        for (int i = 0; i < dim; i++)
        { //update
            if (min_val[i] > vert[dim * iv + i])
                min_val[i] = vert[dim * iv + i];
            if (max_val[i] < vert[dim * iv + i])
                max_val[i] = vert[dim * iv + i];
        }
    }
    for (int i = 0; i < dim; i++)
        scale[i] = max_val[i] - min_val[i];
    for (int iv = 0; iv < nv; iv++) //shift to [0, 1] by subtracting min_val and dividing by scale
        for (int i = 0; i < dim; i++)
            vert[dim * iv + i] = (vert[dim * iv + i] - min_val[i]) / scale[i];

    // cout << "\nshifting_scaling min_val: ";
    // for (int i = 0; i < dim; i++)
    //     cout << min_val[i] << ", ";
    // cout << "\nshifting_scaling max_val: ";
    // for (int i = 0; i < dim; i++)
    //     cout << max_val[i] << ", ";
    // cout << "\nshifting_scaling scale: ";
    // for (int i = 0; i < dim; i++)
    //     cout << scale[i] << ", ";
    // cout << "\n\n";
}

void calc_PHS_RBF_grad_laplace_single_vert_A(vector<double> &vert, PARAMETERS &parameters, Eigen::MatrixXd &A, double *scale)
{ //vert: vertex co-ordinates of this group only (will give global coefficients on that group)
    int dim = parameters.dimension, num_poly_terms = parameters.num_poly_terms;
    int nv = (int)(vert.size() / dim), phs_deg = parameters.phs_deg;
    double r_square, r_phs, d_temp;
    for (int ir = 0; ir < nv; ir++)
    { //top left band of A
        for (int ic = ir + 1; ic < nv; ic++)
        { //due to symmetry, need to compute only strictly upper triangular part (diagonal is zero)
            r_square = 0.0;
            for (int i = 0; i < dim; i++)
            {
                d_temp = vert[dim * ir + i] - vert[dim * ic + i];
                r_square = r_square + (d_temp * d_temp);
            }
            r_phs = pow(r_square, (double)(phs_deg / 2.0)); //instead of computing sqrt separately, here, exponent is divided by 2
            A(ir, ic) = r_phs;                              //symmetry (strict upper triangular)
            A(ic, ir) = r_phs;                              //symmetry (strict lower triangular)
        }
    }
    for (int ir = 0; ir < nv; ir++)
    { //top right and lower left band of A
        for (int j = 0; j < num_poly_terms; j++)
        {
            d_temp = pow(vert[dim * ir], parameters.polynomial_term_exponents(j, 0));
            for (int i = 1; i < dim; i++)
            {
                d_temp = d_temp * pow(vert[dim * ir + i], parameters.polynomial_term_exponents(j, i));
            }
            A(ir, j + nv) = d_temp; //symmetry (upper triangular)
            A(j + nv, ir) = d_temp; //symmetry (lower triangular)
        }
    }
    // Eigen::JacobiSVD<Eigen::MatrixXd> svd(A); //extremely slow: do not use
    // double cond_num = svd.singularValues()(0) / svd.singularValues()(svd.singularValues().size() - 1);
    // printf("\ncalc_PHS_RBF_grad_laplace_single_vert_A: cond number = %e\n\n", cond_num);
}

void calc_PHS_RBF_grad_laplace_single_vert_grad_x_rhs(vector<double> &vert, PARAMETERS &parameters, Eigen::MatrixXd &rhs, double *scale, vector<int> &central_vert_list)
{ //vert: vertex co-ordinates of this group only (will give global coefficients on that group)
    int dim = parameters.dimension, num_poly_terms = parameters.num_poly_terms;
    int nv = (int)(vert.size() / dim), phs_deg = parameters.phs_deg, ic;
    double r_square, r_phs, d_temp;
    for (int i1 = 0; i1 < central_vert_list.size(); i1++)
    { //rhs for grad_x
        for (int ir = 0; ir < nv; ir++)
        { //derivative wrt X operator on the RBFs: r^phs_deg
            r_square = 0.0;
            ic = central_vert_list[i1];
            for (int i = 0; i < dim; i++)
            {
                d_temp = vert[dim * ir + i] - vert[dim * ic + i];
                r_square = r_square + (d_temp * d_temp);
            }
            r_phs = -phs_deg * (vert[dim * ir] - vert[dim * ic]) * pow(r_square, (double)((phs_deg - 2.0) / 2.0)); //instead of computing sqrt separately, here, exponent is divided by 2
            rhs(ir, ic) = r_phs / scale[0];
        }

        for (int ir = nv; ir < nv + num_poly_terms; ir++)
        { //derivative wrt X operator on the appended polynomials
            if (parameters.polynomial_term_exponents(ir - nv, 0) > 0)
            { //if parameters.polynomial_term_exponents(ir, 0) is zero, derivative wrt X is zero
                d_temp = parameters.polynomial_term_exponents(ir - nv, 0) * pow(vert[dim * ic], parameters.polynomial_term_exponents(ir - nv, 0) - 1.0);
                d_temp = d_temp * pow(vert[dim * ic + 1], parameters.polynomial_term_exponents(ir - nv, 1));
                if (dim == 3)
                {
                    d_temp = d_temp * pow(vert[dim * ic + 2], parameters.polynomial_term_exponents(ir - nv, 2));
                }
                rhs(ir, ic) = d_temp / scale[0];
            }
        }
    }
}

void calc_PHS_RBF_grad_laplace_single_vert_grad_y_rhs(vector<double> &vert, PARAMETERS &parameters, Eigen::MatrixXd &rhs, double *scale, vector<int> &central_vert_list)
{ //vert: vertex co-ordinates of this group only (will give global coefficients on that group)
    int dim = parameters.dimension, num_poly_terms = parameters.num_poly_terms;
    int nv = (int)(vert.size() / dim), phs_deg = parameters.phs_deg, ic;
    double r_square, r_phs, d_temp;
    for (int i1 = 0; i1 < central_vert_list.size(); i1++)
    { //rhs for grad_y
        for (int ir = 0; ir < nv; ir++)
        { //derivative wrt Y operator on the RBFs: r^phs_deg
            r_square = 0.0;
            ic = central_vert_list[i1];
            for (int i = 0; i < dim; i++)
            {
                d_temp = vert[dim * ir + i] - vert[dim * ic + i];
                r_square = r_square + (d_temp * d_temp);
            }
            r_phs = -phs_deg * (vert[dim * ir + 1] - vert[dim * ic + 1]) * pow(r_square, (double)((phs_deg - 2.0) / 2.0)); //instead of computing sqrt separately, here, exponent is divided by 2
            rhs(ir, central_vert_list.size() + ic) = r_phs / scale[1];
        }

        for (int ir = nv; ir < nv + num_poly_terms; ir++)
        { //derivative wrt Y operator on the appended polynomials
            if (parameters.polynomial_term_exponents(ir - nv, 1) > 0)
            { //if parameters.polynomial_term_exponents(ir, 1) is zero, derivative wrt Y is zero
                d_temp = pow(vert[dim * ic], parameters.polynomial_term_exponents(ir - nv, 0));
                d_temp = d_temp * parameters.polynomial_term_exponents(ir - nv, 1) * pow(vert[dim * ic + 1], parameters.polynomial_term_exponents(ir - nv, 1) - 1.0);
                if (dim == 3)
                {
                    d_temp = d_temp * pow(vert[dim * ic + 2], parameters.polynomial_term_exponents(ir - nv, 2));
                }
                rhs(ir, central_vert_list.size() + ic) = d_temp / scale[1];
            }
        }
    }
}

void calc_PHS_RBF_grad_laplace_single_vert_grad_z_rhs(vector<double> &vert, PARAMETERS &parameters, Eigen::MatrixXd &rhs, double *scale, vector<int> &central_vert_list)
{ //vert: vertex co-ordinates of this group only (will give global coefficients on that group)
    int dim = parameters.dimension, num_poly_terms = parameters.num_poly_terms;
    int nv = (int)(vert.size() / dim), phs_deg = parameters.phs_deg, ic;
    double r_square, r_phs, d_temp;
    for (int i1 = 0; i1 < central_vert_list.size(); i1++)
    { //rhs for grad_z
        for (int ir = 0; ir < nv; ir++)
        { //derivative wrt Z operator on the RBFs: r^phs_deg
            r_square = 0.0;
            ic = central_vert_list[i1];
            for (int i = 0; i < dim; i++)
            {
                d_temp = vert[dim * ir + i] - vert[dim * ic + i];
                r_square = r_square + (d_temp * d_temp);
            }
            r_phs = -phs_deg * (vert[dim * ir + 2] - vert[dim * ic + 2]) * pow(r_square, (double)((phs_deg - 2.0) / 2.0)); //instead of computing sqrt separately, here, exponent is divided by 2
            rhs(ir, central_vert_list.size() + central_vert_list.size() + ic) = r_phs / scale[2];
        }

        for (int ir = nv; ir < nv + num_poly_terms; ir++)
        { //derivative wrt Z operator on the appended polynomials
            if (parameters.polynomial_term_exponents(ir - nv, 2) > 0)
            { //if parameters.polynomial_term_exponents(ir, 2) is zero, derivative wrt Z is zero
                d_temp = parameters.polynomial_term_exponents(ir - nv, 2) * pow(vert[dim * ic + 2], parameters.polynomial_term_exponents(ir - nv, 2) - 1.0);
                d_temp = d_temp * pow(vert[dim * ic], parameters.polynomial_term_exponents(ir - nv, 0));
                d_temp = d_temp * pow(vert[dim * ic + 1], parameters.polynomial_term_exponents(ir - nv, 1));
                rhs(ir, central_vert_list.size() + central_vert_list.size() + ic) = d_temp / scale[2];
            }
        }
    }
}

void calc_PHS_RBF_grad_laplace_single_vert_laplacian_rhs(vector<double> &vert, PARAMETERS &parameters, Eigen::MatrixXd &rhs, double *scale, vector<int> &central_vert_list)
{ //vert: vertex co-ordinates of this group only (will give global coefficients on that group)
    int dim = parameters.dimension, num_poly_terms = parameters.num_poly_terms;
    int nv = (int)(vert.size() / dim), phs_deg = parameters.phs_deg, ic;
    double r_square, d_temp, grad_xx, grad_yy, grad_zz;
    for (int i1 = 0; i1 < central_vert_list.size(); i1++)
    { //rhs for grad_xx
        for (int ir = 0; ir < nv; ir++)
        { //double derivative wrt X, Y and Z on the RBFs: r^phs_deg
            r_square = 0.0;
            ic = central_vert_list[i1];
            for (int i = 0; i < dim; i++)
            {
                d_temp = vert[dim * ir + i] - vert[dim * ic + i];
                r_square = r_square + (d_temp * d_temp);
            }
            if (fabs(r_square) > 1E-14) //to avoid zero raised to
            {
                grad_xx = phs_deg * phs_deg * pow(vert[dim * ir] - vert[dim * ic], 2.0) * pow(r_square, (double)((phs_deg - 4.0) / 2.0)); //instead of computing sqrt separately, here, exponent is divided by 2
                grad_xx = grad_xx - 2.0 * phs_deg * pow(vert[dim * ir] - vert[dim * ic], 2.0) * pow(r_square, (double)((phs_deg - 4.0) / 2.0));
                grad_xx = grad_xx + phs_deg * pow(r_square, (double)((phs_deg - 2.0) / 2.0));
                grad_xx = grad_xx / (scale[0] * scale[0]);

                grad_yy = phs_deg * phs_deg * pow(vert[dim * ir + 1] - vert[dim * ic + 1], 2.0) * pow(r_square, (double)((phs_deg - 4.0) / 2.0)); //instead of computing sqrt separately, here, exponent is divided by 2
                grad_yy = grad_yy - 2.0 * phs_deg * pow(vert[dim * ir + 1] - vert[dim * ic + 1], 2.0) * pow(r_square, (double)((phs_deg - 4.0) / 2.0));
                grad_yy = grad_yy + phs_deg * pow(r_square, (double)((phs_deg - 2.0) / 2.0));
                grad_yy = grad_yy / (scale[1] * scale[1]);
                grad_zz = 0.0;
                if (dim == 3)
                {
                    grad_zz = phs_deg * phs_deg * pow(vert[dim * ir + 2] - vert[dim * ic + 2], 2.0) * pow(r_square, (double)((phs_deg - 4.0) / 2.0)); //instead of computing sqrt separately, here, exponent is divided by 2
                    grad_zz = grad_zz - 2.0 * phs_deg * pow(vert[dim * ir + 2] - vert[dim * ic + 2], 2.0) * pow(r_square, (double)((phs_deg - 4.0) / 2.0));
                    grad_zz = grad_zz + phs_deg * pow(r_square, (double)((phs_deg - 2.0) / 2.0));
                    grad_zz = grad_zz / (scale[2] * scale[2]);
                }
                rhs(ir, dim * central_vert_list.size() + ic) = grad_xx + grad_yy + grad_zz;
            }
        }

        for (int ir = nv; ir < nv + num_poly_terms; ir++)
        { //double derivative wrt X, Y and Z on the appended polynomials
            grad_xx = 0.0;
            if (parameters.polynomial_term_exponents(ir - nv, 0) > 1)
            { //if parameters.polynomial_term_exponents(ir, 0) is zero or one, derivative wrt X is zero
                grad_xx = parameters.polynomial_term_exponents(ir - nv, 0) * (parameters.polynomial_term_exponents(ir - nv, 0) - 1.0) * pow(vert[dim * ic], parameters.polynomial_term_exponents(ir - nv, 0) - 2.0);
                grad_xx = grad_xx * pow(vert[dim * ic + 1], parameters.polynomial_term_exponents(ir - nv, 1));
                if (dim == 3)
                {
                    grad_xx = grad_xx * pow(vert[dim * ic + 2], parameters.polynomial_term_exponents(ir - nv, 2));
                }
                grad_xx = grad_xx / (scale[0] * scale[0]);
            }

            grad_yy = 0.0;
            if (parameters.polynomial_term_exponents(ir - nv, 1) > 1)
            { //if parameters.polynomial_term_exponents(ir, 1) is zero or one, derivative wrt Y is zero
                grad_yy = parameters.polynomial_term_exponents(ir - nv, 1) * (parameters.polynomial_term_exponents(ir - nv, 1) - 1.0) * pow(vert[dim * ic + 1], parameters.polynomial_term_exponents(ir - nv, 1) - 2.0);
                grad_yy = grad_yy * pow(vert[dim * ic + 0], parameters.polynomial_term_exponents(ir - nv, 0));
                if (dim == 3)
                {
                    grad_yy = grad_yy * pow(vert[dim * ic + 2], parameters.polynomial_term_exponents(ir - nv, 2));
                }
                grad_yy = grad_yy / (scale[1] * scale[1]);
            }

            grad_zz = 0.0;
            if (dim == 3)
            {
                if (parameters.polynomial_term_exponents(ir - nv, 2) > 1)
                { //if parameters.polynomial_term_exponents(ir, 2) is zero or one, derivative wrt Z is zero
                    grad_zz = parameters.polynomial_term_exponents(ir - nv, 2) * (parameters.polynomial_term_exponents(ir - nv, 2) - 1.0) * pow(vert[dim * ic + 2], parameters.polynomial_term_exponents(ir - nv, 2) - 2.0);
                    grad_zz = grad_zz * pow(vert[dim * ic + 0], parameters.polynomial_term_exponents(ir - nv, 0));
                    grad_zz = grad_zz * pow(vert[dim * ic + 1], parameters.polynomial_term_exponents(ir - nv, 1));
                    grad_zz = grad_zz / (scale[2] * scale[2]);
                }
            }
            rhs(ir, dim * central_vert_list.size() + ic) = grad_xx + grad_yy + grad_zz;
        }
    }
}

double calc_PHS_RBF_grad_laplace_single_vert(vector<double> &vert, PARAMETERS &parameters, Eigen::MatrixXd &laplacian, Eigen::MatrixXd &grad_x, Eigen::MatrixXd &grad_y, Eigen::MatrixXd &grad_z, double *scale, vector<int> &central_vert_list)
{ //vert: vertex co-ordinates of this group only (will give global coefficients on that group)
    //central_vert_list: laplacian and grads required only on the central_vert_list (use local numbering in range [0, vert.rows-1])
    int dim = parameters.dimension, num_poly_terms = parameters.num_poly_terms;
    int nv = (int)(vert.size() / dim), phs_deg = parameters.phs_deg;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(nv + num_poly_terms, nv + num_poly_terms);
    Eigen::MatrixXd rhs = Eigen::MatrixXd::Zero(nv + num_poly_terms, central_vert_list.size() * (dim + 1));
    // cout << "\ncalc_PHS_RBF_grad_laplace_single_vert: nv: " << nv << ", num_poly_terms: " << parameters.num_poly_terms << endl;
    // cout << "calc_PHS_RBF_grad_laplace_single_vert: A.shape: " << A.rows() << " X " << A.cols() << ", rhs.shape(): " << rhs.rows() << " X " << rhs.cols() << "\n\n";

    calc_PHS_RBF_grad_laplace_single_vert_A(vert, parameters, A, scale);
    calc_PHS_RBF_grad_laplace_single_vert_grad_x_rhs(vert, parameters, rhs, scale, central_vert_list);
    calc_PHS_RBF_grad_laplace_single_vert_grad_y_rhs(vert, parameters, rhs, scale, central_vert_list);
    if (dim == 3)
        calc_PHS_RBF_grad_laplace_single_vert_grad_z_rhs(vert, parameters, rhs, scale, central_vert_list);
    calc_PHS_RBF_grad_laplace_single_vert_laplacian_rhs(vert, parameters, rhs, scale, central_vert_list);

    Eigen::MatrixXd answer = A.partialPivLu().solve(rhs); //fullPivLu: slow but stable for high condition numbers; partialPivLu: fast but unstable for high condition numbers
    double cond_num = 1.0 / A.partialPivLu().rcond();
    grad_x = answer.block(0, 0, nv, central_vert_list.size()); //block of size (nv,central_vert_list.size()), starting at (0,0)
    grad_x.transposeInPlace();
    grad_y = answer.block(0, central_vert_list.size(), nv, central_vert_list.size()); //block of size (nv,central_vert_list.size()), starting at (0,central_vert_list.size())
    grad_y.transposeInPlace();
    if (dim == 3)
    {
        grad_z = answer.block(0, 2 * central_vert_list.size(), nv, central_vert_list.size()); //block of size (nv,central_vert_list.size()), starting at (0,2*central_vert_list.size())
        grad_z.transposeInPlace();
    }
    laplacian = answer.block(0, dim * central_vert_list.size(), nv, central_vert_list.size()); //block of size (nv,central_vert_list.size()), starting at (0,dim*central_vert_list.size())
    laplacian.transposeInPlace();
    A.resize(0, 0);      //free memory
    rhs.resize(0, 0);    //free memory
    answer.resize(0, 0); //free memory
    return cond_num;
}

void calc_cloud_points_slow_periodic_bc(vector<vector<int>> &cloud_points, vector<double> &xyz_probe, POINTS &points, PARAMETERS &parameters, vector<vector<int>> &periodic_bc_section)
{ //size of xyz_probe: [nv_probe, dim] (nv_probe: no. of points)
    vector<int> ind_p = parameters.periodic_bc_index;
    if (ind_p.size() == 0)
    {
        cout << "\n\nERROR from calc_cloud_points_slow_periodic_bc defined only for peridic bc case\n\n";
        throw bad_exception();
    }
    vector<double> dist_square, k_min_dist_square;
    vector<int> k_min_points;
    for (int iv = 0; iv < points.nv; iv++)
        dist_square.push_back(0.0); //initialize
    int k = parameters.cloud_size, dim = parameters.dimension, nv_probe = ((int)(xyz_probe.size() / dim));
    vector<int> empty_int;
    // vector<vector<int>> periodic_bc_section; //takes values [-1,0,1] for [near_min,middle,near_max] sections respectively: size [nv_probe][no. of periodic axes]
    for (int iv = 0; iv < nv_probe; iv++)
    {
        periodic_bc_section.push_back(empty_int);
        for (int ip = 0; ip < ind_p.size(); ip++)
            periodic_bc_section[iv].push_back(100);
    }
    double xyz_1, xyz_2;
    for (int ivp = 0; ivp < nv_probe; ivp++) //takes values [-1,0,1] for [near_min,middle,near_max] sections respectively
        for (int ip = 0; ip < ind_p.size(); ip++)
        {
            xyz_1 = points.xyz_min[ind_p[ip]] + (points.xyz_length[ind_p[ip]] / 3.0);
            xyz_2 = points.xyz_min[ind_p[ip]] + (2.0 * points.xyz_length[ind_p[ip]] / 3.0);
            if (xyz_probe[dim * ivp + ind_p[ip]] < xyz_1)
                periodic_bc_section[ivp][ip] = -1; //section in range [min, xyz_1)
            else if ((xyz_1 <= xyz_probe[dim * ivp + ind_p[ip]]) && (xyz_probe[dim * ivp + ind_p[ip]] <= xyz_2))
                periodic_bc_section[ivp][ip] = 0; //section in range [xyz_1 -> xyz_2]
            else
                periodic_bc_section[ivp][ip] = 1; //section in range (xyz_2, max]}
        }
    vector<double> xyz_p, xyz;
    for (int id = 0; id < dim; id++)
        xyz_p.push_back(0.0), xyz.push_back(0.0);
    for (int ivp = 0; ivp < nv_probe; ivp++)
    {
        for (int id = 0; id < dim; id++)
            xyz_p[id] = xyz_probe[dim * ivp + id];
        for (int iv = 0; iv < points.nv; iv++)
        {
            for (int id = 0; id < dim; id++)
                xyz[id] = points.xyz[dim * iv + id];
            for (int ip = 0; ip < ind_p.size(); ip++)
                if (periodic_bc_section[ivp][ip] == -points.periodic_bc_section[iv][ip])
                    xyz[ind_p[ip]] = xyz[ind_p[ip]] + (periodic_bc_section[ivp][ip] * points.xyz_length[ind_p[ip]]);
            dist_square[iv] = 0.0;
            for (int id = 0; id < dim; id++)
                dist_square[iv] = dist_square[iv] + (xyz_p[id] - xyz[id]) * (xyz_p[id] - xyz[id]);
        }
        k_smallest_elements(k_min_dist_square, k_min_points, dist_square, k);
        cloud_points.push_back(k_min_points);
    }
}

void calc_cloud_points_slow(vector<vector<int>> &cloud_points, vector<double> &xyz_probe, POINTS &points, PARAMETERS &parameters)
{ //size of xyz_probe: [nv_probe, dim] (nv_probe: no. of points)
    if (parameters.periodic_bc_index.size() > 0)
    {
        cout << "\n\nERROR from calc_cloud_points_slow defined only for non-peridic bc case\n\n";
        throw bad_exception();
    }
    vector<double> dist_square, k_min_dist_square;
    vector<int> k_min_points;
    for (int iv = 0; iv < points.nv; iv++)
        dist_square.push_back(0.0); //initialize
    double xp, yp, zp;
    int k = parameters.cloud_size, dim = parameters.dimension, nv_probe = ((int)(xyz_probe.size() / dim));
    if (dim == 2) //2D problem
        for (int ivp = 0; ivp < nv_probe; ivp++)
        {
            xp = xyz_probe[dim * ivp], yp = xyz_probe[dim * ivp + 1];
            for (int iv = 0; iv < points.nv; iv++)
            {
                dist_square[iv] = (xp - points.xyz[dim * iv]) * (xp - points.xyz[dim * iv]);
                dist_square[iv] = dist_square[iv] + (yp - points.xyz[dim * iv + 1]) * (yp - points.xyz[dim * iv + 1]);
            }
            k_smallest_elements(k_min_dist_square, k_min_points, dist_square, k);
            cloud_points.push_back(k_min_points);
        }
    else //3D problem
        for (int ivp = 0; ivp < nv_probe; ivp++)
        {
            xp = xyz_probe[dim * ivp], yp = xyz_probe[dim * ivp + 1], zp = xyz_probe[dim * ivp + 2];
            for (int iv = 0; iv < points.nv; iv++)
            {
                dist_square[iv] = (xp - points.xyz[dim * iv]) * (xp - points.xyz[dim * iv]);
                dist_square[iv] = dist_square[iv] + (yp - points.xyz[dim * iv + 1]) * (yp - points.xyz[dim * iv + 1]);
                dist_square[iv] = dist_square[iv] + (zp - points.xyz[dim * iv + 2]) * (zp - points.xyz[dim * iv + 2]);
            }
            k_smallest_elements(k_min_dist_square, k_min_points, dist_square, k);
            cloud_points.push_back(k_min_points);
        }
    dist_square.clear();
    k_min_dist_square.clear();
    k_min_points.clear();
}

void calc_PHS_RBF_interp_single_vert_rhs(double *xyz_interp, vector<double> &vert, PARAMETERS &parameters, Eigen::VectorXd &rhs)
{
    int dim = parameters.dimension, num_poly_terms = parameters.num_poly_terms;
    int nv = (int)(vert.size() / dim), phs_deg = parameters.phs_deg;
    double r_square, d_temp;
    for (int ir = 0; ir < nv; ir++)
    { //r^phs_deg
        r_square = 0.0;
        for (int i = 0; i < dim; i++)
        {
            d_temp = vert[dim * ir + i] - xyz_interp[i];
            r_square = r_square + (d_temp * d_temp);
        }
        rhs(ir) = pow(r_square, (phs_deg / 2.0)); //instead of computing sqrt separately, here, exponent is divided by 2
    }

    for (int ir = nv; ir < nv + num_poly_terms; ir++)
    { //appended polynomials
        d_temp = pow(xyz_interp[0], parameters.polynomial_term_exponents(ir - nv, 0));
        d_temp = d_temp * pow(xyz_interp[1], parameters.polynomial_term_exponents(ir - nv, 1));
        if (dim == 3)
            d_temp = d_temp * pow(xyz_interp[2], parameters.polynomial_term_exponents(ir - nv, 2));
        rhs(ir) = d_temp;
    }
}

Eigen::SparseMatrix<double, Eigen::RowMajor> calc_interp_matrix(vector<double> &xyz_probe, POINTS &points, PARAMETERS &parameters)
{
    double cond_num;
    int dim = parameters.dimension, iv_nb, nv_probe = ((int)(xyz_probe.size() / dim));
    Eigen::SparseMatrix<double, Eigen::RowMajor> interp_matrix;
    Eigen::MatrixXd A = Eigen::MatrixXd::Zero(parameters.cloud_size + parameters.num_poly_terms, parameters.cloud_size + parameters.num_poly_terms);
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(parameters.cloud_size + parameters.num_poly_terms);
    Eigen::VectorXd answer = Eigen::VectorXd::Zero(parameters.cloud_size + parameters.num_poly_terms);
    Eigen::VectorXd interp_coeff = Eigen::VectorXd::Zero(parameters.cloud_size);
    vector<vector<int>> cloud_points, periodic_bc_section;
    vector<int> ind_p = parameters.periodic_bc_index;
    if (ind_p.size() == 0) //non-periodic case
        calc_cloud_points_slow(cloud_points, xyz_probe, points, parameters);
    else //periodic case
        calc_cloud_points_slow_periodic_bc(cloud_points, xyz_probe, points, parameters, periodic_bc_section);
    vector<double> vert;
    double scale[3], xyz_p[3], xyz[3];
    vector<Eigen::Triplet<double>> triplet;

    for (int ivp = 0; ivp < nv_probe; ivp++)
    {
        for (int i1 = 0; i1 < cloud_points[ivp].size(); i1++)
        {
            iv_nb = cloud_points[ivp][i1];
            for (int id = 0; id < dim; id++)
                xyz[id] = points.xyz[dim * iv_nb + id];
            if (ind_p.size() > 0)
            { //periodic case
                for (int ip = 0; ip < ind_p.size(); ip++)
                    if (periodic_bc_section[ivp][ip] == -points.periodic_bc_section[iv_nb][ip])
                        xyz[ind_p[ip]] = xyz[ind_p[ip]] + (periodic_bc_section[ivp][ip] * points.xyz_length[ind_p[ip]]);
            }
            for (int id = 0; id < dim; id++)
                vert.push_back(xyz[id]);
        }
        for (int i1 = 0; i1 < dim; i1++)
            xyz_p[i1] = xyz_probe[dim * ivp + i1];
        shifting_scaling(xyz_p, vert, scale, dim);
        calc_PHS_RBF_grad_laplace_single_vert_A(vert, parameters, A, scale);
        calc_PHS_RBF_interp_single_vert_rhs(xyz_p, vert, parameters, rhs);
        answer = A.partialPivLu().solve(rhs); //fullPivLu: slow but stable for high condition numbers; partialPivLu: fast but unstable for high condition numbers
        cond_num = 1.0 / A.partialPivLu().rcond();
        for (int i1 = 0; i1 < parameters.cloud_size; i1++)
            interp_coeff(i1) = answer(i1);

        for (int i1 = 0; i1 < parameters.cloud_size; i1++)
            triplet.push_back(Eigen::Triplet<double>(ivp, cloud_points[ivp][i1], interp_coeff[i1]));

        vert.clear();
    }
    interp_matrix.resize(nv_probe, points.nv);
    interp_matrix.setFromTriplets(triplet.begin(), triplet.end());
    interp_matrix.makeCompressed();

    triplet.clear();  //free memory
    A.resize(0, 0);   //free memory
    rhs.resize(0);    //free memory
    answer.resize(0); //free memory
    for (int i1 = 0; i1 < cloud_points.size(); i1++)
        cloud_points[i1].clear(); //free memory
    cloud_points.clear();         //free memory
    return interp_matrix;
}

void initialize_velocities(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u, Eigen::VectorXd &v)
{
    Eigen::VectorXd rhs = Eigen::VectorXd::Zero(points.nv + 1);
    Eigen::VectorXd u_copy = u, v_copy = v;
    vector<Eigen::Triplet<double>> triplet;
    Eigen::SparseMatrix<double, Eigen::RowMajor> coeff_EIGEN;
    int dim = parameters.dimension, iv_nb;
    if (dim != 2)
    {
        cout << "\n\nERROR from initialize_velocities defined only for 2D problems; current value of parameters.dimension: " << parameters.dimension << "\n\n";
        throw bad_exception();
    }
    double val, absolute_residual, relative_residual;
    for (int iv = 0; iv < points.nv; iv++)
    {
        if (points.boundary_flag[iv]) //lies on boundary
        {
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                iv_nb = cloud.nb_points_col[i1];
                val = points.normal[dim * iv] * cloud.grad_y_coeff[i1] + (-points.normal[dim * iv + 1] * cloud.grad_x_coeff[i1]);
                triplet.push_back(Eigen::Triplet<double>(iv, iv_nb, val));
            }
            rhs[iv] = (points.normal[dim * iv] * u[iv]) + (points.normal[dim * iv + 1] * v[iv]);
        }
        else
            for (int i1 = cloud.nb_points_row[iv]; i1 < cloud.nb_points_row[iv + 1]; i1++)
            {
                iv_nb = cloud.nb_points_col[i1];
                val = cloud.laplacian_coeff[i1]; //diffusion
                triplet.push_back(Eigen::Triplet<double>(iv, iv_nb, val));
            }
    }
    //extra constraint (regularization: http://www-e6.ijs.si/medusa/wiki/index.php/Poisson%27s_equation) for neumann BCs
    for (int iv = 0; iv < points.nv; iv++)
    {
        triplet.push_back(Eigen::Triplet<double>(iv, points.nv, 1.0)); //last column
        triplet.push_back(Eigen::Triplet<double>(points.nv, iv, 1.0)); //last row
    }
    triplet.push_back(Eigen::Triplet<double>(points.nv, points.nv, 0.0)); //last entry
    coeff_EIGEN.resize(points.nv + 1, points.nv + 1);
    coeff_EIGEN.setFromTriplets(triplet.begin(), triplet.end());
    coeff_EIGEN.makeCompressed();
    triplet.clear();
    clock_t clock_t1 = clock();
    Eigen::BiCGSTAB<Eigen::SparseMatrix<double, Eigen::RowMajor>, Eigen::IncompleteLUT<double>> solver_eigen_ilu_bicgstab;
    solver_eigen_ilu_bicgstab.setTolerance(parameters.solver_tolerance); //default is machine precision (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#ac160a444af8998f93da9aa30e858470d)
    solver_eigen_ilu_bicgstab.setMaxIterations(parameters.n_iter);       //default is twice number of columns (https://eigen.tuxfamily.org/dox/classEigen_1_1IterativeSolverBase.html#af83de7a7d31d9d4bd1fef6222b07335b)
    solver_eigen_ilu_bicgstab.preconditioner().setDroptol(1E-6);
    solver_eigen_ilu_bicgstab.compute(coeff_EIGEN);
    Eigen::VectorXd psi = solver_eigen_ilu_bicgstab.solve(rhs);
    absolute_residual = (coeff_EIGEN * psi - rhs).norm();
    relative_residual = absolute_residual / rhs.norm();
    double runtime = ((double)(clock() - clock_t1)) / CLOCKS_PER_SEC;
    printf("\n\ninitialize_velocities abs_res: %g, rel_res: %g, tol: %g in %li iterations; regularization value: %g; runtime: %g seconds\n\n", absolute_residual, relative_residual, parameters.solver_tolerance, solver_eigen_ilu_bicgstab.iterations(), psi[points.nv], runtime);
    u = points.grad_y_matrix_EIGEN * psi.head(points.nv), v = -points.grad_x_matrix_EIGEN * psi.head(points.nv);
    for (int iv = 0; iv < points.nv; iv++)
        if (points.boundary_flag[iv]) //reset boundary values
            u[iv] = u_copy[iv], v[iv] = v_copy[iv];
}