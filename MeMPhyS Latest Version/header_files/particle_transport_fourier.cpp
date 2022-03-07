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
#include "general_functions.hpp"
#include "class.hpp"
#include "nanoflann.hpp"

PARTICLE_TRANSPORT_FOURIER::PARTICLE_TRANSPORT_FOURIER(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, FRACTIONAL_STEP_FOURIER &fractional_step_fourier, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, Eigen::VectorXd &stokes_num, Eigen::VectorXd &density, double flow_charac_time)
{
    np = position.rows();
    int dim = parameters.dimension;
    if (position.cols() != dim + 1)
    {
        printf("\n\nERROR from PARTICLE_TRANSPORT_FOURIER::PARTICLE_TRANSPORT_FOURIER no. of columns of position: %li should be 3\n\n", position.cols());
        throw bad_exception();
    }
    if ((position.rows() != velocity.rows()) || (position.cols() != velocity.cols()))
    {
        printf("\n\nERROR from PARTICLE_TRANSPORT_FOURIER::PARTICLE_TRANSPORT_FOURIER shape of position: (%li, %li) should be same as shape of position: (%li, %li)\n\n", position.rows(), position.cols(), velocity.rows(), velocity.cols());
        throw bad_exception();
    }
    RK_k1 = Eigen::MatrixXd::Zero(np, 6), RK_k2 = RK_k1;
    u_flow = Eigen::VectorXd::Zero(np), v_flow = u_flow, w_flow = u_flow, diameter = u_flow, mass = u_flow;
    acceleration = Eigen::MatrixXd::Zero(np, 3), vel_temp = acceleration, pos_temp = vel_temp;
    for (int ip = 0; ip < np; ip++)
    {
        boundary_flag.push_back(false); // initialize to interior
        diameter[ip] = sqrt(18.0 * parameters.mu * flow_charac_time * stokes_num[ip] / density[ip]);
        mass[ip] = density[ip] * M_PI * pow(diameter[ip], 3.0) / 6.0;
    }

    points_xyz_nf.pts.resize(points.nv);
    int n_point_nb_point = 12; // small set of points
    point_nb_point = Eigen::MatrixXi::Zero(points.nv, n_point_nb_point);
    int n_triangles = 12; // set of triangles originating from each point
    point_triangles = Eigen::MatrixXi::Zero(points.nv, dim * n_triangles);
    point_triangles_2_area = Eigen::MatrixXd::Zero(points.nv, n_triangles);
    particle_nb_triangle = Eigen::MatrixXi::Zero(np, dim + 1);
    particle_z_plane_range = Eigen::VectorXi::Zero(np);
    calc_neighbors_xy_plane(points, cloud, parameters, position);
    calc_triangles_xy_plane(points, cloud, parameters);
    shape_function = Eigen::MatrixXd::Zero(np, dim + 1);
    z_min = fractional_step_fourier.z(0, 0);
    z_max = fractional_step_fourier.z(0, fractional_step_fourier.z.cols() - 1);
}

void PARTICLE_TRANSPORT_FOURIER::calc_neighbors_xy_plane(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position)
{
    int dim = parameters.dimension;
    for (int iv = 0; iv < points.nv; iv++)
    {
        points_xyz_nf.pts[iv].x = points.xyz[dim * iv];
        points_xyz_nf.pts[iv].y = points.xyz[dim * iv + 1];
        if (dim == 3)
            points_xyz_nf.pts[iv].z = points.xyz[dim * iv + 2];
        else
            points_xyz_nf.pts[iv].z = 0.0; // does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
    }
    typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<double, PointCloud<double>>, PointCloud<double>, 3> kd_tree_nf;
    kd_tree_nf index_nf(3, points_xyz_nf, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    index_nf.buildIndex();

    int n_point_nb_point = point_nb_point.cols();
    vector<size_t> nb_vert(n_point_nb_point);
    vector<double> nb_dist(n_point_nb_point);

    double query_pt[3];
    for (int iv = 0; iv < points.nv; iv++)
    {
        query_pt[0] = points.xyz[dim * iv], query_pt[1] = points.xyz[dim * iv + 1];
        if (dim == 3)
            query_pt[2] = points.xyz[dim * iv + 2];
        else
            query_pt[2] = 0.0; // does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
        index_nf.knnSearch(&query_pt[0], n_point_nb_point, &nb_vert[0], &nb_dist[0]);
        for (int i1 = 0; i1 < n_point_nb_point; i1++)
            point_nb_point(iv, i1) = nb_vert[i1];
    }

    for (int i1 = 0; i1 < n_point_nb_point; i1++)
        nb_vert[i1] = -1, nb_dist[i1] = -1;
    for (int ip = 0; ip < np; ip++)
    {
        query_pt[0] = position(ip, 0), query_pt[1] = position(ip, 1);
        if (dim == 3)
            query_pt[2] = position(ip, 2);
        else
            query_pt[2] = 0.0; // does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
        index_nf.knnSearch(&query_pt[0], dim + 1, &nb_vert[0], &nb_dist[0]);
        for (int i1 = 0; i1 < dim + 1; i1++)
            particle_nb_triangle(ip, i1) = nb_vert[i1];
    }
}

void PARTICLE_TRANSPORT_FOURIER::calc_triangles_xy_plane(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
{
    int n_triangles = point_triangles_2_area.cols(), dim = parameters.dimension, iv1, iv2, index, count;
    int n_search = (point_nb_point.cols() - 1) * (point_nb_point.cols() - 2) / 2;
    vector<int> triangle_vert(dim * n_search), argsort(n_search);
    vector<double> max_distance_sq(n_search);
    double dot_prod, dist1, dist2;
    // write_csv(point_nb_point, "point_nb_point.csv");
    for (int iv = 0; iv < points.nv; iv++)
    {
        index = 0, count = 0;
        for (int i1 = 1; i1 < point_nb_point.cols(); i1++) // i1=0 is the point iv
            for (int i2 = i1 + 1; i2 < point_nb_point.cols(); i2++)
            {
                iv1 = point_nb_point(iv, i1), iv2 = point_nb_point(iv, i2);
                triangle_vert[dim * index] = iv1, triangle_vert[dim * index + 1] = iv2;
                dot_prod = 0;
                for (int i3 = 0; i3 < dim; i3++)
                    dot_prod = dot_prod + ((points.xyz[dim * iv1 + i3] - points.xyz[dim * iv + i3]) * (points.xyz[dim * iv2 + i3] - points.xyz[dim * iv + i3]));

                if (dot_prod >= 0)
                { // acute or right angle is formed
                    // triangle_area[index] = calc_triangle_area_double(points.xyz[dim * iv], points.xyz[dim * iv + 1], points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], points.xyz[dim * iv2], points.xyz[dim * iv2 + 1]);
                    dist1 = pow(points.xyz[dim * iv] - points.xyz[dim * iv1], 2.0) + pow(points.xyz[dim * iv + 1] - points.xyz[dim * iv1 + 1], 2.0);
                    dist2 = pow(points.xyz[dim * iv] - points.xyz[dim * iv2], 2.0) + pow(points.xyz[dim * iv + 1] - points.xyz[dim * iv2 + 1], 2.0);
                    max_distance_sq[index] = dist1;
                    if (max_distance_sq[index] < dist2)
                        max_distance_sq[index] = dist2;
                    count++;
                }
                else
                    max_distance_sq[index] = INFINITY; // set to remove obtuse traingles later
                // triangle_area[index] = INFINITY; //set to remove obtuse traingles later
                index++;
            }

        // print_to_terminal(triangle_area, "triangle_area from PARTICLE_TRANSPORT_FOURIER::calc_triangles");
        // reference: https://stackoverflow.com/a/40183830/6932587
        iota(argsort.begin(), argsort.end(), 0); // Initializing
        sort(argsort.begin(), argsort.end(), [&](int i, int j)
             { return max_distance_sq[i] < max_distance_sq[j]; });

        if (count < n_triangles)
        {
            printf("\n\nERROR from PARTICLE_TRANSPORT_FOURIER::calc_triangles iv: %i, found only %i candidates of acute angled triangles; need atleast %i triangles\n", iv, count, n_triangles);
            cout << "point_nb_point.row(iv): " << point_nb_point.row(iv) << "\n";
            print_to_terminal(triangle_vert, n_search, dim, "triangle_vert from PARTICLE_TRANSPORT_FOURIER::calc_triangles");
            print_to_terminal(max_distance_sq, "max_distance_sq from PARTICLE_TRANSPORT_FOURIER::calc_triangles");
            throw bad_exception();
        }

        for (int i1 = 0; i1 < n_triangles; i1++)
        {
            iv1 = triangle_vert[dim * argsort[i1]], iv2 = triangle_vert[dim * argsort[i1] + 1];
            point_triangles_2_area(iv, i1) = calc_triangle_area_double(points.xyz[dim * iv], points.xyz[dim * iv + 1], points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], points.xyz[dim * iv2], points.xyz[dim * iv2 + 1]);
            point_triangles(iv, dim * i1) = iv1, point_triangles(iv, dim * i1 + 1) = iv2;
        }
    }
}

double PARTICLE_TRANSPORT_FOURIER::calc_triangle_area_double(double x1, double y1, double x2, double y2, double x3, double y3)
{ // this gives double of area as division by 2.0 is not performed here
    return fabs(((x1 - x2) * (y1 - y3)) - ((y1 - y2) * (x1 - x3)));
}

void PARTICLE_TRANSPORT_FOURIER::calc_dt(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, Eigen::MatrixXd &velocity)
{
    double u_max, v_max, w_max, temp1, temp2, temp3, speed_max;
    temp1 = u_new.lpNorm<Eigen::Infinity>(), temp2 = u_old.lpNorm<Eigen::Infinity>();
    temp3 = velocity.col(0).lpNorm<Eigen::Infinity>();
    u_max = max(temp1, max(temp2, temp3));
    temp1 = v_new.lpNorm<Eigen::Infinity>(), temp2 = v_old.lpNorm<Eigen::Infinity>();
    temp3 = velocity.col(1).lpNorm<Eigen::Infinity>();
    v_max = max(temp1, max(temp2, temp3));
    temp1 = w_new.lpNorm<Eigen::Infinity>(), temp2 = w_old.lpNorm<Eigen::Infinity>();
    temp3 = velocity.col(2).lpNorm<Eigen::Infinity>();
    w_max = max(temp1, max(temp2, temp3));
    speed_max = sqrt((u_max * u_max) + (v_max * v_max) + (w_max * w_max));
    dt = parameters.min_dx / speed_max;
    if (dt >= parameters.dt)
        dt = parameters.dt, nt = 1;
    else
        nt = (int)(parameters.dt / dt), nt++, dt = parameters.dt / nt;
}

void PARTICLE_TRANSPORT_FOURIER::update_nearest_vert_xy_plane(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, FRACTIONAL_STEP_FOURIER &fractional_step_fourier, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity)
{
    int iv_old, iv_new, dim = parameters.dimension;
    double dist_sq, min_dist_sq;
    for (int ip = 0; ip < np; ip++)
    {
        iv_old = particle_nb_triangle(ip, 0), iv_new = iv_old;
        min_dist_sq = 0.0;
        for (int i2 = 0; i2 < dim; i2++)
            min_dist_sq = min_dist_sq + ((points.xyz[dim * iv_new + i2] - position(ip, i2)) * (points.xyz[dim * iv_new + i2] - position(ip, i2)));
        for (int i1 = 0; i1 < point_nb_point.cols(); i1++)
        {
            iv_new = point_nb_point(iv_old, i1), dist_sq = 0.0;
            for (int i2 = 0; i2 < dim; i2++)
                dist_sq = dist_sq + ((points.xyz[dim * iv_new + i2] - position(ip, i2)) * (points.xyz[dim * iv_new + i2] - position(ip, i2)));
            if (min_dist_sq > dist_sq) // found a closer point
                particle_nb_triangle(ip, 0) = iv_new;
        }
    }
}

void PARTICLE_TRANSPORT_FOURIER::calc_z_plane_range(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, FRACTIONAL_STEP_FOURIER &fractional_step_fourier, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity)
{
    double zp;
    int iz0;
    for (int ip = 0; ip < np; ip++)
    {
        if (position(ip, 2) < z_min) // periodicity in Z direction
            position(ip, 2) = position(ip, 2) + (z_max - z_min);
        else if (position(ip, 2) > z_max) // periodicity in Z direction
            position(ip, 2) = position(ip, 2) - (z_max - z_min);
        iz0 = -1;
        zp = position(ip, 2);
        for (int iz = 0; iz < fractional_step_fourier.z.cols() - 1; iz++)
            if ((fractional_step_fourier.z(0, iz) <= zp) && (zp <= fractional_step_fourier.z(0, iz + 1)))
            {
                iz0 = iz;
                break;
            }
        if (iz0 < 0)
        {
            printf("\n\nERROR from PARTICLE_TRANSPORT_FOURIER::calc_z_plane_range particle no. %i with a Z coordinate: %g cannot be located in the range of (%g, %g)\n\n", ip, zp, z_min, z_max);
            throw bad_exception();
        }
        particle_z_plane_range[ip] = iz0;
    }
}

void PARTICLE_TRANSPORT_FOURIER::update_particle_nb_triangle_shape_function(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity)
{
    int n_triangles = point_triangles_2_area.cols(), iv1, iv2, iv3, dim = parameters.dimension;
    bool found_triangle;
    double area, area1, area2, area3;
    for (int ip = 0; ip < np; ip++)
    {
        found_triangle = false, boundary_flag[ip] = true;
        iv1 = particle_nb_triangle(ip, 0); // nearest point to particle ip
        for (int itr = 0; itr < n_triangles; itr++)
        {
            iv2 = point_triangles(iv1, dim * itr), iv3 = point_triangles(iv1, dim * itr + 1);
            area = point_triangles_2_area(iv1, itr);
            area1 = calc_triangle_area_double(position(ip, 0), position(ip, 1), points.xyz[dim * iv2], points.xyz[dim * iv2 + 1], points.xyz[dim * iv3], points.xyz[dim * iv3 + 1]);
            area2 = calc_triangle_area_double(position(ip, 0), position(ip, 1), points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], points.xyz[dim * iv3], points.xyz[dim * iv3 + 1]);
            area3 = calc_triangle_area_double(position(ip, 0), position(ip, 1), points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], points.xyz[dim * iv2], points.xyz[dim * iv2 + 1]);
            if (area1 + area2 + area3 <= (area + 1E-10))
            { // found a triangle with particle in the interior
                particle_nb_triangle(ip, 1) = iv2, particle_nb_triangle(ip, 2) = iv3;
                shape_function(ip, 0) = area1 / area;
                shape_function(ip, 1) = area2 / area;
                shape_function(ip, 2) = area3 / area;
                found_triangle = true, boundary_flag[ip] = false;
                break;
            }
        }
        if (!found_triangle)
        { // particle does not lie inside any triangle incident on nearest point iv1
            if (!points.boundary_flag[iv1])
            { // nearest point iv1 is inside the domain
                // printf("\n\nPARTICLE_TRANSPORT::update_particle_nb_triangle_shape_function ip: %i, particle_coordinates: (%g, %g)\n", ip, position(ip, 0), position(ip, 1));
                // printf("nearest iv1: %i, points.boundary_flag[iv1]: %i, points.xyz[iv1]: (%g, %g)\n", iv1, (int)(points.boundary_flag[iv1]), points.xyz[dim * iv1], points.xyz[dim * iv1 + 1]);
                // printf("Nearest point is inside the domain but the particle lies outside all the triangles incident on nearest point\n\n");
                // throw bad_exception();

                // recheck this: can happen when some corner point from GMSH grid is deleted: thus that triangle is not defined
                boundary_flag[ip] = false; // particle is inside domain boundary
                shape_function(ip, 0) = 1.0, shape_function(ip, 1) = 0.0, shape_function(ip, 2) = 0.0;
                particle_nb_triangle(ip, 1) = point_triangles(iv1, 0); // smallest triangle
                particle_nb_triangle(ip, 2) = point_triangles(iv1, 1); // smallest triangle
            }
            else
            {                             // nearest point iv1 is on the domain boundary
                boundary_flag[ip] = true; // particle is outside domain boundary
                shape_function(ip, 0) = 1.0, shape_function(ip, 1) = 0.0, shape_function(ip, 2) = 0.0;
                particle_nb_triangle(ip, 1) = point_triangles(iv1, 0); // smallest triangle
                particle_nb_triangle(ip, 2) = point_triangles(iv1, 1); // smallest triangle
            }
        }
    }
}

void PARTICLE_TRANSPORT_FOURIER::interp_flow_velocities(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, FRACTIONAL_STEP_FOURIER &fractional_step_fourier, Eigen::MatrixXd &position, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, double time_offset)
{
    double vel_new, vel_old, z_dist_1, z_dist_2, z_wt_1, z_wt_2;
    int dim = parameters.dimension, iz_1, iz_2;
    vector<int> iv_temp(particle_nb_triangle.cols());
    for (int ip = 0; ip < np; ip++)
    {
        for (int i1 = 0; i1 < particle_nb_triangle.cols(); i1++)
            iv_temp[i1] = particle_nb_triangle(ip, i1);
        iz_1 = particle_z_plane_range[ip], iz_2 = iz_1 + 1;
        z_dist_1 = fabs(position(ip, 2) - fractional_step_fourier.z(0, iz_1));
        z_dist_2 = fabs(position(ip, 2) - fractional_step_fourier.z(0, iz_2));
        z_wt_1 = z_dist_2 / (z_dist_1 + z_dist_2);
        z_wt_2 = z_dist_1 / (z_dist_1 + z_dist_2);

        vel_new = 0.0, vel_old = 0.0;
        for (int i1 = 0; i1 < dim + 1; i1++)
        {
            vel_new = vel_new + (shape_function(ip, i1) * z_wt_1 * u_new(iv_temp[i1], iz_1));
            vel_new = vel_new + (shape_function(ip, i1) * z_wt_2 * u_new(iv_temp[i1], iz_2));
            vel_old = vel_old + (shape_function(ip, i1) * z_wt_1 * u_old(iv_temp[i1], iz_1));
            vel_old = vel_old + (shape_function(ip, i1) * z_wt_2 * u_old(iv_temp[i1], iz_2));
        }
        u_flow[ip] = ((it + time_offset) * vel_new + (nt - it - time_offset) * vel_old) / nt;

        vel_new = 0.0, vel_old = 0.0;
        for (int i1 = 0; i1 < dim + 1; i1++)
        {
            vel_new = vel_new + (shape_function(ip, i1) * z_wt_1 * v_new(iv_temp[i1], iz_1));
            vel_new = vel_new + (shape_function(ip, i1) * z_wt_2 * v_new(iv_temp[i1], iz_2));
            vel_old = vel_old + (shape_function(ip, i1) * z_wt_1 * v_old(iv_temp[i1], iz_1));
            vel_old = vel_old + (shape_function(ip, i1) * z_wt_2 * v_old(iv_temp[i1], iz_2));
        }
        v_flow[ip] = ((it + time_offset) * vel_new + (nt - it - time_offset) * vel_old) / nt;

        vel_new = 0.0, vel_old = 0.0;
        for (int i1 = 0; i1 < dim + 1; i1++)
        {
            vel_new = vel_new + (shape_function(ip, i1) * z_wt_1 * w_new(iv_temp[i1], iz_1));
            vel_new = vel_new + (shape_function(ip, i1) * z_wt_2 * w_new(iv_temp[i1], iz_2));
            vel_old = vel_old + (shape_function(ip, i1) * z_wt_1 * w_old(iv_temp[i1], iz_1));
            vel_old = vel_old + (shape_function(ip, i1) * z_wt_2 * w_old(iv_temp[i1], iz_2));
        }
        w_flow[ip] = ((it + time_offset) * vel_new + (nt - it - time_offset) * vel_old) / nt;
    }
}

void PARTICLE_TRANSPORT_FOURIER::apply_boundary_conditions(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, FRACTIONAL_STEP_FOURIER &fractional_step_fourier, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, double time_offset)
{
    double vel_new, vel_old, z_dist_1, z_dist_2, z_wt_1, z_wt_2;
    int dim = parameters.dimension, iz_1, iz_2;
    vector<int> iv_temp(particle_nb_triangle.cols());
    for (int ip = 0; ip < np; ip++)
        if (boundary_flag[ip])
        { // particle lies on the boundary: set position and velocity
            for (int i1 = 0; i1 < particle_nb_triangle.cols(); i1++)
                iv_temp[i1] = particle_nb_triangle(ip, i1);
            iz_1 = particle_z_plane_range[ip], iz_2 = iz_1 + 1;
            z_dist_1 = fabs(position(ip, 2) - fractional_step_fourier.z(0, iz_1));
            z_dist_2 = fabs(position(ip, 2) - fractional_step_fourier.z(0, iz_2));
            z_wt_1 = z_dist_2 / (z_dist_1 + z_dist_2);
            z_wt_2 = z_dist_1 / (z_dist_1 + z_dist_2);

            position(ip, 0) = 0.0, position(ip, 1) = 0.0; // Z coordinate unchanged
            for (int i1 = 0; i1 < dim + 1; i1++)
            { // Z coordinate unchanged
                position(ip, 0) = position(ip, 0) + (shape_function(ip, i1) * points.xyz[dim * iv_temp[i1]]);
                position(ip, 1) = position(ip, 1) + (shape_function(ip, i1) * points.xyz[dim * iv_temp[i1] + 1]);
            }

            vel_new = 0.0, vel_old = 0.0;
            for (int i1 = 0; i1 < dim + 1; i1++)
            {
                vel_new = vel_new + (shape_function(ip, i1) * z_wt_1 * u_new(iv_temp[i1], iz_1));
                vel_new = vel_new + (shape_function(ip, i1) * z_wt_2 * u_new(iv_temp[i1], iz_2));
                vel_old = vel_old + (shape_function(ip, i1) * z_wt_1 * u_old(iv_temp[i1], iz_1));
                vel_old = vel_old + (shape_function(ip, i1) * z_wt_2 * u_old(iv_temp[i1], iz_2));
            }
            velocity(ip, 0) = ((it + time_offset) * vel_new + (nt - it - time_offset) * vel_old) / nt;

            vel_new = 0.0, vel_old = 0.0;
            for (int i1 = 0; i1 < dim + 1; i1++)
            {
                vel_new = vel_new + (shape_function(ip, i1) * z_wt_1 * v_new(iv_temp[i1], iz_1));
                vel_new = vel_new + (shape_function(ip, i1) * z_wt_2 * v_new(iv_temp[i1], iz_2));
                vel_old = vel_old + (shape_function(ip, i1) * z_wt_1 * v_old(iv_temp[i1], iz_1));
                vel_old = vel_old + (shape_function(ip, i1) * z_wt_2 * v_old(iv_temp[i1], iz_2));
            }
            velocity(ip, 1) = ((it + time_offset) * vel_new + (nt - it - time_offset) * vel_old) / nt;

            vel_new = 0.0, vel_old = 0.0;
            for (int i1 = 0; i1 < dim + 1; i1++)
            {
                vel_new = vel_new + (shape_function(ip, i1) * z_wt_1 * w_new(iv_temp[i1], iz_1));
                vel_new = vel_new + (shape_function(ip, i1) * z_wt_2 * w_new(iv_temp[i1], iz_2));
                vel_old = vel_old + (shape_function(ip, i1) * z_wt_1 * w_old(iv_temp[i1], iz_1));
                vel_old = vel_old + (shape_function(ip, i1) * z_wt_2 * w_old(iv_temp[i1], iz_2));
            }
            velocity(ip, 2) = ((it + time_offset) * vel_new + (nt - it - time_offset) * vel_old) / nt;
        }
}

void PARTICLE_TRANSPORT_FOURIER::calc_acceleration(PARAMETERS &parameters, Eigen::MatrixXd &velocity)
{
    for (int ip = 0; ip < np; ip++)
    {
        acceleration(ip, 0) = -3.0 * M_PI * parameters.mu * diameter[ip] * (velocity(ip, 0) - u_flow[ip]) / mass[ip];
        acceleration(ip, 1) = -3.0 * M_PI * parameters.mu * diameter[ip] * (velocity(ip, 1) - v_flow[ip]) / mass[ip];
        acceleration(ip, 2) = -3.0 * M_PI * parameters.mu * diameter[ip] * (velocity(ip, 2) - w_flow[ip]) / mass[ip];
    }
}

void PARTICLE_TRANSPORT_FOURIER::single_timestep(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, FRACTIONAL_STEP_FOURIER &fractional_step_fourier, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, Eigen::MatrixXd &u_new, Eigen::MatrixXd &v_new, Eigen::MatrixXd &w_new, Eigen::MatrixXd &u_old, Eigen::MatrixXd &v_old, Eigen::MatrixXd &w_old, double &physical_time)
{
    int dim = parameters.dimension;
    calc_dt(points, cloud, parameters, u_new, v_new, w_new, u_old, v_old, w_old, velocity);
    for (it = 0; it < nt; it++)
    {
        update_nearest_vert_xy_plane(points, cloud, parameters, fractional_step_fourier, position, velocity);
        calc_z_plane_range(points, cloud, parameters, fractional_step_fourier, position, velocity);
        update_particle_nb_triangle_shape_function(points, cloud, parameters, position, velocity);
        apply_boundary_conditions(points, cloud, parameters, fractional_step_fourier, position, velocity, u_new, v_new, w_new, u_old, v_old, w_old, 0.0); // old timestep
        interp_flow_velocities(points, cloud, parameters, fractional_step_fourier, position, u_new, v_new, w_new, u_old, v_old, w_old, 0.0);
        calc_acceleration(parameters, velocity);
        for (int i1 = 0; i1 < 3; i1++)
            RK_k1.col(i1) = dt * velocity.col(i1), RK_k1.col(3 + i1) = dt * acceleration.col(i1);

        for (int i1 = 0; i1 < 3; i1++)
        {
            pos_temp.col(i1) = position.col(i1) + RK_k1.col(i1);
            vel_temp.col(i1) = velocity.col(i1) + RK_k1.col(3 + i1);
        }
        update_nearest_vert_xy_plane(points, cloud, parameters, fractional_step_fourier, position, velocity);
        calc_z_plane_range(points, cloud, parameters, fractional_step_fourier, position, velocity);
        update_particle_nb_triangle_shape_function(points, cloud, parameters, pos_temp, vel_temp);
        apply_boundary_conditions(points, cloud, parameters, fractional_step_fourier, pos_temp, vel_temp, u_new, v_new, w_new, u_old, v_old, w_old, 1.0); // new timestep
        interp_flow_velocities(points, cloud, parameters, fractional_step_fourier, pos_temp, u_new, v_new, w_new, u_old, v_old, w_old, 1.0);
        calc_acceleration(parameters, vel_temp);

        for (int i1 = 0; i1 < 3; i1++)
            RK_k2.col(i1) = dt * vel_temp.col(i1), RK_k2.col(3 + i1) = dt * acceleration.col(i1);

        for (int i1 = 0; i1 < 3; i1++)
        {
            position.col(i1) = position.col(i1) + (RK_k1.col(i1) + RK_k2.col(i1)) / 2.0;
            velocity.col(i1) = velocity.col(i1) + (RK_k1.col(3 + i1) + RK_k2.col(3 + i1)) / 2.0;
        }
        physical_time = physical_time + ((it + 1.0) * dt);
    }
    it = nt - 1; // repeat this to avoid particles outside domain
    update_nearest_vert_xy_plane(points, cloud, parameters, fractional_step_fourier, position, velocity);
    calc_z_plane_range(points, cloud, parameters, fractional_step_fourier, position, velocity);
    update_particle_nb_triangle_shape_function(points, cloud, parameters, position, velocity);
    apply_boundary_conditions(points, cloud, parameters, fractional_step_fourier, position, velocity, u_new, v_new, w_new, u_old, v_old, w_old, 1.0); // new timestep
}