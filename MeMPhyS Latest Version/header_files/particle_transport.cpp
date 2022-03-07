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
#include "general_functions.hpp"
#include "class.hpp"
#include "nanoflann.hpp"

PARTICLE_TRANSPORT::PARTICLE_TRANSPORT(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, Eigen::VectorXd &stokes_num, Eigen::VectorXd &density, double flow_charac_time)
{
    np = position.rows();
    int dim = parameters.dimension;
    if (position.cols() != dim)
    {
        printf("\n\nERROR from PARTICLE_TRANSPORT::PARTICLE_TRANSPORT no. of columns of position: %li should be same as dimension: %i\n\n", position.cols(), dim);
        throw bad_exception();
    }
    if ((position.rows() != velocity.rows()) || (position.cols() != velocity.cols()))
    {
        printf("\n\nERROR from PARTICLE_TRANSPORT::PARTICLE_TRANSPORT shape of position: (%li, %li) should be same as shape of position: (%li, %li)\n\n", position.rows(), position.cols(), velocity.rows(), velocity.cols());
        throw bad_exception();
    }
    RK_k1 = Eigen::MatrixXd::Zero(np, 2 * dim), RK_k2 = RK_k1;
    u_flow = Eigen::VectorXd::Zero(np), v_flow = u_flow, diameter = u_flow, mass = u_flow;
    acceleration = Eigen::MatrixXd::Zero(np, dim), vel_temp = acceleration, pos_temp = vel_temp;
    for (int ip = 0; ip < np; ip++)
    {
        boundary_flag.push_back(false); //initialize to interior
        diameter[ip] = sqrt(18.0 * parameters.mu * flow_charac_time * stokes_num[ip] / density[ip]);
        mass[ip] = density[ip] * M_PI * pow(diameter[ip], 3.0) / 6.0;
    }
    // cout << "\n\nPARTICLE_TRANSPORT::PARTICLE_TRANSPORT diameter:\n"
    //      << diameter << "\n";
    // cout << "PARTICLE_TRANSPORT::PARTICLE_TRANSPORT mass:\n"
    //  << mass << "\n";

    points_xyz_nf.pts.resize(points.nv);
    int n_point_nb_point = 12; //small set of points
    point_nb_point = Eigen::MatrixXi::Zero(points.nv, n_point_nb_point);
    int n_triangles = 12; //set of triangles originating from each point
    point_triangles = Eigen::MatrixXi::Zero(points.nv, dim * n_triangles);
    point_triangles_2_area = Eigen::MatrixXd::Zero(points.nv, n_triangles);
    particle_nb_triangle = Eigen::MatrixXi::Zero(np, dim + 1);
    calc_neighbors(points, cloud, parameters, position);
    calc_triangles(points, cloud, parameters);
    shape_function = Eigen::MatrixXd::Zero(np, dim + 1);
    // write_csv(particle_nb_triangle, "particle_nb_triangle.csv");
    // // write_csv(point_nb_point, "point_nb_point.csv");
}

void PARTICLE_TRANSPORT::calc_neighbors(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position)
{
    int dim = parameters.dimension;
    for (int iv = 0; iv < points.nv; iv++)
    {
        points_xyz_nf.pts[iv].x = points.xyz[dim * iv];
        points_xyz_nf.pts[iv].y = points.xyz[dim * iv + 1];
        if (dim == 3)
            points_xyz_nf.pts[iv].z = points.xyz[dim * iv + 2];
        else
            points_xyz_nf.pts[iv].z = 0.0; //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
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
            query_pt[2] = 0.0; //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
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
            query_pt[2] = 0.0; //does not accept dimension as a parameter in KDTreeSingleIndexAdaptor and index
        index_nf.knnSearch(&query_pt[0], dim + 1, &nb_vert[0], &nb_dist[0]);
        for (int i1 = 0; i1 < dim + 1; i1++)
            particle_nb_triangle(ip, i1) = nb_vert[i1];
    }
}

void PARTICLE_TRANSPORT::calc_triangles(POINTS &points, CLOUD &cloud, PARAMETERS &parameters)
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
        for (int i1 = 1; i1 < point_nb_point.cols(); i1++) //i1=0 is the point iv
            for (int i2 = i1 + 1; i2 < point_nb_point.cols(); i2++)
            {
                iv1 = point_nb_point(iv, i1), iv2 = point_nb_point(iv, i2);
                triangle_vert[dim * index] = iv1, triangle_vert[dim * index + 1] = iv2;
                dot_prod = 0;
                for (int i3 = 0; i3 < dim; i3++)
                    dot_prod = dot_prod + ((points.xyz[dim * iv1 + i3] - points.xyz[dim * iv + i3]) * (points.xyz[dim * iv2 + i3] - points.xyz[dim * iv + i3]));
                // if ((fabs(points.xyz[dim * iv1 + 1]) < 0.01) || (fabs(points.xyz[dim * iv2 + 1]) < 0.01))
                //     if (iv == 728)
                //     {
                //         printf("\nPARTICLE_TRANSPORT::calc_triangles iv: %i, dot_prod: %g, index: %i\n", iv, dot_prod, index);
                //         cout << "triangle point: " << iv << ": (" << points.xyz[dim * iv] << ", " << points.xyz[dim * iv + 1] << ")" << endl;
                //         cout << "triangle point: " << iv1 << ": (" << points.xyz[dim * iv1] << ", " << points.xyz[dim * iv1 + 1] << ")" << endl;
                //         cout << "triangle point: " << iv2 << ": (" << points.xyz[dim * iv2] << ", " << points.xyz[dim * iv2 + 1] << ")" << endl;
                //     }

                if (dot_prod >= 0)
                { //acute or right angle is formed
                    // triangle_area[index] = calc_triangle_area_double(points.xyz[dim * iv], points.xyz[dim * iv + 1], points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], points.xyz[dim * iv2], points.xyz[dim * iv2 + 1]);
                    dist1 = pow(points.xyz[dim * iv] - points.xyz[dim * iv1], 2.0) + pow(points.xyz[dim * iv + 1] - points.xyz[dim * iv1 + 1], 2.0);
                    dist2 = pow(points.xyz[dim * iv] - points.xyz[dim * iv2], 2.0) + pow(points.xyz[dim * iv + 1] - points.xyz[dim * iv2 + 1], 2.0);
                    max_distance_sq[index] = dist1;
                    if (max_distance_sq[index] < dist2)
                        max_distance_sq[index] = dist2;
                    count++;
                }
                else
                    max_distance_sq[index] = INFINITY; //set to remove obtuse traingles later
                // triangle_area[index] = INFINITY; //set to remove obtuse traingles later
                index++;
            }

        // print_to_terminal(triangle_area, "triangle_area from PARTICLE_TRANSPORT::calc_triangles");
        //reference: https://stackoverflow.com/a/40183830/6932587
        iota(argsort.begin(), argsort.end(), 0); //Initializing
        sort(argsort.begin(), argsort.end(), [&](int i, int j)
             { return max_distance_sq[i] < max_distance_sq[j]; });
        // print_to_terminal(argsort, "argsort from PARTICLE_TRANSPORT::calc_triangles");
        // print_to_terminal(triangle_area, "triangle_area from PARTICLE_TRANSPORT::calc_triangles");

        if (count < n_triangles)
        {
            printf("\n\nERROR from PARTICLE_TRANSPORT::calc_triangles iv: %i, found only %i candidates of acute angled triangles; need atleast %i triangles\n", iv, count, n_triangles);
            cout << "point_nb_point.row(iv): " << point_nb_point.row(iv) << "\n";
            print_to_terminal(triangle_vert, n_search, dim, "triangle_vert from PARTICLE_TRANSPORT::calc_triangles");
            print_to_terminal(max_distance_sq, "max_distance_sq from PARTICLE_TRANSPORT::calc_triangles");
            throw bad_exception();
        }

        for (int i1 = 0; i1 < n_triangles; i1++)
        {
            iv1 = triangle_vert[dim * argsort[i1]], iv2 = triangle_vert[dim * argsort[i1] + 1];
            // point_triangles_2_area(iv, i1) = triangle_area[argsort[i1]];
            point_triangles_2_area(iv, i1) = calc_triangle_area_double(points.xyz[dim * iv], points.xyz[dim * iv + 1], points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], points.xyz[dim * iv2], points.xyz[dim * iv2 + 1]);
            point_triangles(iv, dim * i1) = iv1, point_triangles(iv, dim * i1 + 1) = iv2;
        }
    }
    // write_csv(point_triangles_2_area, "point_triangles_2_area.csv");
    // write_csv(point_triangles, "point_triangles.csv");
}

void PARTICLE_TRANSPORT::update_nearest_vert(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity)
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
            if (min_dist_sq > dist_sq) //found a closer point
                particle_nb_triangle(ip, 0) = iv_new;
        }
    }
}

void PARTICLE_TRANSPORT::update_particle_nb_triangle_shape_function(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity)
{
    if (parameters.dimension != 2)
    {
        printf("\n\nERROR from PARTICLE_TRANSPORT::apply_boundary_conditions defined only for 2D problems; dimension: %i\n\n", parameters.dimension);
        throw bad_exception();
    }
    int n_triangles = point_triangles_2_area.cols(), iv1, iv2, iv3, dim = parameters.dimension;
    bool found_triangle;
    double area, area1, area2, area3;
    for (int ip = 0; ip < np; ip++)
    {
        found_triangle = false, boundary_flag[ip] = true;
        iv1 = particle_nb_triangle(ip, 0); //nearest point to particle ip
        for (int itr = 0; itr < n_triangles; itr++)
        {
            iv2 = point_triangles(iv1, dim * itr), iv3 = point_triangles(iv1, dim * itr + 1);
            // if (ip == 6109)
            // {
            //     printf("\nPARTICLE_TRANSPORT::update_particle_nb_triangle_shape_function ip: %i, particle_coordinates: (%g, %g)\n", ip, position(ip, 0), position(ip, 1));
            //     cout << "triangle point: " << iv1 << ": (" << points.xyz[dim * iv1] << ", " << points.xyz[dim * iv1 + 1] << ")" << endl;
            //     cout << "triangle point: " << iv2 << ": (" << points.xyz[dim * iv2] << ", " << points.xyz[dim * iv2 + 1] << ")" << endl;
            //     cout << "triangle point: " << iv3 << ": (" << points.xyz[dim * iv3] << ", " << points.xyz[dim * iv3 + 1] << ")" << endl;
            // }
            area = point_triangles_2_area(iv1, itr);
            area1 = calc_triangle_area_double(position(ip, 0), position(ip, 1), points.xyz[dim * iv2], points.xyz[dim * iv2 + 1], points.xyz[dim * iv3], points.xyz[dim * iv3 + 1]);
            area2 = calc_triangle_area_double(position(ip, 0), position(ip, 1), points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], points.xyz[dim * iv3], points.xyz[dim * iv3 + 1]);
            area3 = calc_triangle_area_double(position(ip, 0), position(ip, 1), points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], points.xyz[dim * iv2], points.xyz[dim * iv2 + 1]);
            if (area1 + area2 + area3 <= (area + 1E-10))
            { //found a triangle with particle in the interior
                particle_nb_triangle(ip, 1) = iv2, particle_nb_triangle(ip, 2) = iv3;
                shape_function(ip, 0) = area1 / area;
                shape_function(ip, 1) = area2 / area;
                shape_function(ip, 2) = area3 / area;
                found_triangle = true, boundary_flag[ip] = false;
                break;
            }
        }
        if (!found_triangle)
        { //particle does not lie inside any triangle incident on nearest point iv1
            if (!points.boundary_flag[iv1])
            { //nearest point iv1 is inside the domain
                // printf("\n\nPARTICLE_TRANSPORT::update_particle_nb_triangle_shape_function ip: %i, particle_coordinates: (%g, %g)\n", ip, position(ip, 0), position(ip, 1));
                // printf("nearest iv1: %i, points.boundary_flag[iv1]: %i, points.xyz[iv1]: (%g, %g)\n", iv1, (int)(points.boundary_flag[iv1]), points.xyz[dim * iv1], points.xyz[dim * iv1 + 1]);
                // printf("Nearest point is inside the domain but the particle lies outside all the triangles incident on nearest point\n\n");
                // throw bad_exception();

                //recheck this: can happen when some corner point from GMSH grid is deleted: thus that triangle is not defined
                boundary_flag[ip] = false; //particle is inside domain boundary
                shape_function(ip, 0) = 1.0, shape_function(ip, 1) = 0.0, shape_function(ip, 2) = 0.0;
                particle_nb_triangle(ip, 1) = point_triangles(iv1, 0); //smallest triangle
                particle_nb_triangle(ip, 2) = point_triangles(iv1, 1); //smallest triangle
            }
            else
            {                             //nearest point iv1 is on the domain boundary
                boundary_flag[ip] = true; //particle is outside domain boundary
                shape_function(ip, 0) = 1.0, shape_function(ip, 1) = 0.0, shape_function(ip, 2) = 0.0;
                particle_nb_triangle(ip, 1) = point_triangles(iv1, 0); //smallest triangle
                particle_nb_triangle(ip, 2) = point_triangles(iv1, 1); //smallest triangle
            }
        }
    }
}

// void PARTICLE_TRANSPORT::update_neighbors(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity)
// {
//     vector<int> nb_vert(particle_nb_triangle.cols() * point_nb_point.cols());
//     vector<int>::iterator vec_iter;
//     vector<double> nb_vert_dist_sq(particle_nb_triangle.cols() * point_nb_point.cols());
//     int unique_nb_size, index, dim = parameters.dimension;
//     for (int ip = 0; ip < np; ip++)
//     {
//         // if (ip == 36)
//         //     cout << "\nip: " << ip << " x: " << position(ip, 0) << " y: " << position(ip, 1);
//         index = 0;
//         for (int i1 = 0; i1 < particle_nb_triangle.cols(); i1++)
//             for (int i2 = 0; i2 < point_nb_point.cols(); i2++)
//                 nb_vert[index] = point_nb_point(particle_nb_triangle(ip, i1), i2), index++;
//         // if (ip == 36)
//         // {
//         //     cout << "\nPARTICLE_TRANSPORT::update_neighbors particle_nb_triangle.row(ip) " << particle_nb_triangle.row(ip) << "\n";
//         // }
//         sort(nb_vert.begin(), nb_vert.end());
//         vec_iter = std::unique(nb_vert.begin(), nb_vert.end());
//         unique_nb_size = distance(nb_vert.begin(), vec_iter); //no. of unique neighbors
//         // print_to_terminal(nb_vert, "PARTICLE_TRANSPORT::update_neighbors nb_vert");
//         // cout << "ip: " << ip << " unique_nb_size: " << unique_nb_size << "\n";
//         // cout << "(nb_vert[i1], dist)";
//         for (int i1 = 0; i1 < unique_nb_size; i1++)
//         {
//             nb_vert_dist_sq[i1] = 0.0;
//             for (int i2 = 0; i2 < dim; i2++)
//                 nb_vert_dist_sq[i1] = nb_vert_dist_sq[i1] + pow(position(ip, i2) - points.xyz[dim * nb_vert[i1] + i2], 2.0);
//             // cout << " (" << nb_vert[i1] << ", " << nb_vert_dist_sq[i1] << ") ";
//         }
//         // cout << "\n";
//         for (int i1 = 0; i1 < particle_nb_triangle.cols(); i1++)
//         {
//             index = distance(nb_vert_dist_sq.begin(), min_element(nb_vert_dist_sq.begin(), nb_vert_dist_sq.begin() + unique_nb_size));
//             particle_nb_triangle(ip, i1) = nb_vert[index];
//             nb_vert_dist_sq[index] = numeric_limits<double>::infinity(); //next minimum will not include this element as its set to infinity
//         }
//         // cout << "\n/////////////////////////\n";
//     }
// }

double PARTICLE_TRANSPORT::calc_triangle_area_double(double x1, double y1, double x2, double y2, double x3, double y3)
{ //this gives double of area as division by 2.0 is not performed here
    return fabs(((x1 - x2) * (y1 - y3)) - ((y1 - y2) * (x1 - x3)));
}

// void PARTICLE_TRANSPORT::calc_shape_function(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity)
// { //https://www.sciencedirect.com/topics/engineering/linear-triangular-element
//     //http://www.unm.edu/~bgreen/ME360/2D%20Triangular%20Elements.pdf
//     if (parameters.dimension != 2)
//     {
//         printf("\n\nERROR from PARTICLE_TRANSPORT::calc_shape_function defined only for 2D problems; dimension: %i\n\n", parameters.dimension);
//         throw bad_exception();
//     }
//     int dim = parameters.dimension, iv0, iv1, iv2;
//     // double x1, y1, x2, y2, x3, y3, xp, yp;
//     // double a1, a2, a3, b1, b2, b3, c1, c2, c3, area_twice;
//     double area, area0, area1, area2;
//     for (int ip = 0; ip < np; ip++)
//     {
//         // iv1 = particle_nb_triangle(ip, 0), iv2 = particle_nb_triangle(ip, 1), iv3 = particle_nb_triangle(ip, 2);
//         // x1 = points.xyz[dim * iv1], y1 = points.xyz[dim * iv1 + 1];
//         // x2 = points.xyz[dim * iv2], y2 = points.xyz[dim * iv2 + 1];
//         // x3 = points.xyz[dim * iv3], y3 = points.xyz[dim * iv3 + 1];
//         // xp = position(ip, 0), yp = position(ip, 1);
//         // a1 = x2 * y3 - x3 * y2, b1 = y2 - y3, c1 = x3 - x2;
//         // a2 = x3 * y1 - x1 * y3, b2 = y3 - y1, c2 = x1 - x3;
//         // a3 = x1 * y2 - x2 * y1, b3 = y1 - y2, c3 = x2 - x1;
//         // area_twice = a1 + a2 + a3;
//         // shape_function(ip, 0) = (a1 + b1 * xp + c1 * yp) / area_twice;
//         // shape_function(ip, 1) = (a2 + b2 * xp + c2 * yp) / area_twice;
//         // shape_function(ip, 2) = (a3 + b3 * xp + c3 * yp) / area_twice;

//         iv0 = particle_nb_triangle(ip, 0), iv1 = particle_nb_triangle(ip, 1), iv2 = particle_nb_triangle(ip, 2);
//         area = calc_triangle_area_double(points.xyz[dim * iv0], points.xyz[dim * iv0 + 1], points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], points.xyz[dim * iv2], points.xyz[dim * iv2 + 1]);
//         area0 = calc_triangle_area_double(position(ip, 0), position(ip, 1), points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], points.xyz[dim * iv2], points.xyz[dim * iv2 + 1]);
//         area1 = calc_triangle_area_double(points.xyz[dim * iv0], points.xyz[dim * iv0 + 1], position(ip, 0), position(ip, 1), points.xyz[dim * iv2], points.xyz[dim * iv2 + 1]);
//         area2 = calc_triangle_area_double(points.xyz[dim * iv0], points.xyz[dim * iv0 + 1], points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], position(ip, 0), position(ip, 1));
//         shape_function(ip, 0) = area0 / area;
//         shape_function(ip, 1) = area1 / area;
//         shape_function(ip, 2) = area2 / area;
//         // if (ip == 36)
//         // {
//         //     cout << "\nPARTICLE_TRANSPORT::calc_shape_function ip: " << ip << "\n";
//         //     cout << "\nPARTICLE_TRANSPORT::calc_shape_function particle_nb_triangle.row(ip) " << particle_nb_triangle.row(ip) << "\n";
//         //     cout << "PARTICLE_TRANSPORT::calc_shape_function shape_function: " << shape_function.row(ip) << "\n";
//         // }
//     }
//     // cout << "\n\nPARTICLE_TRANSPORT::calc_shape_function shape_function:\n"
//     //      << shape_function << "\n";
//     // cout << "PARTICLE_TRANSPORT::calc_shape_function shape_function row_sum:\n"
//     //      << shape_function.rowwise().sum() << "\n";
//     // cout << "PARTICLE_TRANSPORT::calc_shape_function particle_nb_triangle:\n"
//     //      << particle_nb_triangle << "\n\n";
// }

void PARTICLE_TRANSPORT::apply_boundary_conditions(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, double time_offset)
{
    if (parameters.dimension != 2)
    {
        printf("\n\nERROR from PARTICLE_TRANSPORT::apply_boundary_conditions defined only for 2D problems; dimension: %i\n\n", parameters.dimension);
        throw bad_exception();
    }
    // int dim = parameters.dimension, unique_nb_size, index, iv0, iv1, iv2, boundary_flag_sum;
    // double shape_func_sum, area, area0, area1, area2;
    // vector<int> nb_vert(dim * point_nb_point.cols()); //keep nearest [dim] points fixed; modify only the last point
    // vector<int>::iterator vec_iter;
    // // cout << "\n\n";
    // for (int ip = 0; ip < np; ip++)
    // {
    //     boundary_flag[ip] = true; //initialize
    //     shape_func_sum = shape_function.row(ip).sum();
    //     if (shape_func_sum > (1.0 + 1E-10))
    //     {
    //         // cout << "PARTICLE_TRANSPORT::apply_boundary_conditions ip: " << ip << "\n";
    //         index = 0;
    //         for (int i1 = 0; i1 < dim; i1++) //keep nearest [dim] points fixed
    //             for (int i2 = 0; i2 < point_nb_point.cols(); i2++)
    //                 nb_vert[index] = point_nb_point(particle_nb_triangle(ip, i1), i2), index++;
    //         for (int i1 = 0; i1 < nb_vert.size(); i1++)
    //         {
    //             for (int i2 = 0; i2 < particle_nb_triangle.cols(); i2++)
    //                 if (nb_vert[i1] == particle_nb_triangle(ip, i2))
    //                     nb_vert[i1] = -10; //existing points set to a negative integer
    //         }
    //         sort(nb_vert.begin(), nb_vert.end());
    //         // if (ip == 36)
    //         // {
    //         //     print_to_terminal(nb_vert, "PARTICLE_TRANSPORT::apply_boundary_conditions nb_vert");
    //         //     cout << "ip: " << ip << " unique_nb_size: " << unique_nb_size << "\n";
    //         // }
    //         vec_iter = std::unique(nb_vert.begin(), nb_vert.end());
    //         unique_nb_size = distance(nb_vert.begin(), vec_iter); //no. of unique neighbors
    //         // if (ip == 36)
    //         // {
    //         //     print_to_terminal(nb_vert, "PARTICLE_TRANSPORT::apply_boundary_conditions nb_vert");
    //         //     cout << "ip: " << ip << " unique_nb_size: " << unique_nb_size << "\n";
    //         // }
    //         iv0 = particle_nb_triangle(ip, 0), iv1 = particle_nb_triangle(ip, 1); //nearest 2 keep same
    //         for (int i1 = 0; i1 < unique_nb_size; i1++)
    //             if (nb_vert[i1] >= 0)
    //             {                      //existing points set to a negative integer
    //                 iv2 = nb_vert[i1]; //new candidate
    //                 area = calc_triangle_area_double(points.xyz[dim * iv0], points.xyz[dim * iv0 + 1], points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], points.xyz[dim * iv2], points.xyz[dim * iv2 + 1]);
    //                 area0 = calc_triangle_area_double(position(ip, 0), position(ip, 1), points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], points.xyz[dim * iv2], points.xyz[dim * iv2 + 1]);
    //                 area1 = calc_triangle_area_double(points.xyz[dim * iv0], points.xyz[dim * iv0 + 1], position(ip, 0), position(ip, 1), points.xyz[dim * iv2], points.xyz[dim * iv2 + 1]);
    //                 area2 = calc_triangle_area_double(points.xyz[dim * iv0], points.xyz[dim * iv0 + 1], points.xyz[dim * iv1], points.xyz[dim * iv1 + 1], position(ip, 0), position(ip, 1));
    //                 if (area0 + area1 + area2 <= (area + 1E-10))
    //                 { //found a triangle with particle in the interior
    //                     particle_nb_triangle(ip, 2) = iv2;
    //                     boundary_flag[ip] = false; //particle is inside the triangulation
    //                     shape_function(ip, 0) = area0 / area;
    //                     shape_function(ip, 1) = area1 / area;
    //                     shape_function(ip, 2) = area2 / area;
    //                     break;
    //                 }
    //             }

    //         // if (ip == 36)
    //         // {
    //         //     cout << "\nPARTICLE_TRANSPORT::update_neighbors particle_nb_triangle.row(ip) " << particle_nb_triangle.row(ip) << "\n";
    //         //     cout << "PARTICLE_TRANSPORT::update_neighbors shape_function: " << shape_function.row(ip) << "\n";
    //         //     printf("PARTICLE_TRANSPORT::apply_boundary_conditions ip: %i, coordinates: (%g, %g) shape_function: (%g, %g, %g), shape_func_sum: %g\n", ip, position(ip, 0), position(ip, 1), shape_function(ip, 0), shape_function(ip, 1), shape_function(ip, 2), shape_function.row(ip).sum());
    //         //     for (int i1 = 0; i1 < particle_nb_triangle.cols(); i1++)
    //         //         printf("Triangle point: %i, coodinates: (%g, %g)\n", particle_nb_triangle(ip, i1), points.xyz[dim * particle_nb_triangle(ip, i1)], points.xyz[dim * particle_nb_triangle(ip, i1) + 1]);
    //         // }
    //     }
    //     else
    //         boundary_flag[ip] = false; //particle is inside the triangulation
    // }
    // for (int ip = 0; ip < np; ip++)
    // { //only to verify that interior particles are correctly triangulated
    //     boundary_flag_sum = 0;
    //     for (int i1 = 0; i1 < particle_nb_triangle.cols(); i1++)
    //         boundary_flag_sum = boundary_flag_sum + ((int)points.boundary_flag[particle_nb_triangle(ip, i1)]);
    //     shape_func_sum = shape_function.row(ip).sum();
    //     if ((boundary_flag_sum == 0) && (shape_func_sum >= (1 + 1E-10)))
    //     { //particle lies outside a completely interior triangle: throw error.
    //         printf("\n\nPARTICLE_TRANSPORT::apply_boundary_conditions ip: %i, particle_boundary_flag: %i, coordinates: (%g, %g) shape_function: (%g, %g, %g), shape_func_sum: %g\n", ip, (int)(boundary_flag[ip]), position(ip, 0), position(ip, 1), shape_function(ip, 0), shape_function(ip, 1), shape_function(ip, 2), shape_func_sum);
    //         for (int i1 = 0; i1 < particle_nb_triangle.cols(); i1++)
    //             printf("Triangle point: %i, coodinates: (%g, %g)\n", particle_nb_triangle(ip, i1), points.xyz[dim * particle_nb_triangle(ip, i1)], points.xyz[dim * particle_nb_triangle(ip, i1) + 1]);
    //         printf("Above triangle is completely interior but the particle lies outside it\n\n");
    //         throw bad_exception();
    //     }
    // }
    // cout << "\n\n";
    // vector<int> iv_temp(particle_nb_triangle.cols());
    // double shape_func_sum, vel_new, vel_old;
    // for (int ip = 0; ip < np; ip++)
    // {
    //     boundary_flag_sum = 0;
    //     for (int i1 = 0; i1 < particle_nb_triangle.cols(); i1++)
    //     {
    //         iv_temp[i1] = particle_nb_triangle(ip, i1);
    //         boundary_flag_sum = boundary_flag_sum + ((int)points.boundary_flag[iv_temp[i1]]);
    //     }
    //     shape_func_sum = shape_function.row(ip).sum();
    //     if (shape_func_sum >= (1 + 1E-6))
    //     { //particle lies outside triangle: assign values of nearest flow point
    //         if (boundary_flag_sum == 0)
    //         { //triangle is completely interior: throw error
    //             printf("\n\nPARTICLE_TRANSPORT::apply_boundary_conditions ip: %i, coordinates: (%g, %g) shape_function: (%g, %g, %g), shape_func_sum: %g\n", ip, position(ip, 0), position(ip, 1), shape_function(ip, 0), shape_function(ip, 1), shape_function(ip, 2), shape_func_sum);
    //             for (int i1 = 0; i1 < particle_nb_triangle.cols(); i1++)
    //                 printf("Triangle point: %i, coodinates: (%g, %g)\n", iv_temp[i1], points.xyz[dim * iv_temp[i1]], points.xyz[dim * iv_temp[i1] + 1]);
    //             printf("Above triangle is completely interior but the particle lies outside it\n\n");
    //             throw bad_exception();
    //         }
    //         boundary_flag[ip] = true; //particle is outside a boundary triangle; i.e., outside domain
    //         //assign values of nearest flow point:
    //         shape_function(ip, 0) = 1.0, shape_function(ip, 1) = 0.0, shape_function(ip, 2) = 0.0;
    //     }
    //     if (boundary_flag_sum == 3)
    //     { //particle is inside a triangle with all 3 vertices on boundary; i.e., particle is outside domain
    //         boundary_flag[ip] = true;
    //         //assign values of nearest flow point:
    //         shape_function(ip, 0) = 1.0, shape_function(ip, 1) = 0.0, shape_function(ip, 2) = 0.0;
    //     }
    //     else if (boundary_flag_sum == 2)
    //     { //2 out of 3 triangle points lie on boundary
    //         for (int i1 = 0; i1 < particle_nb_triangle.cols(); i1++)
    //             if (!points.boundary_flag[iv_temp[i1]] && (shape_function(ip, i1) < 1E-6))
    //             { //iv_temp[i1] is interior and its shape function is zero; so particle lies on boundary egde defined by remaining 2 points
    //                 boundary_flag[ip] = true;
    //                 break;
    //             }
    //     }
    //     else if (boundary_flag_sum == 1)
    //     { //1 out of 3 triangle points lie on boundary
    //         for (int i1 = 0; i1 < particle_nb_triangle.cols(); i1++)
    //             if (points.boundary_flag[iv_temp[i1]] && ((1.0 - shape_function(ip, i1)) < 1E-6))
    //             { //iv_temp[i1] is boundary and its shape function is one; so particle lies on boundary point
    //                 boundary_flag[ip] = true;
    //                 break;
    //             }
    //     }
    // }

    double vel_new, vel_old;
    int dim = parameters.dimension;
    vector<int> iv_temp(particle_nb_triangle.cols());
    for (int ip = 0; ip < np; ip++)
        if (boundary_flag[ip])
        { //particle lies on the boundary: set position and velocity
            vel_new = 0.0, vel_old = 0.0;
            position(ip, 0) = 0.0, position(ip, 1) = 0.0;
            for (int i1 = 0; i1 < particle_nb_triangle.cols(); i1++)
                iv_temp[i1] = particle_nb_triangle(ip, i1);
            for (int i1 = 0; i1 < dim + 1; i1++)
            {
                position(ip, 0) = position(ip, 0) + (shape_function(ip, i1) * points.xyz[dim * iv_temp[i1]]);
                position(ip, 1) = position(ip, 1) + (shape_function(ip, i1) * points.xyz[dim * iv_temp[i1] + 1]);
                vel_new = vel_new + (shape_function(ip, i1) * u_new[iv_temp[i1]]);
                vel_old = vel_old + (shape_function(ip, i1) * u_old[iv_temp[i1]]);
            }
            velocity(ip, 0) = ((it + time_offset) * vel_new + (nt - it - time_offset) * vel_old) / nt;
            vel_new = 0.0, vel_old = 0.0;
            for (int i1 = 0; i1 < dim + 1; i1++)
            {
                vel_new = vel_new + (shape_function(ip, i1) * v_new[iv_temp[i1]]);
                vel_old = vel_old + (shape_function(ip, i1) * v_old[iv_temp[i1]]);
            }
            velocity(ip, 1) = ((it + time_offset) * vel_new + (nt - it - time_offset) * vel_old) / nt;
        }
}

void PARTICLE_TRANSPORT::calc_dt(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, Eigen::MatrixXd &velocity)
{
    double u_max, v_max, temp1, temp2, temp3, speed_max;
    temp1 = u_new.lpNorm<Eigen::Infinity>(), temp2 = u_old.lpNorm<Eigen::Infinity>();
    temp3 = velocity.col(0).lpNorm<Eigen::Infinity>();
    u_max = max(temp1, max(temp2, temp3));
    temp1 = v_new.lpNorm<Eigen::Infinity>(), temp2 = v_old.lpNorm<Eigen::Infinity>();
    temp3 = velocity.col(1).lpNorm<Eigen::Infinity>();
    v_max = max(temp1, max(temp2, temp3));
    speed_max = sqrt((u_max * u_max) + (v_max * v_max));
    dt = parameters.min_dx / speed_max;
    // cout << "\nPARTICLE_TRANSPORT::calc_dt particle speed_max: " << speed_max << ", dt: " << dt << ", flow_dt: " << parameters.dt << "\n\n";
    if (dt >= parameters.dt)
        dt = parameters.dt, nt = 1;
    else
        nt = (int)(parameters.dt / dt), nt++, dt = parameters.dt / nt;
    // cout << "\nPARTICLE_TRANSPORT::calc_dt particle speed_max: " << speed_max << ", dt: " << dt << ", flow_dt: " << parameters.dt << ", nt: " << nt << "\n\n";
}

void PARTICLE_TRANSPORT::interp_flow_velocities(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, double time_offset)
{
    int dim = parameters.dimension;
    double shape_func_sum, vel_new, vel_old;
    for (int ip = 0; ip < np; ip++)
    {
        vel_new = 0.0, vel_old = 0.0;
        for (int i1 = 0; i1 < dim + 1; i1++)
        {
            vel_new = vel_new + (shape_function(ip, i1) * u_new[particle_nb_triangle(ip, i1)]);
            vel_old = vel_old + (shape_function(ip, i1) * u_old[particle_nb_triangle(ip, i1)]);
        }
        u_flow[ip] = ((it + time_offset) * vel_new + (nt - it - time_offset) * vel_old) / nt;
        vel_new = 0.0, vel_old = 0.0;
        for (int i1 = 0; i1 < dim + 1; i1++)
        {
            vel_new = vel_new + (shape_function(ip, i1) * v_new[particle_nb_triangle(ip, i1)]);
            vel_old = vel_old + (shape_function(ip, i1) * v_old[particle_nb_triangle(ip, i1)]);
        }
        v_flow[ip] = ((it + time_offset) * vel_new + (nt - it - time_offset) * vel_old) / nt;
    }
}

void PARTICLE_TRANSPORT::calc_acceleration(PARAMETERS &parameters, Eigen::MatrixXd &velocity)
{
    if (parameters.dimension != 2)
    {
        printf("\n\nERROR from PARTICLE_TRANSPORT::calc_acceleration defined only for 2D problems; dimension: %i\n\n", parameters.dimension);
        throw bad_exception();
    }
    for (int ip = 0; ip < np; ip++)
    {
        acceleration(ip, 0) = -3.0 * M_PI * parameters.mu * diameter[ip] * (velocity(ip, 0) - u_flow[ip]) / mass[ip];
        acceleration(ip, 1) = -3.0 * M_PI * parameters.mu * diameter[ip] * (velocity(ip, 1) - v_flow[ip]) / mass[ip];
    }
}

void PARTICLE_TRANSPORT::single_timestep(POINTS &points, CLOUD &cloud, PARAMETERS &parameters, Eigen::MatrixXd &position, Eigen::MatrixXd &velocity, Eigen::VectorXd &u_new, Eigen::VectorXd &v_new, Eigen::VectorXd &u_old, Eigen::VectorXd &v_old, double &physical_time)
{
    if (parameters.dimension != 2)
    {
        printf("\n\nERROR from PARTICLE_TRANSPORT::single_timestep defined only for 2D problems; dimension: %i\n\n", parameters.dimension);
        throw bad_exception();
    }
    int dim = parameters.dimension;
    calc_dt(points, cloud, parameters, u_new, v_new, u_old, v_old, velocity);
    for (it = 0; it < nt; it++)
    {
        update_nearest_vert(points, cloud, parameters, position, velocity);
        update_particle_nb_triangle_shape_function(points, cloud, parameters, position, velocity);
        // calc_shape_function(points, cloud, parameters, position, velocity);
        apply_boundary_conditions(points, cloud, parameters, position, velocity, u_new, v_new, u_old, v_old, 0.0); //old timestep
        interp_flow_velocities(points, cloud, parameters, u_new, v_new, u_old, v_old, 0.0);
        calc_acceleration(parameters, velocity);
        for (int i1 = 0; i1 < dim; i1++)
            RK_k1.col(i1) = dt * velocity.col(i1), RK_k1.col(dim + i1) = dt * acceleration.col(i1);

        for (int i1 = 0; i1 < dim; i1++)
        {
            pos_temp.col(i1) = position.col(i1) + RK_k1.col(i1);
            vel_temp.col(i1) = velocity.col(i1) + RK_k1.col(dim + i1);
        }
        update_nearest_vert(points, cloud, parameters, position, velocity);
        update_particle_nb_triangle_shape_function(points, cloud, parameters, pos_temp, vel_temp);
        // calc_shape_function(points, cloud, parameters, pos_temp, vel_temp);
        apply_boundary_conditions(points, cloud, parameters, pos_temp, vel_temp, u_new, v_new, u_old, v_old, 1.0); //new timestep
        interp_flow_velocities(points, cloud, parameters, u_new, v_new, u_old, v_old, 1.0);
        calc_acceleration(parameters, vel_temp);

        for (int i1 = 0; i1 < dim; i1++)
            RK_k2.col(i1) = dt * vel_temp.col(i1), RK_k2.col(dim + i1) = dt * acceleration.col(i1);

        for (int i1 = 0; i1 < dim; i1++)
        {
            position.col(i1) = position.col(i1) + (RK_k1.col(i1) + RK_k2.col(i1)) / 2.0;
            velocity.col(i1) = velocity.col(i1) + (RK_k1.col(dim + i1) + RK_k2.col(dim + i1)) / 2.0;
        }
        physical_time = physical_time + ((it + 1.0) * dt);
    }
    it = nt - 1; //repeat this to avoid particles outside domain
    update_nearest_vert(points, cloud, parameters, position, velocity);
    update_particle_nb_triangle_shape_function(points, cloud, parameters, pos_temp, vel_temp);
    // calc_shape_function(points, cloud, parameters, pos_temp, vel_temp);
    apply_boundary_conditions(points, cloud, parameters, pos_temp, vel_temp, u_new, v_new, u_old, v_old, 1.0); //new timestep
}