# Author: Dr. Shantanu Shahane
OBJS = general_functions.o parameters.o points.o coefficient_computations.o cloud.o solver.o fractional_step_1.o particle_transport.o semi_implicit_split_solver.o implicit_scalar_transport_solver.o postprocessing_functions.o #fractional_step_fsi.o coupled_newton.o
CC = mpicxx
DEBUG = -g
HYPRE_DIR = ../../../C_libraries/hypre-2.11.1/src/hypre#HYPRE
Eigen_directory = ../../../C_libraries/Eigen_3_3_7/ #Eigen library location
spectra_directory = ../../../C_libraries/spectra-0.8.1/include/ #Eigen library location
CFLAGS = -std=c++17 -Ofast -Wno-unused-result -march=native -I$(Eigen_directory) -I$(spectra_directory) -I$(HYPRE_DIR)/include -DHAVE_CONFIG_H -DHYPRE_TIMING -c
EFLAGS = -std=c++17
PROFLAGS = -pg #for code profiling
LIBS      = -L$(HYPRE_DIR)/lib -lHYPRE -lm
LFLAGS    = $(LIBS) -lstdc++
INCLUDE = ../../header_files

########### Dependencies started
general_functions.o : ../../header_files/general_functions.cpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/general_functions.cpp

parameters.o : ../../header_files/parameters.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/parameters.cpp

points.o : ../../header_files/points.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/points.cpp

solver.o : ../../header_files/solver.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/solver.cpp

coefficient_computations.o : ../../header_files/coefficient_computations.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/coefficient_computations.cpp

fractional_step_1.o : ../../header_files/fractional_step_1.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/fractional_step_1.cpp

particle_transport.o : ../../header_files/particle_transport.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/particle_transport.cpp

postprocessing_functions.o : ../../header_files/postprocessing_functions.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/postprocessing_functions.cpp

cloud.o : ../../header_files/cloud.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/cloud.cpp

fractional_step_fsi.o : ../../header_files/fractional_step_fsi.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/fractional_step_fsi.cpp

semi_implicit_split_solver.o : ../../header_files/semi_implicit_split_solver.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/semi_implicit_split_solver.cpp

implicit_scalar_transport_solver.o : ../../header_files/implicit_scalar_transport_solver.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/implicit_scalar_transport_solver.cpp

coupled_newton.o : ../../header_files/coupled_newton.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) ../../header_files/coupled_newton.cpp
########### Dependencies completed

########### Individual code files

kovasznay_flow.o : kovasznay_flow.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) kovasznay_flow.cpp
kovasznay_flow : $(OBJS) kovasznay_flow.o
	$(CC) $(PROFLAGS) -o out kovasznay_flow.o $(OBJS) $(LIBS)

couette_flow.o : couette_flow.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) couette_flow.cpp
couette_flow : $(OBJS) couette_flow.o
	$(CC) $(PROFLAGS) -o out couette_flow.o $(OBJS) $(LIBS)

couette_2_cylinders_ellipse.o : couette_2_cylinders_ellipse.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) couette_2_cylinders_ellipse.cpp
couette_2_cylinders_ellipse : $(OBJS) couette_2_cylinders_ellipse.o
	$(CC) $(PROFLAGS) -o out couette_2_cylinders_ellipse.o $(OBJS) $(LIBS)

couette_2_cylinders_rectangle.o : couette_2_cylinders_rectangle.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) couette_2_cylinders_rectangle.cpp
couette_2_cylinders_rectangle : $(OBJS) couette_2_cylinders_rectangle.o
	$(CC) $(PROFLAGS) -o out couette_2_cylinders_rectangle.o $(OBJS) $(LIBS)

couette_flow_coupled_newton.o : couette_flow_coupled_newton.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) couette_flow_coupled_newton.cpp
couette_flow_coupled_newton : $(OBJS) couette_flow_coupled_newton.o
	$(CC) $(PROFLAGS) -o out couette_flow_coupled_newton.o $(OBJS) $(LIBS)

eccentric_couette_flow.o : eccentric_couette_flow.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) eccentric_couette_flow.cpp
eccentric_couette_flow : $(OBJS) eccentric_couette_flow.o
	$(CC) $(PROFLAGS) -o out eccentric_couette_flow.o $(OBJS) $(LIBS)

eccentric_couette_flow_batch_ml.o : eccentric_couette_flow_batch_ml.cpp eccentric_couette_flow_batch_ml_func.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) eccentric_couette_flow_batch_ml.cpp
eccentric_couette_flow_batch_ml : $(OBJS) eccentric_couette_flow_batch_ml.o
	$(CC) $(PROFLAGS) -o out eccentric_couette_flow_batch_ml.o $(OBJS) $(LIBS)

elliptic_couette_flow.o : elliptic_couette_flow.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) elliptic_couette_flow.cpp
elliptic_couette_flow : $(OBJS) elliptic_couette_flow.o
	$(CC) $(PROFLAGS) -o out elliptic_couette_flow.o $(OBJS) $(LIBS)

flow_over_cylinder.o : flow_over_cylinder.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) flow_over_cylinder.cpp
flow_over_cylinder : $(OBJS) flow_over_cylinder.o
	$(CC) $(PROFLAGS) -o out flow_over_cylinder.o $(OBJS) $(LIBS)

flow_over_cylinder_fsi.o : flow_over_cylinder_fsi.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) flow_over_cylinder_fsi.cpp
flow_over_cylinder_fsi : $(OBJS) flow_over_cylinder_fsi.o
	$(CC) $(PROFLAGS) -o out flow_over_cylinder_fsi.o $(OBJS) $(LIBS)

backward_facing_step.o : backward_facing_step.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) backward_facing_step.cpp
backward_facing_step : $(OBJS) backward_facing_step.o
	$(CC) $(PROFLAGS) -o out backward_facing_step.o $(OBJS) $(LIBS)

driven_cavity.o : driven_cavity.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) driven_cavity.cpp
driven_cavity : $(OBJS) driven_cavity.o
	$(CC) $(PROFLAGS) -o out driven_cavity.o $(OBJS) $(LIBS)

channel_over_cavity.o : channel_over_cavity.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) channel_over_cavity.cpp
channel_over_cavity : $(OBJS) channel_over_cavity.o
	$(CC) $(PROFLAGS) -o out channel_over_cavity.o $(OBJS) $(LIBS)

natural_convection.o : natural_convection.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) natural_convection.cpp
natural_convection : $(OBJS) natural_convection.o
	$(CC) $(PROFLAGS) -o out natural_convection.o $(OBJS) $(LIBS)

channel_flow.o : channel_flow.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) channel_flow.cpp
channel_flow : $(OBJS) channel_flow.o
	$(CC) $(PROFLAGS) -o out channel_flow.o $(OBJS) $(LIBS)

channel_flow_periodic.o : channel_flow_periodic.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) channel_flow_periodic.cpp
channel_flow_periodic : $(OBJS) channel_flow_periodic.o
	$(CC) $(PROFLAGS) -o out channel_flow_periodic.o $(OBJS) $(LIBS)

bellow_periodic.o : bellow_periodic.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) bellow_periodic.cpp
bellow_periodic : $(OBJS) bellow_periodic.o
	$(CC) $(PROFLAGS) -o out bellow_periodic.o $(OBJS) $(LIBS)

bellow_in_out.o : bellow_in_out.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) bellow_in_out.cpp
bellow_in_out : $(OBJS) bellow_in_out.o
	$(CC) $(PROFLAGS) -o out bellow_in_out.o $(OBJS) $(LIBS)

taylor_green_modified.o : taylor_green_modified.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) taylor_green_modified.cpp
taylor_green_modified : $(OBJS) taylor_green_modified.o
	$(CC) $(PROFLAGS) -o out taylor_green_modified.o $(OBJS) $(LIBS)

taylor_green.o : taylor_green.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) taylor_green.cpp
taylor_green : $(OBJS) taylor_green.o
	$(CC) $(PROFLAGS) -o out taylor_green.o $(OBJS) $(LIBS)

taylor_green_steady.o : taylor_green_steady.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) taylor_green_steady.cpp
taylor_green_steady : $(OBJS) taylor_green_steady.o
	$(CC) $(PROFLAGS) -o out taylor_green_steady.o $(OBJS) $(LIBS)

bell_colella_JCP.o : bell_colella_JCP.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) bell_colella_JCP.cpp
bell_colella_JCP : $(OBJS) bell_colella_JCP.o
	$(CC) $(PROFLAGS) -o out bell_colella_JCP.o $(OBJS) $(LIBS)

bell_colella_JCP_periodic.o : bell_colella_JCP_periodic.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) bell_colella_JCP_periodic.cpp
bell_colella_JCP_periodic : $(OBJS) bell_colella_JCP_periodic.o
	$(CC) $(PROFLAGS) -o out bell_colella_JCP_periodic.o $(OBJS) $(LIBS)

tesla_valve.o : tesla_valve.cpp ../../header_files/class.hpp ../../header_files/general_functions.hpp
	$(CC) $(CFLAGS) $(LFLAGS) $(LIBS) -I $(INCLUDE) tesla_valve.cpp
tesla_valve : $(OBJS) tesla_valve.o
	$(CC) $(PROFLAGS) -o out tesla_valve.o $(OBJS) $(LIBS)

########### Individual code completed

clean:
	rm -f *.o out *.out
rm_sim_files:
	find . -maxdepth 1 ! -name 'parameters_file.csv' -name '*.csv' -delete
	rm -f *.plt