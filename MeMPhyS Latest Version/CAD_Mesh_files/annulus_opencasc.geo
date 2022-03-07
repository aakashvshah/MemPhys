ecc_ratio=0.5; //range: [0.0, 0.5]
r_i=0.6; //range: [0.1, 0.6]
n=200; //min value: 100

SetFactory("OpenCASCADE");
r_o=1; //fixed parameter
ecc=ecc_ratio*(r_o-r_i);
Circle(1) = {ecc, 0, 0, r_i, 0, 2*Pi};
Circle(2) = {0, 0, 0, r_o, 0, 2*Pi};
Curve Loop(1) = {1};
Curve Loop(2) = {2};
Plane Surface(1) = {1,2};
Mesh.CharacteristicLengthMax=2*Pi*r_o/n;