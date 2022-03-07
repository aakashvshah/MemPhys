from cmath import pi
import pygmsh
import numpy as np
import os
import csv
import pandas as pd
n=400
arr=np.array([np.random.uniform(0.0, 0.5, 5), np.random.uniform(0.1, 0.6, 5)]) #2x5 (modify the column number to increase the number of files generated) (First Row=ecc_ratio, Second Row=r_i)
arr2=np.array([np.random.uniform(10.0, 100.0, 5), np.random.uniform(10.0, 100.0, 5), np.random.uniform(0.1, 2.0, 5)]) #3x5 (modify the column number to increase the number of files generated) (First Row=Re_o, Second Row=Re_i, Third Row=Pr)
fields=["ecc_ratio", "r_i", "Re_o", "Re_i", "Pr"]

with open('param_table.csv', 'w', newline='') as csvfile:                                          #The parameter table creation and initialisation
    csvwriter = csv.writer(csvfile, dialect='excel', delimiter=',')
    csvwriter.writerow(fields)
    for i in range(0, 5):
        row=[str(arr[0][i]), str(arr[1][i]), str(arr2[0][i]), str(arr2[1][i]), str(arr2[2][i])]
        csvwriter.writerow(row)

read_file = pd.read_csv (r'/Users/admin/Desktop/URAP/param_table.csv')             #Needed to convert from csv to excel (maybe just local issue on my mac)
read_file.to_excel (r'/Users/admin/Desktop/URAP/param_table.xlsx', index = None, header=True)

for i in range(0, 5): #Loop for creating multiple files (Modify the range to fit the number of files generated)
    with pygmsh.occ.Geometry() as geom:
        ecc=(1-arr[1][i])*arr[0][i]                   #Creation of the data (of the one circle substracted from the other) using pygmsh
        geom.characteristic_length_max=2*pi/n
        circle1=geom.add_disk([ecc, 0.0], arr[1][i])
        circle2=geom.add_disk([0.0, 0.0], 1)
        geom.boolean_difference(circle2, circle1)
        mesh = geom.generate_mesh()
        s="test"+str(i)+".msh"                        #Creation of the mesh file and its movement into its own directory
        pygmsh.write(s)
        dir="test"+str(i)
        parent_dir="/Users/admin/Desktop/URAP/" #Replace with required diretory (makes the folder and moves the file in there)
        new_path=os.path.join(parent_dir, dir)
        os.mkdir(new_path)
        os.rename(os.path.join(parent_dir, s), os.path.join(new_path, s))
        param_file="parameter"+str(i)+".txt"    #Create the file, that contains parameters
        param_path=os.path.join(dir, param_file)
        fo = open(param_path, "w")
        param_str="\nRe_o, "+str(arr2[0][i])+"\nRe_i, "+str(arr2[1][i])+"\nPr, "+str(arr2[2][i])+"\necc_ratio, "+str(ecc)+"\nr_i, "+str(arr[1][i])
        fo.write(param_str)
        fo.close()




    
    
    
    