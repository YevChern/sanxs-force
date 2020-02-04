import numpy as np
from math import sqrt

# This script will calculate and output chi^2 and the values of the form factor at the very beginning and the end of simulation, togather with experimental data.

work_dir = "./"         # Working directory
num_points_skip = 5     # Read only every num_points_skip's point and skip the rest

sys_number = "0"        # The number of system that produce output. For example, if we chose rank "0" as the one that writes output, the file name would be "0ff_xray.dat".
relative_chi = False    # Instead of calculating a normal chi^2, normalize it not by the experimental uncertainty, but by initial value of form factor

# Set experimental data files names
xray_file_names = ["POPS_ULV@25Cin0D.xff"]
neutron_file_names = ["POPS_ULV@25Cin100D.nff", "POPS_ULV@25Cin75D.nff", "POPS_ULV@25Cin50D.nff"]

exp_files = []
for ff_type in ["xray", "neutron"]:
    if (ff_type=="xray"):
        exp_files = xray_file_names
    else:
        exp_files = neutron_file_names
        
    num_points_ofset = 0        # Skip first num_points_ofset points of the experimental data
    f_in_name1 = work_dir + sys_number + "ff_" + ff_type +".dat"
    f_in1 = open(f_in_name1,"r")
    print("Reading file " + f_in_name1)

    set_number = 0              # The column number of the data set in a file
    for exp_file_name in exp_files:
        f_out_exp_scaled = open(work_dir + "scaled" + exp_file_name, "w")
        f_in2 = open(work_dir + exp_file_name, "r")
        f_out_name = work_dir + "chi" + sys_number + exp_file_name +".dat"
        f_out = open(f_out_name, "w")

        f_in3 = open(work_dir + sys_number + "scale_" + ff_type +".dat", "r")
        scale = []
        for line in f_in3:
            if (len(line.split())>0):
                scale.append(float(line.split()[set_number]))

        q = []
        ref_val = []
        delta_ref_val = []

        for line in f_in2:
            q.append(float(line.split()[0]))
            ref_val.append(float(line.split()[1]))
            delta_ref_val.append(float(line.split()[2]))
            f_out_exp_scaled.write(str(float(line.split()[0])) + "   " + str(float(line.split()[1]) * scale[-1]) + "    " + str(float(line.split()[2]) * scale[-1]) + "\n")
            
        f_out_exp_scaled.close()
        f_in2.close()
        print("Number of q points in the data set "+ exp_file_name +" = " + str(len(q)))
                
        norm = 0.0
        first_line = ""
        
        val_number = 0
        with open(f_in_name1) as f:
            first_line = f.readline()
            for val in first_line.split():
                if ((val_number < len(q)+num_points_ofset) and (val_number >= num_points_ofset)):
                    norm += abs(float(val) - ref_val[val_number - num_points_ofset]*scale[0])
                val_number += 1     
        f_in1.seek(0)
        
        chi = 0.0
        chi_array = []
        val_array = [[]]
        line_number = 0
        buff = ""
        for line in f_in1:
            if (line_number % num_points_skip == 0):
                if (len(line.split())>0):
                    buff = line
                    tmp_array = []
                    val_number = 0
                    for val in line.split():
                        if ((val_number < len(q)+num_points_ofset) and (val_number >= num_points_ofset)):
                            tmp_array.append(float(val))
                            if (relative_chi):
                                # Normalized by initial deviation
                                chi += (abs(float(val) - ref_val[val_number - num_points_ofset]*scale[line_number])/norm)
                            else:
                                # Real chi_square (normalized by delta_ref_val)
                                chi += ((float(val) - ref_val[val_number - num_points_ofset]*scale[line_number])*(float(val) - ref_val[val_number - num_points_ofset]*scale[line_number]))/((delta_ref_val[val_number - num_points_ofset]*scale[line_number])*(delta_ref_val[val_number - num_points_ofset]*scale[line_number]))
                        val_number += 1
                    if (len(tmp_array)!=0):
                        val_array.append(tmp_array)
                    if (not relative_chi):
                        chi = sqrt(chi/(len(q)-1))
                    f_out.write(str(chi) + '\n')
                    chi_array.append(chi)
                    chi = 0.0
            line_number += 1
        f_in1.seek(0)
        f_out.close()

        f_out2_name = work_dir + "last_line"+ exp_file_name +".dat"
        f_out2 = open(f_out2_name, "w")
        first_line_array = []
        last_line_array = []
        i=0
        val_number=0
        for val in buff.split():
            if ((val_number < len(q)+num_points_ofset) and (val_number >= num_points_ofset)):
                f_out2.write(str(q[i]) + "   " + val + '\n')
                i+=1
            val_number += 1    
        f_out2.close()

        first_line_array = []
        f_out3_name = work_dir + "first_line"+ exp_file_name +".dat"
        f_out3 = open(f_out3_name, "w")
        i=0
        val_number=0
        for val in first_line.split():
            if ((val_number < len(q)+num_points_ofset) and (val_number >= num_points_ofset)):
                f_out3.write(str(q[i]) + "   " + val + '\n')
                i+=1
            val_number += 1   
            first_line_array.append(float(val))
        f_out3.close()

        num_points_ofset += len(q)
        set_number += 1

    f_in1.close()
