# 
# Usage: To re-create this platform project launch xsct with below options.
# xsct C:\Users\brand\Documents\engs109-omp-fpga\SW\VitisWorkspace\engs109_fpgaOMP\platform.tcl
# 
# OR launch xsct and run below command.
# source C:\Users\brand\Documents\engs109-omp-fpga\SW\VitisWorkspace\engs109_fpgaOMP\platform.tcl
# 
# To create the platform in a different location, modify the -out option of "platform create" command.
# -out option specifies the output directory of the platform project.

platform create -name {engs109_fpgaOMP}\
-hw {C:\Users\brand\Documents\engs109-omp-fpga\SW\processor_alone.xsa}\
-proc {ps7_cortexa9_0} -os {standalone} -out {C:/Users/brand/Documents/engs109-omp-fpga/SW/VitisWorkspace}

platform write
platform generate -domains 
platform active {engs109_fpgaOMP}
platform generate
platform clean
platform generate
platform clean
platform clean
platform generate
platform clean
platform generate
platform clean
platform generate
platform clean
platform clean
platform generate
platform clean
platform generate
platform clean
platform generate
platform clean
platform generate
platform clean
platform generate
platform clean
platform clean
platform generate
platform clean
platform generate
platform clean
platform generate
platform clean
platform generate
platform clean
platform generate
