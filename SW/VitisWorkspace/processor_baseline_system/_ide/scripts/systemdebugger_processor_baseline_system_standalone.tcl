# Usage with Vitis IDE:
# In Vitis IDE create a Single Application Debug launch configuration,
# change the debug type to 'Attach to running target' and provide this 
# tcl script in 'Execute Script' option.
# Path of this script: C:\Users\brand\Documents\engs109-omp-fpga\SW\VitisWorkspace\processor_baseline_system\_ide\scripts\systemdebugger_processor_baseline_system_standalone.tcl
# 
# 
# Usage with xsct:
# To debug using xsct, launch xsct and run below command
# source C:\Users\brand\Documents\engs109-omp-fpga\SW\VitisWorkspace\processor_baseline_system\_ide\scripts\systemdebugger_processor_baseline_system_standalone.tcl
# 
connect -url tcp:127.0.0.1:3121
targets -set -nocase -filter {name =~"APU*"}
rst -system
after 3000
targets -set -filter {jtag_cable_name =~ "Digilent Zybo Z7 210351BE5C15A" && level==0 && jtag_device_ctx=="jsn-Zybo Z7-210351BE5C15A-23727093-0"}
fpga -file C:/Users/brand/Documents/engs109-omp-fpga/SW/VitisWorkspace/processor_baseline/_ide/bitstream/processor_alone.bit
targets -set -nocase -filter {name =~"APU*"}
loadhw -hw C:/Users/brand/Documents/engs109-omp-fpga/SW/VitisWorkspace/engs109_fpgaOMP/export/engs109_fpgaOMP/hw/processor_alone.xsa -mem-ranges [list {0x40000000 0xbfffffff}] -regs
configparams force-mem-access 1
targets -set -nocase -filter {name =~"APU*"}
source C:/Users/brand/Documents/engs109-omp-fpga/SW/VitisWorkspace/processor_baseline/_ide/psinit/ps7_init.tcl
ps7_init
ps7_post_config
targets -set -nocase -filter {name =~ "*A9*#0"}
dow C:/Users/brand/Documents/engs109-omp-fpga/SW/VitisWorkspace/processor_baseline/Debug/processor_baseline.elf
configparams force-mem-access 0
targets -set -nocase -filter {name =~ "*A9*#0"}
con
