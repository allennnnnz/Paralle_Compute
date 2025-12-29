make clean
make
srun -p nvidia nvprof --metrics \
flop_count_sp,flop_count_sp_add,flop_count_sp_mul,flop_count_sp_fma,\
sm_efficiency,achieved_occupancy,ipc,inst_integer,inst_executed,\
gld_throughput,gst_throughput,dram_read_throughput,dram_write_throughput,\
l2_read_throughput,l2_write_throughput,shared_efficiency,\
shared_load_throughput,shared_store_throughput,\
eligible_warps_per_cycle,warp_execution_efficiency,issue_slot_utilization,\
stall_inst_fetch,stall_exec_dependency,stall_memory_dependency,\
stall_constant_memory_dependency,stall_texture,stall_sync,stall_pipe_busy,stall_other \
./hw4 ./testcases/t28 ./out

