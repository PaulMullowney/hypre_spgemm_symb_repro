# hypre_spgemm_symb_repro
reproducer for symbolic matrix matrix multiply differences between ROCm 5.4.0 and 5.4.3

tar zxvf input_files.gz
	
module load rocm/5.4.3
hipcc -O2 -g -Wall -I/opt/rocm-5.4.3/include/hip/  hypre_repro.cpp
srun -n 1 ./a.out

mullowne@crusher145:~/SM/hypre_spgemm_symb_repro> srun -n 1 ./a.out
	rows_name=d_ia.row_offsets.bin, cols_name=d_ja.columns.bin
	n=245636, nnz=2408677
	rows_name=d_ib.row_offsets.bin, cols_name=d_jb.columns.bin
	n=786433, nnz=25144689
	hypre_repro.cpp[1125], BIN[5]: m 245635 k 786432 n 245635, HASH D, SHMEM_HASH_SIZE 512, GROUP_SIZE 32, can_fail 1, need_ghash 0, ghash (nil) size 0
	kernel spec [220 1 1] x [2 16 16] num_groups_per_block=16
	[hypre_repro.cpp, 1249]: num of failed rows 1 (0.00)

module load rocm/5.4.0
hipcc -O2 -g -Wall -I/opt/rocm-5.4.0/include/hip/  hypre_repro.cpp
srun -n 1 ./a.out

mullowne@crusher145:~/SM/hypre_spgemm_symb_repro> srun -n 1 ./a.out
	rows_name=d_ia.row_offsets.bin, cols_name=d_ja.columns.bin
	n=245636, nnz=2408677
	rows_name=d_ib.row_offsets.bin, cols_name=d_jb.columns.bin
	n=786433, nnz=25144689
	hypre_repro.cpp[1125], BIN[5]: m 245635 k 786432 n 245635, HASH D, SHMEM_HASH_SIZE 512, GROUP_SIZE 32, can_fail 1, need_ghash 0, ghash (nil) size 0
	kernel spec [220 1 1] x [2 16 16] num_groups_per_block=16
	[hypre_repro.cpp, 1249]: num of failed rows 0 (0.00)