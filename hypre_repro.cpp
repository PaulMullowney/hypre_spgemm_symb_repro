#include "hip_runtime.h"
#include <cassert>
#include <vector>
#include <algorithm>

#ifndef DEBUG
#define DEBUG
#endif
//#undef DEBUG

#ifndef DEV_DEBUG
#define DEV_DEBUG
#endif
#undef DEV_DEBUG

#define HIP_CALL(call)                                                         \
  do {                                                                         \
    hipError_t err = call;                                                     \
    if (hipSuccess != err) {                                                   \
      printf("HIP ERROR (code = %d, %s) at %s:%d\n", err,                      \
             hipGetErrorString(err), __FILE__, __LINE__);                      \
      assert(0);                                                               \
      exit(1);                                                                 \
    }                                                                          \
  } while (0)

#define HYPRE_WARP_SIZE       64
#define HYPRE_WARP_BITSHIFT   6
#define HYPRE_WARP_FULL_MASK  0xFFFFFFFFFFFFFFFF
#define HYPRE_1D_BLOCK_SIZE   512
#define HYPRE_SPGEMM_MAX_NBIN 10

int spgemm_block_num_dim[4][HYPRE_SPGEMM_MAX_NBIN + 1];
#define hypre_SpgemmBlockNumDim() spgemm_block_num_dim

template <int GROUP_SIZE>
static constexpr int
hypre_spgemm_get_num_groups_per_block()
{
   return HYPRE_1D_BLOCK_SIZE / GROUP_SIZE > 1 ? HYPRE_1D_BLOCK_SIZE / GROUP_SIZE : 1;
}


template <typename T>
static __device__ __forceinline__
T warp_reduce_sum(T in)
{
#pragma unroll
   for (int d = HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
		in += __shfl_down(in, d, HYPRE_WARP_SIZE);
   }
   return in;
}


/* the warp id in the group */
template <int GROUP_SIZE>
static __device__ __forceinline__
int get_warp_in_group_id()
{
   if (GROUP_SIZE <= HYPRE_WARP_SIZE)
   {
      return 0;
   }
   else
   {
      return (threadIdx.y * blockDim.x + threadIdx.x) >> HYPRE_WARP_BITSHIFT;
   }
}

/* group reads 2 values from ptr to v1 and v2
 * GROUP_SIZE must be >= 2
 */
template <int GROUP_SIZE>
static __device__ __forceinline__
void group_read(const int *ptr, bool valid_ptr, int &v1,
                int &v2)
{
   if (GROUP_SIZE >= HYPRE_WARP_SIZE)
   {
      /* lane = warp_lane
       * Note: use "2" since assume HYPRE_WARP_SIZE divides (blockDim.x * blockDim.y) */
      const int lane = (threadIdx.y * blockDim.x + threadIdx.x) & (HYPRE_WARP_SIZE - 1);

      if (lane < 2)
      {
         v1 = ptr[lane];
      }
      v2 = __shfl(v1, 1, HYPRE_WARP_SIZE);
      v1 = __shfl(v1, 0, HYPRE_WARP_SIZE);
   }
   else
   {
      /* lane = group_lane */
      const int lane = threadIdx.y * blockDim.x + threadIdx.x;

      if (valid_ptr && lane < 2)
      {
         v1 = ptr[lane];
      }
      v2 = __shfl(v1, 1, GROUP_SIZE);
      v1 = __shfl(v1, 0, GROUP_SIZE);
   }
}

/* group reads a value from ptr to v1
 * GROUP_SIZE must be >= 2
 */
template <int GROUP_SIZE>
static __device__ __forceinline__
void group_read(const int *ptr, bool valid_ptr, int &v1)
{
   if (GROUP_SIZE >= HYPRE_WARP_SIZE)
   {
      /* lane = warp_lane
       * Note: use "2" since assume HYPRE_WARP_SIZE divides (blockDim.x * blockDim.y) */
      const int lane = (threadIdx.y * blockDim.x + threadIdx.x) & (HYPRE_WARP_SIZE - 1);

      if (!lane)
      {
         v1 = *ptr;
      }
		v1 = __shfl(v1, 0, HYPRE_WARP_SIZE);
   }
   else
   {
      /* lane = group_lane */
      const int lane = threadIdx.y * blockDim.x + threadIdx.x;

      if (valid_ptr && !lane)
      {
         v1 = *ptr;
      }
      v1 = __shfl(v1, 0, GROUP_SIZE);
   }
}

template <typename T, int NUM_GROUPS_PER_BLOCK, int GROUP_SIZE>
static __device__ __forceinline__
T group_reduce_sum(T in)
{
#pragma unroll
   for (int d = GROUP_SIZE / 2; d > 0; d >>= 1)
   {
		in += __shfl_down(in, d, HYPRE_WARP_SIZE);
   }

   return in;
}

/* s_WarpData[NUM_GROUPS_PER_BLOCK * GROUP_SIZE / HYPRE_WARP_SIZE] */
template <typename T, int NUM_GROUPS_PER_BLOCK, int GROUP_SIZE>
static __device__ __forceinline__
T group_reduce_sum(T in, volatile T *s_WarpData)
{
   T out = warp_reduce_sum(in);

	const int warp_lane_id = (threadIdx.y * blockDim.x + threadIdx.x) & (HYPRE_WARP_SIZE - 1);

	const int warp_id =(threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
							  threadIdx.x) >> HYPRE_WARP_BITSHIFT;

   if (warp_lane_id == 0)
   {
      s_WarpData[warp_id] = out;
   }

   __syncthreads();

   if (get_warp_in_group_id<GROUP_SIZE>() == 0)
   {
      const T a = warp_lane_id < GROUP_SIZE / HYPRE_WARP_SIZE ? s_WarpData[warp_id + warp_lane_id] : 0.0;
      out = warp_reduce_sum(a);
   }

   __syncthreads();

   return out;
}

template <int GROUP_SIZE>
static __device__ __forceinline__
void group_sync()
{
   if (GROUP_SIZE > HYPRE_WARP_SIZE)
      __syncthreads();
}

/* Hash functions */
static __device__ __forceinline__
int Hash2Func(int key)
{
   //return ( (key << 1) | 1 );
   //TODO: 6 --> should depend on hash1 size
   return ( (key >> 6) | 1 );
}


template <int SHMEM_HASH_SIZE>
static __device__ __forceinline__
int HashFunc(int key, int i, int prev)
{
	int hashval = ( prev + Hash2Func(key) ) & (SHMEM_HASH_SIZE - 1);
   return hashval;
}


template <int SHMEM_HASH_SIZE, int UNROLL_FACTOR>
static __device__ __forceinline__
int
hypre_spgemm_hash_insert_symbl(
   volatile int *HashKeys,
   int           key,
   int          &count )
{
   int j = 0;
   int old = -1;

#if HIP_VERSION_MAJOR == 5 && HIP_VERSION_MINOR == 4 && HIP_VERSION_PATCH == 22804
#pragma unroll 8
#else
#pragma unroll UNROLL_FACTOR
#endif
   for (int i = 0; i < SHMEM_HASH_SIZE; i++)
   {
      /* compute the hash value of key */
      if (i == 0)
      {
         j = key & (SHMEM_HASH_SIZE - 1);
      }
      else
      {
         j = HashFunc<SHMEM_HASH_SIZE>(key, i, j);
      }

      /* try to insert key into slot j */
      old = atomicCAS((int*)(HashKeys + j), -1, key);
      if (old == -1)
      {
         count++;
         return j;
      }
      if (old == key)
      {
         return j;
      }
   }
   return -1;
}

template <int SHMEM_HASH_SIZE, int GROUP_SIZE, bool IA1, int UNROLL_FACTOR>
static __device__ __forceinline__
int
hypre_spgemm_compute_row_symbl( int           istart_a,
                                int           iend_a,
                                const int    *ja,
                                const int    *ib,
                                const int    *jb,
                                volatile int *s_HashKeys,
                                char               &failed,
										  int ii)
{
   int threadIdx_x = threadIdx.x;
   int threadIdx_y = threadIdx.y;
#ifdef DEV_DEBUG
   int threadIdx_z = threadIdx.z;
#endif
   int blockDim_x = blockDim.x;
   int blockDim_y = blockDim.y;
   int num_new_insert = 0;

   /* load column idx and values of row i of A */
   for (int i = istart_a + threadIdx_y; __any(i < iend_a); i += blockDim_y)
   {
      int rowB = -1;

      if (threadIdx_x == 0 && i < iend_a)
      {
         rowB = ja[i];
      }

      /* threads in the same ygroup work on one row together */
		rowB = __shfl(rowB, 0, blockDim_x);

      /* open this row of B, collectively */
      int tmp = 0;
      if (rowB != -1 && threadIdx_x < 2)
      {
         tmp = ib[rowB + threadIdx_x];
      }
		const int rowB_start = __shfl(tmp, 0, blockDim_x);
		const int rowB_end   = __shfl(tmp, 1, blockDim_x);

      for (int k = rowB_start + threadIdx_x; __any(k < rowB_end); k += blockDim_x)
      {
         if (k < rowB_end)
         {
            if (IA1)
            {
               num_new_insert ++;
            }
            else
            {
               const int k_idx = jb[k];
               /* first try to insert into shared memory hash table */
               int pos = hypre_spgemm_hash_insert_symbl<SHMEM_HASH_SIZE, UNROLL_FACTOR>
						(s_HashKeys, k_idx, num_new_insert);

               /* if failed again, both hash tables must have been full
                  (hash table size estimation was too small).
                  Increase the counter anyhow (will lead to over-counting) */
               if (pos == -1)
               {
                  num_new_insert ++;
                  failed = 1;
               }
            }
         }
      }
#ifdef DEV_DEBUG
		__syncthreads();
		if (ii==29916)
		{
			printf("threadIdx=%d, %d, %d : num_new_insert=%d, failed=%d, rowB=%d, nnz in rowB=%d\n",threadIdx_x,threadIdx_y,threadIdx_z,num_new_insert,failed,rowB,rowB_end-rowB_start);
		}
		__syncthreads();
#endif
   }

   return num_new_insert;
}

template <int NUM_GROUPS_PER_BLOCK, int GROUP_SIZE, int SHMEM_HASH_SIZE>
__global__ void
hypre_spgemm_symbolic( const int               M,
                       const int* __restrict__ ia,
                       const int* __restrict__ ja,
                       const int* __restrict__ ib,
                       const int* __restrict__ jb,
                       int*       __restrict__ rc,
                       char*      __restrict__ rf )
{
   /* number of groups in the grid */
   volatile const int grid_num_groups = blockDim.z * gridDim.x;
   /* group id inside the block */
   volatile const int group_id = threadIdx.z;
   /* group id in the grid */
   volatile const int grid_group_id = blockIdx.x * blockDim.z + group_id;
   /* lane id inside the group */
   volatile const int lane_id = threadIdx.y * blockDim.x + threadIdx.x;
   /* shared memory hash table */
   __shared__ volatile int s_HashKeys[NUM_GROUPS_PER_BLOCK * SHMEM_HASH_SIZE];
   /* shared memory hash table for this group */
   volatile int *group_s_HashKeys = s_HashKeys + group_id * SHMEM_HASH_SIZE;

   int valid_ptr;

   /* WM: note - in cuda/hip, exited threads are not required to reach collective calls like
    *            syncthreads(), but this is not true for sycl (all threads must call the collective).
    *            Thus, all threads in the block must enter the loop (which is not ensured for cuda). */
   for (int i = grid_group_id; __any(i<M); i += grid_num_groups)
   {
      valid_ptr = GROUP_SIZE >= HYPRE_WARP_SIZE || i < M;

      char failed = 0;

      /* initialize group's shared memory hash table */
      if (valid_ptr)
      {
#pragma unroll SHMEM_HASH_SIZE
			for (int k = lane_id; k < SHMEM_HASH_SIZE; k += GROUP_SIZE)
			{
				group_s_HashKeys[k] = -1;
			}
      }
      group_sync<GROUP_SIZE>();

      /* start/end position of row of A */
      int istart_a = 0, iend_a = 0;

      /* load the start and end position of row i of A */
      group_read<GROUP_SIZE>(ia + i, valid_ptr, istart_a, iend_a);

      /* work with two hash tables */
      int jsum;

      if (iend_a == istart_a + 1)
      {
			jsum = hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, GROUP_SIZE, true, SHMEM_HASH_SIZE>
				(istart_a, iend_a, ja, ib, jb, group_s_HashKeys, failed, i);
      }
      else
      {
			jsum = hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, GROUP_SIZE, false, SHMEM_HASH_SIZE>
				(istart_a, iend_a, ja, ib, jb, group_s_HashKeys, failed, i);
      }

      /* num of nonzeros of this row (an upper bound)
       * use s_HashKeys as shared memory workspace */
      if (GROUP_SIZE <= HYPRE_WARP_SIZE)
      {
         jsum = group_reduce_sum<int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(jsum);
      }
      else
      {
         group_sync<GROUP_SIZE>();

         jsum = group_reduce_sum<int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(jsum, s_HashKeys);
      }

      /* if this row failed */
		if (GROUP_SIZE <= HYPRE_WARP_SIZE)
		{
			failed = (char) group_reduce_sum<int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>((int) failed);
		}
		else
		{
			group_sync<GROUP_SIZE>();
			failed = (char) group_reduce_sum<int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>((int) failed,
																											s_HashKeys);
      }

      if ((valid_ptr) && lane_id == 0)
      {
         rc[i] = jsum;
			rf[i] = failed > 0;
     }
   }
}

template <int SHMEM_HASH_SIZE, int GROUP_SIZE>
int
hypre_spgemm_symbolic_rownnz( int  m,
                              int  k,
                              int  n,
                              int *d_ia,
                              int *d_ja,
                              int *d_ib,
                              int *d_jb,
                              int *d_rc,
                              char      *d_rf  /* output: if symbolic mult. failed for each row */ )
{
   constexpr int num_groups_per_block = hypre_spgemm_get_num_groups_per_block<GROUP_SIZE>();
   const int BDIMX                = std::min(2, GROUP_SIZE);
   const int BDIMY                = GROUP_SIZE / BDIMX;

   /* CUDA kernel configurations: bDim.z is the number of groups in block */
   dim3 bDim(BDIMX, BDIMY, num_groups_per_block);
	assert(bDim.x * bDim.y == GROUP_SIZE);


	// The grid dim is hardcoded for how Hypre runs it. I need to dynamically determine it from SpgemmBlockNumDim.
	const int num_blocks = 220; // std::min( hypre_SpgemmBlockNumDim()[0][BIN],(int) ((m + bDim.z - 1) / bDim.z) );
   dim3 gDim( num_blocks );

   /* ---------------------------------------------------------------------------
    * build hash table (no values)
    * ---------------------------------------------------------------------------*/

   printf("%s[%d], m %d k %d n %d : SHMEM_HASH_SIZE %d, GROUP_SIZE %d\n", __FILE__, __LINE__, m, k, n,
			 SHMEM_HASH_SIZE, GROUP_SIZE);
   printf("kernel spec [%d %d %d] x [%d %d %d] num_groups_per_block=%d\n", gDim.x, gDim.y, gDim.z, bDim.x, bDim.y,
			 bDim.z, num_groups_per_block);

   const size_t shmem_bytes = 0;

   /* ---------------------------------------------------------------------------
    * symbolic multiplication:
    * On output, it provides an upper bound of nnz in rows of C
    * ---------------------------------------------------------------------------*/
	hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE>
		<<<gDim, bDim, shmem_bytes, 0>>>(m, d_ia, d_ja, d_ib, d_jb, d_rc, d_rf );
	HIP_CALL(hipGetLastError());

   return 0;
}


void initDeviceRowsAndCols(const char * rows_name, const char * cols_name, int ** hrows, int ** hcols, int ** drows, int ** dcols, int n, int ncols)
{
	*hrows = (int *)malloc(n*sizeof(int));
	FILE * fid = fopen(rows_name,"rb");
	fread(*hrows, sizeof(int), n, fid);
	fclose(fid);
	int nnz = (*hrows)[n-1];
	HIP_CALL(hipMalloc((void **)drows, n*sizeof(int)));
	HIP_CALL(hipMemcpy(*drows, *hrows, n*sizeof(int), hipMemcpyHostToDevice));

	*hcols = (int *)malloc(nnz*sizeof(int));
	fid = fopen(cols_name,"rb");
	fread(*hcols, sizeof(int), nnz, fid);
	fclose(fid);
	HIP_CALL(hipMalloc((void **)dcols, nnz*sizeof(int)));
	HIP_CALL(hipMemcpy(*dcols, *hcols, nnz*sizeof(int), hipMemcpyHostToDevice));

	printf("rows_name=%s, cols_name=%s num_rows=%d, nnz=%d\n",rows_name, cols_name,n-1,nnz);
	for (int i=0; i<n-1; ++i)
	{
		for (int j=(*hrows)[i]; j<(*hrows)[i+1]; ++j) {
			if ((*hcols)[j]<0 || (*hcols)[j]>=ncols)
			{
				printf("row=%d, bad column %d (index=%d)\n",i,(*hcols)[j],j);
			}
		}
	}

	return;
}

int main(int argc, char * argv[])
{
	int m, k;
	if (atoi(argv[1])==1)
	{
		m=245635;
		k=786432;
	}
	else
	{
		m=33825;
		k=245635;
	}
	int n=m;
	int *h_ia, *h_ja;
	int *h_ib, *h_jb;
	int *d_ia, *d_ja;
	int *d_ib, *d_jb;
	int *d_rc;
	char *d_rf;

	if (atoi(argv[1])==1)
	{
		initDeviceRowsAndCols("d_ia.row_offsets.bin", "d_ja.columns.bin", &h_ia, &h_ja, &d_ia, &d_ja, m+1, k);
		initDeviceRowsAndCols("d_ib.row_offsets.bin", "d_jb.columns.bin", &h_ib, &h_jb, &d_ib, &d_jb, k+1, n);
	}
	else
	{
		initDeviceRowsAndCols("d_ia2.row_offsets.bin", "d_ja2.columns.bin", &h_ia, &h_ja, &d_ia, &d_ja, m+1, k);
		initDeviceRowsAndCols("d_ib2.row_offsets.bin", "d_jb2.columns.bin", &h_ib, &h_jb, &d_ib, &d_jb, k+1, n);
	}

	HIP_CALL(hipMalloc((void **)&d_rc, m*sizeof(int)));
	HIP_CALL(hipMemset(d_rc, 0, m*sizeof(int)));
	HIP_CALL(hipMalloc((void **)&d_rf, m*sizeof(char)));
	HIP_CALL(hipMemset(d_rf, 0, m*sizeof(char)));

   constexpr int SHMEM_HASH_SIZE = 512; //1024;
   constexpr int GROUP_SIZE = 32; //64;

   hypre_spgemm_symbolic_rownnz<SHMEM_HASH_SIZE, GROUP_SIZE>
   (m, k, n, d_ia, d_ja, d_ib, d_jb, d_rc, d_rf);

	/* row nnz is exact if no row failed */
	char * h_rf = (char *) malloc(m*sizeof(char));
	HIP_CALL(hipMemcpy(h_rf, d_rf, m*sizeof(char), hipMemcpyDeviceToHost));

	int * h_rc = (int *) malloc(m*sizeof(int));
	HIP_CALL(hipMemcpy(h_rc, d_rc, m*sizeof(int), hipMemcpyDeviceToHost));

	int num_failed_rows = 0;
	for (int i=0; i<m; ++i)
	{
		std::vector<int> collisions(0);
		if (h_rf[i]==1)
			num_failed_rows++;
		for (int j=h_ia[i]; j<h_ia[i+1]; ++j)
		{
			int col = h_ja[j];
			for (int k=h_ib[col]; k<h_ib[col+1]; ++k)
			{
				collisions.push_back(h_jb[k]);
			}
		}
		std::sort(collisions.begin(), collisions.end());
		auto last = std::unique(collisions.begin(), collisions.end());
#ifdef DEBUG
		if (h_rc[i]!=last-collisions.begin())
		{
			printf("failed row = %d of %d\n",i,m);
			printf("\tGPU row count=%d\n",h_rc[i]);
			printf("\tCPU row count=%ld\n",last-collisions.begin());
		}
#endif
	}
#ifdef DEBUG
	printf("[%s, %d]: num of failed rows %d (%.2f)\n", __FILE__, __LINE__,
			 num_failed_rows, num_failed_rows / (m + 0.0) );
#endif
	free(h_rf);
	free(h_rc);

	HIP_CALL(hipFree(d_rc));
	HIP_CALL(hipFree(d_rf));

	HIP_CALL(hipFree(d_ia));
	HIP_CALL(hipFree(d_ja));
	HIP_CALL(hipFree(d_ib));
	HIP_CALL(hipFree(d_jb));
	free(h_ia);
	free(h_ja);
	free(h_ib);
	free(h_jb);
   return 0;
}
