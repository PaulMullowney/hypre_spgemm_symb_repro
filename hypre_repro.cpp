#include "hip_runtime.h"
#include <cassert>
#include <vector>
#include <algorithm>

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
#define HYPRE_WARP_FULL_MASK  0xFFFFFFFF
#define HYPRE_1D_BLOCK_SIZE   512
#define HYPRE_SPGEMM_MAX_NBIN 10

static const char HYPRE_SPGEMM_HASH_TYPE = 'D';
constexpr int SYMBL_HASH_SIZE[11] = { 0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
constexpr int T_GROUP_SIZE[11]    = { 0,  2,  4,   8,  16,  32,   64,  128,  256,  512,  1024 };

/* unroll factor in the kernels */
#define HYPRE_SPGEMM_SYMBL_UNROLL 512

int                         spgemm_block_num_dim[4][HYPRE_SPGEMM_MAX_NBIN + 1];
#define hypre_SpgemmBlockNumDim() spgemm_block_num_dim

template <int GROUP_SIZE>
static constexpr int
hypre_spgemm_get_num_groups_per_block()
{
   return std::max(512 / GROUP_SIZE, 1);
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
#if defined(HYPRE_DEBUG)
   hypre_device_assert(GROUP_SIZE <= HYPRE_WARP_SIZE);
#endif

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
#if defined(HYPRE_DEBUG)
   hypre_device_assert(GROUP_SIZE > HYPRE_WARP_SIZE);
#endif

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

template <char HASHTYPE>
static __device__ __forceinline__
int HashFunc(int m, int key, int i, int prev)
{
   int hashval = 0;

   /* assume m is power of 2 */
   if (HASHTYPE == 'L')
   {
      //hashval = (key + i) % m;
      hashval = ( prev + 1 ) & (m - 1);
   }
   else if (HASHTYPE == 'Q')
   {
      //hashval = (key + (i + i*i)/2) & (m-1);
      hashval = ( prev + i ) & (m - 1);
   }
   else if (HASHTYPE == 'D')
   {
      //hashval = (key + i*Hash2Func(key) ) & (m - 1);
      hashval = ( prev + Hash2Func(key) ) & (m - 1);
   }

   return hashval;
}

template <int SHMEM_HASH_SIZE, char HASHTYPE>
static __device__ __forceinline__
int HashFunc(int key, int i, int prev)
{
   int hashval = 0;

   /* assume m is power of 2 */
   if (HASHTYPE == 'L')
   {
      //hashval = (key + i) % SHMEM_HASH_SIZE;
      hashval = ( prev + 1 ) & (SHMEM_HASH_SIZE - 1);
   }
   else if (HASHTYPE == 'Q')
   {
      //hashval = (key + (i + i*i)/2) & (SHMEM_HASH_SIZE-1);
      hashval = ( prev + i ) & (SHMEM_HASH_SIZE - 1);
   }
   else if (HASHTYPE == 'D')
   {
      //hashval = (key + i*Hash2Func(key) ) & (SHMEM_HASH_SIZE - 1);
      hashval = ( prev + Hash2Func(key) ) & (SHMEM_HASH_SIZE - 1);
   }

   return hashval;
}


template <int SHMEM_HASH_SIZE, char HASHTYPE, int UNROLL_FACTOR>
static __device__ __forceinline__
int
hypre_spgemm_hash_insert_symbl(
   volatile int *HashKeys,
   int           key,
   int          &count )
{
   int j = 0;
   int old = -1;

#pragma unroll UNROLL_FACTOR
   for (int i = 0; i < SHMEM_HASH_SIZE; i++)
   {
      /* compute the hash value of key */
      if (i == 0)
      {
         j = key & (SHMEM_HASH_SIZE - 1);
      }
      else
      {
         j = HashFunc<SHMEM_HASH_SIZE, HASHTYPE>(key, i, j);
      }

      /* try to insert key+1 into slot j */
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

template <char HASHTYPE>
static __device__ __forceinline__
int
hypre_spgemm_hash_insert_symbl( int           HashSize,
                                volatile int *HashKeys,
                                int           key,
                                int          &count )
{
   int j = 0;
   int old = -1;

   for (int i = 0; i < HashSize; i++)
   {
      /* compute the hash value of key */
      if (i == 0)
      {
         j = key & (HashSize - 1);
      }
      else
      {
         j = HashFunc<HASHTYPE>(HashSize, key, i, j);
      }

      /* try to insert key+1 into slot j */
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

template <int SHMEM_HASH_SIZE, char HASHTYPE, int GROUP_SIZE, bool HAS_GHASH, bool IA1, int UNROLL_FACTOR>
static __device__ __forceinline__
int
hypre_spgemm_compute_row_symbl( int           istart_a,
                                int           iend_a,
                                const int    *ja,
                                const int    *ib,
                                const int    *jb,
                                volatile int *s_HashKeys,
                                int           g_HashSize,
                                int          *g_HashKeys,
                                char               &failed )
{
   int threadIdx_x = threadIdx.x;
   int threadIdx_y = threadIdx.y;
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

#if 0
      //const int ymask = get_mask<4>(...);
      // TODO: need to confirm the behavior of __ballot_sync, leave it here for now
      //const int num_valid_rows = __popc(__ballot_sync(ymask, valid_i));
      //for (int j = 0; j < num_valid_rows; j++)
#endif

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
               int pos = hypre_spgemm_hash_insert_symbl<SHMEM_HASH_SIZE, HASHTYPE, UNROLL_FACTOR>
                               (s_HashKeys, k_idx, num_new_insert);

               if (HAS_GHASH && -1 == pos)
               {
                  pos = hypre_spgemm_hash_insert_symbl<HASHTYPE>
                        (g_HashSize, g_HashKeys, k_idx, num_new_insert);
               }
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
   }

   return num_new_insert;
}

template <int NUM_GROUPS_PER_BLOCK, int GROUP_SIZE, int SHMEM_HASH_SIZE, bool HAS_RIND,
          bool CAN_FAIL, char HASHTYPE, bool HAS_GHASH>
__global__ void
hypre_spgemm_symbolic( const int               M,
                       const int* __restrict__ rind,
                       const int* __restrict__ ia,
                       const int* __restrict__ ja,
                       const int* __restrict__ ib,
                       const int* __restrict__ jb,
                       const int* __restrict__ ig,
                       int*       __restrict__ jg,
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

   //const int UNROLL_FACTOR = min(HYPRE_SPGEMM_SYMBL_UNROLL, SHMEM_HASH_SIZE);
   int valid_ptr;

#if defined(HYPRE_DEBUG)
   hypre_device_assert(blockDim.x * blockDim.y == GROUP_SIZE);
#endif

   /* WM: note - in cuda/hip, exited threads are not required to reach collective calls like
    *            syncthreads(), but this is not true for sycl (all threads must call the collective).
    *            Thus, all threads in the block must enter the loop (which is not ensured for cuda). */
   for (int i = grid_group_id; __any(i<M); i += grid_num_groups)
   {
      valid_ptr = GROUP_SIZE >= HYPRE_WARP_SIZE || i < M;

      int ii = -1;
      char failed = 0;

      if (HAS_RIND)
      {
         group_read<GROUP_SIZE>(rind + i, valid_ptr, ii);
      }
      else
      {
         ii = i;
      }

      /* start/end position of global memory hash table */
      int istart_g = 0, iend_g = 0, ghash_size = 0;

      if (HAS_GHASH)
      {
         group_read<GROUP_SIZE>(ig + grid_group_id, valid_ptr,
                                istart_g, iend_g);

         /* size of global hash table allocated for this row
           (must be power of 2 and >= the actual size of the row of C - shmem hash size) */
         ghash_size = iend_g - istart_g;

         /* initialize group's global memory hash table */
         for (int k = lane_id; k < ghash_size; k += GROUP_SIZE)
         {
            jg[istart_g + k] = -1;
         }
      }

      /* initialize group's shared memory hash table */
      if (valid_ptr)
      {
			if (HYPRE_SPGEMM_SYMBL_UNROLL<SHMEM_HASH_SIZE)
			{
#pragma unroll HYPRE_SPGEMM_SYMBL_UNROLL
				for (int k = lane_id; k < SHMEM_HASH_SIZE; k += GROUP_SIZE)
				{
					group_s_HashKeys[k] = -1;
				}
			}
			else
			{
#pragma unroll SHMEM_HASH_SIZE
				for (int k = lane_id; k < SHMEM_HASH_SIZE; k += GROUP_SIZE)
				{
					group_s_HashKeys[k] = -1;
				}
			}
      }
      group_sync<GROUP_SIZE>();

      /* start/end position of row of A */
      int istart_a = 0, iend_a = 0;

      /* load the start and end position of row ii of A */
      group_read<GROUP_SIZE>(ia + ii, valid_ptr, istart_a, iend_a);

      /* work with two hash tables */
      int jsum;

      if (iend_a == istart_a + 1)
      {
			if (HYPRE_SPGEMM_SYMBL_UNROLL<SHMEM_HASH_SIZE)
				jsum = hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH, true, HYPRE_SPGEMM_SYMBL_UNROLL>
					(istart_a, iend_a, ja, ib, jb, group_s_HashKeys, ghash_size, jg + istart_g, failed);
			else
				jsum = hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH, true, SHMEM_HASH_SIZE>
					(istart_a, iend_a, ja, ib, jb, group_s_HashKeys, ghash_size, jg + istart_g, failed);
      }
      else
      {
			if (HYPRE_SPGEMM_SYMBL_UNROLL<SHMEM_HASH_SIZE)
				jsum = hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH, false, HYPRE_SPGEMM_SYMBL_UNROLL>
					(istart_a, iend_a, ja, ib, jb, group_s_HashKeys, ghash_size, jg + istart_g, failed);
			else
				jsum = hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH, false, SHMEM_HASH_SIZE>
					(istart_a, iend_a, ja, ib, jb, group_s_HashKeys, ghash_size, jg + istart_g, failed);
      }

#if defined(HYPRE_DEBUG)
      hypre_device_assert(CAN_FAIL || failed == 0);
#endif
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
      if (CAN_FAIL)
      {
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
      }

      if ((valid_ptr) && lane_id == 0)
      {
#if defined(HYPRE_DEBUG)
         hypre_device_assert(ii >= 0);
#endif
         rc[ii] = jsum;

         if (CAN_FAIL)
         {
            rf[ii] = failed > 0;
         }
      }
   }
}

template <int BIN, int SHMEM_HASH_SIZE, int GROUP_SIZE, bool HAS_RIND>
int
hypre_spgemm_symbolic_rownnz( int  m,
                              int *row_ind, /* input: row indices (length of m) */
                              int  k,
                              int  n,
                              bool       need_ghash,
                              int *d_ia,
                              int *d_ja,
                              int *d_ib,
                              int *d_jb,
                              int *d_rc,
                              bool       can_fail,
                              char      *d_rf  /* output: if symbolic mult. failed for each row */ )
{
   //const int num_groups_per_block = hypre_spgemm_get_num_groups_per_block<GROUP_SIZE>();
	constexpr int num_groups_per_block = 16;
   const int BDIMX                = std::min(2, GROUP_SIZE);
   const int BDIMY                = GROUP_SIZE / BDIMX;

   /* CUDA kernel configurations: bDim.z is the number of groups in block */
   dim3 bDim(BDIMX, BDIMY, num_groups_per_block);
	assert(bDim.x * bDim.y == GROUP_SIZE);


	// The grid dim is hardcoded for how Hypre runs it. I need to dynamically determine it from SpgemmBlockNumDim.
	const int num_blocks = 220; // std::min( hypre_SpgemmBlockNumDim()[0][BIN],(int) ((m + bDim.z - 1) / bDim.z) );
   dim3 gDim( num_blocks );

   const char HASH_TYPE = HYPRE_SPGEMM_HASH_TYPE;
   if (HASH_TYPE != 'L' && HASH_TYPE != 'Q' && HASH_TYPE != 'D')
   {
      printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
   }

   /* ---------------------------------------------------------------------------
    * build hash table (no values)
    * ---------------------------------------------------------------------------*/
   int *d_ghash_i = NULL;
   int *d_ghash_j = NULL;
   int  ghash_size = 0;

   printf("%s[%d], BIN[%d]: m %d k %d n %d, HASH %c, SHMEM_HASH_SIZE %d, GROUP_SIZE %d, "
			 "can_fail %d, need_ghash %d, ghash %p size %d\n",
			 __FILE__, __LINE__, BIN, m, k, n,
			 HASH_TYPE, SHMEM_HASH_SIZE, GROUP_SIZE, can_fail, need_ghash, d_ghash_i, ghash_size);
   printf("kernel spec [%d %d %d] x [%d %d %d] num_groups_per_block=%d\n", gDim.x, gDim.y, gDim.z, bDim.x, bDim.y,
			 bDim.z, num_groups_per_block);

   const size_t shmem_bytes = 0;

   /* ---------------------------------------------------------------------------
    * symbolic multiplication:
    * On output, it provides an upper bound of nnz in rows of C
    * ---------------------------------------------------------------------------*/
   assert(HAS_RIND == (row_ind != NULL) );

   if (can_fail)
   {
		hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, true, HASH_TYPE, false>
			<<<gDim, bDim, shmem_bytes, 0>>>(m, row_ind, d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
   }
   else
   {
		hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, false, HASH_TYPE, false>
			<<<gDim, bDim, shmem_bytes,0>>>(m, row_ind, d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
   }
	HIP_CALL(hipGetLastError());

   return 0;
}


void initDeviceRowsAndCols(const char * rows_name, const char * cols_name, int ** hrows, int ** hcols, int ** drows, int ** dcols, int n)
{
	printf("rows_name=%s, cols_name=%s\n",rows_name, cols_name);
	*hrows = (int *)malloc(n*sizeof(int));
	FILE * fid = fopen(rows_name,"rb");
	fread(*hrows, sizeof(int), n, fid);
	fclose(fid);
	int nnz = (*hrows)[n-1];
	printf("n=%d, nnz=%d\n",n,nnz);
	HIP_CALL(hipMalloc((void **)drows, n*sizeof(int)));
	HIP_CALL(hipMemcpy(*drows, *hrows, n*sizeof(int), hipMemcpyHostToDevice));

	*hcols = (int *)malloc(nnz*sizeof(int));
	fid = fopen(cols_name,"rb");
	fread(*hcols, sizeof(int), nnz, fid);
	fclose(fid);

	HIP_CALL(hipMalloc((void **)dcols, nnz*sizeof(int)));
	HIP_CALL(hipMemcpy(*dcols, *hcols, nnz*sizeof(int), hipMemcpyHostToDevice));
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
	int *h_ia;
	int *h_ja;
	int *h_ib;
	int *h_jb;
	int *d_ia;
	int *d_ja;
	int *d_ib;
	int *d_jb;
	int  in_rc=0;
	int *d_rc;
	char *d_rf;

	if (atoi(argv[1])==1)
	{
		initDeviceRowsAndCols("d_ia.row_offsets.bin", "d_ja.columns.bin", &h_ia, &h_ja, &d_ia, &d_ja, m+1);
		initDeviceRowsAndCols("d_ib.row_offsets.bin", "d_jb.columns.bin", &h_ib, &h_jb, &d_ib, &d_jb, k+1);
	}
	else
	{
		initDeviceRowsAndCols("d_ia2.row_offsets.bin", "d_ja2.columns.bin", &h_ia, &h_ja, &d_ia, &d_ja, m+1);
		initDeviceRowsAndCols("d_ib2.row_offsets.bin", "d_jb2.columns.bin", &h_ib, &h_jb, &d_ib, &d_jb, k+1);
	}

	HIP_CALL(hipMalloc((void **)&d_rc, m*sizeof(int)));
	HIP_CALL(hipMemset(d_rc, 0, m*sizeof(int)));
	HIP_CALL(hipMalloc((void **)&d_rf, m*sizeof(char)));
	HIP_CALL(hipMemset(d_rf, 0, m*sizeof(char)));

   constexpr int SHMEM_HASH_SIZE = SYMBL_HASH_SIZE[5];
   constexpr int GROUP_SIZE = T_GROUP_SIZE[5];
   const int BIN = 5;

   const bool need_ghash = in_rc > 0;
   const bool can_fail = in_rc < 2;

   hypre_spgemm_symbolic_rownnz<BIN, SHMEM_HASH_SIZE, GROUP_SIZE, false>
   (m, NULL, k, n, need_ghash, d_ia, d_ja, d_ib, d_jb, d_rc, can_fail, d_rf);

   if (can_fail)
   {
      /* row nnz is exact if no row failed */
		char * h_rf = (char *) malloc(m*sizeof(char));
		HIP_CALL(hipMemcpy(h_rf, d_rf, m*sizeof(char), hipMemcpyDeviceToHost));

		int * h_rc = (int *) malloc(m*sizeof(int));
		HIP_CALL(hipMemcpy(h_rc, d_rc, m*sizeof(int), hipMemcpyDeviceToHost));

		int num_failed_rows = 0;
		for (int i=0; i<m; ++i)
		{
			if (h_rf[i]==1) {
				std::vector<int> collisions(0);
				num_failed_rows++;
#ifdef DEBUG
				printf("rc=%d, failed row=%d of %d : %d %d\n",h_rc[i],i,m,h_ia[i],h_ia[i+1]);
#endif
				for (int j=h_ia[i]; j<h_ia[i+1]; ++j)
				{
					int col = h_ja[j];
#ifdef DEBUG
					printf("\t%d : col A, row B=%d  nnz in row B=%d\n",j,col,h_ib[col+1]-h_ib[col]);
					printf("\t\tcolumns in B :");
#endif
					for (int k=h_ib[col]; k<h_ib[col+1]; ++k)
					{
#ifdef DEBUG
						printf(" %d",h_jb[k]);
#endif
						collisions.push_back(h_jb[k]);
					}
#ifdef DEBUG
					printf("\n");
#endif
				}
				std::sort(collisions.begin(), collisions.end());
				auto last = std::unique(collisions.begin(), collisions.end());
				printf("failed row = %d or %d\n",i,m);
				printf("\tnumber of unique collisions found on GPU=%d\n",h_rc[i]);
				printf("\tnumber of unique collisions found on CPU=%ld\n",last-collisions.begin());
			}
		}

		printf("[%s, %d]: num of failed rows %d (%.2f)\n", __FILE__, __LINE__,
				 num_failed_rows, num_failed_rows / (m + 0.0) );
		free(h_rf);
   }

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
