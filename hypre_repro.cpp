#include "hip_runtime.h"
#include <cassert>

#define HYPRE_WARP_SIZE       64
#define HYPRE_WARP_BITSHIFT   6
#define HYPRE_WARP_FULL_MASK  0xFFFFFFFF
#define HYPRE_1D_BLOCK_SIZE   512
#define HYPRE_SPGEMM_MAX_NBIN 10

static const char HYPRE_SPGEMM_HASH_TYPE = 'D';
/* bin settings                             0   1   2    3    4    5     6     7     8     9     10 */
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


/* return the number of threads in block */
template <int dim>
static __device__ __forceinline__
int hypre_gpu_get_num_threads()
{
   switch (dim)
   {
      case 1:
         return (blockDim.x);
      case 2:
         return (blockDim.x * blockDim.y);
      case 3:
         return (blockDim.x * blockDim.y * blockDim.z);
   }

   return -1;
}

/* return the flattened thread id in block */
template <int dim>
static __device__ __forceinline__
int hypre_gpu_get_thread_id()
{
   switch (dim)
   {
      case 1:
         return (threadIdx.x);
      case 2:
         return (threadIdx.y * blockDim.x + threadIdx.x);
      case 3:
         return (threadIdx.z * blockDim.x * blockDim.y + threadIdx.y * blockDim.x +
                 threadIdx.x);
   }

   return -1;
}

/* return the number of warps in block */
template <int dim>
static __device__ __forceinline__
int hypre_gpu_get_num_warps()
{
   return hypre_gpu_get_num_threads<dim>() >> HYPRE_WARP_BITSHIFT;
}

/* return the warp id in block */
template <int dim>
static __device__ __forceinline__
int hypre_gpu_get_warp_id()
{
   return hypre_gpu_get_thread_id<dim>() >> HYPRE_WARP_BITSHIFT;
}

/* return the thread lane id in warp */
template <int dim>
static __device__ __forceinline__
int hypre_gpu_get_lane_id()
{
   return hypre_gpu_get_thread_id<dim>() & (HYPRE_WARP_SIZE - 1);
}

/* return the num of blocks in grid */
template <int dim>
static __device__ __forceinline__
int hypre_gpu_get_num_blocks()
{
   switch (dim)
   {
      case 1:
         return (gridDim.x);
      case 2:
         return (gridDim.x * gridDim.y);
      case 3:
         return (gridDim.x * gridDim.y * gridDim.z);
   }

   return -1;
}

/* return the flattened block id in grid */
template <int dim>
static __device__ __forceinline__
int hypre_gpu_get_block_id()
{
   switch (dim)
   {
      case 1:
         return (blockIdx.x);
      case 2:
         return (blockIdx.y * gridDim.x + blockIdx.x);
      case 3:
         return (blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x +
                 blockIdx.x);
   }

   return -1;
}

/* return the number of threads in grid */
template <int bdim, int gdim>
static __device__ __forceinline__
int hypre_gpu_get_grid_num_threads()
{
   return hypre_gpu_get_num_blocks<gdim>() * hypre_gpu_get_num_threads<bdim>();
}

/* return the flattened thread id in grid */
template <int bdim, int gdim>
static __device__ __forceinline__
int hypre_gpu_get_grid_thread_id()
{
   return hypre_gpu_get_block_id<gdim>() * hypre_gpu_get_num_threads<bdim>() +
          hypre_gpu_get_thread_id<bdim>();
}

/* return the number of warps in grid */
template <int bdim, int gdim>
static __device__ __forceinline__
int hypre_gpu_get_grid_num_warps()
{
   return hypre_gpu_get_num_blocks<gdim>() * hypre_gpu_get_num_warps<bdim>();
}

/* return the flattened warp id in grid */
template <int bdim, int gdim>
static __device__ __forceinline__
int hypre_gpu_get_grid_warp_id()
{
   return hypre_gpu_get_block_id<gdim>() * hypre_gpu_get_num_warps<bdim>() +
          hypre_gpu_get_warp_id<bdim>();
}

static __device__ __forceinline__
int __any_sync(unsigned mask, int predicate)
{
   return __any(predicate);
}

static __device__ __forceinline__
int warp_any_sync(unsigned mask, int predicate)
{
   return __any_sync(mask, predicate);
}

template <typename T>
static __device__ __forceinline__
T __shfl_sync(unsigned mask, T val, int src_line, int width = HYPRE_WARP_SIZE)
{
   return __shfl(val, src_line, width);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_sync(unsigned mask, T val, int src_line,
                    int width = HYPRE_WARP_SIZE)
{
   return __shfl_sync(mask, val, src_line, width);
}


template <typename T>
static __device__ __forceinline__
T __shfl_down_sync(unsigned mask, T val, unsigned delta, int width = HYPRE_WARP_SIZE)
{
   return __shfl_down(val, delta, width);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_down_sync(unsigned mask, T val, int delta,
                         int width = HYPRE_WARP_SIZE)
{
   return __shfl_down_sync(mask, val, delta, width);
}

/* sync the thread block */
static __device__ __forceinline__
void block_sync()
{
   __syncthreads();
}

static __device__ __forceinline__
void __syncwarp()
{
}

/* sync the warp */
static __device__ __forceinline__
void warp_sync()
{
   __syncwarp();
}

template <typename T>
static __device__ __forceinline__
T read_only_load( const T *ptr )
{
   return *ptr;
}


template <typename T>
static __device__ __forceinline__
T warp_reduce_sum(T in)
{
#pragma unroll
   for (int d = HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
   {
      in += __shfl_down_sync(HYPRE_WARP_FULL_MASK, in, d);
   }
   return in;
}

dim3
hypre_GetDefaultDeviceBlockDimension()
{
   dim3 bDim(HYPRE_1D_BLOCK_SIZE, 1, 1);
   return bDim;
}

/*--------------------------------------------------------------------
 * hypre_GetDefaultDeviceGridDimension
 *--------------------------------------------------------------------*/

dim3
hypre_GetDefaultDeviceGridDimension( int   n,
                                     const char *granularity,
                                     dim3        bDim )
{
   int num_blocks = 0;
   int num_threads_per_block = bDim.x * bDim.y * bDim.z;

   if (granularity[0] == 't')
   {
      num_blocks = (n + num_threads_per_block - 1) / num_threads_per_block;
   }
   else if (granularity[0] == 'w')
   {
      int num_warps_per_block = num_threads_per_block >> HYPRE_WARP_BITSHIFT;

      assert(num_warps_per_block * HYPRE_WARP_SIZE == num_threads_per_block);

      num_blocks = (n + num_warps_per_block - 1) / num_warps_per_block;
   }
   else
   {
      printf("Error %s %d: Unknown granularity !\n", __FILE__, __LINE__);
      assert(0);
   }

   dim3 gDim = dim3(num_blocks);
   return gDim;
}

#if defined(HYPRE_DEBUG)
#define GPU_LAUNCH_SYNC { hypre_SyncComputeStream(hypre_handle()); hypre_GetDeviceLastError(); }
#else
#define GPU_LAUNCH_SYNC
#endif

#define HYPRE_GPU_LAUNCH2(kernel_name, gridsize, blocksize, shmem_size, ...) \
{                                                                                                                                   \
   if ( gridsize.x  == 0 || gridsize.y  == 0 || gridsize.z  == 0 ||                                                                 \
        blocksize.x == 0 || blocksize.y == 0 || blocksize.z == 0 )                                                                  \
   {                                                                                                                                \
      /* printf("Warning %s %d: Zero CUDA grid/block (%d %d %d) (%d %d %d)\n",                                                      \
                 __FILE__, __LINE__,                                                                                                \
                 gridsize.x, gridsize.y, gridsize.z, blocksize.x, blocksize.y, blocksize.z); */                                     \
   }                                                                                                                                \
   else                                                                                                                             \
   {                                                                                                                                \
      (kernel_name) <<< (gridsize), (blocksize), shmem_size, 0 >>> (__VA_ARGS__);                                             \
      GPU_LAUNCH_SYNC;                                                                                                              \
   }                                                                                                                                \
}

#define HYPRE_GPU_LAUNCH(kernel_name, gridsize, blocksize, ...) HYPRE_GPU_LAUNCH2(kernel_name, gridsize, blocksize, 0, __VA_ARGS__)

/* the number of groups in block */
static __device__ __forceinline__
int get_num_groups()
{
   return blockDim.z;
}

/* the group id in the block */
static __device__ __forceinline__
int get_group_id()
{
   return threadIdx.z;
}

/* the thread id (lane) in the group */
static __device__ __forceinline__
int get_group_lane_id()
{
   return hypre_gpu_get_thread_id<2>();
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
      return hypre_gpu_get_warp_id<2>();
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
      const int lane = hypre_gpu_get_lane_id<2>();

      if (lane < 2)
      {
         v1 = read_only_load(ptr + lane);
      }
      v2 = warp_shuffle_sync(HYPRE_WARP_FULL_MASK, v1, 1);
      v1 = warp_shuffle_sync(HYPRE_WARP_FULL_MASK, v1, 0);
   }
   else
   {
      /* lane = group_lane */
      const int lane = get_group_lane_id();

      if (valid_ptr && lane < 2)
      {
         v1 = read_only_load(ptr + lane);
      }
      v2 = warp_shuffle_sync(HYPRE_WARP_FULL_MASK, v1, 1, GROUP_SIZE);
      v1 = warp_shuffle_sync(HYPRE_WARP_FULL_MASK, v1, 0, GROUP_SIZE);
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
      const int lane = hypre_gpu_get_lane_id<2>();

      if (!lane)
      {
         v1 = read_only_load(ptr);
      }
      v1 = warp_shuffle_sync(HYPRE_WARP_FULL_MASK, v1, 0);
   }
   else
   {
      /* lane = group_lane */
      const int lane = get_group_lane_id();

      if (valid_ptr && !lane)
      {
         v1 = read_only_load(ptr);
      }
      v1 = warp_shuffle_sync(HYPRE_WARP_FULL_MASK, v1, 0, GROUP_SIZE);
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
      in += warp_shuffle_down_sync(HYPRE_WARP_FULL_MASK, in, d);
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

   const int warp_lane_id = hypre_gpu_get_lane_id<2>();
   const int warp_id = hypre_gpu_get_warp_id<3>();

   if (warp_lane_id == 0)
   {
      s_WarpData[warp_id] = out;
   }

   block_sync();

   if (get_warp_in_group_id<GROUP_SIZE>() == 0)
   {
      const T a = warp_lane_id < GROUP_SIZE / HYPRE_WARP_SIZE ? s_WarpData[warp_id + warp_lane_id] : 0.0;
      out = warp_reduce_sum(a);
   }

   block_sync();

   return out;
}

/* GROUP_SIZE must <= HYPRE_WARP_SIZE */
template <typename T, int GROUP_SIZE>
static __device__ __forceinline__
T group_prefix_sum(int lane_id, T in, T &all_sum)
{
#pragma unroll
   for (int d = 2; d <= GROUP_SIZE; d <<= 1)
   {
      T t = warp_shuffle_up_sync(HYPRE_WARP_FULL_MASK, in, d >> 1, GROUP_SIZE);
      if ( (lane_id & (d - 1)) == (d - 1) )
      {
         in += t;
      }
   }

   all_sum = warp_shuffle_sync(HYPRE_WARP_FULL_MASK, in, GROUP_SIZE - 1, GROUP_SIZE);

   if (lane_id == GROUP_SIZE - 1)
   {
      in = 0;
   }

#pragma unroll
   for (int d = GROUP_SIZE >> 1; d > 0; d >>= 1)
   {
      T t = warp_shuffle_xor_sync(HYPRE_WARP_FULL_MASK, in, d, GROUP_SIZE);

      if ( (lane_id & (d - 1)) == (d - 1))
      {
         if ( (lane_id & ((d << 1) - 1)) == ((d << 1) - 1) )
         {
            in += t;
         }
         else
         {
            in = t;
         }
      }
   }
   return in;
}

template <int GROUP_SIZE>
static __device__ __forceinline__
void group_sync()
{
   if (GROUP_SIZE <= HYPRE_WARP_SIZE)
   {
      warp_sync();
   }
   else
   {
      block_sync();
   }
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
         j = HashFunc<HASHTYPE>(SHMEM_HASH_SIZE, key, i, j);
         //j = HashFunc<SHMEM_HASH_SIZE, HASHTYPE>(key, i, j);
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
   for (int i = istart_a + threadIdx_y; warp_any_sync(HYPRE_WARP_FULL_MASK, i < iend_a);
        i += blockDim_y)
   {
      int rowB = -1;

      if (threadIdx_x == 0 && i < iend_a)
      {
         rowB = read_only_load(ja + i);
      }

#if 0
      //const int ymask = get_mask<4>(...);
      // TODO: need to confirm the behavior of __ballot_sync, leave it here for now
      //const int num_valid_rows = __popc(__ballot_sync(ymask, valid_i));
      //for (int j = 0; j < num_valid_rows; j++)
#endif

      /* threads in the same ygroup work on one row together */
      rowB = warp_shuffle_sync(HYPRE_WARP_FULL_MASK, rowB, 0, blockDim_x);
      /* open this row of B, collectively */
      int tmp = 0;
      if (rowB != -1 && threadIdx_x < 2)
      {
         tmp = read_only_load(ib + rowB + threadIdx_x);
      }
      const int rowB_start = warp_shuffle_sync(HYPRE_WARP_FULL_MASK, tmp, 0, blockDim_x);
      const int rowB_end   = warp_shuffle_sync(HYPRE_WARP_FULL_MASK, tmp, 1, blockDim_x);

      for (int k = rowB_start + threadIdx_x;
           warp_any_sync(HYPRE_WARP_FULL_MASK, k < rowB_end);
           k += blockDim_x)
      {
         if (k < rowB_end)
         {
            if (IA1)
            {
               num_new_insert ++;
            }
            else
            {
               const int k_idx = read_only_load(jb + k);
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
                       char*            __restrict__ rf )
{
   /* number of groups in the grid */
   volatile const int grid_num_groups = get_num_groups() * gridDim.x;
   /* group id inside the block */
   volatile const int group_id = get_group_id();
   /* group id in the grid */
   volatile const int grid_group_id = blockIdx.x * get_num_groups() + group_id;
   /* lane id inside the group */
   volatile const int lane_id = get_group_lane_id();
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
   for (int i = grid_group_id; warp_any_sync(HYPRE_WARP_FULL_MASK, i < M);
        i += grid_num_groups)
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

template <int SHMEM_HASH_SIZE, int GROUP_SIZE>
int hypre_spgemm_symbolic_max_num_blocks( int  multiProcessorCount,
                                                int *num_blocks_ptr,
                                                int *block_size_ptr )
{
   const char HASH_TYPE = HYPRE_SPGEMM_HASH_TYPE;
   const int num_groups_per_block = hypre_spgemm_get_num_groups_per_block<GROUP_SIZE>();
   const int block_size = num_groups_per_block * GROUP_SIZE;
   int numBlocksPerSm = 0;
   int dynamic_shmem_size = 0;

   hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm,
      hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, false, HASH_TYPE, true>,
      block_size, dynamic_shmem_size);

   *num_blocks_ptr = multiProcessorCount * numBlocksPerSm;
   *block_size_ptr = block_size;

   return 0;
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
   // grid dimension (number of blocks)
	const int num_blocks = 220; // std::min( hypre_SpgemmBlockNumDim()[0][BIN],(int) ((m + bDim.z - 1) / bDim.z) );
   dim3 gDim( num_blocks );
   // number of active groups
   //int num_act_groups = std::min((int) (bDim.z * gDim.x), m);

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

   // if (need_ghash)
   // {
   //    hypre_SpGemmCreateGlobalHashTable(m, row_ind, num_act_groups, d_rc, SHMEM_HASH_SIZE,
   //                                      &d_ghash_i, &d_ghash_j, NULL, &ghash_size);
   // }

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

   /* <NUM_GROUPS_PER_BLOCK, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, CAN_FAIL, HASHTYPE, HAS_GHASH> */

   if (can_fail)
   {
      // if (ghash_size)
      // {
      //    HYPRE_GPU_LAUNCH2(
      //       (hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, true, HASH_TYPE, true>),
      //       gDim, bDim, shmem_bytes,
      //       m, row_ind, d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
      // }
      // else
      // {
         HYPRE_GPU_LAUNCH2(
            (hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, true, HASH_TYPE, false>),
            gDim, bDim, shmem_bytes,
            m, row_ind, d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
			// }
   }
   else
   {
      // if (ghash_size)
      // {
      //    HYPRE_GPU_LAUNCH2(
      //       (hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, false, HASH_TYPE, true>),
      //       gDim, bDim, shmem_bytes,
      //       m, row_ind, d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
      // }
      // else
      // {
         HYPRE_GPU_LAUNCH2(
            (hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, HAS_RIND, false, HASH_TYPE, false>),
            gDim, bDim, shmem_bytes,
            m, row_ind, d_ia, d_ja, d_ib, d_jb, d_ghash_i, d_ghash_j, d_rc, d_rf );
			//}
   }

   //hipFree(d_ghash_i);
	//hipFree(d_ghash_j);

   return 0;
}


void initDeviceRowsAndCols(const char * rows_name, const char * cols_name, int ** drows, int ** dcols, int n)
{
	printf("rows_name=%s, cols_name=%s\n",rows_name, cols_name);
	int * hrows = (int *)malloc(n*sizeof(int));
	FILE * fid = fopen(rows_name,"rb");
	fread(hrows, sizeof(int), n, fid);
	fclose(fid);
	int nnz = hrows[n-1];
	printf("n=%d, nnz=%d\n",n,nnz);
	hipMalloc((void **)drows, n*sizeof(int));
	hipMemcpy(*drows, hrows, n*sizeof(int), hipMemcpyHostToDevice);
	free(hrows);

	int * hcols = (int *)malloc(nnz*sizeof(int));
	fid = fopen(cols_name,"rb");
	fread(hcols, sizeof(int), nnz, fid);
	fclose(fid);

	hipMalloc((void **)dcols, nnz*sizeof(int));
	hipMemcpy(*dcols, hcols, nnz*sizeof(int), hipMemcpyHostToDevice);
	free(hcols);
	return;
}

int main(int argc, char * argv[])
{
   int  m=245635;
	int  k=786432;
	int  n=245635;
	int *d_ia;
	int *d_ja;
	int *d_ib;
	int *d_jb;
	int  in_rc=0;
	int *d_rc;
	char *d_rf;

	initDeviceRowsAndCols("d_ia.row_offsets.bin", "d_ja.columns.bin", &d_ia, &d_ja, m+1);
	initDeviceRowsAndCols("d_ib.row_offsets.bin", "d_jb.columns.bin", &d_ib, &d_jb, k+1);

	hipMalloc((void **)&d_rc, m*sizeof(int));
	hipMemset(d_rc, 0, m*sizeof(int));
	hipMalloc((void **)&d_rf, m*sizeof(char));
	hipMemset(d_rf, 0, m*sizeof(char));

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
		hipMemcpy(h_rf, d_rf, m*sizeof(char), hipMemcpyDeviceToHost);

		int num_failed_rows = 0;
		for (int i=0; i<m; ++i)
		{
			if (h_rf[i]==1) num_failed_rows++;
		}

		printf("[%s, %d]: num of failed rows %d (%.2f)\n", __FILE__, __LINE__,
				 num_failed_rows, num_failed_rows / (m + 0.0) );
		free(h_rf);
   }

	hipFree(d_rc);
	hipFree(d_rf);

	hipFree(d_ia);
	hipFree(d_ja);
	hipFree(d_ib);
	hipFree(d_jb);
   return 0;
}
