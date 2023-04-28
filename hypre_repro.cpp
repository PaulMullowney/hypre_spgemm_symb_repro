/* HashKeys: assumed to be initialized as all -1's
 * Key:      assumed to be nonnegative
 * increase by 1 if is a new entry
 */
#include "hip_runtime.h"
#include <cassert>

/*
#include <thrust/execution_policy.h>
#include <thrust/system/hip/execution_policy.h>
#include <thrust/count.h>
#include <thrust/device_ptr.h>
#include <thrust/unique.h>
#include <thrust/sort.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include <thrust/gather.h>
#include <thrust/scan.h>
#include <thrust/fill.h>
#include <thrust/adjacent_difference.h>
#include <thrust/inner_product.h>
#include <thrust/logical.h>
#include <thrust/replace.h>
#include <thrust/sequence.h>
#include <thrust/for_each.h>
#include <thrust/remove.h>
*/

using hypre_DeviceItem = void*;

typedef int HYPRE_Int;
typedef int hypre_int;
typedef double HYPRE_Complex;

dim3
hypre_dim3(HYPRE_Int x)
{
   dim3 d(x);
   return d;
}

dim3
hypre_dim3(HYPRE_Int x, HYPRE_Int y)
{
   dim3 d(x, y);
   return d;
}

dim3
hypre_dim3(HYPRE_Int x, HYPRE_Int y, HYPRE_Int z)
{
   dim3 d(x, y, z);
   return d;
}


#define HYPRE_WARP_SIZE       64
#define HYPRE_WARP_BITSHIFT   6
#define HYPRE_WARP_FULL_MASK  0xFFFFFFFF
#define HYPRE_MAX_NUM_WARPS   (64 * 64 * 32)
#define HYPRE_FLT_LARGE       1e30
#define HYPRE_1D_BLOCK_SIZE   512
#define HYPRE_MAX_NUM_STREAMS 10
#define HYPRE_SPGEMM_MAX_NBIN 10

#define COHEN_USE_SHMEM 0
static const char HYPRE_SPGEMM_HASH_TYPE = 'D';
/* bin settings                             0   1   2    3    4    5     6     7     8     9     10 */
constexpr HYPRE_Int SYMBL_HASH_SIZE[11] = { 0, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384 };
constexpr HYPRE_Int T_GROUP_SIZE[11]    = { 0,  2,  4,   8,  16,  32,   64,  128,  256,  512,  1024 };
#define HYPRE_SPGEMM_DEFAULT_BIN 6
/* unroll factor in the kernels */
#define HYPRE_SPGEMM_NUMER_UNROLL 256
#define HYPRE_SPGEMM_SYMBL_UNROLL 512

HYPRE_Int                         spgemm_block_num_dim[4][HYPRE_SPGEMM_MAX_NBIN + 1];
#define hypre_SpgemmBlockNumDim() spgemm_block_num_dim

template <HYPRE_Int GROUP_SIZE>
static constexpr HYPRE_Int
hypre_spgemm_get_num_groups_per_block()
{
   return std::max(512 / GROUP_SIZE, 1);
}


/* return the number of threads in block */
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_gpu_get_num_threads(hypre_DeviceItem &item)
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
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_gpu_get_thread_id(hypre_DeviceItem &item)
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
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_gpu_get_num_warps(hypre_DeviceItem &item)
{
   return hypre_gpu_get_num_threads<dim>(item) >> HYPRE_WARP_BITSHIFT;
}

/* return the warp id in block */
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_gpu_get_warp_id(hypre_DeviceItem &item)
{
   return hypre_gpu_get_thread_id<dim>(item) >> HYPRE_WARP_BITSHIFT;
}

/* return the thread lane id in warp */
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_gpu_get_lane_id(hypre_DeviceItem &item)
{
   return hypre_gpu_get_thread_id<dim>(item) & (HYPRE_WARP_SIZE - 1);
}

/* return the num of blocks in grid */
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_gpu_get_num_blocks()
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
template <hypre_int dim>
static __device__ __forceinline__
hypre_int hypre_gpu_get_block_id(hypre_DeviceItem &item)
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
template <hypre_int bdim, hypre_int gdim>
static __device__ __forceinline__
hypre_int hypre_gpu_get_grid_num_threads(hypre_DeviceItem &item)
{
   return hypre_gpu_get_num_blocks<gdim>() * hypre_gpu_get_num_threads<bdim>(item);
}

/* return the flattened thread id in grid */
template <hypre_int bdim, hypre_int gdim>
static __device__ __forceinline__
hypre_int hypre_gpu_get_grid_thread_id(hypre_DeviceItem &item)
{
   return hypre_gpu_get_block_id<gdim>(item) * hypre_gpu_get_num_threads<bdim>(item) +
          hypre_gpu_get_thread_id<bdim>(item);
}

/* return the number of warps in grid */
template <hypre_int bdim, hypre_int gdim>
static __device__ __forceinline__
hypre_int hypre_gpu_get_grid_num_warps(hypre_DeviceItem &item)
{
   return hypre_gpu_get_num_blocks<gdim>() * hypre_gpu_get_num_warps<bdim>(item);
}

/* return the flattened warp id in grid */
template <hypre_int bdim, hypre_int gdim>
static __device__ __forceinline__
hypre_int hypre_gpu_get_grid_warp_id(hypre_DeviceItem &item)
{
   return hypre_gpu_get_block_id<gdim>(item) * hypre_gpu_get_num_warps<bdim>(item) +
          hypre_gpu_get_warp_id<bdim>(item);
}

static __device__ __forceinline__
hypre_int __any_sync(unsigned mask, hypre_int predicate)
{
   return __any(predicate);
}

static __device__ __forceinline__
hypre_int warp_any_sync(hypre_DeviceItem &item, unsigned mask, hypre_int predicate)
{
   return __any_sync(mask, predicate);
}

template <typename T>
static __device__ __forceinline__
T __shfl_sync(unsigned mask, T val, hypre_int src_line, hypre_int width = HYPRE_WARP_SIZE)
{
   return __shfl(val, src_line, width);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_sync(hypre_DeviceItem &item, unsigned mask, T val, hypre_int src_line,
                    hypre_int width = HYPRE_WARP_SIZE)
{
   return __shfl_sync(mask, val, src_line, width);
}


template <typename T>
static __device__ __forceinline__
T __shfl_down_sync(unsigned mask, T val, unsigned delta, hypre_int width = HYPRE_WARP_SIZE)
{
   return __shfl_down(val, delta, width);
}

template <typename T>
static __device__ __forceinline__
T warp_shuffle_down_sync(hypre_DeviceItem &item, unsigned mask, T val, hypre_int delta,
                         hypre_int width = HYPRE_WARP_SIZE)
{
   return __shfl_down_sync(mask, val, delta, width);
}

/* sync the thread block */
static __device__ __forceinline__
void block_sync(hypre_DeviceItem &item)
{
   __syncthreads();
}

static __device__ __forceinline__
void __syncwarp()
{
}

/* sync the warp */
static __device__ __forceinline__
void warp_sync(hypre_DeviceItem &item)
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
T warp_reduce_sum(hypre_DeviceItem &item, T in)
{
#pragma unroll
   for (hypre_int d = HYPRE_WARP_SIZE / 2; d > 0; d >>= 1)
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
hypre_GetDefaultDeviceGridDimension( HYPRE_Int   n,
                                     const char *granularity,
                                     dim3        bDim )
{
   HYPRE_Int num_blocks = 0;
   HYPRE_Int num_threads_per_block = bDim.x * bDim.y * bDim.z;

   if (granularity[0] == 't')
   {
      num_blocks = (n + num_threads_per_block - 1) / num_threads_per_block;
   }
   else if (granularity[0] == 'w')
   {
      HYPRE_Int num_warps_per_block = num_threads_per_block >> HYPRE_WARP_BITSHIFT;

      assert(num_warps_per_block * HYPRE_WARP_SIZE == num_threads_per_block);

      num_blocks = (n + num_warps_per_block - 1) / num_warps_per_block;
   }
   else
   {
      printf("Error %s %d: Unknown granularity !\n", __FILE__, __LINE__);
      assert(0);
   }

   dim3 gDim = hypre_dim3(num_blocks);

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
      hypre_DeviceItem item = NULL;                                                                                                 \
      (kernel_name) <<< (gridsize), (blocksize), shmem_size, 0 >>> (item, __VA_ARGS__);                                             \
      GPU_LAUNCH_SYNC;                                                                                                              \
   }                                                                                                                                \
}

#define HYPRE_GPU_LAUNCH(kernel_name, gridsize, blocksize, ...) HYPRE_GPU_LAUNCH2(kernel_name, gridsize, blocksize, 0, __VA_ARGS__)

// HYPRE_Int
// hypre_SpGemmCreateGlobalHashTable( HYPRE_Int       num_rows,        /* number of rows */
//                                    HYPRE_Int      *row_id,          /* row_id[i] is index of ith row; i if row_id == NULL */
//                                    HYPRE_Int       num_ghash,       /* number of hash tables <= num_rows */
//                                    HYPRE_Int      *row_sizes,       /* row_sizes[rowid[i]] is the size of ith row */
//                                    HYPRE_Int       SHMEM_HASH_SIZE,
//                                    HYPRE_Int     **ghash_i_ptr,     /* of length num_ghash + 1 */
//                                    HYPRE_Int     **ghash_j_ptr,
//                                    HYPRE_Complex **ghash_a_ptr,
//                                    HYPRE_Int      *ghash_size_ptr )
// {
//    assert(num_ghash <= num_rows);

//    HYPRE_Int *ghash_i, ghash_size;
//    dim3 bDim = hypre_GetDefaultDeviceBlockDimension();

// 	hipMalloc((void **)&ghash_i, sizeof(HYPRE_Int)*(num_ghash + 1));
//    //ghash_i = hypre_TAlloc(HYPRE_Int, num_ghash + 1, HYPRE_MEMORY_DEVICE);
// 	hipMemset(ghash_i, 0, sizeof(HYPRE_Int)*(num_ghash + 1));
//    //hypre_Memset(ghash_i + num_ghash, 0, sizeof(HYPRE_Int), HYPRE_MEMORY_DEVICE);
//    dim3 gDim = hypre_GetDefaultDeviceGridDimension(num_ghash, "thread", bDim);
//    HYPRE_GPU_LAUNCH( hypre_SpGemmGhashSize, gDim, bDim,
//                      num_rows, row_id, num_ghash, row_sizes, ghash_i, SHMEM_HASH_SIZE );

// 	thrust::exclusive_scan(ghash_i, ghash_i + num_ghash + 1, ghash_i);

// 	hipMemcpy(&ghash_size, ghash_i + num_ghash, sizeof(HYPRE_Int), hipMemcpyDeviceToHost);
//    //hypre_TMemcpy(&ghash_size, ghash_i + num_ghash, HYPRE_Int, 1, HYPRE_MEMORY_HOST,
// 	//             HYPRE_MEMORY_DEVICE);

//    if (!ghash_size)
//    {
//       hipFree(ghash_i);  assert(ghash_i == NULL);
//    }

//    if (ghash_i_ptr)
//    {
//       *ghash_i_ptr = ghash_i;
//    }

//    if (ghash_j_ptr)
//    {
// 		hipMalloc((void **)&ghash_j_ptr , sizeof(HYPRE_Int)*ghash_size);
//       //*ghash_j_ptr = hypre_TAlloc(HYPRE_Int, ghash_size, HYPRE_MEMORY_DEVICE);
//    }

//    if (ghash_a_ptr)
//    {
// 		hipMalloc((void **)&ghash_a_ptr , sizeof(HYPRE_Complex)*ghash_size);
//       //*ghash_a_ptr = hypre_TAlloc(HYPRE_Complex, ghash_size, HYPRE_MEMORY_DEVICE);
//    }

//    if (ghash_size_ptr)
//    {
//       *ghash_size_ptr = ghash_size;
//    }

//    return 0;
// }
// *

/* the number of groups in block */
static __device__ __forceinline__
hypre_int get_num_groups(hypre_DeviceItem &item)
{
   return blockDim.z;
}

/* the group id in the block */
static __device__ __forceinline__
hypre_int get_group_id(hypre_DeviceItem &item)
{
   return threadIdx.z;
}

/* the thread id (lane) in the group */
static __device__ __forceinline__
hypre_int get_group_lane_id(hypre_DeviceItem &item)
{
   return hypre_gpu_get_thread_id<2>(item);
}

/* the warp id in the group */
template <HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
hypre_int get_warp_in_group_id(hypre_DeviceItem &item)
{
   if (GROUP_SIZE <= HYPRE_WARP_SIZE)
   {
      return 0;
   }
   else
   {
      return hypre_gpu_get_warp_id<2>(item);
   }
}

/* group reads 2 values from ptr to v1 and v2
 * GROUP_SIZE must be >= 2
 */
template <HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
void group_read(hypre_DeviceItem &item, const HYPRE_Int *ptr, bool valid_ptr, HYPRE_Int &v1,
                HYPRE_Int &v2)
{
   if (GROUP_SIZE >= HYPRE_WARP_SIZE)
   {
      /* lane = warp_lane
       * Note: use "2" since assume HYPRE_WARP_SIZE divides (blockDim.x * blockDim.y) */
      const HYPRE_Int lane = hypre_gpu_get_lane_id<2>(item);

      if (lane < 2)
      {
         v1 = read_only_load(ptr + lane);
      }
      v2 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, v1, 1);
      v1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, v1, 0);
   }
   else
   {
      /* lane = group_lane */
      const HYPRE_Int lane = get_group_lane_id(item);

      if (valid_ptr && lane < 2)
      {
         v1 = read_only_load(ptr + lane);
      }
      v2 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, v1, 1, GROUP_SIZE);
      v1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, v1, 0, GROUP_SIZE);
   }
}

/* group reads a value from ptr to v1
 * GROUP_SIZE must be >= 2
 */
template <HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
void group_read(hypre_DeviceItem &item, const HYPRE_Int *ptr, bool valid_ptr, HYPRE_Int &v1)
{
   if (GROUP_SIZE >= HYPRE_WARP_SIZE)
   {
      /* lane = warp_lane
       * Note: use "2" since assume HYPRE_WARP_SIZE divides (blockDim.x * blockDim.y) */
      const HYPRE_Int lane = hypre_gpu_get_lane_id<2>(item);

      if (!lane)
      {
         v1 = read_only_load(ptr);
      }
      v1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, v1, 0);
   }
   else
   {
      /* lane = group_lane */
      const HYPRE_Int lane = get_group_lane_id(item);

      if (valid_ptr && !lane)
      {
         v1 = read_only_load(ptr);
      }
      v1 = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, v1, 0, GROUP_SIZE);
   }
}

template <typename T, HYPRE_Int NUM_GROUPS_PER_BLOCK, HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
T group_reduce_sum(hypre_DeviceItem &item, T in)
{
#if defined(HYPRE_DEBUG)
   hypre_device_assert(GROUP_SIZE <= HYPRE_WARP_SIZE);
#endif

#pragma unroll
   for (hypre_int d = GROUP_SIZE / 2; d > 0; d >>= 1)
   {
      in += warp_shuffle_down_sync(item, HYPRE_WARP_FULL_MASK, in, d);
   }

   return in;
}

/* s_WarpData[NUM_GROUPS_PER_BLOCK * GROUP_SIZE / HYPRE_WARP_SIZE] */
template <typename T, HYPRE_Int NUM_GROUPS_PER_BLOCK, HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
T group_reduce_sum(hypre_DeviceItem &item, T in, volatile T *s_WarpData)
{
#if defined(HYPRE_DEBUG)
   hypre_device_assert(GROUP_SIZE > HYPRE_WARP_SIZE);
#endif

   T out = warp_reduce_sum(item, in);

   const HYPRE_Int warp_lane_id = hypre_gpu_get_lane_id<2>(item);
   const HYPRE_Int warp_id = hypre_gpu_get_warp_id<3>(item);

   if (warp_lane_id == 0)
   {
      s_WarpData[warp_id] = out;
   }

   block_sync(item);

   if (get_warp_in_group_id<GROUP_SIZE>(item) == 0)
   {
      const T a = warp_lane_id < GROUP_SIZE / HYPRE_WARP_SIZE ? s_WarpData[warp_id + warp_lane_id] : 0.0;
      out = warp_reduce_sum(item, a);
   }

   block_sync(item);

   return out;
}

/* GROUP_SIZE must <= HYPRE_WARP_SIZE */
template <typename T, HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
T group_prefix_sum(hypre_DeviceItem &item, hypre_int lane_id, T in, T &all_sum)
{
#pragma unroll
   for (hypre_int d = 2; d <= GROUP_SIZE; d <<= 1)
   {
      T t = warp_shuffle_up_sync(item, HYPRE_WARP_FULL_MASK, in, d >> 1, GROUP_SIZE);
      if ( (lane_id & (d - 1)) == (d - 1) )
      {
         in += t;
      }
   }

   all_sum = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, in, GROUP_SIZE - 1, GROUP_SIZE);

   if (lane_id == GROUP_SIZE - 1)
   {
      in = 0;
   }

#pragma unroll
   for (hypre_int d = GROUP_SIZE >> 1; d > 0; d >>= 1)
   {
      T t = warp_shuffle_xor_sync(item, HYPRE_WARP_FULL_MASK, in, d, GROUP_SIZE);

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

template <HYPRE_Int GROUP_SIZE>
static __device__ __forceinline__
void group_sync(hypre_DeviceItem &item)
{
   if (GROUP_SIZE <= HYPRE_WARP_SIZE)
   {
      warp_sync(item);
   }
   else
   {
      block_sync(item);
   }
}

/* Hash functions */
static __device__ __forceinline__
HYPRE_Int Hash2Func(HYPRE_Int key)
{
   //return ( (key << 1) | 1 );
   //TODO: 6 --> should depend on hash1 size
   return ( (key >> 6) | 1 );
}

template <char HASHTYPE>
static __device__ __forceinline__
HYPRE_Int HashFunc(HYPRE_Int m, HYPRE_Int key, HYPRE_Int i, HYPRE_Int prev)
{
   HYPRE_Int hashval = 0;

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

template <HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE>
static __device__ __forceinline__
HYPRE_Int HashFunc(HYPRE_Int key, HYPRE_Int i, HYPRE_Int prev)
{
   HYPRE_Int hashval = 0;

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


template <HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE, HYPRE_Int UNROLL_FACTOR>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_hash_insert_symbl(
   volatile HYPRE_Int *HashKeys,
   HYPRE_Int           key,
   HYPRE_Int          &count )
{
   HYPRE_Int j = 0;
   HYPRE_Int old = -1;

#pragma unroll UNROLL_FACTOR
   for (HYPRE_Int i = 0; i < SHMEM_HASH_SIZE; i++)
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
      old = atomicCAS((HYPRE_Int*)(HashKeys + j), -1, key);
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
HYPRE_Int
hypre_spgemm_hash_insert_symbl( HYPRE_Int           HashSize,
                                volatile HYPRE_Int *HashKeys,
                                HYPRE_Int           key,
                                HYPRE_Int          &count )
{
   HYPRE_Int j = 0;
   HYPRE_Int old = -1;

   for (HYPRE_Int i = 0; i < HashSize; i++)
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
      old = atomicCAS((HYPRE_Int*)(HashKeys + j), -1, key);

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

template <HYPRE_Int SHMEM_HASH_SIZE, char HASHTYPE, HYPRE_Int GROUP_SIZE, bool HAS_GHASH, bool IA1, HYPRE_Int UNROLL_FACTOR>
static __device__ __forceinline__
HYPRE_Int
hypre_spgemm_compute_row_symbl( hypre_DeviceItem   &item,
                                HYPRE_Int           istart_a,
                                HYPRE_Int           iend_a,
                                const HYPRE_Int    *ja,
                                const HYPRE_Int    *ib,
                                const HYPRE_Int    *jb,
                                volatile HYPRE_Int *s_HashKeys,
                                HYPRE_Int           g_HashSize,
                                HYPRE_Int          *g_HashKeys,
                                char               &failed )
{
   HYPRE_Int threadIdx_x = threadIdx.x;
   HYPRE_Int threadIdx_y = threadIdx.y;
   HYPRE_Int blockDim_x = blockDim.x;
   HYPRE_Int blockDim_y = blockDim.y;
   HYPRE_Int num_new_insert = 0;

   /* load column idx and values of row i of A */
   for (HYPRE_Int i = istart_a + threadIdx_y; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < iend_a);
        i += blockDim_y)
   {
      HYPRE_Int rowB = -1;

      if (threadIdx_x == 0 && i < iend_a)
      {
         rowB = read_only_load(ja + i);
      }

#if 0
      //const HYPRE_Int ymask = get_mask<4>(...);
      // TODO: need to confirm the behavior of __ballot_sync, leave it here for now
      //const HYPRE_Int num_valid_rows = __popc(__ballot_sync(ymask, valid_i));
      //for (HYPRE_Int j = 0; j < num_valid_rows; j++)
#endif

      /* threads in the same ygroup work on one row together */
      rowB = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, rowB, 0, blockDim_x);
      /* open this row of B, collectively */
      HYPRE_Int tmp = 0;
      if (rowB != -1 && threadIdx_x < 2)
      {
         tmp = read_only_load(ib + rowB + threadIdx_x);
      }
      const HYPRE_Int rowB_start = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, tmp, 0, blockDim_x);
      const HYPRE_Int rowB_end   = warp_shuffle_sync(item, HYPRE_WARP_FULL_MASK, tmp, 1, blockDim_x);

      for (HYPRE_Int k = rowB_start + threadIdx_x;
           warp_any_sync(item, HYPRE_WARP_FULL_MASK, k < rowB_end);
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
               const HYPRE_Int k_idx = read_only_load(jb + k);
               /* first try to insert into shared memory hash table */
               HYPRE_Int pos = hypre_spgemm_hash_insert_symbl<SHMEM_HASH_SIZE, HASHTYPE, UNROLL_FACTOR>
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

template <HYPRE_Int NUM_GROUPS_PER_BLOCK, HYPRE_Int GROUP_SIZE, HYPRE_Int SHMEM_HASH_SIZE, bool HAS_RIND,
          bool CAN_FAIL, char HASHTYPE, bool HAS_GHASH>
__global__ void
hypre_spgemm_symbolic( hypre_DeviceItem             &item,
                       const HYPRE_Int               M,
                       const HYPRE_Int* __restrict__ rind,
                       const HYPRE_Int* __restrict__ ia,
                       const HYPRE_Int* __restrict__ ja,
                       const HYPRE_Int* __restrict__ ib,
                       const HYPRE_Int* __restrict__ jb,
                       const HYPRE_Int* __restrict__ ig,
                       HYPRE_Int*       __restrict__ jg,
                       HYPRE_Int*       __restrict__ rc,
                       char*            __restrict__ rf )
{
   /* number of groups in the grid */
   volatile const HYPRE_Int grid_num_groups = get_num_groups(item) * gridDim.x;
   /* group id inside the block */
   volatile const HYPRE_Int group_id = get_group_id(item);
   /* group id in the grid */
   volatile const HYPRE_Int grid_group_id = blockIdx.x * get_num_groups(item) + group_id;
   /* lane id inside the group */
   volatile const HYPRE_Int lane_id = get_group_lane_id(item);
   /* shared memory hash table */
#if defined(HYPRE_SPGEMM_DEVICE_USE_DSHMEM)
   extern __shared__ volatile HYPRE_Int shared_mem[];
   volatile HYPRE_Int *s_HashKeys = shared_mem;
#else
   __shared__ volatile HYPRE_Int s_HashKeys[NUM_GROUPS_PER_BLOCK * SHMEM_HASH_SIZE];
#endif
   /* shared memory hash table for this group */
   volatile HYPRE_Int *group_s_HashKeys = s_HashKeys + group_id * SHMEM_HASH_SIZE;

   //const HYPRE_Int UNROLL_FACTOR = min(HYPRE_SPGEMM_SYMBL_UNROLL, SHMEM_HASH_SIZE);
   HYPRE_Int valid_ptr;

#if defined(HYPRE_DEBUG)
   hypre_device_assert(blockDim.x * blockDim.y == GROUP_SIZE);
#endif

   /* WM: note - in cuda/hip, exited threads are not required to reach collective calls like
    *            syncthreads(), but this is not true for sycl (all threads must call the collective).
    *            Thus, all threads in the block must enter the loop (which is not ensured for cuda). */
   for (HYPRE_Int i = grid_group_id; warp_any_sync(item, HYPRE_WARP_FULL_MASK, i < M);
        i += grid_num_groups)
   {
      valid_ptr = GROUP_SIZE >= HYPRE_WARP_SIZE || i < M;

      HYPRE_Int ii = -1;
      char failed = 0;

      if (HAS_RIND)
      {
         group_read<GROUP_SIZE>(item, rind + i, valid_ptr, ii);
      }
      else
      {
         ii = i;
      }

      /* start/end position of global memory hash table */
      HYPRE_Int istart_g = 0, iend_g = 0, ghash_size = 0;

      if (HAS_GHASH)
      {
         group_read<GROUP_SIZE>(item, ig + grid_group_id, valid_ptr,
                                istart_g, iend_g);

         /* size of global hash table allocated for this row
           (must be power of 2 and >= the actual size of the row of C - shmem hash size) */
         ghash_size = iend_g - istart_g;

         /* initialize group's global memory hash table */
         for (HYPRE_Int k = lane_id; k < ghash_size; k += GROUP_SIZE)
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
				for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += GROUP_SIZE)
				{
					group_s_HashKeys[k] = -1;
				}
			}
			else
			{
#pragma unroll SHMEM_HASH_SIZE
				for (HYPRE_Int k = lane_id; k < SHMEM_HASH_SIZE; k += GROUP_SIZE)
				{
					group_s_HashKeys[k] = -1;
				}
			}
      }

      group_sync<GROUP_SIZE>(item);

      /* start/end position of row of A */
      HYPRE_Int istart_a = 0, iend_a = 0;

      /* load the start and end position of row ii of A */
      group_read<GROUP_SIZE>(item, ia + ii, valid_ptr, istart_a, iend_a);

      /* work with two hash tables */
      HYPRE_Int jsum;

      if (iend_a == istart_a + 1)
      {
			if (HYPRE_SPGEMM_SYMBL_UNROLL<SHMEM_HASH_SIZE)
				jsum = hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH, true, HYPRE_SPGEMM_SYMBL_UNROLL>
					(item, istart_a, iend_a, ja, ib, jb, group_s_HashKeys, ghash_size, jg + istart_g, failed);
			else
				jsum = hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH, true, SHMEM_HASH_SIZE>
					(item, istart_a, iend_a, ja, ib, jb, group_s_HashKeys, ghash_size, jg + istart_g, failed);
      }
      else
      {
			if (HYPRE_SPGEMM_SYMBL_UNROLL<SHMEM_HASH_SIZE)
				jsum = hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH, false, HYPRE_SPGEMM_SYMBL_UNROLL>
					(item, istart_a, iend_a, ja, ib, jb, group_s_HashKeys, ghash_size, jg + istart_g, failed);
			else
				jsum = hypre_spgemm_compute_row_symbl<SHMEM_HASH_SIZE, HASHTYPE, GROUP_SIZE, HAS_GHASH, false, SHMEM_HASH_SIZE>
					(item, istart_a, iend_a, ja, ib, jb, group_s_HashKeys, ghash_size, jg + istart_g, failed);
      }

#if defined(HYPRE_DEBUG)
      hypre_device_assert(CAN_FAIL || failed == 0);
#endif
      /* num of nonzeros of this row (an upper bound)
       * use s_HashKeys as shared memory workspace */
      if (GROUP_SIZE <= HYPRE_WARP_SIZE)
      {
         jsum = group_reduce_sum<HYPRE_Int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(item, jsum);
      }
      else
      {
         group_sync<GROUP_SIZE>(item);

         jsum = group_reduce_sum<HYPRE_Int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(item, jsum, s_HashKeys);
      }

      /* if this row failed */
      if (CAN_FAIL)
      {
         if (GROUP_SIZE <= HYPRE_WARP_SIZE)
         {
            failed = (char) group_reduce_sum<hypre_int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(item,
                                                                                          (hypre_int) failed);
         }
         else
         {
         group_sync<GROUP_SIZE>(item);
            failed = (char) group_reduce_sum<hypre_int, NUM_GROUPS_PER_BLOCK, GROUP_SIZE>(item,
                                                                                          (hypre_int) failed,
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

template <HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE>
HYPRE_Int hypre_spgemm_symbolic_max_num_blocks( HYPRE_Int  multiProcessorCount,
                                                HYPRE_Int *num_blocks_ptr,
                                                HYPRE_Int *block_size_ptr )
{
   const char HASH_TYPE = HYPRE_SPGEMM_HASH_TYPE;
   const HYPRE_Int num_groups_per_block = hypre_spgemm_get_num_groups_per_block<GROUP_SIZE>();
   const HYPRE_Int block_size = num_groups_per_block * GROUP_SIZE;
   hypre_int numBlocksPerSm = 0;
#if defined(HYPRE_SPGEMM_DEVICE_USE_DSHMEM)
   const hypre_int shmem_bytes = num_groups_per_block * SHMEM_HASH_SIZE * sizeof(HYPRE_Int);
   hypre_int dynamic_shmem_size = shmem_bytes;
#else
   hypre_int dynamic_shmem_size = 0;
#endif

   hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &numBlocksPerSm,
      hypre_spgemm_symbolic<num_groups_per_block, GROUP_SIZE, SHMEM_HASH_SIZE, true, false, HASH_TYPE, true>,
      block_size, dynamic_shmem_size);

   *num_blocks_ptr = multiProcessorCount * numBlocksPerSm;
   *block_size_ptr = block_size;

   return 0;
}

template <HYPRE_Int BIN, HYPRE_Int SHMEM_HASH_SIZE, HYPRE_Int GROUP_SIZE, bool HAS_RIND>
HYPRE_Int
hypre_spgemm_symbolic_rownnz( HYPRE_Int  m,
                              HYPRE_Int *row_ind, /* input: row indices (length of m) */
                              HYPRE_Int  k,
                              HYPRE_Int  n,
                              bool       need_ghash,
                              HYPRE_Int *d_ia,
                              HYPRE_Int *d_ja,
                              HYPRE_Int *d_ib,
                              HYPRE_Int *d_jb,
                              HYPRE_Int *d_rc,
                              bool       can_fail,
                              char      *d_rf  /* output: if symbolic mult. failed for each row */ )
{
   //const HYPRE_Int num_groups_per_block = hypre_spgemm_get_num_groups_per_block<GROUP_SIZE>();
	constexpr HYPRE_Int num_groups_per_block = 16;
   const HYPRE_Int BDIMX                = std::min(2, GROUP_SIZE);
   const HYPRE_Int BDIMY                = GROUP_SIZE / BDIMX;

   /* CUDA kernel configurations: bDim.z is the number of groups in block */
   dim3 bDim(BDIMX, BDIMY, num_groups_per_block);
	assert(bDim.x * bDim.y == GROUP_SIZE);
   // grid dimension (number of blocks)
   const HYPRE_Int num_blocks = 220; //std::min( hypre_SpgemmBlockNumDim()[0][BIN],
	//    (HYPRE_Int) ((m + bDim.z - 1) / bDim.z) );
   dim3 gDim( num_blocks );
   // number of active groups
   //HYPRE_Int num_act_groups = std::min((HYPRE_Int) (bDim.z * gDim.x), m);

   const char HASH_TYPE = HYPRE_SPGEMM_HASH_TYPE;
   if (HASH_TYPE != 'L' && HASH_TYPE != 'Q' && HASH_TYPE != 'D')
   {
      printf("Unrecognized hash type ... [L(inear), Q(uadratic), D(ouble)]\n");
   }

   /* ---------------------------------------------------------------------------
    * build hash table (no values)
    * ---------------------------------------------------------------------------*/
   HYPRE_Int *d_ghash_i = NULL;
   HYPRE_Int *d_ghash_j = NULL;
   HYPRE_Int  ghash_size = 0;

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


void initDeviceRowsAndCols(const char * rows_name, const char * cols_name, HYPRE_Int ** drows, HYPRE_Int ** dcols, HYPRE_Int n)
{
	printf("rows_name=%s, cols_name=%s\n",rows_name, cols_name);
	HYPRE_Int * hrows = (HYPRE_Int *)malloc(n*sizeof(HYPRE_Int));
	FILE * fid = fopen(rows_name,"rb");
	fread(hrows, sizeof(HYPRE_Int), n, fid);
	fclose(fid);
	HYPRE_Int nnz = hrows[n-1];
	printf("n=%d, nnz=%d\n",n,nnz);
	hipMalloc((void **)drows, n*sizeof(HYPRE_Int));
	hipMemcpy(*drows, hrows, n*sizeof(HYPRE_Int), hipMemcpyHostToDevice);
	free(hrows);

	HYPRE_Int * hcols = (HYPRE_Int *)malloc(nnz*sizeof(HYPRE_Int));
	fid = fopen(cols_name,"rb");
	fread(hcols, sizeof(HYPRE_Int), nnz, fid);
	fclose(fid);

	hipMalloc((void **)dcols, nnz*sizeof(HYPRE_Int));
	hipMemcpy(*dcols, hcols, nnz*sizeof(HYPRE_Int), hipMemcpyHostToDevice);
	free(hcols);
	return;
}

int main(int argc, char * argv[])
{
   HYPRE_Int  m=245635;
	HYPRE_Int  k=786432;
	HYPRE_Int  n=245635;
	HYPRE_Int *d_ia;
	HYPRE_Int *d_ja;
	HYPRE_Int *d_ib;
	HYPRE_Int *d_jb;
	HYPRE_Int  in_rc=0;
	HYPRE_Int *d_rc;
	char      *d_rf;

	initDeviceRowsAndCols("d_ia.row_offsets.bin", "d_ja.columns.bin", &d_ia, &d_ja, m+1);
	initDeviceRowsAndCols("d_ib.row_offsets.bin", "d_jb.columns.bin", &d_ib, &d_jb, k+1);

	hipMalloc((void **)&d_rc, m*sizeof(HYPRE_Int));
	hipMemset(d_rc, 0, m*sizeof(HYPRE_Int));
	hipMalloc((void **)&d_rf, m*sizeof(char));
	hipMemset(d_rf, 0, m*sizeof(char));

   constexpr HYPRE_Int SHMEM_HASH_SIZE = SYMBL_HASH_SIZE[5];
   constexpr HYPRE_Int GROUP_SIZE = T_GROUP_SIZE[5];
   const HYPRE_Int BIN = 5;

   const bool need_ghash = in_rc > 0;
   const bool can_fail = in_rc < 2;

   hypre_spgemm_symbolic_rownnz<BIN, SHMEM_HASH_SIZE, GROUP_SIZE, false>
   (m, NULL, k, n, need_ghash, d_ia, d_ja, d_ib, d_jb, d_rc, can_fail, d_rf);

   if (can_fail)
   {
      /* row nnz is exact if no row failed */
		char * h_rf = (char *) malloc(m*sizeof(char));
		hipMemcpy(h_rf, d_rf, m*sizeof(char), hipMemcpyDeviceToHost);

		HYPRE_Int num_failed_rows = 0;
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