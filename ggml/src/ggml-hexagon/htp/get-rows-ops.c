#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#include <HAP_farf.h>
#include <HAP_perf.h>

#include <math.h>
#include <string.h>

#define GGML_COMMON_DECL_C
#include "ggml-common.h"
#include "htp-ctx.h"
#include "htp-ops.h"
#include "htp-ops.h"
#include "hvx-utils.h"

struct get_rows_context {
    struct htp_ops_context * octx;
    uint32_t tasks_per_thread;
    uint32_t total_tasks;
    uint32_t chunks_per_row;
    uint32_t chunk_size;
    struct fastdiv_values get_rows_div_ne10;
    struct fastdiv_values get_rows_div_ne10_ne11;
    struct fastdiv_values get_rows_div_chunks_per_row;
};

#define get_rows_preamble \
    const uint32_t ne00 = octx->src[0]->ne[0]; \
    const uint32_t ne01 = octx->src[0]->ne[1]; \
    const uint32_t ne02 = octx->src[0]->ne[2]; \
    const uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    const uint32_t ne10 = octx->src[1]->ne[0]; \
    const uint32_t ne11 = octx->src[1]->ne[1]; \
    const uint32_t ne12 = octx->src[1]->ne[2]; \
    const uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    const uint32_t ne0 = octx->dst->ne[0];     \
    const uint32_t ne1 = octx->dst->ne[1];     \
    const uint32_t ne2 = octx->dst->ne[2];     \
    const uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    const uint32_t nb01 = octx->src[0]->nb[1]; \
    const uint32_t nb02 = octx->src[0]->nb[2]; \
    const uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    const uint32_t nb10 = octx->src[1]->nb[0]; \
    const uint32_t nb11 = octx->src[1]->nb[1]; \
    const uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    const uint32_t nb1 = octx->dst->nb[1];     \
    const uint32_t nb2 = octx->dst->nb[2];     \
    const uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    const uint32_t nr = ne10 * ne11 * ne12;

static void get_rows_thread_f32_f32_dma(unsigned int nth, unsigned int ith, void *data) {
    struct get_rows_context * grctx = (struct get_rows_context *)data;
    struct htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    uint64_t qt = HAP_perf_get_qtimer_count();

    const uint32_t dr  = grctx->tasks_per_thread;
    const uint32_t ir0 = dr * ith;
    if (ir0 >= grctx->total_tasks) {
        return;
    }
    const uint32_t ir1 = MIN(ir0 + dr, grctx->total_tasks);

    const bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    dma_queue * dma_queue = octx->ctx->dma[ith];
    for (uint32_t i = ir0; i < ir1; ++i) {
        const uint32_t i12 = fastdiv(i, &grctx->get_rows_div_ne10_ne11);
        const uint32_t rem = i - i12 * ne11 * ne10;
        const uint32_t i11 = fastdiv(rem, &grctx->get_rows_div_ne10);
        const uint32_t i10 = rem - i11 * ne10;

        const uintptr_t src1_addr = octx->src[1]->data + i10*nb10 + i11*nb11 + i12*nb12;
        uint32_t i01 = is_i32 ? *(int32_t *)src1_addr : *(int64_t *)src1_addr;

        if (i01 >= ne01) {
            continue;
        }

        const uintptr_t src0_ptr = octx->src[0]->data + i01*nb01 + i11*nb02 + i12*nb03;
        const uintptr_t dst_ptr  = octx->dst->data    + i10*nb1  + i11*nb2  + i12*nb3;

        while (!dma_queue_push(dma_queue, dma_make_ptr((void *)dst_ptr, (const void *)src0_ptr), nb1, nb01, ne00 * sizeof(float), 1)) {
            dma_queue_pop(dma_queue);
        }
    }
    dma_queue_flush(dma_queue);

    qt = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - qt);
    FARF(HIGH, "get-rows-f32-f32-dma %d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         ne00, ne01, ne02, ne03, ir0, ir1, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, (unsigned) qt);
}

static void get_rows_thread_f32_f32_hvx(unsigned int nth, unsigned int ith, void *data) {
    struct get_rows_context * grctx = (struct get_rows_context *)data;
    struct htp_ops_context * octx = grctx->octx;
    get_rows_preamble;

    uint64_t qt = HAP_perf_get_qtimer_count();

    const uint32_t dr  = grctx->tasks_per_thread;
    const uint32_t ir0 = dr * ith;
    if (ir0 >= grctx->total_tasks) {
        return;
    }
    const uint32_t ir1 = MIN(ir0 + dr, grctx->total_tasks);

    const bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    const uint32_t chunks_per_row = grctx->chunks_per_row;
    const uint32_t chunk_size     = grctx->chunk_size;
    for (uint32_t i = ir0; i < ir1; ++i) {
        const uint32_t row_idx   = fastdiv(i, &grctx->get_rows_div_chunks_per_row);
        const uint32_t chunk_idx = i - row_idx * chunks_per_row;

        const uint32_t i12 = fastdiv(row_idx, &grctx->get_rows_div_ne10_ne11);
        const uint32_t rem = row_idx - i12 * ne11 * ne10;
        const uint32_t i11 = fastdiv(rem, &grctx->get_rows_div_ne10);
        const uint32_t i10 = rem - i11 * ne10;

        const uintptr_t src1_addr = octx->src[1]->data + i10*nb10 + i11*nb11 + i12*nb12;
        uint32_t i01 = is_i32 ? *(int32_t *)src1_addr : *(int64_t *)src1_addr;

        if (i01 >= ne01) {
            continue;
        }

        const uint32_t offset = chunk_idx * chunk_size;
        if (offset < ne00) {
            const uint32_t copy_size = MIN(chunk_size, ne00 - offset);
            const uintptr_t src0_ptr = octx->src[0]->data + i01*nb01 + i11*nb02 + i12*nb03 + offset * sizeof(float);
            const uintptr_t dst_ptr  = octx->dst->data    + i10*nb1  + i11*nb2  + i12*nb3  + offset * sizeof(float);
            hvx_copy_f32_uu((uint8_t *)dst_ptr, (const uint8_t *)src0_ptr, copy_size);
        }
    }

    qt = HAP_perf_qtimer_count_to_us(HAP_perf_get_qtimer_count() - qt);
    FARF(HIGH, "get-rows-f32-f32-hvx %d/%d: %ux%ux%ux%u (%u:%u) x %ux%ux%ux%u -> %ux%ux%ux%u usec %u\n", ith, nth,
         ne00, ne01, ne02, ne03, ir0, ir1, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, (unsigned) qt);
}

// Q4_0 tiled-REPACK constants (must match HTP_MM_WEIGHT_TILE_SIZE_Q4_0 = 576 in matmul-ops.h).
// Layout per tile: 512 bytes quants (32 rows x 16 column-pairs x 2 nibbles, packed column-major)
//                 + 64 bytes fp16 scales (32 rows x 2 bytes)
// Tile ordering: tile[col_tile * n_k_tiles + k_tile] at stride GET_ROWS_Q4_0_TILE_SIZE.
// Within tile: byte[cp * 32 + row_in_tile] = (q[row][2*cp+1] << 4) | q[row][2*cp]
// kernel_params[0]=1 selects this path; [1]=n_k_tiles is set by the host.
#define GET_ROWS_Q4_0_TILE_QUANTS_SIZE 512
#define GET_ROWS_Q4_0_TILE_SIZE        576

static inline float get_rows_fp16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h >> 15) << 31;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        f = sign;  // zero / denormal → treat as 0 for this experimental path
    } else if (exp == 31) {
        f = sign | 0x7F800000u | (mant << 13);  // Inf / NaN passthrough
    } else {
        f = sign | ((exp + (127u - 15u)) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, sizeof(f));
    return result;
}

// Dequantize one row from a tiled Q4_0 REPACK buffer into a float32 output row.
// One thread per output row (nr total tasks, no chunking).
static void get_rows_thread_q4_0_tiled(unsigned int nth, unsigned int ith, void *data) {
    struct get_rows_context * grctx = (struct get_rows_context *)data;
    struct htp_ops_context  * octx  = grctx->octx;
    get_rows_preamble;

    const uint32_t n_k_tiles = (uint32_t) octx->kernel_params[1];

    const uint32_t dr  = grctx->tasks_per_thread;
    const uint32_t ir0 = dr * ith;
    if (ir0 >= grctx->total_tasks) {
        return;
    }
    const uint32_t ir1 = MIN(ir0 + dr, grctx->total_tasks);

    const bool     is_i32    = (octx->src[1]->type == HTP_TYPE_I32);
    const uint8_t *src0_data = (const uint8_t *) octx->src[0]->data;

    for (uint32_t i = ir0; i < ir1; ++i) {
        const uint32_t i12 = fastdiv(i, &grctx->get_rows_div_ne10_ne11);
        const uint32_t rem = i - i12 * ne11 * ne10;
        const uint32_t i11 = fastdiv(rem, &grctx->get_rows_div_ne10);
        const uint32_t i10 = rem - i11 * ne10;

        const uintptr_t src1_addr = octx->src[1]->data + i10*nb10 + i11*nb11 + i12*nb12;
        uint32_t i01 = is_i32 ? *(const int32_t *)src1_addr
                               : (uint32_t)*(const int64_t *)src1_addr;

        if (i01 >= ne01) {
            continue;  // out-of-vocab index, skip
        }

        float *dst_row = (float *)((uint8_t *)octx->dst->data + i10*nb1 + i11*nb2 + i12*nb3);

        // Locate this row's tile group.
        // Tiled layout: rows [ct*32 .. ct*32+31] share the same col-tile group.
        const uint32_t ct          = i01 / 32;
        const uint32_t row_in_tile = i01 % 32;

        // Iterate over k-tiles (covers ne0 elements in groups of 32).
        for (uint32_t kt = 0; kt < n_k_tiles; kt++) {
            const uint8_t *tile = src0_data +
                (ct * n_k_tiles + kt) * GET_ROWS_Q4_0_TILE_SIZE;

            // Scale (fp16) for this row within the tile.
            const uint16_t scale_raw = *(const uint16_t *)(tile + GET_ROWS_Q4_0_TILE_QUANTS_SIZE
                                                            + row_in_tile * 2u);
            const float scale = get_rows_fp16_to_f32(scale_raw);

            float *out = dst_row + kt * 32;
            for (uint32_t e = 0; e < 32; e++) {
                // byte[cp * 32 + row] = (q[2*cp+1] << 4) | q[2*cp]
                const uint32_t cp   = e >> 1;
                const uint8_t  byte = tile[cp * 32 + row_in_tile];
                const int      q    = (e & 1) ? (byte >> 4) : (byte & 0xF);
                out[e] = (float)(q - 8) * scale;
            }
        }
    }
}

int op_get_rows(struct htp_ops_context * octx) {
    get_rows_preamble;

    // Q4_0 tiled-REPACK path (FORCE_GET_ROWS_HTP=1 on host side).
    // kernel_params[0]==1 is set by ggml-hexagon.cpp when src0 is Q4_0 in REPACK buffer.
    if (octx->src[0]->type == HTP_TYPE_Q4_0 && octx->kernel_params[0] == 1) {
        if (octx->dst->type != HTP_TYPE_F32) {
            return HTP_STATUS_NO_SUPPORT;
        }
        if (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
            return HTP_STATUS_NO_SUPPORT;
        }
        if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
            return HTP_STATUS_OK;
        }

        struct get_rows_context grctx;
        grctx.octx                       = octx;
        grctx.get_rows_div_ne10          = init_fastdiv_values(octx->src[1]->ne[0]);
        grctx.get_rows_div_ne10_ne11     = init_fastdiv_values(octx->src[1]->ne[0] * octx->src[1]->ne[1]);
        grctx.get_rows_div_chunks_per_row = init_fastdiv_values(1);
        grctx.chunks_per_row             = 1;
        grctx.chunk_size                 = ne00;
        grctx.total_tasks                = nr;

        const uint32_t n_threads = MIN(nr, octx->n_threads);
        grctx.tasks_per_thread   = (nr + n_threads - 1) / n_threads;

        worker_pool_run_func(octx->ctx->worker_pool, get_rows_thread_q4_0_tiled, &grctx, n_threads);
        return HTP_STATUS_OK;
    }

    // Original F32 path.
    if (octx->src[0]->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->dst->type != HTP_TYPE_F32) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->src[1]->type != HTP_TYPE_I32 && octx->src[1]->type != HTP_TYPE_I64) {
        return HTP_STATUS_NO_SUPPORT;
    }

    if (octx->flags & HTP_OPFLAGS_SKIP_COMPUTE) {
        return HTP_STATUS_OK;
    }

    const uint32_t nb00 = octx->src[0]->nb[0];
    const uint32_t nb0  = octx->dst->nb[0];

    const bool can_use_dma = (nb00 == sizeof(float)) && (nb0 == sizeof(float));
    const bool use_dma = can_use_dma && (ne00 >= 2048);

    struct get_rows_context grctx;
    grctx.octx = octx;
    grctx.get_rows_div_ne10      = init_fastdiv_values(octx->src[1]->ne[0]);
    grctx.get_rows_div_ne10_ne11 = init_fastdiv_values(octx->src[1]->ne[0] * octx->src[1]->ne[1]);

    if (use_dma) {
        grctx.chunks_per_row = 1;
        grctx.chunk_size = ne00;
        grctx.total_tasks = nr;
        grctx.get_rows_div_chunks_per_row = init_fastdiv_values(1);

        const uint32_t n_threads = MIN(nr, octx->n_threads);
        grctx.tasks_per_thread = (nr + n_threads - 1) / n_threads;

        worker_pool_run_func(octx->ctx->worker_pool, get_rows_thread_f32_f32_dma, &grctx, n_threads);
    } else {
        uint32_t chunks_per_row = 1;
        uint32_t chunk_size = ne00;
        uint32_t total_tasks = nr;

        if (nr < octx->n_threads) {
            const uint32_t min_chunk_size = 1024;
            uint32_t max_chunks = ne00 / min_chunk_size;
            if (max_chunks == 0) {
                max_chunks = 1;
            }
            chunks_per_row = MIN((octx->n_threads + nr - 1) / nr, max_chunks);
            chunk_size = (ne00 + chunks_per_row - 1) / chunks_per_row;
            total_tasks = nr * chunks_per_row;
        }

        grctx.chunks_per_row = chunks_per_row;
        grctx.chunk_size = chunk_size;
        grctx.total_tasks = total_tasks;
        grctx.get_rows_div_chunks_per_row = init_fastdiv_values(chunks_per_row);

        const uint32_t n_threads = MIN(total_tasks, octx->n_threads);
        grctx.tasks_per_thread = (total_tasks + n_threads - 1) / n_threads;

        worker_pool_run_func(octx->ctx->worker_pool, get_rows_thread_f32_f32_hvx, &grctx, n_threads);
    }
    return HTP_STATUS_OK;
}
