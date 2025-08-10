#include <rknn_matmul_api.h>
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <span>
#include <cstring>
#include <fstream>
#include <csignal>
#include <unistd.h>
#include <cstdlib>
static inline bool debug_enabled()
{
  const char* lvl = std::getenv("RKNN_LOG_LEVEL");
  return lvl != nullptr && std::strcmp(lvl, "5") == 0;
}

// Pack int8 int4-values (-8..7) into 4-bit pairs (2 values per byte).
// dst length must be num_values/2 bytes; src length must be num_values bytes.
static inline void set_mem_from_int8_to_int4(int8_t* dst, const int8_t* src, size_t num_values)
{
    size_t out_bytes = num_values / 2;
    for (size_t i = 0, di = 0; di < out_bytes; i += 2, ++di)
    {
        uint8_t lo = static_cast<uint8_t>(src[i]   & 0x0f);
        uint8_t hi = static_cast<uint8_t>(src[i+1] & 0x0f);
        dst[di] = static_cast<int8_t>(lo | (hi << 4));
    }
}

// Convert A from normal layout [M, K] (int4 values provided as int8 -8..7)
// to performance/native layout [K/subK, M, subK] for RK3588 INT4.
// The destination buffer holds packed int4 values (two per byte).
static inline void norm_layout_to_perf_layout_int4(const int8_t* src, int8_t* dst,
                                                   int32_t M, int32_t K, int32_t subK)
{
    // Ensure destination starts clean since we OR nibbles into bytes
    // attr.A.size is already allocated; zero only the portion we will touch
    // Bytes needed = ((ceil(K/subK) * M * subK) + 1) / 2
    // But to keep it simple and safe, zero a conservative upper bound: M*K/2
    // (the allocated buffer size comes from io_attr and is sufficient)
    size_t approx_bytes = static_cast<size_t>(M) * static_cast<size_t>(K) / 2;
    memset(dst, 0, approx_bytes);

    int32_t outer = (K + subK - 1) / subK;
    for (int32_t i = 0; i < outer; ++i)
    {
        for (int32_t m = 0; m < M; ++m)
        {
            for (int32_t j = 0; j < subK; ++j)
            {
                int32_t ki = i * subK + j;
                int8_t v4 = 0;
                if (ki < K)
                {
                    int32_t input_index = m * K + ki;
                    v4 = static_cast<int8_t>(src[input_index] & 0x0f);
                }
                int32_t output_index = i * M * subK + m * subK + j;
                if ((output_index & 1) == 0)
                {
                    dst[output_index / 2] = v4;
                }
                else
                {
                    int8_t temp = dst[output_index / 2];
                    dst[output_index / 2] = static_cast<int8_t>(temp | static_cast<int8_t>(v4 << 4));
                }
            }
        }
    }
}

template <typename T, typename U>
void fill_random(std::span<T> data, U min, U max)
{
    using Dist = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>;
    std::random_device rd;
    std::mt19937 gen(rd());
    Dist dis(min, max);
    for (auto &x : data)
    {
        x = dis(gen);
    }
}

struct RKNNMatMul
{
    RKNNMatMul(int m, int k, int n, rknn_matmul_type type, bool ac_native, bool b_native)
        : m(m), k(k), n(n), type(type)
    {
    if (debug_enabled()) {
      std::cerr << "[DEBUG] RKNNMatMul ctor enter: M=" << m
                << ", K=" << k << ", N=" << n
                << ", type=" << get_matmul_type_string(type)
                << " (" << static_cast<int>(type) << ")"
                << ", AC_layout=" << ac_native
                << ", B_layout=" << b_native << "\n";
    }
        memset(&info, 0, sizeof(info));
        info.M = m;
        info.K = k;
        info.N = n;
        info.type = type;
        info.B_layout = b_native;
        info.AC_layout = ac_native;

        memset(&attr, 0, sizeof(attr));
    if (debug_enabled()) {
      std::cerr << "[DEBUG] calling rknn_matmul_create with info: {M=" << info.M
                << ", K=" << info.K << ", N=" << info.N
                << ", type=" << static_cast<int>(info.type)
                << ", B_layout=" << info.B_layout
                << ", AC_layout=" << info.AC_layout << "}" << std::endl;
    }
    int ret = rknn_matmul_create(&ctx, &info, &attr);
        if (ret != 0)
        {
            std::cerr << "rknn_matmul_create failed: " << ret << std::endl;
            return;
        }

    if (debug_enabled()) {
      std::cerr << "[DEBUG] rknn_matmul_create ok. IO attr:" << std::endl;
      auto print_tensor_attr = [](const char* name, const rknn_matmul_tensor_attr& t) {
        std::cerr << "  [" << name << "] n_dims=" << t.n_dims
                  << ", dims=[";
        for (uint32_t i = 0; i < t.n_dims && i < (uint32_t)RKNN_MAX_DIMS; ++i)
        {
          std::cerr << t.dims[i] << (i + 1 < t.n_dims ? "," : "");
        }
        std::cerr << "]"
                  << ", size=" << t.size
                  << ", type=" << static_cast<int>(t.type)
                  << "\n";
      };
      print_tensor_attr("A", attr.A);
      print_tensor_attr("B", attr.B);
      print_tensor_attr("C", attr.C);
    }

        void *a = nullptr, *b = nullptr;
        if(type == RKNN_INT8_MM_INT8_TO_INT32)
        {
            a = malloc(m * k);
            b = malloc(k * n);

            fill_random(std::span<int8_t>((int8_t*)a, m * k), -128, 127);
            fill_random(std::span<int8_t>((int8_t*)b, k * n), -128, 127);
        }
        else if(type == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32)
        {
            a = malloc(m * k * 2);
            b = malloc(k * n * 2);

            fill_random(std::span<uint16_t>((uint16_t*)a, m * k), -1.0, 1.0);
            fill_random(std::span<uint16_t>((uint16_t*)b, k * n), -1.0, 1.0);
        }
        else if(type == RKNN_INT4_MM_INT4_TO_INT16)
        {
            // Generate full int4 value streams (one int8 per int4 value), then pack later
            a = malloc(m * k);
            b = malloc(k * n);

            fill_random(std::span<int8_t>((int8_t*)a, m * k), -8, 7);
            fill_random(std::span<int8_t>((int8_t*)b, k * n), -8, 7);
        }
        else
        {
            std::cerr << "Unsupported type: " << type << std::endl;
            return;
        }

        A = rknn_create_mem(ctx, attr.A.size);
    if(A == nullptr)
        {
            std::cerr << "rknn_create_mem failed" << std::endl;
            free(a);
            free(b);
            return;
        }

        B = rknn_create_mem(ctx, attr.B.size);
    if(B == nullptr)
        {
            std::cerr << "rknn_create_mem failed" << std::endl;
            free(a);
            free(b);
            return;
        }
        C = rknn_create_mem(ctx, attr.C.size);
    if(C == nullptr)
        {
            std::cerr << "rknn_create_mem failed" << std::endl;
            free(a);
            free(b);
            return;
        }

    if (debug_enabled()) {
      std::cerr << "[DEBUG] allocated mem sizes: A->size=" << A->size
                << ", B->size=" << B->size
                << ", C->size=" << C->size << std::endl;
    }
    if (A->size != attr.A.size || B->size != attr.B.size || C->size != attr.C.size)
    {
      std::cerr << "[WARN] mem size != attr.size: "
                << "A(" << A->size << " vs " << attr.A.size << "), "
                << "B(" << B->size << " vs " << attr.B.size << "), "
                << "C(" << C->size << " vs " << attr.C.size << ")" << std::endl;
    }

    if (type == RKNN_INT4_MM_INT4_TO_INT16)
    {
        // Pack A according to AC_layout
        if (info.AC_layout == 0)
        {
            set_mem_from_int8_to_int4(static_cast<int8_t*>(A->virt_addr), static_cast<int8_t*>(a), static_cast<size_t>(m) * k);
        }
        else // AC_layout == 1 (perf/native layout)
        {
            int32_t subK = static_cast<int32_t>(attr.A.dims[2]);
            norm_layout_to_perf_layout_int4(static_cast<int8_t*>(a), static_cast<int8_t*>(A->virt_addr), m, k, subK);
        }

        // B packing: support both normal (B_layout==0) and native (B_layout==1)
        if (info.B_layout == 0)
        {
            set_mem_from_int8_to_int4(static_cast<int8_t*>(B->virt_addr), static_cast<int8_t*>(b), static_cast<size_t>(k) * n);
        }
        else if (info.B_layout == 1)
        {
            // Create a temporary packed-normal buffer, then convert to native layout
            std::vector<int8_t> b_int4_packed(static_cast<size_t>(k) * n / 2);
            set_mem_from_int8_to_int4(b_int4_packed.data(), static_cast<int8_t*>(b), static_cast<size_t>(k) * n);
            int ret_n = rknn_B_normal_layout_to_native_layout(b_int4_packed.data(), B->virt_addr, k, n, &info);
            if (ret_n != 0)
            {
                std::cerr << "rknn_B_normal_layout_to_native_layout failed: " << ret_n << std::endl;
            }
        }
        else
        {
            // Transposed layout not handled in this bench; default to normal packing
            set_mem_from_int8_to_int4(static_cast<int8_t*>(B->virt_addr), static_cast<int8_t*>(b), static_cast<size_t>(k) * n);
        }
    }
    else
    {
        memcpy(A->virt_addr, a, std::min<size_t>(A->size, attr.A.size));
        memcpy(B->virt_addr, b, std::min<size_t>(B->size, attr.B.size));
    }

        free(a);
        free(b);

    if (debug_enabled()) {
      std::cerr << "[DEBUG] set_io_mem A: mem_size=" << A->size
                << ", attr.size=" << attr.A.size
                << ", attr.type=" << static_cast<int>(attr.A.type) << std::endl;
    }
    ret = rknn_matmul_set_io_mem(ctx, A, &attr.A);
        if(ret != 0)
        {
            std::cerr << "rknn_matmul_set_io_mem failed: " << ret << std::endl;
            return;
        }
    if (debug_enabled()) {
      std::cerr << "[DEBUG] set_io_mem B: mem_size=" << B->size
                << ", attr.size=" << attr.B.size
                << ", attr.type=" << static_cast<int>(attr.B.type) << std::endl;
    }
    ret = rknn_matmul_set_io_mem(ctx, B, &attr.B);
        if(ret != 0)
        {
            std::cerr << "rknn_matmul_set_io_mem failed: " << ret << std::endl;
            return;
        }
    if (debug_enabled()) {
      std::cerr << "[DEBUG] set_io_mem C: mem_size=" << C->size
                << ", attr.size=" << attr.C.size
                << ", attr.type=" << static_cast<int>(attr.C.type) << std::endl;
    }
    ret = rknn_matmul_set_io_mem(ctx, C, &attr.C);
        if(ret != 0)
        {
            std::cerr << "rknn_matmul_set_io_mem failed: " << ret << std::endl;
            return;
        }
    if (debug_enabled()) {
      std::cerr << "[DEBUG] RKNNMatMul ctor exit OK" << std::endl;
    }
    }

    void run()
    {
    if (debug_enabled()) {
      std::cerr << "[DEBUG] calling rknn_matmul_run" << std::endl;
    }
    int ret = rknn_matmul_run(ctx);
        if(ret != 0)
        {
            std::cerr << "rknn_matmul_compute failed: " << ret << std::endl;
            return;
        }
    if (debug_enabled()) {
      std::cerr << "[DEBUG] rknn_matmul_run OK" << std::endl;
    }
    }

    ~RKNNMatMul()
    {
        rknn_destroy_mem(ctx, A);
        rknn_destroy_mem(ctx, B);
        rknn_destroy_mem(ctx, C);
        rknn_matmul_destroy(ctx);
    }

    int m, k, n;
    rknn_matmul_type type;
    rknn_matmul_ctx ctx;
    rknn_matmul_info info;
    rknn_matmul_io_attr attr;
    rknn_tensor_mem *A = nullptr, *B = nullptr, *C = nullptr;
};

int main()
{
  // make RKNN print verbose logs and flush our stdio
//   setenv("RKNN_LOG_LEVEL", "5", 1);
  std::setvbuf(stdout, nullptr, _IONBF, 0);
  std::setvbuf(stderr, nullptr, _IONBF, 0);

  // simple SIGABRT handler to dump a message before aborting
  std::signal(SIGABRT, +[](int){
    if (debug_enabled()) {
      std::cerr << "\n[DEBUG] Caught SIGABRT (abort). See the last '[DEBUG] BEGIN CASE' above for the failing configuration." << std::endl;
    }
    _exit(134);
  });
    size_t run_count = 30;
    // std::vector<int> m = {1, 2, 4, 8, 16, 32, 64, 128};
    std::vector<int> m = {128};
    std::vector<int> k = {64, 128, 256, 512, 1024, 2048, 4096, 8192};
    std::vector<int> n = {64, 128, 256, 512, 1024, 2048, 4096, 8192};

    std::ofstream file("result.csv");
    if(!file.good())
    {
        std::cerr << "Failed to open result.csv" << std::endl;
        return 1;
    }

    std::ofstream initfile("init.csv");
    if(!initfile.good())
    {
        std::cerr << "Failed to open init.csv" << std::endl;
        return 1;
    }
    file << "count,m,k,n,type,ac_native,b_native,time_ns,gops\n";
    initfile << "m,k,n,type,ac_native,b_native,time_ns\n";
    for(auto m_ : m)
    {
        for(auto k_ : k)
        {
            for(auto n_ : n)
            {
                for(int type = 0; type < 3; type++)
                {
                    for(int ac_native = 0; ac_native < 2; ac_native++)
                    {
                        for(int b_native = 0; b_native < 2; b_native++)
                        {
                            rknn_matmul_type t;
                            std::string type_str;
                            if(type == 0)
                            {
                                t = RKNN_INT8_MM_INT8_TO_INT32;
                                type_str = "RKNN_INT8_MM_INT8_TO_INT32";
                            }
                            else if(type == 1)
                            {
                                t = RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32;
                                type_str = "RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32";
                            }
                            else
                            {
                                t = RKNN_INT4_MM_INT4_TO_INT16;
                                type_str = "RKNN_INT4_MM_INT4_TO_INT16";
                            }
                            // RK3588 matmul alignment constraints (from rknn_matmul_api.h):
                            // - INT8  (A:int8 x B:int8 -> C:int32):   K % 32 == 0, N % 32 == 0
                            // - FP16  (A:fp16 x B:fp16 -> C:fp32):    K % 32 == 0, N % 16 == 0
                            // - INT4  (A:int4 x B:int4 -> C:int16):   K % 64 == 0, N % 128 == 0
                            // Enforce these before creating the matmul; skip invalid shapes.
                            bool shape_ok = true;
                            if (t == RKNN_INT8_MM_INT8_TO_INT32)
                            {
                                shape_ok = (k_ % 32 == 0) && (n_ % 32 == 0);
                            }
                            else if (t == RKNN_FLOAT16_MM_FLOAT16_TO_FLOAT32)
                            {
                                // RK3588/3576 FP16 alignment: K aligned to 32 bytes (=> 16 elems), N aligned to 16 bytes (=> 8 elems)
                                shape_ok = (k_ % 16 == 0) && (n_ % 8 == 0);
                            }
                            else if (t == RKNN_INT4_MM_INT4_TO_INT16)
                            {
                                // Note: This implicitly forces INT4 tests to use N starting at 128
                                // and multiples thereof since N % 128 == 0 excludes 64.
                                shape_ok = (k_ % 64 == 0) && (n_ % 128 == 0);
                            }

                            if (!shape_ok)
                            {
                                std::cout << "SKIP m: " << m_ << ", k: " << k_ << ", n: " << n_
                                          << ", type: " << type_str
                                          << ", ac_native: " << ac_native << ", b_native: " << b_native
                                          << " (violates RK3588 alignment constraints)" << "\n";
                                continue;
                            }

                            // AC_layout=1 is now supported for INT4 in this bench

                            if (t == RKNN_INT4_MM_INT4_TO_INT16 && b_native == 0)
                            {
                                std::cout << "SKIP m: " << m_ << ", k: " << k_ << ", n: " << n_
                                          << ", type: " << type_str
                                          << ", ac_native: " << ac_native << ", b_native: " << b_native
                                          << " (INT4 B normal layout unsupported on current runtime; use native)" << "\n";
                                continue;
                            }

                            if (debug_enabled()) {
                              std::cerr << "[DEBUG] BEGIN CASE: M=" << m_ << ", K=" << k_ << ", N=" << n_
                                        << ", type=" << type_str << " (" << static_cast<int>(t) << ")"
                                        << ", ac_native=" << ac_native << ", b_native=" << b_native << "\n";
                            }
                            auto start = std::chrono::high_resolution_clock::now();
                            RKNNMatMul matmul(m_, k_, n_, t, ac_native, b_native);
                            auto end = std::chrono::high_resolution_clock::now();
                            auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                            initfile << m_ << "," << k_ << "," << n_ << "," << type_str << "," << ac_native << "," << b_native << "," << duration.count() << "\n";
                            std::cout << "INIT m: " << m_ << ", k: " << k_ << ", n: " << n_ << ", type: " << type_str << ", ac_native: " << ac_native << ", b_native: " << b_native << ", init time: " << duration.count() << "ns" << "\n";

                            for(size_t i = 0; i < run_count; i++)
                            {
                                auto start = std::chrono::high_resolution_clock::now();
                                matmul.run();
                                auto end = std::chrono::high_resolution_clock::now();
                                auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
                                auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
                                auto gops = (uint64_t)m_ * n_ * (2 * k_ - 1) / 1000UL / ((double)duration_us.count());
                                file << i << "," << m_ << "," << k_ << "," << n_ << "," << type_str << "," << ac_native << "," << b_native << "," << duration.count() << "," << gops << "\n";
                                std::cout << "m: " << m_ << ", k: " << k_ << ", n: " << n_ << ", type: " << type_str << ", ac_native: " << ac_native << ", b_native: " << b_native << ", time: " << duration.count() << "ns, " << gops << "GOPS" << "\n";
                            }
                        }
                    }
                }
            }
        }
    }
}
