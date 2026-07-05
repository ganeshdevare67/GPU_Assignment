// tumor_pipeline_cpu.cpp
// Compile: g++ -O3 -march=native -fopenmp tumor_pipeline_cpu.cpp -o cpu_pipeline

#include <iostream>
#include <vector>
#include <cmath>
#include <chrono>
#include <algorithm>
#include <random>
#include <omp.h>

struct VolumeDims {
    int D, H, W;
    size_t N() const { return (size_t)D * H * W; }
};

// ---------- Stage 1: Intensity Normalization ----------
void normalizeVolume(std::vector<float>& vol) {
    float minVal = *std::min_element(vol.begin(), vol.end());
    float maxVal = *std::max_element(vol.begin(), vol.end());
    float range = std::max(maxVal - minVal, 1e-6f);

    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)vol.size(); ++i) {
        vol[i] = (vol[i] - minVal) / range;
        vol[i] = std::min(1.0f, std::max(0.0f, vol[i]));
    }
}

// ---------- Stage 2: 3x3x3 Smoothing (separable-free, direct stencil) ----------
void smoothVolume(const std::vector<float>& in, std::vector<float>& out, const VolumeDims& dims) {
    const int D = dims.D, H = dims.H, W = dims.W;
    const float invKernel = 1.0f / 27.0f; // 3x3x3 box filter

    #pragma omp parallel for collapse(2) schedule(static)
    for (int z = 0; z < D; ++z) {
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float sum = 0.0f;
                for (int dz = -1; dz <= 1; ++dz) {
                    int zz = std::min(D - 1, std::max(0, z + dz));
                    for (int dy = -1; dy <= 1; ++dy) {
                        int yy = std::min(H - 1, std::max(0, y + dy));
                        for (int dx = -1; dx <= 1; ++dx) {
                            int xx = std::min(W - 1, std::max(0, x + dx));
                            sum += in[(size_t)zz * H * W + yy * W + xx];
                        }
                    }
                }
                out[(size_t)z * H * W + y * W + x] = sum * invKernel;
            }
        }
    }
}

// ---------- Stage 3: Simulated CNN Layer (dense MAC + activation) ----------
// Simulates a 1x1x1 conv across K "feature maps" collapsed to one output channel,
// followed by a sigmoid activation -> pseudo "tumor probability map"
void inferenceMock(const std::vector<float>& in, std::vector<float>& out,
                    const std::vector<float>& weights, int K) {
    #pragma omp parallel for schedule(static)
    for (long i = 0; i < (long)in.size(); ++i) {
        float acc = 0.0f;
        // Simulate K synthetic input channels derived from neighborhood taps
        for (int k = 0; k < K; ++k) {
            acc += in[i] * weights[k]; // dense MAC
        }
        out[i] = 1.0f / (1.0f + std::exp(-acc)); // sigmoid activation
    }
}

int main() {
    VolumeDims dims{256, 256, 256};
    size_t N = dims.N();

    std::vector<float> volume(N), smoothed(N), detection(N);

    // Synthetic scanner data (12-bit range)
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 4095.0f);
    for (auto& v : volume) v = dist(rng);

    const int K = 16; // simulated feature-map depth
    std::vector<float> weights(K, 0.05f);

    auto t0 = std::chrono::high_resolution_clock::now();
    normalizeVolume(volume);
    auto t1 = std::chrono::high_resolution_clock::now();
    smoothVolume(volume, smoothed, dims);
    auto t2 = std::chrono::high_resolution_clock::now();
    inferenceMock(smoothed, detection, weights, K);
    auto t3 = std::chrono::high_resolution_clock::now();

    auto ms = [](auto a, auto b) {
        return std::chrono::duration<double, std::milli>(b - a).count();
    };

    std::cout << "Volume size: " << dims.D << "x" << dims.H << "x" << dims.W
              << " (" << N << " voxels)\n";
    std::cout << "Normalization: " << ms(t0, t1) << " ms\n";
    std::cout << "Smoothing:     " << ms(t1, t2) << " ms\n";
    std::cout << "Inference:     " << ms(t2, t3) << " ms\n";
    std::cout << "Total:         " << ms(t0, t3) << " ms\n";

    return 0;
}
