#include <chrono>
#include <immintrin.h>
#include <iostream>
#include <omp.h>
#include <string>

using namespace std;
using namespace std::chrono;

#define NO_OPTIMIZATION 0
#define GENERAL_OPTIMIZATION 1
#define SIMD_OPTIMIZATION 2
#define OPENMP_OPTIMIZATION 3
#define ALL_OPTIMIZATION 4

const int DIM = 1000;
const char *optimizationTypes[] = { "No", "General", "SIMD", "OpenMP", "All" };

steady_clock::time_point start;
steady_clock::time_point stop;
double ellapsedTime = 0;

// populates every variable in the array to be 1
void populateBlock(int *db[DIM]) {
    for (int i = 0; i < DIM; i++) {
        db[i] = (int*)malloc(DIM * sizeof(int));

        for (int j = 0; j < DIM; j++) {
            db[i][j] = 1;
        }
    }
}

// starts the chrono clock
void startClock(steady_clock::time_point& start) {
    start = steady_clock::now();
}

// stops the chrono clock by getting current time and subtracting the previously obtained start time, then print it
void stopClock(steady_clock::time_point& start, steady_clock::time_point& stop, double& ellapsedTime) {
    stop = steady_clock::now();
    ellapsedTime = duration<double, std::milli>(stop - start).count();

    printf("\nOperation took %f milliseconds\n\n", ellapsedTime);
}


// regular optimization with a simple nested for loop to add all
int noOptimizationSum(int* db[DIM]) {
    int counter = 0;

    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            counter += db[i][j];
        }
    }

    return counter;
}

// general optimization with loop unrolling
int generalOptimizationSum(int* db[DIM]) {
    int counter = 0;
    int i, j;

    for (i = 0; i < DIM; i += 4) {
        for (j = 0; j < DIM; j += 4) {
            counter += db[i][j];
            counter += db[i][j + 1];
            counter += db[i][j + 2];
            counter += db[i][j + 3];

            counter += db[i + 1][j];
            counter += db[i + 1][j + 1];
            counter += db[i + 1][j + 2];
            counter += db[i + 1][j + 3];

            counter += db[i + 2][j];
            counter += db[i + 2][j + 1];
            counter += db[i + 2][j + 2];
            counter += db[i + 2][j + 3];

            counter += db[i + 3][j];
            counter += db[i + 3][j + 1];
            counter += db[i + 3][j + 2];
            counter += db[i + 3][j + 3];
        }
    }

    return counter;
}

// simd optimization, using 4 128-bit variables to add every value in groups of 16 and avoid resetting after 65535
int simdOptimizationSum(int* db[DIM]) {
    __m128i counter1 = _mm_set1_epi32(0);
    __m128i counter2 = _mm_set1_epi32(0);
    __m128i counter3 = _mm_set1_epi32(0);
    __m128i counter4 = _mm_set1_epi32(0);

    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j += 16) {
            counter1 = _mm_add_epi32(counter1, _mm_load_si128((__m128i*) & db[i][j]));
            counter2 = _mm_add_epi32(counter2, _mm_load_si128((__m128i*) & db[i][j + 4]));

            if (j < DIM - 16) {
                counter3 = _mm_add_epi32(counter3, _mm_load_si128((__m128i*) & db[i][j + 8]));
                counter4 = _mm_add_epi32(counter4, _mm_load_si128((__m128i*) & db[i][j + 12]));
            }            
        }
    }

    int sum1 = _mm_extract_epi16(counter1, 0) + _mm_extract_epi16(counter1, 2) + _mm_extract_epi16(counter1, 4) + _mm_extract_epi16(counter1, 6);
    int sum2 = _mm_extract_epi16(counter2, 0) + _mm_extract_epi16(counter2, 2) + _mm_extract_epi16(counter2, 4) + _mm_extract_epi16(counter2, 6);
    int sum3 = _mm_extract_epi16(counter3, 0) + _mm_extract_epi16(counter3, 2) + _mm_extract_epi16(counter3, 4) + _mm_extract_epi16(counter3, 6);
    int sum4 = _mm_extract_epi16(counter4, 0) + _mm_extract_epi16(counter4, 2) + _mm_extract_epi16(counter4, 4) + _mm_extract_epi16(counter4, 6);

    return sum1 + sum2 + sum3 + sum4;
}

// openmp optimization with 2 threads and reduction
int openMPOptimization(int* db[DIM]) {
    int counter = 0;

    #pragma omp parallel num_threads(2) reduction(+:counter)
    for (int i = 0; i < DIM; i++) {
        for (int j = 0; j < DIM; j++) {
            counter += db[i][j];
        }
    }

    return counter;
}

// all optimizations combined
int allOptimizations(int* db[DIM]) {
    __m128i counter1 = _mm_set1_epi32(0);
    __m128i counter2 = _mm_set1_epi32(0);
    __m128i counter3 = _mm_set1_epi32(0);
    __m128i counter4 = _mm_set1_epi32(0);

    #pragma omp parallel num_threads(2) reduction(+:counter)
    for (int i = 0; i < DIM; i += 4) {
        for (int j = 0; j < DIM; j += 16) {
            counter1 = _mm_add_epi32(counter1, _mm_load_si128((__m128i*) & db[i][j]));
            counter1 = _mm_add_epi32(counter1, _mm_load_si128((__m128i*) & db[i + 1][j]));
            counter1 = _mm_add_epi32(counter1, _mm_load_si128((__m128i*) & db[i + 2][j]));
            counter1 = _mm_add_epi32(counter1, _mm_load_si128((__m128i*) & db[i + 3][j]));

            counter2 = _mm_add_epi32(counter2, _mm_load_si128((__m128i*) & db[i][j + 4]));
            counter2 = _mm_add_epi32(counter2, _mm_load_si128((__m128i*) & db[i + 1][j + 4]));
            counter2 = _mm_add_epi32(counter2, _mm_load_si128((__m128i*) & db[i + 2][j + 4]));
            counter2 = _mm_add_epi32(counter2, _mm_load_si128((__m128i*) & db[i + 3][j + 4]));

            if (j < DIM - 16) {
                counter3 = _mm_add_epi32(counter3, _mm_load_si128((__m128i*) & db[i][j + 8]));
                counter3 = _mm_add_epi32(counter3, _mm_load_si128((__m128i*) & db[i + 1][j + 8]));
                counter3 = _mm_add_epi32(counter3, _mm_load_si128((__m128i*) & db[i + 2][j + 8]));
                counter3 = _mm_add_epi32(counter3, _mm_load_si128((__m128i*) & db[i + 3][j + 8]));

                counter4 = _mm_add_epi32(counter4, _mm_load_si128((__m128i*) & db[i][j + 12]));
                counter4 = _mm_add_epi32(counter4, _mm_load_si128((__m128i*) & db[i + 1][j + 12]));
                counter4 = _mm_add_epi32(counter4, _mm_load_si128((__m128i*) & db[i + 2][j + 12]));
                counter4 = _mm_add_epi32(counter4, _mm_load_si128((__m128i*) & db[i + 3][j + 12]));
            }
        }
    }

    int sum1 = _mm_extract_epi16(counter1, 0) + _mm_extract_epi16(counter1, 2) + _mm_extract_epi16(counter1, 4) + _mm_extract_epi16(counter1, 6);
    int sum2 = _mm_extract_epi16(counter2, 0) + _mm_extract_epi16(counter2, 2) + _mm_extract_epi16(counter2, 4) + _mm_extract_epi16(counter2, 6);
    int sum3 = _mm_extract_epi16(counter3, 0) + _mm_extract_epi16(counter3, 2) + _mm_extract_epi16(counter3, 4) + _mm_extract_epi16(counter3, 6);
    int sum4 = _mm_extract_epi16(counter4, 0) + _mm_extract_epi16(counter4, 2) + _mm_extract_epi16(counter4, 4) + _mm_extract_epi16(counter4, 6);

    return sum1 + sum2 + sum3 + sum4;
}

// call each optimization with a switch and display their elapsed time
void getSum(int *db[DIM], int type) {
    startClock(start);

    int sum = 0;

    switch (type) {
        case NO_OPTIMIZATION:
            sum = noOptimizationSum(db);
        break;
        case GENERAL_OPTIMIZATION:
            sum = generalOptimizationSum(db);
        break;
        case SIMD_OPTIMIZATION:
            sum = simdOptimizationSum(db);
        break;
        case OPENMP_OPTIMIZATION:
            sum = openMPOptimization(db);
        break;
        case ALL_OPTIMIZATION:
            sum = allOptimizations(db);
        break;
        default:
            printf("Please choose a proper optimization type\n\n");
        break;
    }

    printf("Total sum with %s optimizations is: %d", optimizationTypes[type], sum);

    stopClock(start, stop, ellapsedTime);
}

int main() {
    int* dataBlock[DIM];

    populateBlock(dataBlock);

    getSum(dataBlock, NO_OPTIMIZATION);
    getSum(dataBlock, GENERAL_OPTIMIZATION);
    getSum(dataBlock, SIMD_OPTIMIZATION);
    getSum(dataBlock, OPENMP_OPTIMIZATION);
    getSum(dataBlock, ALL_OPTIMIZATION);
}

