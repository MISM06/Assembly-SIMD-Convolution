#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <dirent.h>
#include <string.h> 
#include <immintrin.h> // برای دستورات اسمبلی AVX

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

// لود کردن وزن‌هایی که از پایتون گرفتیم
#include "weights.h" 

// یه عکس 28 در 28 میگیره، 4 تا کرنل 3*3 اعمال میکنه و 4 تا عکس 26*26 میده
void convolution_c(float* input, float* output, float* kernels) {
    for (int c = 0; c < 4; c++) { // 4 تا کرنل داریم
        for (int i = 0; i < 26; i++) { // ارتفاع خروجی
            for (int j = 0; j < 26; j++) { // عرض خروجی
                float sum = 0.0f;
                // حرکت دادن کرنل 3x3 روی تصویر
                for (int ki = 0; ki < 3; ki++) {
                    for (int kj = 0; kj < 3; kj++) {
                        int in_row = i + ki;
                        int in_col = j + kj;
                        // ضرب پیکسل تصویر در وزن کرنل
                        sum += input[in_row * 28 + in_col] * kernels[c * 9 + ki * 3 + kj];
                    }
                }
                output[c * 26 * 26 + i * 26 + j] = sum;
            }
        }
    }
}

void convolution_asm(float* input, float* output, float* kernels) {
    for (int c = 0; c < 4; c++) { 
        for (int i = 0; i < 26; i++) {
            // حلقه اصلی با گام 8 تایی (چون رجیستر 256 بیتی AVX جا برای 8 تا float داره)
            int j = 0;
            for (; j <= 26 - 8; j += 8) {
                __m256 sum = _mm256_setzero_ps(); // vxorps
                
                for (int ki = 0; ki < 3; ki++) {
                    for (int kj = 0; kj < 3; kj++) {
                        // vmovups
                        __m256 pixels = _mm256_loadu_ps(&input[(i + ki) * 28 + (j + kj)]);
                        // broadcast
                        __m256 weight = _mm256_set1_ps(kernels[c * 9 + ki * 3 + kj]);
                        // vmulps and vaddps
                        sum = _mm256_fmadd_ps(pixels, weight, sum);
                    }
                }
                // vmovps
                _mm256_storeu_ps(&output[c * 26 * 26 + i * 26 + j], sum);
            }
            
            // یه دور هم 8 پیکسل آخر رو حساب میکنیم که 2 پیکسل باقی مونده هم محاسبه بشوند
            j = 26 - 8;
            __m256 sum = _mm256_setzero_ps(); // vxorps
                
            for (int ki = 0; ki < 3; ki++) {
                for (int kj = 0; kj < 3; kj++) {
                    // vmovups
                    __m256 pixels = _mm256_loadu_ps(&input[(i + ki) * 28 + (j + kj)]);
                    // broadcast
                    __m256 weight = _mm256_set1_ps(kernels[c * 9 + ki * 3 + kj]);
                    // vmulps and vaddps
                    sum = _mm256_fmadd_ps(pixels, weight, sum);
                }
            }
            // vmovps
            _mm256_storeu_ps(&output[c * 26 * 26 + i * 26 + j], sum);

        }
    }
}

// اعداد کمتر از 0 رو 0 میکنه(فعالساز)
void relu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0) data[i] = 0.0f;
    }
}

// هر مربع 2*2 رو میگیره و بیشترین عدد آن را انتخاب میکند، ابعاد عکس نصف میشود(13*13)
void max_pooling(float* input, float* output) {
    for (int c = 0; c < 4; c++) {
        for (int i = 0; i < 13; i++) {
            for (int j = 0; j < 13; j++) {
                float max_val = -1e9;
                for (int ki = 0; ki < 2; ki++) {
                    for (int kj = 0; kj < 2; kj++) {
                        float val = input[c * 26 * 26 + (i * 2 + ki) * 26 + (j * 2 + kj)];
                        if (val > max_val) max_val = val;
                    }
                }
                output[c * 13 * 13 + i * 13 + j] = max_val;
            }
        }
    }
}

// لایه تصمیم‌گیری نهایی (Fully Connected)
// 13 * 13 * 4 تا عدد رو میگیره(4 تا ماتریس 13*13 داشتیم) و امتیاز هر رقم رو حساب میکنه
int predict_digit(float* flat_input) {
    float max_score = -1e9;
    int best_digit = -1;
    
    for (int i = 0; i < 10; i++) {
        float score = fc_bias[i];
        for (int j = 0; j < 4 * 13 * 13; j++) { 
            score += fc_weight[i][j] * flat_input[j];
        }
        
        // printf("Class %d: Score = %f\n", i, score);
        if (score > max_score) {
            max_score = score;
            best_digit = i;
        }
    }
    return best_digit;
}


double predict_c(unsigned char* img) {
    int width, height, channels;

    // اختصاص حافظه
    float* img_float = (float*)malloc(28 * 28 * sizeof(float));
    float* conv_output = (float*)malloc(4 * 26 * 26 * sizeof(float));
    float* pooling_output = (float*)malloc(4 * 13 * 13 * sizeof(float));
    
    // نرمال‌سازی عکس (0 تا 1)
    for (int i = 0; i < 28 * 28; i++) {
        img_float[i] = (img[i] / 255.0f);
    }

    clock_t start, end;
    double time_c;

    start = clock();
    convolution_c(img_float, conv_output, conv_kernels);
    
    end = clock();
    time_c = ((double)(end - start)) / (CLOCKS_PER_SEC / 1000.0);
    relu(conv_output, 4 * 26 * 26);
    max_pooling(conv_output, pooling_output);
    int predicted_digit = predict_digit(pooling_output);

    free(img_float);
    free(conv_output);
    free(pooling_output);
    stbi_image_free(img);
    return time_c;
}

double predict_asm(unsigned char* img) {
    int width, height, channels;

    // اختصاص حافظه
    float* img_float = (float*)malloc(28 * 28 * sizeof(float));
    float* conv_output = (float*)malloc(4 * 26 * 26 * sizeof(float));
    float* pooling_output = (float*)malloc(4 * 13 * 13 * sizeof(float));
    
    // نرمال‌سازی عکس (0 تا 1)
    for (int i = 0; i < 28 * 28; i++) {
        img_float[i] = (img[i] / 255.0f);
    }

    clock_t start, end;
    double time_asm;

    start = clock();
    convolution_asm(img_float, conv_output, conv_kernels);
    
    end = clock();
    time_asm = ((double)(end - start)) / (CLOCKS_PER_SEC / 1000.0);
    relu(conv_output, 4 * 26 * 26);
    max_pooling(conv_output, pooling_output);
    int predicted_digit = predict_digit(pooling_output);
    
    free(img_float);
    free(conv_output);
    free(pooling_output);
    stbi_image_free(img);
    return time_asm;
}

int main() {
    int width, height, channels;
    
    // خواندن لیست عکس‌ ها
    DIR *d;
    struct dirent *dir;
    char filenames[2000][256]; 
    int file_count = 0;

    d = opendir("pictures/tests");
    
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            // فیلتر کردن فایل‌های مخفی سیستم و پوشه‌های . و ..
            if (dir->d_name[0] != '.') {
                strcpy(filenames[file_count], dir->d_name);
                file_count++;
            }
        }
        closedir(d);
    } else {
        printf("Error: Could not open directory 'tests'.\n");
        return 0;
    }

    if (file_count == 0) {
        printf("No images found in the folder!\n");
        return 0;
    }

    double total_time_c = 0.0;
    double total_time_asm = 0.0;

    int proccesed_images = 0;

    for (int image_choice = 0; image_choice < file_count; image_choice++) {
            char input_path[512];
        sprintf(input_path, "pictures/tests/%s", filenames[image_choice]);

        unsigned char *img = stbi_load(input_path, &width, &height, &channels, 1);
        if (img == NULL) {
            printf("Error: Could not load image %s for C version!\n", filenames[image_choice]);
            continue;
        }
        // فرض می‌کنیم یه عکس به اسم test.jpg داری (سایز حتما باید 28*28 باشه)
        if (width != 28 || height != 28) {
            printf("Error: Image %s is %d * %d but it should be 28 * 28\n", filenames[image_choice], height, width);
            continue;
        }

        double time_c, time_asm;
        time_c = predict_c(img);
        total_time_c += time_c;

        // دوباره عکس رو بارگذاری می‌کنیم چون تابع predict_c حافظه عکس رو آزاد می‌کنه
        img = stbi_load(input_path, &width, &height, &channels, 1);
        if (img == NULL) {
            printf("Error: Could not load image %s for ASM version!\n", filenames[image_choice]);
            continue;
        }

        ++proccesed_images;

        time_asm = predict_asm(img);
        total_time_asm += time_asm;
       
    }
    printf("\n==================== Performance Summary ====================\n");
    printf("program ran over %d images.\n", proccesed_images);
    printf("⏱️ C version took: %.2f miliseconds\n", total_time_c);
    printf("⏱️ Assembly version took: %.2f miliseconds\n", total_time_asm);
    double speed_up = total_time_c / total_time_asm;
    printf("\n🚀 SPEEDUP: Assembly is %.2f times faster than C!\n\n", speed_up);

    return 0;
}