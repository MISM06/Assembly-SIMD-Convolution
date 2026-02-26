#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <dirent.h>
#include <string.h> 

//پیاده سازی کد های داخل کتابخونه در همین جا
//بخش پیاده سازی یعنی فعال سازی کد های داخل header
#define STB_IMAGE_IMPLEMENTATION 
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

extern void apply_convolution_avx(float *input, float *output, int width, int height, float *kernel);

void convolution_c(float *input, float *output, int width, int height, float *kernel) {
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            float sum = 0.0;
            
            //اعمال کرنل روی پیکسل (i,j)
            for (int ki = -1; ki <= 1; ki++) {
                for (int kj = -1; kj <= 1; kj++) {
                    int pixel_index = (i + ki) * width + (j + kj);
                    int kernel_index = (ki + 1) * 3 + (kj + 1);
                    sum += input[pixel_index] * kernel[kernel_index];
                }
            }
            
            output[i * width + j] = sum;
        }
    }
}

int main() {

    float kernel_edge[9] = {
        -1.0f, -1.0f, -1.0f,
        -1.0f,  8.0f, -1.0f,
        -1.0f, -1.0f, -1.0f
    };

    float kernel_sharpen[9] = {
         0.0f, -1.0f,  0.0f,
        -1.0f,  5.0f, -1.0f,
         0.0f, -1.0f,  0.0f
    };

    float kernel_blur[9] = {
        0.111f, 0.111f, 0.111f,
        0.111f, 0.111f, 0.111f,
        0.111f, 0.111f, 0.111f
    };

    float kernel_emboss[9] = {
        -2.0f, -1.0f,  0.0f,
        -1.0f,  1.0f,  1.0f,
         0.0f,  1.0f,  2.0f
    };

    int width, height, channels;

    // خواندن لیست عکس‌ها و انتخاب توسط کاربر
    DIR *d;
    struct dirent *dir;
    char filenames[50][256]; // آرایه‌ای برای ذخیره نام حداکثر 50 عکس
    int file_count = 0;

    printf("\n--- Available Images in 'pictures/inputs/' ---\n");
    d = opendir("pictures/inputs");
    
    if (d) {
        while ((dir = readdir(d)) != NULL) {
            // فیلتر کردن فایل‌های مخفی سیستم و پوشه‌های . و ..
            if (dir->d_name[0] != '.') {
                strcpy(filenames[file_count], dir->d_name);
                printf("%d. %s\n", file_count + 1, filenames[file_count]);
                file_count++;
            }
        }
        closedir(d);
    } else {
        printf("Error: Could not open directory 'pictures/inputs/'.\n");
        return 1;
    }

    if (file_count == 0) {
        printf("No images found in the folder!\n");
        return 0;
    }

    printf("Select an image (1-%d): ", file_count);

    int img_choice = 0;

    scanf("%d", &img_choice);

    if (img_choice < 1 || img_choice > file_count) {
        printf("Invalid image choice!\n");
        return 1;
    }


    // ساختن مسیر کامل فایل ورودی
    char input_path[512];
    sprintf(input_path, "pictures/inputs/%s", filenames[img_choice - 1]); // ساختن مسیر کامل فایل ورودی (مثلاً اگر ورودی car.jpg باشه، مسیر میشه pictures/inputs/car.jpg)

    // ساختن مسیر کامل فایل خروجی (مثلاً اگر ورودی car.jpg باشه، خروجی میشه out_car.jpg.png)
    char output_path[512];
    sprintf(output_path, "pictures/outputs/out_%s.png", filenames[img_choice - 1]);

    printf("\nLoading %s...\n", input_path);
    
    // خواندن عکس به صورت سیاه و سفید (Grayscale)
    unsigned char *img = stbi_load(input_path, &width, &height, &channels, 1); 
    
    if (img == NULL) {
        printf("Error: Could not load image!\n");
        return 1;
    }
    printf("Image \"%s\" loaded! Width: %d, Height: %d\n", filenames[img_choice - 1], width, height);

    // تبدیل اعداد صحیح به اعداد اعشاری برای کار با کرنل
    int total_pixels = width * height;
    float *input_float = (float *)malloc(total_pixels * sizeof(float)); // اختصاص داده حافظه به آرایه ورودی
    float *output_float = (float *)malloc(total_pixels * sizeof(float)); // اختصاص دادن حافظه به آرایه خروجی
    for (int i = 0; i < total_pixels; i++) {
        input_float[i] = (float)img[i];
        output_float[i] = 0.0f; 
    }

    printf("please enter your filter choice (1: Edge Detection, 2: Sharpen, 3: Blur, 4: Emboss): ");
    int choice;
    scanf("%d", &choice);
    float *selected_kernel;
    switch(choice) {
        case 1: selected_kernel = kernel_edge; printf("Edge Detection selected.\n"); break;
        case 2: selected_kernel = kernel_sharpen; printf("Sharpen selected.\n"); break;
        case 3: selected_kernel = kernel_blur; printf("Blur selected.\n"); break;
        case 4: selected_kernel = kernel_emboss; printf("Emboss selected.\n"); break;
        default: selected_kernel = kernel_edge; printf("Invalid choice! Defaulting to Edge.\n"); break;
    }

    printf("Warming up the cache...\n"); // اجرای اولیه برای پر کردن کش و جلوگیری از تاثیر آن روی تست سرعت
    convolution_c(input_float, output_float, width, height, selected_kernel);
    apply_convolution_avx(input_float, output_float, width, height, selected_kernel);


    printf("\nRunning C implementation...\n");
    clock_t start_c = clock();
    convolution_c(input_float, output_float, width, height, selected_kernel);
    clock_t end_c = clock();
    double time_c = ((double)(end_c - start_c)) / CLOCKS_PER_SEC;
    printf("⏱️ C version took: %f seconds\n", time_c);

    // ۲. صفر کردن آرایه خروجی (تا تقلب نشه و اسمبلی از صفر شروع کنه)
    for (int i = 0; i < total_pixels; i++) {
        output_float[i] = 0.0f;
    }

    // ۳. تست سرعت اسمبلی (AVX)
    printf("Running Assembly (AVX) implementation...\n");
    clock_t start_asm = clock();
    apply_convolution_avx(input_float, output_float, width, height, selected_kernel);
    clock_t end_asm = clock();
    double time_asm = ((double)(end_asm - start_asm)) / CLOCKS_PER_SEC;
    printf("⏱️ Assembly version took: %f seconds\n", time_asm);

    // ۴. محاسبه ریاضیِ نسبت تسریع
    if (time_asm > 0) {
        double speedup = time_c / time_asm;
        printf("\n🚀 SPEEDUP: Assembly is %.2f times faster than C!\n\n", speedup);
    } else {
        printf("\n⚠️ Assembly version time is too small to calculate speedup accurately.\n\n");
    }

    unsigned char *final_img = (unsigned char *)malloc(total_pixels); // عکس خروجی
    for (int i = 0; i < total_pixels; i++) {
        float val = output_float[i];
        if (val > 255.0f) val = 255.0f; // اگر خروجی از 255 بیشتر شد روی 255 قفل بشه
        if (val < 0.0f) val = 0.0f; // اگر کمتر از 0 شد روی 0
        
        final_img[i] = (unsigned char) val; // تبدیل به عدد صحیح برای ذخیره عکس
    }

    //ذخیره عکس خروجی
    printf("Saving image...\n");
    // ذخیره عکس به صورت PNG
    stbi_write_png(output_path, width, height, 1, final_img, width);
     // عرض تصویر * تعداد کانال رنگ * سایز هر پیکسل = stride_bytes(گام)
    // تعریف stride = پرش به خط بعد ممکنه عکست 1920*1080 باشه ولی تو یه بخش 100*100 رو بخوای اینطوری طول و عرض همون 100 عه اما پرش 1920 هست
    printf("Done!\n");

    //آزاد سازی حافظه
    stbi_image_free(img); // آزاد سازی حافظه عکس ورودی
    free(input_float); // آزاد سازی حافظه آرایه ورودی
    free(output_float); // آزاد سازی حافظه آرایه خروجی
    free(final_img); // آزاد سازی حافظه عکس خروجی

    return 0;
}