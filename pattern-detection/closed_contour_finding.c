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

int load_img(char* output_path, int* w, int* h, int* c, unsigned char** img) {
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
        return 0;
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
        return 0;
    }


    // ساختن مسیر کامل فایل ورودی
    char input_path[512];
    sprintf(input_path, "pictures/inputs/%s", filenames[img_choice - 1]); // ساختن مسیر کامل فایل ورودی (مثلاً اگر ورودی car.jpg باشه، مسیر میشه pictures/inputs/car.jpg)

    // ساختن مسیر کامل فایل خروجی (مثلاً اگر ورودی car.jpg باشه، خروجی میشه out_car.jpg.png)
    sprintf(output_path, "pictures/outputs/out_%s", filenames[img_choice - 1]);

    printf("\nLoading %s...\n", input_path);
    
    // خواندن عکس به صورت سیاه و سفید (Grayscale)
    *img = stbi_load(input_path, &width, &height, &channels, 1); 
    
    if (img == NULL) {
        printf("Error: Could not load image!\n");
        return 0;
    }
    printf("Image \"%s\" loaded! Width: %d, Height: %d\n", filenames[img_choice - 1], width, height);

    *w = width;
    *h = height;
    *c = channels;    

    return 1;
}

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


// پارامتر is_dilation: اگر 1 باشد عمل گسترش و اگر 0 باشد عمل سایش انجام می‌شود
void morphology(unsigned char* src, unsigned char* dst, int width, int height, int is_dilation) {
    // از سطر و ستون 1 تا یکی مانده به آخر می‌رویم تا از لبه‌های عکس خارج نشویم
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int idx = y * width + x;
            
            unsigned char extreme_val = is_dilation ? 0 : 255; 

            // بررسی پنجره ۳ در ۳ (۸ همسایه + خودش)
            for(int dy = -1; dy <= 1; dy++) {
                for(int dx = -1; dx <= 1; dx++) {
                    int neighbor_idx = (y + dy) * width + (x + dx);
                    unsigned char val = src[neighbor_idx];
                    
                    if (is_dilation) {
                        if (val > extreme_val) extreme_val = val; // پیدا کردن روشن‌ترین
                    } else {
                        if (val < extreme_val) extreme_val = val; // پیدا کردن تاریک‌ترین
                    }
                }
            }
            dst[idx] = extreme_val;
        }
    }
}

int find_contour(int p0_x, int p0_y, int* contour_x, int* contour_y, unsigned char* img, int width, int height) { //طول خم را برمیگرداند، اگر خم بسته بود، در غیر این صورت -1 برمیگرداند(البته همیشه خم بسته حساب میکنه و فقط وقتی نویز باشه -1 میده)
    //پیدا کردن خم بسته
    int contour_len = 0;
    int dx[8] = { 0,  1,  1,  1,  0, -1, -1, -1};
    int dy[8] = {-1, -1,  0,  1,  1,  1,  0, -1};   

    int current_x = p0_x;
    int current_y = p0_y;
    int backtrack_dir = 6; // خونه قبلی سمت چپ بوده  

    int p1_x = -1, p1_y = -1; // نقطه بعد از شروع که باید پیدا بشه تا بدونیم خم بسته شده یا نه
    while(1) {
        contour_x[contour_len] = current_x;
        contour_y[contour_len] = current_y;
        contour_len++;

        int next_x = -1, next_y = -1;
        int found_next = 0;

        for (int i = 0; i < 8; i++) {
            // رادار از پیکسل خالی شروع به چرخش ساعت‌گرد می‌کند
            int check_dir = (backtrack_dir + i) % 8;
            int nx = current_x + dx[check_dir];
            int ny = current_y + dy[check_dir];

            if (img[ny * width + nx]) {
                // پیکسل مرزی بعدی پیدا شد
                next_x = nx;
                next_y = ny;
                
                // پیکسل پشتیبان برای حرکت بعدی، می‌شه پیکسلی که دقیقاً قبل از این چک کردیم (و خالی بود)
                if (check_dir % 2 == 0) {
                    // اگر حرکت مستقیم بوده (بالا، راست، پایین، چپ)
                    // باید جهتِ مخالف + 2 قدم ساعت‌گرد بچرخیم
                    backtrack_dir = (check_dir + 4 + 2) % 8; // معادل (check_dir + 6) % 8
                } else {
                    // اگر حرکت مورب (گوشه‌ها) بوده
                    // باید جهتِ مخالف + 1 قدم ساعت‌گرد بچرخیم
                    backtrack_dir = (check_dir + 4 + 1) % 8; // معادل (check_dir + 5) % 8
                }
                found_next = 1;
                break;
            }
        }

        if (!found_next) {
            return -1; // شکل فقط ۱ پیکسل بوده (نویز)
        }

        // اگر P1 هنوز مقداردهی نشده، الان اولین قدم ماست، پس ثبتش می‌کنیم
        if (p1_x == -1) {
            p1_x = next_x;
            p1_y = next_y;
        } else if (current_x == p0_x && current_y == p0_y && next_x == p1_x && next_y == p1_y) {
            //  بررسی شرط توقف جیکوب (Jacob's Stopping Criterion)
            // آیا الان روی P0 هستیم و قدم بعدی‌مان دقیقاً P1 است؟
            contour_len--; // چون آخرین نقطه‌ای که اضافه کردیم P0 است، باید یک واحد از طول کم کنیم
            break;
        }

        // حرکت به پیکسل بعدی
        current_x = next_x;
        current_y = next_y;
    }
    return contour_len;
}

void flood_erase(unsigned char* img, int width, int height, int start_x, int start_y) {
    // الگوریتم bfs
    // تخصیص حافظه برای Stack (بدترین حالت اینه که کل عکس روشن باشه)
    int* stack_x = (int*)malloc(width * height * sizeof(int));
    int* stack_y = (int*)malloc(width * height * sizeof(int));
    
    if (stack_x == NULL || stack_y == NULL) {
        return; 
    }

    int top = 0;

    // قرار دادن نقطه شروع در پشته
    stack_x[top] = start_x;
    stack_y[top] = start_y;
    top++;

    // آرایه‌های جهت برای بررسی 8 همسایه (بالا، پایین، چپ، راست و 4 تا قطری)
    int dx[8] = {-1,  0,  1, -1, 1, -1, 0, 1};
    int dy[8] = {-1, -1, -1,  0, 0,  1, 1, 1};

    // تا زمانی که پشته خالی نشده، ادامه بده
    while (top > 0) {
        // برداشتن آخرین پیکسل از پشته (Pop)
        top--;
        int x = stack_x[top];
        int y = stack_y[top];

        // بررسی اینکه از کادر عکس خارج نشده باشیم
        if (x < 0 || x >= width || y < 0 || y >= height) {
            continue;
        }

        int index = y * width + x;

        //  اگر پیکسل روشن است (یعنی جزئی از مرز یا دمِ چسبیده به آن است)
        if (img[index]) { 
            
            // نابودش کن! (تاریک کردن پیکسل)
            img[index] = 0;

            // 3. حالا هر 8 همسایه این پیکسل را به پشته اضافه کن تا چک بشن
            for (int i = 0; i < 8; i++) {
                int nx = x + dx[i];
                int ny = y + dy[i];
                if (nx >= 0 && nx < width && ny >= 0 && ny < height && img[ny * width + nx]) {
                    stack_x[top] = nx;
                    stack_y[top] = ny;
                    top++;
                }
            }
        }
    }

    // آزادسازی حافظه برای جلوگیری از Memory Leak
    free(stack_x);
    free(stack_y);
}

int main() {
    
    float sobel_kernel_v[9] = {
        -1.0f,  0.0f,  1.0f,
        -2.0f,  0.0f,  2.0f,
        -1.0f,  0.0f,  1.0f
    };
    float sobel_kernel_h[9] = {
        -1.0f, -2.0f, -1.0f,
         0.0f,  0.0f,  0.0f,
         1.0f,  2.0f,  1.0f
    };

    int width, height, channels;

    unsigned char *img = NULL; // اشاره‌گری برای ذخیره عکس ورودی
    char output_path[512]; // آرایه‌ای برای ذخیره مسیر فایل خروجی
    if(load_img(output_path, &width, &height, &channels, &img) <= 0){ // بارگذاری عکس و دریافت ابعاد و کانال‌ها    
        return 0;
    }

    // تبدیل اعداد صحیح به اعداد اعشاری برای کار با کرنل
    int total_pixels = width * height;
    float *input_float = (float *)malloc(total_pixels * sizeof(float));
    float *result_v = (float *)malloc(total_pixels * sizeof(float)); 
    float *result_h = (float *)malloc(total_pixels * sizeof(float)); 
    float *sobel_result = (float *)malloc(total_pixels * sizeof(float)); 
    unsigned char *final_img = (unsigned char *)malloc(total_pixels);
    unsigned char *temp1 = (unsigned char *)malloc(total_pixels);
    unsigned char *temp2 = (unsigned char *)malloc(total_pixels);
    unsigned char *output_img = (unsigned char *)malloc(total_pixels * 3); // آرایه‌ای برای ذخیره تصویر خروجی رنگی (RGB)


    for (int i = 0; i < total_pixels; i++) {
        input_float[i] = (float)img[i];
        result_v[i] = 0.0f; 
        result_h[i] = 0.0f;
        sobel_result[i] = 0.0f;
        temp1[i] = 0;
        temp2[i] = 0;
        final_img[i] = 0;
    }

    clock_t start_c = clock();
    convolution_c(input_float, result_v, width, height, sobel_kernel_v); // محاسبه لبه‌های عمودی
    convolution_c(input_float, result_h, width, height, sobel_kernel_h); // محاسبه لبه‌های افقی
    for (int i = 0; i < total_pixels; i++) {
        sobel_result[i] = sqrtf(result_v[i] * result_v[i] + result_h[i] * result_h[i]); // ترکیب لبه‌های عمودی و افقی
    }

    for (int i = 0; i < total_pixels; i++) {
        float val = sobel_result[i];
        if (val > 255.0f) val = 255.0f; // اگر خروجی از 255 بیشتر شد روی 255 قفل بشه
        if (val < 128.0) val = 0.0f; // اگر مقدار لبه ضعیف بود (کمتر از 100) اون رو صفر کنیم تا نویز حذف بشه
        
        temp1[i] = (unsigned char) val; // تبدیل به عدد صحیح برای ذخیره عکس
    }

    // closing (بستن شکاف ها) : dilation -> erosion
    morphology(temp1, temp2, width, height, 1); // گسترش
    morphology(temp2, final_img, width, height, 0); // سایش
    
    // for (int i = 0; i < total_pixels; i++) {
    //     output_img[i * 3 + 0] = final_img[i]; // کانال R
    //     output_img[i * 3 + 2] = final_img[i]; // کانال B
    //     output_img[i * 3 + 1] = final_img[i]; // کانال G
    // }
   
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            if (final_img[y * width + x] ) {
                int contour_x[10000]; // آرایه‌ای برای ذخیره مختصات x نقاط مرزی
                int contour_y[10000]; // آرایه‌ای برای ذخیره مختصات y نقاط مرزی
                int contour_length = find_contour(x, y, contour_x, contour_y, final_img, width, height);
                
                if (contour_length > 0) {
                    printf("Found a closed contour with length and start at (%d, %d): %d\n", x, y, contour_length);
                    for (int i = 0; i < contour_length; i++) {
                        int idx = contour_y[i] * width + contour_x[i];
                        output_img[idx * 3 + 0] = 255; // رنگ سفید برای نقاط مرزی (در کانال R)
                        output_img[idx * 3 + 1] = 0;   // رنگ سیاه برای نقاط مرزی (در کانال G)
                        output_img[idx * 3 + 2] = 0;   // رنگ سیاه برای نقاط مرزی (در کانال B)
                    }
                }
                //پاک کردن نقاط مرزی از تصویر نهایی تا در جستجوی بعدی دوباره پیدا نشوند
                flood_erase(final_img, width, height, x, y);
            }
        }
    }

    clock_t end_c = clock();

    double elapsed_time = ((double)(end_c - start_c)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %f seconds\n", elapsed_time);

    //ذخیره عکس خروجی
    printf("Saving image...\n");
    // ذخیره عکس به صورت PNG
    stbi_write_png(output_path, width, height, 3, output_img, width * 3);
    // char *output_path_2 = (char *)malloc(600 * sizeof(char));
    // sprintf(output_path_2, "%s_org.png", output_path);
    // stbi_write_png(output_path_2, width, height, 1, final_img, width);
    // عرض تصویر * تعداد کانال رنگ * سایز هر پیکسل = stride_bytes(گام)
    // تعریف stride = پرش به خط بعد ممکنه عکست 1920*1080 باشه ولی تو یه بخش 100*100 رو بخوای اینطوری طول و عرض همون 100 عه اما پرش 1920 هست
    printf("Done!\n");

    //آزاد سازی حافظه
    stbi_image_free(img);
    free(input_float);
    free(final_img);
    free(temp1);
    free(temp2);
    free(result_v);
    free(result_h);
    free(sobel_result);
    free(output_img);
    // free(output_path_2);

    return 0;
}