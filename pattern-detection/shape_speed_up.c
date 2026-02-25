#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <dirent.h>
#include <string.h> 
#include <math.h>

//پیاده سازی کد های داخل کتابخونه در همین جا
//بخش پیاده سازی یعنی فعال سازی کد های داخل header
#define STB_IMAGE_IMPLEMENTATION 
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

// فراخوانی تابع اسمبلی ما
extern void apply_convolution_avx(float *input, float *output, int width, int height, float *kernel);
extern void sobel_magnitude_asm(float* res_v, float* res_h, unsigned char* out_img, int total_pixels);
extern void erosion_asm(unsigned char* src, unsigned char* dst, int width, int height);
extern void dilation_asm(unsigned char* src, unsigned char* dst, int width, int height);

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

double calc_area(int* contour_x, int* contour_y, int contour_len) {
    // محاسبه مساحت داخل خم بسته با استفاده از فرمول شانون (Shoelace formula)
    double area = 0.0;
    for (int i = 0; i < contour_len; i++) {
        int j = (i + 1) % contour_len; // اندیس بعدی (با حلقه زدن)
        area += contour_x[i] * contour_y[j] - contour_x[j] * contour_y[i];
    }
    return fabs(area) / 2.0; // مساحت مثبت است
}

double calc_perimeter(int* contour_x, int* contour_y, int contour_len) {
    // محاسبه محیط خم بسته با جمع کردن فاصله بین نقاط متوالی
    double rad2 = sqrt(2.0); // فاصله بین نقاط مورب
    double perimeter = 0.0;
    for (int i = 0; i < contour_len; i++) {
        int j = (i + 1) % contour_len; // اندیس بعدی (با حلقه زدن)
        double dx = contour_x[j] - contour_x[i];
        double dy = contour_y[j] - contour_y[i];
        // perimeter += sqrt(dx * dx + dy * dy);
        if (dx == 0 || dy == 0) {
            perimeter += 1.0; // اگر حرکت مستقیم باشد، فاصله 1 است
        } else {
            perimeter += rad2; // اگر حرکت مورب باشد، فاصله sqrt(2) است
        }
    }
    return perimeter;
}

void calc_oriented_bbox(int* contour_x, int* contour_y, int contour_len, double* best_h, double* best_w, double* sym_score) {
    if (contour_len == 0) return;

    //پیدا کردن مرکز ثقل
    double cx = 0, cy = 0;
    for (int i = 0; i < contour_len; i++) {
        cx += contour_x[i];
        cy += contour_y[i];
    }
    cx /= contour_len;
    cy /= contour_len;

    // محاسبه گشتاور های مرکزی برای محاسبه زاویه چرخش
    double mu20 = 0, mu02 = 0, mu11 = 0;
    for (int i = 0; i < contour_len; i++) {
        double dx = contour_x[i] - cx;
        double dy = contour_y[i] - cy;
        
        // Var(X), Var(Y), Cov(X,Y) 
        mu20 += dx * dx;       
        mu02 += dy * dy;       
        mu11 += dx * dy;       
    }

    //بدست آوردن زاویه چرخش بر حسب گرادیان
    double theta = 0.5 * atan2(2.0 * mu11, mu20 - mu02);
    

    // دوران دادن نقاط خم به اندازه قرینه زاویه چرخش تا طول و عرض واقعی را در حالت افقی حساب کنیم
    double min_x_rot = 1e9, max_x_rot = -1e9; 
    double min_y_rot = 1e9, max_y_rot = -1e9;

    //حالت غیر چرخشی هم حساب میکنیم برای حالتایی مثل دایره یا مربع تقریبی که تانژات 0/0 میشود به صورت تئوری اما عملی چرخش 90 درجه میدهد که ابعاد را خراب میکند
    double min_x_norm = 1e9, max_x_norm = -1e9; 
    double min_y_norm = 1e9, max_y_norm = -1e9;

    double cos_t = cos(-theta);
    double sin_t = sin(-theta);

    for (int i = 0; i < contour_len; i++) {
        // انتقال نقطه به مرکز (مبدأ مختصات)
        double dx = contour_x[i] - cx;
        double dy = contour_y[i] - cy;

        // ضرب در ماتریس دوران
        double rx = dx * cos_t - dy * sin_t;
        double ry = dx * sin_t + dy * cos_t;
        /*
            [ cos(t)  -sin(t) ]
            [ sin(t)   cos(t) ]
        */

        if (rx < min_x_rot) min_x_rot = rx;
        if (rx > max_x_rot) max_x_rot = rx;
        if (ry < min_y_rot) min_y_rot = ry;
        if (ry > max_y_rot) max_y_rot = ry;

        int x = contour_x[i];
        int y = contour_y[i];
        if (x < min_x_norm) min_x_norm = x;
        if (x > max_x_norm) max_x_norm = x;
        if (y < min_y_norm) min_y_norm = y;
        if (y > max_y_norm) max_y_norm = y;
    }

    double h_rot = max_y_rot - min_y_rot;
    double w_rot = max_x_rot - min_x_rot;
    double h_norm = max_y_norm - min_y_norm;
    double w_norm = max_x_norm - min_x_norm;

    double area_obb = h_rot * w_rot;
    double area_bbox = h_norm * w_norm;
    if (area_obb < area_bbox) {
        //چرخیده
        *best_h = h_rot;
        *best_w = w_rot;

        // محاسبه تقارن برای جعبه چرخیده (در فضای شیفت داده شده مبدأ 0و0 است)
        double box_cx = (min_x_rot + max_x_rot) / 2.0;
        double box_cy = (min_y_rot + max_y_rot) / 2.0;
        double max_dim = (w_rot > h_rot) ? w_rot : h_rot;
        *sym_score = sqrt(box_cx*box_cx + box_cy*box_cy) / max_dim;
    } else {
        //نچرخیده
        *best_h = h_norm;
        *best_w = w_norm;
        // محاسبه تقارن برای جعبه عادی
        double box_cx = (min_x_norm + max_x_norm) / 2.0;
        double box_cy = (min_y_norm + max_y_norm) / 2.0;
        double dx = box_cx - cx;
        double dy = box_cy - cy;
        double max_dim = (w_norm > h_norm) ? w_norm : h_norm;
        *sym_score = sqrt(dx*dx + dy*dy) / max_dim;
    }

}

void classify_shape(double area, double perimeter, double real_height, double real_width, double symmetry_score) {
    
    // اگر مساحت خیلی کوچک بود، شکلی نیست
    if (area < 20.0) {
        // printf("Shape: Unknown / Noise\n");
        return;
    }

    double PI = 3.14159265358979323846;


    double obb_area = real_width * real_height;
    if (obb_area == 0.0) return;

    // شاخص پرشدگی
    double extent = area / obb_area;

    // شاخص گردی (دایره=1.0 ، مربع=0.78 ، مثلث=0.6)
    double circularity = (4.0 * PI * area) / (perimeter * perimeter);
    
    // نسبت ابعاد (عدد کوچکتر تقسیم بر بزرگتر تا بین 0 و 1 باشه)
    double min_dim = (real_width < real_height) ? real_width : real_height;
    double max_dim = (real_width > real_height) ? real_width : real_height;
    double aspect_ratio = min_dim / max_dim;


    //دایره و بیضی
    //گردی بالا
    if (circularity > 0.84) {
        if (aspect_ratio > 0.80) {
            // printf("Shape: Circle\n");
        } else {
            // printf("Shape: Ellipse\n");
        }
    }
    //مربع و مستطیل
    //extent بالا
    else if (extent > 0.80) {
        if (aspect_ratio > 0.80) {
            // printf("Shape: Square\n");
        } else {
            // printf("Shape: Rectangle\n");
        }
    }
    //لوزی و مثلث
    // extent حدود 0.5
    else if (extent > 0.40 && extent < 0.60) {
        // آستانه تقارنِ اصلاح شده برای هندل کردن مثلث‌های قائم‌الزاویه
        if (symmetry_score < 0.04) {
            // printf("Shape: Rhombus\n");
        } else {
            // printf("Shape: Triangle\n");
        }
    }
    //نامعلموم
    else {
        // printf("Shape: Unknown or Complex Polygon\n");
    }
}

double calc_asm(unsigned char* img, int width, int height) {
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
    // تبدیل اعداد صحیح به اعداد اعشاری برای کار با کرنل
    int total_pixels = width * height;
    float *input_float = (float *)malloc(total_pixels * sizeof(float));
    float *result_v = (float *)malloc(total_pixels * sizeof(float)); 
    float *result_h = (float *)malloc(total_pixels * sizeof(float)); 
    float *sobel_result = (float *)malloc(total_pixels * sizeof(float)); 
    unsigned char *final_img = (unsigned char *)malloc(total_pixels);
    unsigned char *temp1 = (unsigned char *)malloc(total_pixels);
    unsigned char *temp2 = (unsigned char *)malloc(total_pixels);

    for (int i = 0; i < total_pixels; i++) {
        input_float[i] = (float)img[i];
        result_v[i] = 0.0f; 
        result_h[i] = 0.0f;
        sobel_result[i] = 0.0f;
        temp1[i] = 0;
        temp2[i] = 0;
        final_img[i] = 0;
    }

    clock_t start_asm = clock();
    apply_convolution_avx(input_float, result_v, width, height, sobel_kernel_v); //لبه های عمودی
    apply_convolution_avx(input_float, result_h, width, height, sobel_kernel_h); //لبه های افقی
    sobel_magnitude_asm(result_v, result_h, temp1, total_pixels); //ترکیب لبه ها افقی و عمودی

    // closing (بستن شکاف ها) : dilation -> erosion
    dilation_asm(temp1, temp2, width, height);
    erosion_asm(temp2, final_img, width, height);

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            if (final_img[y * width + x] ) {
                int contour_x[10000]; // آرایه‌ای برای ذخیره مختصات x نقاط مرزی
                int contour_y[10000]; // آرایه‌ای برای ذخیره مختصات y نقاط مرزی
                int contour_length = find_contour(x, y, contour_x, contour_y, final_img, width, height);
                if (contour_length <= 0) continue; // اگر خم بسته‌ای پیدا نشد یا فقط یک پیکسل بود، ردش کن
              
                //پاک کردن نقاط مرزی از تصویر نهایی تا در جستجوی بعدی دوباره پیدا نشوند
                flood_erase(final_img, width, height, x, y);
                double area = calc_area(contour_x, contour_y, contour_length);
                if (area <= 20.0) continue; // اگر مساحت خیلی کوچک بود، احتمالاً نویز است، ردش کن
                for (int i = 0; i < contour_length; i++) {
                    int idx = contour_y[i] * width + contour_x[i];
                }
                double perimeter = calc_perimeter(contour_x, contour_y, contour_length);
                double best_h, best_w, sym_score;
                calc_oriented_bbox(contour_x, contour_y, contour_length, &best_h, &best_w, &sym_score);
                classify_shape(area, perimeter, best_h, best_w, sym_score);
                break; // اگر فقط میخوایم اولین شکل رو پیدا کنیم، این break رو بردار تا همه شکل‌ها رو پیدا کنه
            }
        }
    }
    

    clock_t end_asm = clock();

    double time = ((double)(end_asm - start_asm)) / (CLOCKS_PER_SEC / 1000.0); // زمان به میلی‌ثانیه

    //آزاد سازی حافظه
    stbi_image_free(img);
    free(input_float);
    free(final_img);
    free(temp1);
    free(temp2);
    free(result_v);
    free(result_h);
    free(sobel_result);

    return time;
}

double calc_c(unsigned char* img, int width, int height) {
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
    // تبدیل اعداد صحیح به اعداد اعشاری برای کار با کرنل
    int total_pixels = width * height;
    float *input_float = (float *)malloc(total_pixels * sizeof(float));
    float *result_v = (float *)malloc(total_pixels * sizeof(float)); 
    float *result_h = (float *)malloc(total_pixels * sizeof(float)); 
    float *sobel_result = (float *)malloc(total_pixels * sizeof(float)); 
    unsigned char *final_img = (unsigned char *)malloc(total_pixels);
    unsigned char *temp1 = (unsigned char *)malloc(total_pixels);
    unsigned char *temp2 = (unsigned char *)malloc(total_pixels);

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

    // آستانه‌گذاری برای حذف نویزهای ضعیف و قفل کردن مقادیر بالاتر از 255
    for (int i = 0; i < total_pixels; i++) {
        float val = sobel_result[i];
        if (val > 255.0f) val = 255.0f; // اگر خروجی از 255 بیشتر شد روی 255 قفل بشه
        if (val < 60.0) val = 0.0f; // اگر مقدار لبه ضعیف بود (کمتر از 60) اون رو صفر کنیم تا نویز حذف بشه
        
        temp1[i] = (unsigned char) val; // تبدیل به عدد صحیح برای ذخیره عکس
        
    }

    // closing (بستن شکاف ها) : dilation -> erosion
    morphology(temp1, temp2, width, height, 1); // گسترش
    morphology(temp2, final_img, width, height, 0); // سایش
   
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            if (final_img[y * width + x] ) {
                int contour_x[10000]; // آرایه‌ای برای ذخیره مختصات x نقاط مرزی
                int contour_y[10000]; // آرایه‌ای برای ذخیره مختصات y نقاط مرزی
                int contour_length = find_contour(x, y, contour_x, contour_y, final_img, width, height);
                if (contour_length <= 0) continue; // اگر خم بسته‌ای پیدا نشد یا فقط یک پیکسل بود، ردش کن
              
                //پاک کردن نقاط مرزی از تصویر نهایی تا در جستجوی بعدی دوباره پیدا نشوند
                flood_erase(final_img, width, height, x, y);
                double area = calc_area(contour_x, contour_y, contour_length);
                if (area <= 20.0) continue; // اگر مساحت خیلی کوچک بود، احتمالاً نویز است، ردش کن
                for (int i = 0; i < contour_length; i++) {
                    int idx = contour_y[i] * width + contour_x[i];
                }
                double perimeter = calc_perimeter(contour_x, contour_y, contour_length);
                double best_h, best_w, sym_score;
                calc_oriented_bbox(contour_x, contour_y, contour_length, &best_h, &best_w, &sym_score);
                classify_shape(area, perimeter, best_h, best_w, sym_score);
                break; // اگر فقط میخوایم اولین شکل رو پیدا کنیم، این break رو بردار تا همه شکل‌ها رو پیدا کنه
            }
        }
    }
    

    clock_t end_c = clock();

    double time = ((double)(end_c - start_c)) / (CLOCKS_PER_SEC / 1000.0); // تبدیل به میلی‌ثانیه

    //آزاد سازی حافظه
    stbi_image_free(img);
    free(input_float);
    free(final_img);
    free(temp1);
    free(temp2);
    free(result_v);
    free(result_h);
    free(sobel_result);
    
    return time;
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

        double time_c, time_asm;
        time_c = calc_c(img, width, height);
        total_time_c += time_c;

        // دوباره عکس رو بارگذاری می‌کنیم چون تابع calc_c حافظه عکس رو آزاد می‌کنه
        img = stbi_load(input_path, &width, &height, &channels, 1);
        if (img == NULL) {
            printf("Error: Could not load image %s for ASM version!\n", filenames[image_choice]);
            continue;
        }

        ++proccesed_images;

        time_asm = calc_asm(img, width, height);
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