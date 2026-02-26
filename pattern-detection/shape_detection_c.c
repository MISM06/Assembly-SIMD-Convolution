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

# define M_PI   3.14159265358979323846

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

typedef struct {
    double x;
    double y;
} Point;

// محاسبه ضرب خارجی (Cross Product) برای تشخیص جهت چرخش
double cross_product(Point o, Point a, Point b) {
    return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

// تابع مقایسه برای qsort (مرتب سازی برحسب x و سپس y)
int compare_points(const void* a, const void* b) {
    Point* p1 = (Point*)a;
    Point* p2 = (Point*)b;
    if (p1->x < p2->x) return -1;
    if (p1->x > p2->x) return 1;
    if (p1->y < p2->y) return -1;
    if (p1->y > p2->y) return 1;
    return 0;
}

// الگوریتم Monotone Chain برای استخراج Convex Hull
int convex_hull(Point* points, int n, Point* hull) {
    if (n <= 3) {
        for (int i = 0; i < n; i++) hull[i] = points[i];
        return n;
    }

    // مرتب‌سازی نقاط
    qsort(points, n, sizeof(Point), compare_points);

    int k = 0;
    // ساخت نیمه پایینی پوش محدب
    for (int i = 0; i < n; ++i) {
        while (k >= 2 && cross_product(hull[k - 2], hull[k - 1], points[i]) <= 0) k--;
        hull[k++] = points[i];
    }

    // ساخت نیمه بالایی پوش محدب
    for (int i = n - 2, t = k + 1; i >= 0; i--) {
        while (k >= t && cross_product(hull[k - 2], hull[k - 1], points[i]) <= 0) k--;
        hull[k++] = points[i];
    }

    // نقطه آخر تکراری است (همان نقطه اول)، پس یکی کم می‌کنیم
    return k - 1;
}

void min_bounding_box(Point* hull, int h, double* out_width, double* out_height, double* bbox_cx, double* bbox_cy) {
    double min_area = 1e9;
    double best_w = 0, best_h = 0;
    double best_cx = 0, best_cy = 0;

    // روی تک‌تک اضلاع پوش محدب حلقه می‌زنیم
    for (int i = 0; i < h; i++) {
        int next = (i + 1) % h;
        double dx = hull[next].x - hull[i].x;
        double dy = hull[next].y - hull[i].y;
        double length = sqrt(dx * dx + dy * dy);

        if (length == 0.0) continue;

        // محاسبه بردارهای یکه (مماس و عمود)
        double ux = dx / length;
        double uy = dy / length;
        double vx = -uy;
        double vy = ux;

        double min_u = 1e9, max_u = -1e9;
        double min_v = 1e9, max_v = -1e9;

        // تصویر کردن تمام نقاط پوش محدب روی این دو محور
        for (int j = 0; j < h; j++) {
            double proj_u = hull[j].x * ux + hull[j].y * uy;
            double proj_v = hull[j].x * vx + hull[j].y * vy;

            if (proj_u < min_u) min_u = proj_u;
            if (proj_u > max_u) max_u = proj_u;
            if (proj_v < min_v) min_v = proj_v;
            if (proj_v > max_v) max_v = proj_v;
        }

        // محاسبه طول، عرض و مساحت برای زاویه فعلی
        double width = max_u - min_u;
        double height = max_v - min_v;
        double area = width * height;

        // جواب کمترین مساحت است
        if (area < min_area) {
            min_area = area;
            best_w = width;
            best_h = height;
            // مرکز در مختصات جدید
            double center_u = (min_u + max_u) / 2.0;
            double center_v = (min_v + max_v) / 2.0;

            // تبدیل مرکز به محور های x , y
            best_cx = (center_u * ux) + (center_v * vx);
            best_cy = (center_u * uy) + (center_v * vy);
        }
    }

    *out_width = best_w;
    *out_height = best_h;
    *bbox_cx = best_cx;
    *bbox_cy = best_cy;
}

double calc_sym_score(int* contour_x, int* contour_y, int n, double bbox_cx, double bbox_cy, double best_h, double best_w) {

    double cx = 0, cy = 0;
    for (int i = 0; i < n; i++) {
        cx += contour_x[i];
        cy += contour_y[i];
    }
    cx /= n;
    cy /= n;

    double max_dim = (best_w > best_h) ? best_w : best_h;
    double dx = bbox_cx - cx;
    double dy = bbox_cy - cy;
    return sqrt(dx * dx + dy * dy) / max_dim;
}

void classify_shape(double area, double perimeter, double real_height, double real_width, double symmetry_score) {
    
    // اگر مساحت خیلی کوچک بود، شکلی نیست
    if (area < 20.0) {
        printf("Shape: Unknown / Noise\n");
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
    
    
    printf("Area: %.1f | Perim: %.1f\n", area, perimeter);
    printf("Extent: %.2f | Circ: %.2f | AR: %.2f | Sym: %.2f\n", extent, circularity, aspect_ratio, symmetry_score);
    
    double theoretical_rhombus_circ = (PI / 2.0) * aspect_ratio / (1.0 + (aspect_ratio * aspect_ratio));
    printf("theo_circ: %.1f\n", theoretical_rhombus_circ);
    // دایره گردی بالا و ابعاد برابر
    if (circularity > 0.84 && aspect_ratio > 0.85) {
        printf(">> Shape: Circle\n");
    }
    // مربع و مستطیل extent زیاد دارند
    else if (extent > 0.83) {
        if (aspect_ratio > 0.80) {
            printf(">> Shape: Square\n");
        } else {
            printf(">> Shape: Rectangle\n");
        }
    }
    
    else if (extent > 0.44 && extent < 0.56 && symmetry_score >= 0.04) {
        printf(">> Shape: Triangle\n");
    }
    // محاسبه گردی تئوری برای یک لوزی با همین نسبت ابعاد

    // محاسبه گردی تئوری برای یک لوزی با همین نسبت ابعاد
    else if (circularity > theoretical_rhombus_circ + 0.08) {
        printf(">> Shape: Ellipse\n");
    } 
    // اگر به عدد تئوری نزدیک بود و متقارن بود، لوزی است
    else if (symmetry_score < 0.04 && circularity > theoretical_rhombus_circ - 0.15) {
        printf(">> Shape: Rhombus\n");
    }
    else {
        printf(">> Shape: Unknown or Complex Polygon\n");
    }
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
        output_img[i * 3 + 0] = 0; 
        output_img[i * 3 + 1] = 0; 
        output_img[i * 3 + 2] = 0; 
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
        if (val < 100.0) val = 0.0f; // اگر مقدار لبه ضعیف بود (کمتر از 60) اون رو صفر کنیم تا نویز حذف بشه
        
        temp1[i] = (unsigned char) val; // تبدیل به عدد صحیح برای ذخیره عکس
        
    }

    // closing (بستن شکاف ها) : dilation -> erosion
    morphology(temp1, temp2, width, height, 1); // گسترش
    morphology(temp2, final_img, width, height, 0); // سایش
    
    for (int i = 0; i < total_pixels; i++) {
        output_img[i * 3 + 0] = temp1[i]; // کانال R
        output_img[i * 3 + 2] = temp1[i]; // کانال B
        output_img[i * 3 + 1] = temp1[i]; // کانال G
    }
   
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            if (final_img[y * width + x] ) {
                int contour_x[10000]; // آرایه‌ای برای ذخیره مختصات x نقاط مرزی
                int contour_y[10000]; // آرایه‌ای برای ذخیره مختصات y نقاط مرزی
                int contour_length = find_contour(x, y, contour_x, contour_y, final_img, width, height);
                
                if (contour_length > 0) {
                    //پاک کردن نقاط مرزی از تصویر نهایی تا در جستجوی بعدی دوباره پیدا نشوند
                    flood_erase(final_img, width, height, x, y);
                    double area = calc_area(contour_x, contour_y, contour_length);
                    if (area >= 20.0) {
                        printf("Found a closed contour with length and start at (%d, %d): %d\n", x, y, contour_length);
                        for (int i = 0; i < contour_length; i++) {
                            int idx = contour_y[i] * width + contour_x[i];
                            output_img[idx * 3 + 0] = 255; // کانال R
                            output_img[idx * 3 + 1] = 0;   // کانال G
                            output_img[idx * 3 + 2] = 0;   // کانال B
                
                        }
                        
                        double perimeter = calc_perimeter(contour_x, contour_y, contour_length);
                        double best_h, best_w, bbox_cx, bbox_cy, sym_score;
                        
                        Point* points = (Point*)malloc(contour_length * sizeof(Point));
                        for (int i = 0; i < contour_length; i++) {
                            points[i].x = contour_x[i];
                            points[i].y = contour_y[i];
                        }
                        Point* hull = (Point*)malloc(2 * contour_length * sizeof(Point));
                        
                        int h_count = convex_hull(points, contour_length, hull);
                        min_bounding_box(hull, h_count, &best_w, &best_h, &bbox_cx, &bbox_cy);
                        sym_score = calc_sym_score(contour_x, contour_y, contour_length, bbox_cx, bbox_cy, best_h, best_w);
                        classify_shape(area, perimeter, best_h, best_w, sym_score);

                        free(points);
                        free(hull);
                    }
                }
            }
        }
    }

    clock_t end_c = clock();

    double token_time = ((double)(end_c - start_c)) / CLOCKS_PER_SEC;
    printf("Token time: %f seconds\n", token_time);

    //ذخیره عکس خروجی
    printf("Saving image...\n");
    // ذخیره عکس به صورت PNG
    stbi_write_png(output_path, width, height, 3, output_img, width * 3);
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