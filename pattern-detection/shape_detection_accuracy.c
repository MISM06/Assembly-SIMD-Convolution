#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <dirent.h>
#include <string.h> 
#include <math.h>
#include <ctype.h>

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

int classify_shape(double area, double perimeter, double real_height, double real_width, double symmetry_score) {
    
    // اگر مساحت خیلی کوچک بود، شکلی نیست

    double PI = 3.14159265358979323846;


    double obb_area = real_width * real_height;
    if (obb_area == 0.0) return 0;

    // شاخص پرشدگی
    double extent = area / obb_area;

    // شاخص گردی (دایره=1.0 ، مربع=0.78 ، مثلث=0.6)
    double circularity = (4.0 * PI * area) / (perimeter * perimeter);
    
    // نسبت ابعاد (عدد کوچکتر تقسیم بر بزرگتر تا بین 0 و 1 باشه)
    double min_dim = (real_width < real_height) ? real_width : real_height;
    double max_dim = (real_width > real_height) ? real_width : real_height;
    double aspect_ratio = min_dim / max_dim;


    // printf("Area: %.1f | Perim: %.1f\n", area, perimeter);
    // printf("Extent: %.2f | Circ: %.2f | AR: %.2f | Sym: %.2f\n", extent, circularity, aspect_ratio, symmetry_score);

    // دایره گردی بالا و ابعاد برابر
    if (circularity > 0.84 && aspect_ratio > 0.85) {
        return 1; // Circle
    }
    // مربع و مستطیل extent زیاد دارند
    else if (extent > 0.83) {
        if (aspect_ratio > 0.80) {
            return 3; // Square
        } else {
            return 4; // Rectangle
        }
    }
    
    else if (extent > 0.44 && extent < 0.56 && symmetry_score >= 0.04) {
        return 6; // Triangle
    }
    // محاسبه گردی تئوری برای یک لوزی با همین نسبت ابعاد
    double theoretical_rhombus_circ = (PI / 2.0) * aspect_ratio / (1.0 + (aspect_ratio * aspect_ratio));

    // محاسبه گردی تئوری برای یک لوزی با همین نسبت ابعاد
    if (circularity > theoretical_rhombus_circ + 0.08) {
        return 2; // Ellipse
    } 
    // اگر به عدد تئوری نزدیک بود و متقارن بود، لوزی است
    else if (symmetry_score < 0.04 && circularity > theoretical_rhombus_circ - 0.15) {
        return 5; // Rhombus
    }
    return 0;
}

int calc_asm(unsigned char* img, int width, int height) {
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
    float kernel_sharpen[9] = {
         0.0f, -1.0f,  0.0f,
        -1.0f,  5.0f, -1.0f,
         0.0f, -1.0f,  0.0f
    };
    // تبدیل اعداد صحیح به اعداد اعشاری برای کار با کرنل
    int total_pixels = width * height;
    float *input_float = (float *)malloc(total_pixels * sizeof(float));
    float *sharpned_float = (float *)malloc(total_pixels * sizeof(float));
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

    apply_convolution_avx(input_float, sharpned_float, width, height, kernel_sharpen); // تیز کردن تصویر برای بهتر کردن لبه ها
    apply_convolution_avx(sharpned_float, result_v, width, height, sobel_kernel_v); //لبه های عمودی
    apply_convolution_avx(sharpned_float, result_h, width, height, sobel_kernel_h); //لبه های افقی
    sobel_magnitude_asm(result_v, result_h, temp1, total_pixels); //ترکیب لبه ها افقی و عمودی

    // closing (بستن شکاف ها) : dilation -> erosion
    dilation_asm(temp1, temp2, width, height);
    erosion_asm(temp2, final_img, width, height);

    int res = 0;
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
                if (area <= 10.0) continue; // اگر مساحت خیلی کوچک بود، احتمالاً نویز است، ردش کن
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
                res = classify_shape(area, perimeter, best_h, best_w, sym_score);

                free(points);
                free(hull);
                break; // اگر فقط میخوایم اولین شکل رو پیدا کنیم، این break رو بردار تا همه شکل‌ها رو پیدا کنه
            }
        }
    }
    
    //آزاد سازی حافظه
    stbi_image_free(img);
    free(input_float);
    free(final_img);
    free(temp1);
    free(temp2);
    free(result_v);
    free(result_h);
    free(sobel_result);

    return res;
}

int main() {
    
    int width, height, channels;

    // خواندن لیست عکس‌ها و انتخاب توسط کاربر
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


    int proccesed_images = 0;
    int correct_detections = 0;

    int class_procced[7];
    int class_correct[7];
    for (int i = 0; i < 7; i++) {
        class_correct[i] = 0;
        class_procced[i] = 0;
    }

    for (int image_choice = 0; image_choice < file_count; image_choice++) {
            char input_path[512];
        sprintf(input_path, "pictures/tests/%s", filenames[image_choice]);

        unsigned char *img = stbi_load(input_path, &width, &height, &channels, 1);

        img = stbi_load(input_path, &width, &height, &channels, 1);
        if (img == NULL) {
            printf("Error: Could not load image %s!\n", filenames[image_choice]);
            continue;
        }

        ++proccesed_images;
        int detected_shape = calc_asm(img, width, height);
        int correct_shape = 0; //unkown or nothing
        char *name = (char *)malloc(256);
        strcpy(name, filenames[image_choice]);
        for (int i = 0; name[i]; i++) {
            name[i] = tolower(name[i]);
        }
        if (strstr(name, "circle")) {
            correct_shape = 1;
        } else if (strstr(name, "ellipse")) {
            correct_shape = 2;
        } else if (strstr(name, "square")) {
            correct_shape = 3;
        } else if (strstr(name, "rectangle")) {
            correct_shape = 4;
        } else if (strstr(name, "rhombus")) {
            correct_shape = 5;
        } else if (strstr(name, "triangle")) {
            correct_shape = 6;
        }
        class_procced[correct_shape]++;
        if (detected_shape == correct_shape) {
            correct_detections++;
            class_correct[detected_shape]++;
        } else {
            printf("res: %d, %s\n", detected_shape, filenames[image_choice]);
        }
        free(name);
    }
    printf("\n==================== Accuracy Summary ====================\n");
    printf("Processed %d images.\n", proccesed_images);
    printf("Correct Detections: %d\n", correct_detections);
    double accuracy = (double)correct_detections / proccesed_images * 100;
    printf("Accuracy for each shape:\n");
    printf(">> Circle : %.2f%%\n", (double)class_correct[1] / class_procced[1] * 100);
    printf(">> Ellipse : %.2f%%\n", (double)class_correct[2] / class_procced[2] * 100);
    printf(">> Square : %.2f%%\n", (double)class_correct[3] / class_procced[3] * 100);
    printf(">> Rectangle : %.2f%%\n", (double)class_correct[4] / class_procced[4] * 100);
    printf(">> Rhombus : %.2f%%\n", (double)class_correct[5] / class_procced[5] * 100);
    printf(">> Triangle : %.2f%%\n", (double)class_correct[6] / class_procced[6] * 100);

    printf("✅ Total accuracy: %.2f%%\n", accuracy);

    return 0;
}