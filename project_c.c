#include <stdio.h>
#include <stdlib.h>

//پیاده سازی کد های داخل کتابخونه در همین جا
//بخش پیاده سازی یعنی فعال سازی کد های داخل header
#define STB_IMAGE_IMPLEMENTATION 
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

void convolution_c(float *input, float *output, int width, int height, float *kernel, int kernel_size) { // kernel size should be odd (e.g., 3 for 3*3)
    for (int i = 1; i < height - 1; i++) {
        for (int j = 1; j < width - 1; j++) {
            float sum = 0.0;
            
            //اعمال کرنل روی پیکسل (i,j)
            for (int ki = -(kernel_size / 2); ki <= (kernel_size / 2); ki++) {
                for (int kj = -(kernel_size / 2); kj <= (kernel_size / 2); kj++) {
                    int pixel_index = (i + ki) * width + (j + kj);
                    int kernel_index = (ki + kernel_size/2) * kernel_size + (kj + kernel_size/2);
                    sum += input[pixel_index] * kernel[kernel_index];
                }
            }
            
            // مقادیر رنگ باید بین 0 تا 255 باشه
            if (sum < 0.0) sum = 0.0;
            if (sum > 255.0) sum = 255.0;
            
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

    //خواندن عکس
    printf("Loading image...\n");
    // لود کردن عکس به صورت آرایه ای از پیکسل های سیاه سفید (Grayscale)
    unsigned char *img = stbi_load("pictures/inputs/pic1.jpg", &width, &height, &channels, 1); 
    if (img == NULL) {
        printf("Error: Could not load image!\n");
        return 1;
    }
    printf("Image loaded! Width: %d, Height: %d\n", width, height);

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


    // do filters here
    convolution_c(input_float, output_float, width, height, selected_kernel, 3); // اعمال فیلتر انتخاب شده  

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
    stbi_write_png("pictures/outputs/pic1.png", width, height, 1, final_img, width); // عرض تصویر * تعداد کانال رنگ * سایز هر پیکسل = stride_bytes(گام)
    // تعریف stride = پرش به خط بعد ممکنه عکست 1920*1080 باشه ولی تو یه بخش 100*100 رو بخوای اینطوری طول و عرض همون 100 عه اما پرش 1920 هست
    printf("Done!\n");

    //آزاد سازی حافظه
    stbi_image_free(img); // آزاد سازی حافظه عکس ورودی
    free(input_float); // آزاد سازی حافظه آرایه ورودی
    free(output_float); // آزاد سازی حافظه آرایه خروجی
    free(final_img); // آزاد سازی حافظه عکس خروجی

    return 0;
}