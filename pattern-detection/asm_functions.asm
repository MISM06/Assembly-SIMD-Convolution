section .rodata ; read-only data
    align 32
    const_255 dd 255.0
    const_thershold dd 100.0

section .text
    global apply_convolution_avx
    global sobel_magnitude_asm
    global dilation_asm
    global erosion_asm


%macro DO_CONVOLUTION 0
    ; Index = (j * width) + i
    mov rax, r13
    imul rax, rdx           ; rax = j * width
    add rax, r14            ; rax = (j * width) + i
    shl rax, 2              ; rax = Index * 4 (تبدیل به بایت برای float)

    lea rbx, [rdi + rax]    ; rbx = آدرس دقیق پیکسل مرکزی در input
    lea r15, [rsi + rax]    ; r15 = آدرس دقیق پیکسل مرکزی در output

    sub rbx, r10            ; rbx = آدرس پیکسل بالای پیکسل مرکزی

    vxorps ymm0, ymm0, ymm0 

    ;ردیف بالا
    vbroadcastss ymm2, dword [r8]          ; بالا-چپ (kernel[0])
    vmovups ymm1, [rbx - 4]
    vmulps ymm1, ymm1, ymm2
    vaddps ymm0, ymm0, ymm1

    vbroadcastss ymm2, dword [r8 + 4]      ; بالا-وسط (kernel[1])
    vmovups ymm1, [rbx]
    vmulps ymm1, ymm1, ymm2
    vaddps ymm0, ymm0, ymm1

    vbroadcastss ymm2, dword [r8 + 8]      ; بالا-راست (kernel[2])
    vmovups ymm1, [rbx + 4]
    vmulps ymm1, ymm1, ymm2
    vaddps ymm0, ymm0, ymm1

    ; ردیف وسط
    vbroadcastss ymm2, dword [r8 + 12]     ; وسط-چپ (kernel[3])
    vmovups ymm1, [rbx + r10 - 4]
    vmulps ymm1, ymm1, ymm2
    vaddps ymm0, ymm0, ymm1

    vbroadcastss ymm2, dword [r8 + 16]     ; مرکز (kernel[4])
    vmovups ymm1, [rbx + r10]
    vmulps ymm1, ymm1, ymm2
    vaddps ymm0, ymm0, ymm1

    vbroadcastss ymm2, dword [r8 + 20]     ; وسط-راست (kernel[5])
    vmovups ymm1, [rbx + r10 + 4]
    vmulps ymm1, ymm1, ymm2
    vaddps ymm0, ymm0, ymm1

    ; ردیف پایین
    vbroadcastss ymm2, dword [r8 + 24]     ; پایین-چپ (kernel[6])
    vmovups ymm1, [rbx + r10 * 2 - 4]
    vmulps ymm1, ymm1, ymm2
    vaddps ymm0, ymm0, ymm1

    vbroadcastss ymm2, dword [r8 + 28]     ; پایین-وسط (kernel[7])
    vmovups ymm1, [rbx + r10 * 2]
    vmulps ymm1, ymm1, ymm2
    vaddps ymm0, ymm0, ymm1

    vbroadcastss ymm2, dword [r8 + 32]     ; پایین-راست (kernel[8])
    vmovups ymm1, [rbx + r10 * 2 + 4]
    vmulps ymm1, ymm1, ymm2
    vaddps ymm0, ymm0, ymm1

    ; ذخیره نتیجه در خروجی
    vmovups [r15], ymm0
%endmacro

;void apply_convolution_avx(float* input, float* output, int width, int height, float* kernel)
; rdi = input image pointer
; rsi = output image pointer
; rdx = width
; rcx = height
; r8  = kernel pointer
apply_convolution_avx:
    ; تنظیم چارچوب استک برای تابع
    push rbp
    mov rbp, rsp

    ;ذخیره رجیسترهای مورد استفاده در استک بازگردانی مقادیر آنها در پایان تابع
    push rbx
    push r12
    push r13
    push r14
    push r15

    movsxd rdx, edx ; تبدیل عرض به 64 بیت
    movsxd rcx, ecx ; تبدیل ارتفاع به 64 بیت

    mov r10, rdx
    shl r10, 2              ; r10 = width * 4 (فاصله بین ردیف‌ها به بایت)

    mov r11, rcx
    dec r11                 ; r11 = height - 1 (حد نهایی حلقه j)

    
    mov r12, rdx
    sub r12, 9              ; r12 = width - 9 (آخرین ایندکسِ مجاز برای شروع یک پک 8 تایی)

    mov r13, 1              ; r13 = متغیر j (شروع از 1)

.loop_j:
    cmp r13, r11            ; آیا به ردیف یکی مانده به آخر رسیدیم؟
    jge .end_j

    mov r14, 1              ; r14 = متغیر i (شروع از 1)

.loop_i:
    cmp r14, r12            ; آیا از آخرین حد مجاز عبور کردیم؟
    jge .tail_check          ; اگر بله، برو بخش دُم رو چک کن!

    DO_CONVOLUTION

    add r14, 8              ; i += 8
    jmp .loop_i            

.tail_check:
    
    mov r14, r12            ; i = width - 9
    DO_CONVOLUTION          ; محاسبه کانولوشن برای 8 پیکسل مجاز پایانی

.end_i:
    inc r13                 ; متغیر j را 1 واحد جلو ببر (برو ردیف بعدی)
    jmp .loop_j             ; تکرار حلقه بیرونی

.end_j:
    
    vzeroupper              ; صفر کردن بالای ymmxرها برای جلوگیری از جریمه عملکرد در دستورات SSE بعدی

    ; بازگردانی رجیسترها
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx

    mov rsp, rbp
    pop rbp
    ret
 


; void sobel_magnitude_asm(float* res_v, float* res_h, unsigned char* out_img, int total_pixels);
; محاسبه گرادیان لبه های افقی و عمودی و همچنین آستانه گذاری برای روشن و خاموش بودن پیکسل ها
; rdi = float* res_v
; rsi = float* res_h
; rdx = unsigned char* out_img
; rcx = int total_pixels
sobel_magnitude_asm:
    movsxd rcx, ecx         ; اطمینان از 64-بیتی بودن تعداد پیکسل‌ها

    test rcx, rcx           ; اگر تعداد پیکسل‌ها 0 یا منفی بود، خروج
    jle .done

    ; کپی کردن مقادیر ثابت در تمام 8 خانه رجیسترها 
    vbroadcastss ymm14, [rel const_255] ;آدرس نسبی
    vbroadcastss ymm15, [rel const_thershold]

    ; محاسبه تعداد تکرار حلقه‌ی 8 تایی و طول دم (باقیمانده)
    mov r8, rcx
    shr r8, 3               ; r8 = total_pixels / 8 (تعداد اجرای حلقه AVX)
    mov r9, rcx
    and r9, 7               ; r9 = total_pixels % 8 (تعداد پیکسل‌های باقی‌مانده)

    xor rax, rax            ; rax نقش ایندکس دارد

.avx_loop:
    test r8, r8
    jz .tail_setup          


    vmovups ymm0, [rdi + rax*4]   ; ymm0 = 8 تا پیکسل از لبه عمودی
    vmovups ymm1, [rsi + rax*4]   ; ymm1 = 8 تا پیکسل از لبه افقی

    ; محسابه گرادیان(sqrt(Gx^2 + Gy^2))
    vmulps ymm0, ymm0, ymm0       
    vmulps ymm1, ymm1, ymm1       
    vaddps ymm0, ymm0, ymm1       
    vsqrtps ymm0, ymm0

    ; thersholding
    vcmpgeps ymm2, ymm0, ymm15    ; cmp greater or equal: هر عنصر توی ymm0 اگه از 128 بیشتر مساوی باشه -1(0xFFFFFFFF) میشه وگرنه 0
                                  ; اعداد کمتر از آستانه 0، بیشتر از آستانه 255
    vcvttps2dq ymm0, ymm0         ; تبدیل 8 تا float به 8 تا integer (32-bit)

    ;8 تا int 32 بیتی داریم که هرکدوم 0 تا 255 هستن پس هرکدوم توی 8 بیت جا میشن
    vextracti128 xmm1, ymm2, 1    ; جدا کردن 4 عددِ بالایی به یک رجیستر کمکی
    vpackssdw xmm0, xmm2, xmm1    ; فشرده کردن 32-بیت به 16-بیت
    vpacksswb xmm0, xmm0, xmm0    ; فشرده کردن 16-بیت به 8-بیت (unsigned char)
                                  ; ما از دستورات ss (signed saturate) استفاده کردیم چون در اینجاعدد -1 که همه بیت ها 1 هستند رو داریم فشرده میکنیم
                                  ; اگر از us استفاده کنیم چون بی علامته اعداد منفی رو صفر میکنه
    
    vmovq [rdx + rax], xmm0       ; نوشتن پیکسل ها در حافظه

    add rax, 8                    ; 8 پیکسل رفتیم جلو
    dec r8                        ; یکی از تعداد حلقه‌ها کم کن
    jnz .avx_loop                 ; تکرار تا اتمام بسته‌های 8 تایی

.tail_setup:
    test r9, r9
    jz .done                      ; اگر دُم نداشتیم، پایان

.tail_loop:
    ; پردازش پیکسل‌های باقی‌مانده یکی‌یکی (Scalar)
    movss xmm0, [rdi + rax*4]
    movss xmm1, [rsi + rax*4]

    mulss xmm0, xmm0
    mulss xmm1, xmm1
    addss xmm0, xmm1

    sqrtss xmm0, xmm0

    vcmpgess xmm0, xmm0, xmm15   ; مقایسه با آستانه
    vmovd r10d, xmm0       ; تبدیل نتیجه مقایسه به integer  

    mov byte [rdx + rax], r10b    ; نوشتنِ تک بایت در آرایه خروجی

    inc rax
    dec r9
    jnz .tail_loop

.done:
    vzeroupper                    ; پاکسازی رجیسترهای AVX (بسیار مهم برای جلوگیری از افت سرعت C)
    ret


; void dilation_asm(unsigned char* src, unsigned char* dst, int width, int height);
; گرفتن ماکسیمم مربع 3*3 دور هر پیکسل
; چون فقط 0 بودن یا نبودن پیکسل ها مهم است به جای ماکسیمم از or استفاده میکنیم که سریعتر است
; rdi = src, rsi = dst, edx = width, ecx = height
dilation_asm:
    movsxd rdx, edx         ; width (64-bit)
    movsxd rcx, ecx         ; height (64-bit)

    cmp rcx, 2
    jle .d_done             ; اگر ارتفاع کمتر از 3 بود، خروج
    cmp rdx, 34
    jl .d_done              ; این الگوریتم برای عکس‌های با عرض بزرگتر از 34 نوشته شده

    ; تنظیم اشاره‌گرهای 3 سطر (بالا، وسط، پایین)
    mov r8, rdi             ; r8 = سطر بالا (y-1)
    mov r9, rdi
    add r9, rdx             ; r9 = سطر وسط (y)
    mov r10, r9
    add r10, rdx            ; r10 = سطر پایین (y+1)

    mov r11, rsi
    add r11, rdx            ; r11 = سطر وسط در عکس خروجی (dst)

    sub rcx, 2              ; تعداد سطرهایی که باید پردازش بشن (height - 2)
    mov rdi, rdx
    sub rdi, 2              ; rdi = تعداد پیکسل‌های قابل پردازش در هر سطر (width - 2)

.d_row_loop:
    xor rax, rax            ; rax = 0 (آفست ستون در سطر فعلی)

.d_avx_loop:
    mov rsi, rdi
    sub rsi, rax            ; پیکسل‌های باقی‌مانده در این سطر = کل - آفست فعلی
    cmp rsi, 32
    jl .d_tail_check        ; اگر کمتر از 32 تا مونده بود، برو برای ترفند همپوشانی

    ; سطر بالا
    vmovdqu ymm0, [r8 + rax]       ; چپ
    vmovdqu ymm1, [r8 + rax + 1]   ; وسط
    vpor ymm0, ymm0, ymm1       ; ماکزیمم(چپ، وسط)
    vmovdqu ymm1, [r8 + rax + 2]   ; راست
    vpor ymm0, ymm0, ymm1       ; ماکزیمم با راست

    ; سطر وسط
    vmovdqu ymm2, [r9 + rax]
    vmovdqu ymm1, [r9 + rax + 1]
    vpor ymm2, ymm2, ymm1
    vmovdqu ymm1, [r9 + rax + 2]
    vpor ymm2, ymm2, ymm1
    vpor ymm0, ymm0, ymm2       ; ترکیب ماکزیمم سطر بالا و وسط

    ; سطر پایین
    vmovdqu ymm3, [r10 + rax]
    vmovdqu ymm1, [r10 + rax + 1]
    vpor ymm3, ymm3, ymm1
    vmovdqu ymm1, [r10 + rax + 2]
    vpor ymm3, ymm3, ymm1
    vpor ymm0, ymm0, ymm3       ; ترکیب نهایی هر 9 خانه

    ;ذخیره نتیجه 32 پیکسل
    vmovdqu [r11 + rax + 1], ymm0

    add rax, 32                    
    jmp .d_avx_loop

.d_tail_check:
    test rsi, rsi
    jz .d_next_row                 ; اگر دقیقاً صفر شده بود برو سطر بعد

    ; برای 32 پیکسل آخر سطر محاسبه میکنی
    ; ممکن است بعضی پیکسل هارا مجددا حساب کنیم اما به دلیل استفاده از یک دستور برداری ارزش دارد
    mov rax, rdi
    sub rax, 32
    
    vmovdqu ymm0, [r8 + rax]
    vmovdqu ymm1, [r8 + rax + 1]
    vpor ymm0, ymm0, ymm1
    vmovdqu ymm1, [r8 + rax + 2]
    vpor ymm0, ymm0, ymm1

    vmovdqu ymm2, [r9 + rax]
    vmovdqu ymm1, [r9 + rax + 1]
    vpor ymm2, ymm2, ymm1
    vmovdqu ymm1, [r9 + rax + 2]
    vpor ymm2, ymm2, ymm1
    vpor ymm0, ymm0, ymm2

    vmovdqu ymm3, [r10 + rax]
    vmovdqu ymm1, [r10 + rax + 1]
    vpor ymm3, ymm3, ymm1
    vmovdqu ymm1, [r10 + rax + 2]
    vpor ymm3, ymm3, ymm1
    vpor ymm0, ymm0, ymm3

    vmovdqu [r11 + rax + 1], ymm0

.d_next_row:
    ; رفتن به سطر بعدی
    add r8, rdx
    add r9, rdx
    add r10, rdx
    add r11, rdx
    dec rcx
    jnz .d_row_loop

.d_done:
    vzeroupper
    ret



; void erosion_asm(unsigned char* src, unsigned char* dst, int width, int height);
; پیدا کردن تاریک ترین پیکسل در مربع 3*3 دور هر پیکسل
; به جای مینیمم and میگیریم
; rdi = src, rsi = dst, edx = width, ecx = height
erosion_asm:
    movsxd rdx, edx         ; width (64-bit)
    movsxd rcx, ecx         ; height (64-bit)

    cmp rcx, 2
    jle .e_done             ; اگر ارتفاع کمتر از 3 بود، خروج
    cmp rdx, 34
    jl .e_done              ; این الگوریتم برای عکس‌های با عرض بزرگتر از 34 نوشته شده

    ; تنظیم اشاره‌گرهای 3 سطر (بالا، وسط، پایین)
    mov r8, rdi             ; r8 = سطر بالا (y-1)
    mov r9, rdi
    add r9, rdx             ; r9 = سطر وسط (y)
    mov r10, r9
    add r10, rdx            ; r10 = سطر پایین (y+1)

    mov r11, rsi
    add r11, rdx            ; r11 = سطر وسط در عکس خروجی (dst)

    sub rcx, 2              ; تعداد سطرهایی که باید پردازش بشن (height - 2)
    mov rdi, rdx
    sub rdi, 2              ; rdi = تعداد پیکسل‌های قابل پردازش در هر سطر (width - 2)

.e_row_loop:
    xor rax, rax            ; rax = 0 (آفست ستون در سطر فعلی)

.e_avx_loop:
    mov rsi, rdi
    sub rsi, rax            ; پیکسل‌های باقی‌مانده در این سطر = کل - آفست فعلی
    cmp rsi, 32
    jl .e_tail_check        ; اگر کمتر از 32 تا مونده بود، برو برای ترفند همپوشانی

    ; سطر بالا
    vmovdqu ymm0, [r8 + rax]       ; چپ
    vmovdqu ymm1, [r8 + rax + 1]   ; وسط
    vpand ymm0, ymm0, ymm1       ; ماکزیمم(چپ، وسط)
    vmovdqu ymm1, [r8 + rax + 2]   ; راست
    vpand ymm0, ymm0, ymm1       ; ماکزیمم با راست

    ; سطر وسط
    vmovdqu ymm2, [r9 + rax]
    vmovdqu ymm1, [r9 + rax + 1]
    vpand ymm2, ymm2, ymm1
    vmovdqu ymm1, [r9 + rax + 2]
    vpand ymm2, ymm2, ymm1
    vpand ymm0, ymm0, ymm2       ; ترکیب ماکزیمم سطر بالا و وسط

    ; سطر پایین
    vmovdqu ymm3, [r10 + rax]
    vmovdqu ymm1, [r10 + rax + 1]
    vpand ymm3, ymm3, ymm1
    vmovdqu ymm1, [r10 + rax + 2]
    vpand ymm3, ymm3, ymm1
    vpand ymm0, ymm0, ymm3       ; ترکیب نهایی هر 9 خانه

    ;ذخیره نتیجه 32 پیکسل
    vmovdqu [r11 + rax + 1], ymm0

    add rax, 32                    
    jmp .e_avx_loop

.e_tail_check:
    test rsi, rsi
    jz .e_next_row                 ; اگر دقیقاً صفر شده بود برو سطر بعد

    ; برای 32 پیکسل آخر سطر محاسبه میکنی
    ; ممکن است بعضی پیکسل هارا مجددا حساب کنیم اما به دلیل استفاده از یک دستور برداری ارزش دارد
    mov rax, rdi
    sub rax, 32
    
    vmovdqu ymm0, [r8 + rax]
    vmovdqu ymm1, [r8 + rax + 1]
    vpand ymm0, ymm0, ymm1
    vmovdqu ymm1, [r8 + rax + 2]
    vpand ymm0, ymm0, ymm1

    vmovdqu ymm2, [r9 + rax]
    vmovdqu ymm1, [r9 + rax + 1]
    vpand ymm2, ymm2, ymm1
    vmovdqu ymm1, [r9 + rax + 2]
    vpand ymm2, ymm2, ymm1
    vpand ymm0, ymm0, ymm2

    vmovdqu ymm3, [r10 + rax]
    vmovdqu ymm1, [r10 + rax + 1]
    vpand ymm3, ymm3, ymm1
    vmovdqu ymm1, [r10 + rax + 2]
    vpand ymm3, ymm3, ymm1
    vpand ymm0, ymm0, ymm3

    vmovdqu [r11 + rax + 1], ymm0

.e_next_row:
    ; رفتن به سطر بعدی
    add r8, rdx
    add r9, rdx
    add r10, rdx
    add r11, rdx
    dec rcx
    jnz .e_row_loop

.e_done:
    vzeroupper
    ret

section .note.GNU-stack noalloc noexec nowrite progbits