section .text
global apply_convolution_avx


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

section .note.GNU-stack noalloc noexec nowrite progbits  
