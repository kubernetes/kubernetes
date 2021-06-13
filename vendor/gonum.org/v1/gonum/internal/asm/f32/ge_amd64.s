// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

#define SIZE 4
#define BITSIZE 2
#define KERNELSIZE 3

#define M_DIM m+0(FP)
#define M CX
#define N_DIM n+8(FP)
#define N BX

#define TMP1 R14
#define TMP2 R15

#define X_PTR SI
#define Y y_base+56(FP)
#define Y_PTR DX
#define A_ROW AX
#define A_PTR DI

#define INC_X R8
#define INC3_X R9

#define INC_Y R10
#define INC3_Y R11

#define LDA R12
#define LDA3 R13

#define ALPHA X0
#define ALPHA_SPILL al-16(SP)

#define LOAD_ALPHA \
	MOVSS  alpha+16(FP), ALPHA \
	SHUFPS $0, ALPHA, ALPHA

#define LOAD_SCALED4 \
	PREFETCHNTA 16*SIZE(X_PTR)    \
	MOVDDUP     (X_PTR), X1       \
	MOVDDUP     2*SIZE(X_PTR), X3 \
	MOVSHDUP    X1, X2            \
	MOVSHDUP    X3, X4            \
	MOVSLDUP    X1, X1            \
	MOVSLDUP    X3, X3            \
	MULPS       ALPHA, X1         \
	MULPS       ALPHA, X2         \
	MULPS       ALPHA, X3         \
	MULPS       ALPHA, X4

#define LOAD_SCALED2 \
	MOVDDUP  (X_PTR), X1 \
	MOVSHDUP X1, X2      \
	MOVSLDUP X1, X1      \
	MULPS    ALPHA, X1   \
	MULPS    ALPHA, X2

#define LOAD_SCALED1 \
	MOVSS  (X_PTR), X1 \
	SHUFPS $0, X1, X1  \
	MULPS  ALPHA, X1

#define LOAD_SCALED4_INC \
	PREFETCHNTA (X_PTR)(INC_X*8)      \
	MOVSS       (X_PTR), X1           \
	MOVSS       (X_PTR)(INC_X*1), X2  \
	MOVSS       (X_PTR)(INC_X*2), X3  \
	MOVSS       (X_PTR)(INC3_X*1), X4 \
	SHUFPS      $0, X1, X1            \
	SHUFPS      $0, X2, X2            \
	SHUFPS      $0, X3, X3            \
	SHUFPS      $0, X4, X4            \
	MULPS       ALPHA, X1             \
	MULPS       ALPHA, X2             \
	MULPS       ALPHA, X3             \
	MULPS       ALPHA, X4

#define LOAD_SCALED2_INC \
	MOVSS  (X_PTR), X1          \
	MOVSS  (X_PTR)(INC_X*1), X2 \
	SHUFPS $0, X1, X1           \
	SHUFPS $0, X2, X2           \
	MULPS  ALPHA, X1            \
	MULPS  ALPHA, X2

#define KERNEL_LOAD8 \
	MOVUPS (Y_PTR), X5       \
	MOVUPS 4*SIZE(Y_PTR), X6

#define KERNEL_LOAD8_INC \
	MOVSS    (Y_PTR), X5             \
	MOVSS    (Y_PTR)(INC_Y*1), X6    \
	MOVSS    (Y_PTR)(INC_Y*2), X7    \
	MOVSS    (Y_PTR)(INC3_Y*1), X8   \
	UNPCKLPS X6, X5                  \
	UNPCKLPS X8, X7                  \
	MOVLHPS  X7, X5                  \
	LEAQ     (Y_PTR)(INC_Y*4), Y_PTR \
	MOVSS    (Y_PTR), X6             \
	MOVSS    (Y_PTR)(INC_Y*1), X7    \
	MOVSS    (Y_PTR)(INC_Y*2), X8    \
	MOVSS    (Y_PTR)(INC3_Y*1), X9   \
	UNPCKLPS X7, X6                  \
	UNPCKLPS X9, X8                  \
	MOVLHPS  X8, X6

#define KERNEL_LOAD4 \
	MOVUPS (Y_PTR), X5

#define KERNEL_LOAD4_INC \
	MOVSS    (Y_PTR), X5           \
	MOVSS    (Y_PTR)(INC_Y*1), X6  \
	MOVSS    (Y_PTR)(INC_Y*2), X7  \
	MOVSS    (Y_PTR)(INC3_Y*1), X8 \
	UNPCKLPS X6, X5                \
	UNPCKLPS X8, X7                \
	MOVLHPS  X7, X5

#define KERNEL_LOAD2 \
	MOVSD (Y_PTR), X5

#define KERNEL_LOAD2_INC \
	MOVSS    (Y_PTR), X5          \
	MOVSS    (Y_PTR)(INC_Y*1), X6 \
	UNPCKLPS X6, X5

#define KERNEL_4x8 \
	MOVUPS X5, X7  \
	MOVUPS X6, X8  \
	MOVUPS X5, X9  \
	MOVUPS X6, X10 \
	MOVUPS X5, X11 \
	MOVUPS X6, X12 \
	MULPS  X1, X5  \
	MULPS  X1, X6  \
	MULPS  X2, X7  \
	MULPS  X2, X8  \
	MULPS  X3, X9  \
	MULPS  X3, X10 \
	MULPS  X4, X11 \
	MULPS  X4, X12

#define STORE_4x8 \
	MOVUPS ALPHA, ALPHA_SPILL         \
	MOVUPS (A_PTR), X13               \
	ADDPS  X13, X5                    \
	MOVUPS 4*SIZE(A_PTR), X14         \
	ADDPS  X14, X6                    \
	MOVUPS (A_PTR)(LDA*1), X15        \
	ADDPS  X15, X7                    \
	MOVUPS 4*SIZE(A_PTR)(LDA*1), X0   \
	ADDPS  X0, X8                     \
	MOVUPS (A_PTR)(LDA*2), X13        \
	ADDPS  X13, X9                    \
	MOVUPS 4*SIZE(A_PTR)(LDA*2), X14  \
	ADDPS  X14, X10                   \
	MOVUPS (A_PTR)(LDA3*1), X15       \
	ADDPS  X15, X11                   \
	MOVUPS 4*SIZE(A_PTR)(LDA3*1), X0  \
	ADDPS  X0, X12                    \
	MOVUPS X5, (A_PTR)                \
	MOVUPS X6, 4*SIZE(A_PTR)          \
	MOVUPS X7, (A_PTR)(LDA*1)         \
	MOVUPS X8, 4*SIZE(A_PTR)(LDA*1)   \
	MOVUPS X9, (A_PTR)(LDA*2)         \
	MOVUPS X10, 4*SIZE(A_PTR)(LDA*2)  \
	MOVUPS X11, (A_PTR)(LDA3*1)       \
	MOVUPS X12, 4*SIZE(A_PTR)(LDA3*1) \
	MOVUPS ALPHA_SPILL, ALPHA         \
	ADDQ   $8*SIZE, A_PTR

#define KERNEL_4x4 \
	MOVUPS X5, X6 \
	MOVUPS X5, X7 \
	MOVUPS X5, X8 \
	MULPS  X1, X5 \
	MULPS  X2, X6 \
	MULPS  X3, X7 \
	MULPS  X4, X8

#define STORE_4x4 \
	MOVUPS (A_PTR), X13         \
	ADDPS  X13, X5              \
	MOVUPS (A_PTR)(LDA*1), X14  \
	ADDPS  X14, X6              \
	MOVUPS (A_PTR)(LDA*2), X15  \
	ADDPS  X15, X7              \
	MOVUPS (A_PTR)(LDA3*1), X13 \
	ADDPS  X13, X8              \
	MOVUPS X5, (A_PTR)          \
	MOVUPS X6, (A_PTR)(LDA*1)   \
	MOVUPS X7, (A_PTR)(LDA*2)   \
	MOVUPS X8, (A_PTR)(LDA3*1)  \
	ADDQ   $4*SIZE, A_PTR

#define KERNEL_4x2 \
	MOVUPS X5, X6 \
	MOVUPS X5, X7 \
	MOVUPS X5, X8 \
	MULPS  X1, X5 \
	MULPS  X2, X6 \
	MULPS  X3, X7 \
	MULPS  X4, X8

#define STORE_4x2 \
	MOVSD (A_PTR), X9          \
	ADDPS X9, X5               \
	MOVSD (A_PTR)(LDA*1), X10  \
	ADDPS X10, X6              \
	MOVSD (A_PTR)(LDA*2), X11  \
	ADDPS X11, X7              \
	MOVSD (A_PTR)(LDA3*1), X12 \
	ADDPS X12, X8              \
	MOVSD X5, (A_PTR)          \
	MOVSD X6, (A_PTR)(LDA*1)   \
	MOVSD X7, (A_PTR)(LDA*2)   \
	MOVSD X8, (A_PTR)(LDA3*1)  \
	ADDQ  $2*SIZE, A_PTR

#define KERNEL_4x1 \
	MOVSS (Y_PTR), X5 \
	MOVSS X5, X6      \
	MOVSS X5, X7      \
	MOVSS X5, X8      \
	MULSS X1, X5      \
	MULSS X2, X6      \
	MULSS X3, X7      \
	MULSS X4, X8

#define STORE_4x1 \
	ADDSS (A_PTR), X5         \
	ADDSS (A_PTR)(LDA*1), X6  \
	ADDSS (A_PTR)(LDA*2), X7  \
	ADDSS (A_PTR)(LDA3*1), X8 \
	MOVSS X5, (A_PTR)         \
	MOVSS X6, (A_PTR)(LDA*1)  \
	MOVSS X7, (A_PTR)(LDA*2)  \
	MOVSS X8, (A_PTR)(LDA3*1) \
	ADDQ  $SIZE, A_PTR

#define KERNEL_2x8 \
	MOVUPS X5, X7 \
	MOVUPS X6, X8 \
	MULPS  X1, X5 \
	MULPS  X1, X6 \
	MULPS  X2, X7 \
	MULPS  X2, X8

#define STORE_2x8 \
	MOVUPS (A_PTR), X9               \
	ADDPS  X9, X5                    \
	MOVUPS 4*SIZE(A_PTR), X10        \
	ADDPS  X10, X6                   \
	MOVUPS (A_PTR)(LDA*1), X11       \
	ADDPS  X11, X7                   \
	MOVUPS 4*SIZE(A_PTR)(LDA*1), X12 \
	ADDPS  X12, X8                   \
	MOVUPS X5, (A_PTR)               \
	MOVUPS X6, 4*SIZE(A_PTR)         \
	MOVUPS X7, (A_PTR)(LDA*1)        \
	MOVUPS X8, 4*SIZE(A_PTR)(LDA*1)  \
	ADDQ   $8*SIZE, A_PTR

#define KERNEL_2x4 \
	MOVUPS X5, X6 \
	MULPS  X1, X5 \
	MULPS  X2, X6

#define STORE_2x4 \
	MOVUPS (A_PTR), X9         \
	ADDPS  X9, X5              \
	MOVUPS (A_PTR)(LDA*1), X11 \
	ADDPS  X11, X6             \
	MOVUPS X5, (A_PTR)         \
	MOVUPS X6, (A_PTR)(LDA*1)  \
	ADDQ   $4*SIZE, A_PTR

#define KERNEL_2x2 \
	MOVSD X5, X6 \
	MULPS X1, X5 \
	MULPS X2, X6

#define STORE_2x2 \
	MOVSD (A_PTR), X7        \
	ADDPS X7, X5             \
	MOVSD (A_PTR)(LDA*1), X8 \
	ADDPS X8, X6             \
	MOVSD X5, (A_PTR)        \
	MOVSD X6, (A_PTR)(LDA*1) \
	ADDQ  $2*SIZE, A_PTR

#define KERNEL_2x1 \
	MOVSS (Y_PTR), X5 \
	MOVSS X5, X6      \
	MULSS X1, X5      \
	MULSS X2, X6

#define STORE_2x1 \
	ADDSS (A_PTR), X5        \
	ADDSS (A_PTR)(LDA*1), X6 \
	MOVSS X5, (A_PTR)        \
	MOVSS X6, (A_PTR)(LDA*1) \
	ADDQ  $SIZE, A_PTR

#define KERNEL_1x8 \
	MULPS X1, X5 \
	MULPS X1, X6

#define STORE_1x8 \
	MOVUPS (A_PTR), X7       \
	ADDPS  X7, X5            \
	MOVUPS 4*SIZE(A_PTR), X8 \
	ADDPS  X8, X6            \
	MOVUPS X5, (A_PTR)       \
	MOVUPS X6, 4*SIZE(A_PTR) \
	ADDQ   $8*SIZE, A_PTR

#define KERNEL_1x4 \
	MULPS X1, X5 \
	MULPS X1, X6

#define STORE_1x4 \
	MOVUPS (A_PTR), X7    \
	ADDPS  X7, X5         \
	MOVUPS X5, (A_PTR)    \
	ADDQ   $4*SIZE, A_PTR

#define KERNEL_1x2 \
	MULPS X1, X5

#define STORE_1x2 \
	MOVSD (A_PTR), X6    \
	ADDPS X6, X5         \
	MOVSD X5, (A_PTR)    \
	ADDQ  $2*SIZE, A_PTR

#define KERNEL_1x1 \
	MOVSS (Y_PTR), X5 \
	MULSS X1, X5

#define STORE_1x1 \
	ADDSS (A_PTR), X5  \
	MOVSS X5, (A_PTR)  \
	ADDQ  $SIZE, A_PTR

// func Ger(m, n uintptr, alpha float32,
//	x []float32, incX uintptr,
//	y []float32, incY uintptr,
//	a []float32, lda uintptr)
TEXT ·Ger(SB), 0, $16-120
	MOVQ M_DIM, M
	MOVQ N_DIM, N
	CMPQ M, $0
	JE   end
	CMPQ N, $0
	JE   end

	LOAD_ALPHA

	MOVQ x_base+24(FP), X_PTR
	MOVQ y_base+56(FP), Y_PTR
	MOVQ a_base+88(FP), A_ROW
	MOVQ A_ROW, A_PTR
	MOVQ lda+112(FP), LDA     // LDA = LDA * sizeof(float32)
	SHLQ $BITSIZE, LDA
	LEAQ (LDA)(LDA*2), LDA3   // LDA3 = LDA * 3

	CMPQ incY+80(FP), $1 // Check for dense vector Y (fast-path)
	JNE  inc
	CMPQ incX+48(FP), $1 // Check for dense vector X (fast-path)
	JNE  inc

	SHRQ $2, M
	JZ   r2

r4:

	// LOAD 4
	LOAD_SCALED4

	MOVQ N_DIM, N
	SHRQ $KERNELSIZE, N
	JZ   r4c4

r4c8:
	// 4x8 KERNEL
	KERNEL_LOAD8
	KERNEL_4x8
	STORE_4x8

	ADDQ $8*SIZE, Y_PTR

	DECQ N
	JNZ  r4c8

r4c4:
	TESTQ $4, N_DIM
	JZ    r4c2

	// 4x4 KERNEL
	KERNEL_LOAD4
	KERNEL_4x4
	STORE_4x4

	ADDQ $4*SIZE, Y_PTR

r4c2:
	TESTQ $2, N_DIM
	JZ    r4c1

	// 4x2 KERNEL
	KERNEL_LOAD2
	KERNEL_4x2
	STORE_4x2

	ADDQ $2*SIZE, Y_PTR

r4c1:
	TESTQ $1, N_DIM
	JZ    r4end

	// 4x1 KERNEL
	KERNEL_4x1
	STORE_4x1

	ADDQ $SIZE, Y_PTR

r4end:
	ADDQ $4*SIZE, X_PTR
	MOVQ Y, Y_PTR
	LEAQ (A_ROW)(LDA*4), A_ROW
	MOVQ A_ROW, A_PTR

	DECQ M
	JNZ  r4

r2:
	TESTQ $2, M_DIM
	JZ    r1

	// LOAD 2
	LOAD_SCALED2

	MOVQ N_DIM, N
	SHRQ $KERNELSIZE, N
	JZ   r2c4

r2c8:
	// 2x8 KERNEL
	KERNEL_LOAD8
	KERNEL_2x8
	STORE_2x8

	ADDQ $8*SIZE, Y_PTR

	DECQ N
	JNZ  r2c8

r2c4:
	TESTQ $4, N_DIM
	JZ    r2c2

	// 2x4 KERNEL
	KERNEL_LOAD4
	KERNEL_2x4
	STORE_2x4

	ADDQ $4*SIZE, Y_PTR

r2c2:
	TESTQ $2, N_DIM
	JZ    r2c1

	// 2x2 KERNEL
	KERNEL_LOAD2
	KERNEL_2x2
	STORE_2x2

	ADDQ $2*SIZE, Y_PTR

r2c1:
	TESTQ $1, N_DIM
	JZ    r2end

	// 2x1 KERNEL
	KERNEL_2x1
	STORE_2x1

	ADDQ $SIZE, Y_PTR

r2end:
	ADDQ $2*SIZE, X_PTR
	MOVQ Y, Y_PTR
	LEAQ (A_ROW)(LDA*2), A_ROW
	MOVQ A_ROW, A_PTR

r1:
	TESTQ $1, M_DIM
	JZ    end

	// LOAD 1
	LOAD_SCALED1

	MOVQ N_DIM, N
	SHRQ $KERNELSIZE, N
	JZ   r1c4

r1c8:
	// 1x8 KERNEL
	KERNEL_LOAD8
	KERNEL_1x8
	STORE_1x8

	ADDQ $8*SIZE, Y_PTR

	DECQ N
	JNZ  r1c8

r1c4:
	TESTQ $4, N_DIM
	JZ    r1c2

	// 1x4 KERNEL
	KERNEL_LOAD4
	KERNEL_1x4
	STORE_1x4

	ADDQ $4*SIZE, Y_PTR

r1c2:
	TESTQ $2, N_DIM
	JZ    r1c1

	// 1x2 KERNEL
	KERNEL_LOAD2
	KERNEL_1x2
	STORE_1x2

	ADDQ $2*SIZE, Y_PTR

r1c1:
	TESTQ $1, N_DIM
	JZ    end

	// 1x1 KERNEL
	KERNEL_1x1
	STORE_1x1

end:
	RET

inc:  // Algorithm for incY != 0 ( split loads in kernel )

	MOVQ incX+48(FP), INC_X       // INC_X = incX * sizeof(float32)
	SHLQ $BITSIZE, INC_X
	MOVQ incY+80(FP), INC_Y       // INC_Y = incY * sizeof(float32)
	SHLQ $BITSIZE, INC_Y
	LEAQ (INC_X)(INC_X*2), INC3_X // INC3_X = INC_X * 3
	LEAQ (INC_Y)(INC_Y*2), INC3_Y // INC3_Y = INC_Y * 3

	XORQ    TMP2, TMP2
	MOVQ    M, TMP1
	SUBQ    $1, TMP1
	IMULQ   INC_X, TMP1
	NEGQ    TMP1
	CMPQ    INC_X, $0
	CMOVQLT TMP1, TMP2
	LEAQ    (X_PTR)(TMP2*SIZE), X_PTR

	XORQ    TMP2, TMP2
	MOVQ    N, TMP1
	SUBQ    $1, TMP1
	IMULQ   INC_Y, TMP1
	NEGQ    TMP1
	CMPQ    INC_Y, $0
	CMOVQLT TMP1, TMP2
	LEAQ    (Y_PTR)(TMP2*SIZE), Y_PTR

	SHRQ $2, M
	JZ   inc_r2

inc_r4:
	// LOAD 4
	LOAD_SCALED4_INC

	MOVQ N_DIM, N
	SHRQ $KERNELSIZE, N
	JZ   inc_r4c4

inc_r4c8:
	// 4x4 KERNEL
	KERNEL_LOAD8_INC
	KERNEL_4x8
	STORE_4x8

	LEAQ (Y_PTR)(INC_Y*4), Y_PTR
	DECQ N
	JNZ  inc_r4c8

inc_r4c4:
	TESTQ $4, N_DIM
	JZ    inc_r4c2

	// 4x4 KERNEL
	KERNEL_LOAD4_INC
	KERNEL_4x4
	STORE_4x4

	LEAQ (Y_PTR)(INC_Y*4), Y_PTR

inc_r4c2:
	TESTQ $2, N_DIM
	JZ    inc_r4c1

	// 4x2 KERNEL
	KERNEL_LOAD2_INC
	KERNEL_4x2
	STORE_4x2

	LEAQ (Y_PTR)(INC_Y*2), Y_PTR

inc_r4c1:
	TESTQ $1, N_DIM
	JZ    inc_r4end

	// 4x1 KERNEL
	KERNEL_4x1
	STORE_4x1

	ADDQ INC_Y, Y_PTR

inc_r4end:
	LEAQ (X_PTR)(INC_X*4), X_PTR
	MOVQ Y, Y_PTR
	LEAQ (A_ROW)(LDA*4), A_ROW
	MOVQ A_ROW, A_PTR

	DECQ M
	JNZ  inc_r4

inc_r2:
	TESTQ $2, M_DIM
	JZ    inc_r1

	// LOAD 2
	LOAD_SCALED2_INC

	MOVQ N_DIM, N
	SHRQ $KERNELSIZE, N
	JZ   inc_r2c4

inc_r2c8:
	// 2x8 KERNEL
	KERNEL_LOAD8_INC
	KERNEL_2x8
	STORE_2x8

	LEAQ (Y_PTR)(INC_Y*4), Y_PTR
	DECQ N
	JNZ  inc_r2c8

inc_r2c4:
	TESTQ $4, N_DIM
	JZ    inc_r2c2

	// 2x4 KERNEL
	KERNEL_LOAD4_INC
	KERNEL_2x4
	STORE_2x4

	LEAQ (Y_PTR)(INC_Y*4), Y_PTR

inc_r2c2:
	TESTQ $2, N_DIM
	JZ    inc_r2c1

	// 2x2 KERNEL
	KERNEL_LOAD2_INC
	KERNEL_2x2
	STORE_2x2

	LEAQ (Y_PTR)(INC_Y*2), Y_PTR

inc_r2c1:
	TESTQ $1, N_DIM
	JZ    inc_r2end

	// 2x1 KERNEL
	KERNEL_2x1
	STORE_2x1

	ADDQ INC_Y, Y_PTR

inc_r2end:
	LEAQ (X_PTR)(INC_X*2), X_PTR
	MOVQ Y, Y_PTR
	LEAQ (A_ROW)(LDA*2), A_ROW
	MOVQ A_ROW, A_PTR

inc_r1:
	TESTQ $1, M_DIM
	JZ    end

	// LOAD 1
	LOAD_SCALED1

	MOVQ N_DIM, N
	SHRQ $KERNELSIZE, N
	JZ   inc_r1c4

inc_r1c8:
	// 1x8 KERNEL
	KERNEL_LOAD8_INC
	KERNEL_1x8
	STORE_1x8

	LEAQ (Y_PTR)(INC_Y*4), Y_PTR
	DECQ N
	JNZ  inc_r1c8

inc_r1c4:
	TESTQ $4, N_DIM
	JZ    inc_r1c2

	// 1x4 KERNEL
	KERNEL_LOAD4_INC
	KERNEL_1x4
	STORE_1x4

	LEAQ (Y_PTR)(INC_Y*4), Y_PTR

inc_r1c2:
	TESTQ $2, N_DIM
	JZ    inc_r1c1

	// 1x2 KERNEL
	KERNEL_LOAD2_INC
	KERNEL_1x2
	STORE_1x2

	LEAQ (Y_PTR)(INC_Y*2), Y_PTR

inc_r1c1:
	TESTQ $1, N_DIM
	JZ    inc_end

	// 1x1 KERNEL
	KERNEL_1x1
	STORE_1x1

inc_end:
	RET
