// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !noasm,!appengine,!safe

#include "textflag.h"

#define SIZE 8

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

#define LOAD4 \
	PREFETCHNTA (X_PTR )(INC_X*8)     \
	MOVDDUP     (X_PTR), X1           \
	MOVDDUP     (X_PTR)(INC_X*1), X2  \
	MOVDDUP     (X_PTR)(INC_X*2), X3  \
	MOVDDUP     (X_PTR)(INC3_X*1), X4 \
	MULPD       ALPHA, X1             \
	MULPD       ALPHA, X2             \
	MULPD       ALPHA, X3             \
	MULPD       ALPHA, X4

#define LOAD2 \
	MOVDDUP (X_PTR), X1          \
	MOVDDUP (X_PTR)(INC_X*1), X2 \
	MULPD   ALPHA, X1            \
	MULPD   ALPHA, X2

#define LOAD1 \
	MOVDDUP (X_PTR), X1 \
	MULPD   ALPHA, X1

#define KERNEL_LOAD4 \
	MOVUPS (Y_PTR), X5       \
	MOVUPS 2*SIZE(Y_PTR), X6

#define KERNEL_LOAD4_INC \
	MOVLPD (Y_PTR), X5           \
	MOVHPD (Y_PTR)(INC_Y*1), X5  \
	MOVLPD (Y_PTR)(INC_Y*2), X6  \
	MOVHPD (Y_PTR)(INC3_Y*1), X6

#define KERNEL_LOAD2 \
	MOVUPS (Y_PTR), X5

#define KERNEL_LOAD2_INC \
	MOVLPD (Y_PTR), X5          \
	MOVHPD (Y_PTR)(INC_Y*1), X5

#define KERNEL_4x4 \
	MOVUPS X5, X7  \
	MOVUPS X6, X8  \
	MOVUPS X5, X9  \
	MOVUPS X6, X10 \
	MOVUPS X5, X11 \
	MOVUPS X6, X12 \
	MULPD  X1, X5  \
	MULPD  X1, X6  \
	MULPD  X2, X7  \
	MULPD  X2, X8  \
	MULPD  X3, X9  \
	MULPD  X3, X10 \
	MULPD  X4, X11 \
	MULPD  X4, X12

#define STORE_4x4 \
	MOVUPS (A_PTR), X13               \
	ADDPD  X13, X5                    \
	MOVUPS 2*SIZE(A_PTR), X14         \
	ADDPD  X14, X6                    \
	MOVUPS (A_PTR)(LDA*1), X15        \
	ADDPD  X15, X7                    \
	MOVUPS 2*SIZE(A_PTR)(LDA*1), X0   \
	ADDPD  X0, X8                     \
	MOVUPS (A_PTR)(LDA*2), X13        \
	ADDPD  X13, X9                    \
	MOVUPS 2*SIZE(A_PTR)(LDA*2), X14  \
	ADDPD  X14, X10                   \
	MOVUPS (A_PTR)(LDA3*1), X15       \
	ADDPD  X15, X11                   \
	MOVUPS 2*SIZE(A_PTR)(LDA3*1), X0  \
	ADDPD  X0, X12                    \
	MOVUPS X5, (A_PTR)                \
	MOVUPS X6, 2*SIZE(A_PTR)          \
	MOVUPS X7, (A_PTR)(LDA*1)         \
	MOVUPS X8, 2*SIZE(A_PTR)(LDA*1)   \
	MOVUPS X9, (A_PTR)(LDA*2)         \
	MOVUPS X10, 2*SIZE(A_PTR)(LDA*2)  \
	MOVUPS X11, (A_PTR)(LDA3*1)       \
	MOVUPS X12, 2*SIZE(A_PTR)(LDA3*1) \
	ADDQ   $4*SIZE, A_PTR

#define KERNEL_4x2 \
	MOVUPS X5, X6 \
	MOVUPS X5, X7 \
	MOVUPS X5, X8 \
	MULPD  X1, X5 \
	MULPD  X2, X6 \
	MULPD  X3, X7 \
	MULPD  X4, X8

#define STORE_4x2 \
	MOVUPS (A_PTR), X9          \
	ADDPD  X9, X5               \
	MOVUPS (A_PTR)(LDA*1), X10  \
	ADDPD  X10, X6              \
	MOVUPS (A_PTR)(LDA*2), X11  \
	ADDPD  X11, X7              \
	MOVUPS (A_PTR)(LDA3*1), X12 \
	ADDPD  X12, X8              \
	MOVUPS X5, (A_PTR)          \
	MOVUPS X6, (A_PTR)(LDA*1)   \
	MOVUPS X7, (A_PTR)(LDA*2)   \
	MOVUPS X8, (A_PTR)(LDA3*1)  \
	ADDQ   $2*SIZE, A_PTR

#define KERNEL_4x1 \
	MOVSD (Y_PTR), X5 \
	MOVSD X5, X6      \
	MOVSD X5, X7      \
	MOVSD X5, X8      \
	MULSD X1, X5      \
	MULSD X2, X6      \
	MULSD X3, X7      \
	MULSD X4, X8

#define STORE_4x1 \
	ADDSD (A_PTR), X5         \
	ADDSD (A_PTR)(LDA*1), X6  \
	ADDSD (A_PTR)(LDA*2), X7  \
	ADDSD (A_PTR)(LDA3*1), X8 \
	MOVSD X5, (A_PTR)         \
	MOVSD X6, (A_PTR)(LDA*1)  \
	MOVSD X7, (A_PTR)(LDA*2)  \
	MOVSD X8, (A_PTR)(LDA3*1) \
	ADDQ  $SIZE, A_PTR

#define KERNEL_2x4 \
	MOVUPS X5, X7 \
	MOVUPS X6, X8 \
	MULPD  X1, X5 \
	MULPD  X1, X6 \
	MULPD  X2, X7 \
	MULPD  X2, X8

#define STORE_2x4 \
	MOVUPS (A_PTR), X9               \
	ADDPD  X9, X5                    \
	MOVUPS 2*SIZE(A_PTR), X10        \
	ADDPD  X10, X6                   \
	MOVUPS (A_PTR)(LDA*1), X11       \
	ADDPD  X11, X7                   \
	MOVUPS 2*SIZE(A_PTR)(LDA*1), X12 \
	ADDPD  X12, X8                   \
	MOVUPS X5, (A_PTR)               \
	MOVUPS X6, 2*SIZE(A_PTR)         \
	MOVUPS X7, (A_PTR)(LDA*1)        \
	MOVUPS X8, 2*SIZE(A_PTR)(LDA*1)  \
	ADDQ   $4*SIZE, A_PTR

#define KERNEL_2x2 \
	MOVUPS X5, X6 \
	MULPD  X1, X5 \
	MULPD  X2, X6

#define STORE_2x2 \
	MOVUPS (A_PTR), X7        \
	ADDPD  X7, X5             \
	MOVUPS (A_PTR)(LDA*1), X8 \
	ADDPD  X8, X6             \
	MOVUPS X5, (A_PTR)        \
	MOVUPS X6, (A_PTR)(LDA*1) \
	ADDQ   $2*SIZE, A_PTR

#define KERNEL_2x1 \
	MOVSD (Y_PTR), X5 \
	MOVSD X5, X6      \
	MULSD X1, X5      \
	MULSD X2, X6

#define STORE_2x1 \
	ADDSD (A_PTR), X5        \
	ADDSD (A_PTR)(LDA*1), X6 \
	MOVSD X5, (A_PTR)        \
	MOVSD X6, (A_PTR)(LDA*1) \
	ADDQ  $SIZE, A_PTR

#define KERNEL_1x4 \
	MULPD X1, X5 \
	MULPD X1, X6

#define STORE_1x4 \
	MOVUPS (A_PTR), X7       \
	ADDPD  X7, X5            \
	MOVUPS 2*SIZE(A_PTR), X8 \
	ADDPD  X8, X6            \
	MOVUPS X5, (A_PTR)       \
	MOVUPS X6, 2*SIZE(A_PTR) \
	ADDQ   $4*SIZE, A_PTR

#define KERNEL_1x2 \
	MULPD X1, X5

#define STORE_1x2 \
	MOVUPS (A_PTR), X6    \
	ADDPD  X6, X5         \
	MOVUPS X5, (A_PTR)    \
	ADDQ   $2*SIZE, A_PTR

#define KERNEL_1x1 \
	MOVSD (Y_PTR), X5 \
	MULSD X1, X5

#define STORE_1x1 \
	ADDSD (A_PTR), X5  \
	MOVSD X5, (A_PTR)  \
	ADDQ  $SIZE, A_PTR

// func Ger(m, n uintptr, alpha float64,
//	x []float64, incX uintptr,
//	y []float64, incY uintptr,
//	a []float64, lda uintptr)
TEXT ·Ger(SB), NOSPLIT, $0
	MOVQ M_DIM, M
	MOVQ N_DIM, N
	CMPQ M, $0
	JE   end
	CMPQ N, $0
	JE   end

	MOVDDUP alpha+16(FP), ALPHA

	MOVQ x_base+24(FP), X_PTR
	MOVQ y_base+56(FP), Y_PTR
	MOVQ a_base+88(FP), A_ROW
	MOVQ incX+48(FP), INC_X       // INC_X = incX * sizeof(float64)
	SHLQ $3, INC_X
	MOVQ lda+112(FP), LDA         // LDA = LDA * sizeof(float64)
	SHLQ $3, LDA
	LEAQ (LDA)(LDA*2), LDA3       // LDA3 = LDA * 3
	LEAQ (INC_X)(INC_X*2), INC3_X // INC3_X = INC_X * 3
	MOVQ A_ROW, A_PTR

	XORQ    TMP2, TMP2
	MOVQ    M, TMP1
	SUBQ    $1, TMP1
	IMULQ   INC_X, TMP1
	NEGQ    TMP1
	CMPQ    INC_X, $0
	CMOVQLT TMP1, TMP2
	LEAQ    (X_PTR)(TMP2*SIZE), X_PTR

	CMPQ incY+80(FP), $1 // Check for dense vector Y (fast-path)
	JG   inc
	JL   end

	SHRQ $2, M
	JZ   r2

r4:
	// LOAD 4
	LOAD4

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   r4c2

r4c4:
	// 4x4 KERNEL
	KERNEL_LOAD4
	KERNEL_4x4
	STORE_4x4

	ADDQ $4*SIZE, Y_PTR

	DECQ N
	JNZ  r4c4

	// Reload ALPHA after it's clobbered by STORE_4x4
	MOVDDUP alpha+16(FP), ALPHA

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
	LEAQ (X_PTR)(INC_X*4), X_PTR
	MOVQ Y, Y_PTR
	LEAQ (A_ROW)(LDA*4), A_ROW
	MOVQ A_ROW, A_PTR

	DECQ M
	JNZ  r4

r2:
	TESTQ $2, M_DIM
	JZ    r1

	// LOAD 2
	LOAD2

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   r2c2

r2c4:
	// 2x4 KERNEL
	KERNEL_LOAD4
	KERNEL_2x4
	STORE_2x4

	ADDQ $4*SIZE, Y_PTR

	DECQ N
	JNZ  r2c4

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
	LEAQ (X_PTR)(INC_X*2), X_PTR
	MOVQ Y, Y_PTR
	LEAQ (A_ROW)(LDA*2), A_ROW
	MOVQ A_ROW, A_PTR

r1:
	TESTQ $1, M_DIM
	JZ    end

	// LOAD 1
	LOAD1

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   r1c2

r1c4:
	// 1x4 KERNEL
	KERNEL_LOAD4
	KERNEL_1x4
	STORE_1x4

	ADDQ $4*SIZE, Y_PTR

	DECQ N
	JNZ  r1c4

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

	ADDQ $SIZE, Y_PTR

end:
	RET

inc:  // Algorithm for incY != 1 ( split loads in kernel )

	MOVQ incY+80(FP), INC_Y       // INC_Y = incY * sizeof(float64)
	SHLQ $3, INC_Y
	LEAQ (INC_Y)(INC_Y*2), INC3_Y // INC3_Y = INC_Y * 3

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
	LOAD4

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   inc_r4c2

inc_r4c4:
	// 4x4 KERNEL
	KERNEL_LOAD4_INC
	KERNEL_4x4
	STORE_4x4

	LEAQ (Y_PTR)(INC_Y*4), Y_PTR
	DECQ N
	JNZ  inc_r4c4

	// Reload ALPHA after it's clobbered by STORE_4x4
	MOVDDUP alpha+16(FP), ALPHA

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
	LOAD2

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   inc_r2c2

inc_r2c4:
	// 2x4 KERNEL
	KERNEL_LOAD4_INC
	KERNEL_2x4
	STORE_2x4

	LEAQ (Y_PTR)(INC_Y*4), Y_PTR
	DECQ N
	JNZ  inc_r2c4

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
	LOAD1

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   inc_r1c2

inc_r1c4:
	// 1x4 KERNEL
	KERNEL_LOAD4_INC
	KERNEL_1x4
	STORE_1x4

	LEAQ (Y_PTR)(INC_Y*4), Y_PTR
	DECQ N
	JNZ  inc_r1c4

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
	JZ    end

	// 1x1 KERNEL
	KERNEL_1x1
	STORE_1x1

	ADDQ INC_Y, Y_PTR

inc_end:
	RET
