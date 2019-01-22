// Copyright ©2017 The gonum Authors. All rights reserved.
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
#define X x_base+56(FP)
#define INC_X R8
#define INC3_X R9

#define Y_PTR DX
#define Y y_base+96(FP)
#define INC_Y R10
#define INC3_Y R11

#define A_ROW AX
#define A_PTR DI
#define LDA R12
#define LDA3 R13

#define ALPHA X15
#define BETA X14

#define INIT4 \
	XORPS X0, X0 \
	XORPS X1, X1 \
	XORPS X2, X2 \
	XORPS X3, X3

#define INIT2 \
	XORPS X0, X0 \
	XORPS X1, X1

#define INIT1 \
	XORPS X0, X0

#define KERNEL_LOAD4 \
	MOVUPS (X_PTR), X12       \
	MOVUPS 2*SIZE(X_PTR), X13

#define KERNEL_LOAD2 \
	MOVUPS (X_PTR), X12

#define KERNEL_LOAD4_INC \
	MOVSD  (X_PTR), X12           \
	MOVHPD (X_PTR)(INC_X*1), X12  \
	MOVSD  (X_PTR)(INC_X*2), X13  \
	MOVHPD (X_PTR)(INC3_X*1), X13

#define KERNEL_LOAD2_INC \
	MOVSD  (X_PTR), X12          \
	MOVHPD (X_PTR)(INC_X*1), X12

#define KERNEL_4x4 \
	MOVUPS (A_PTR), X4                \
	MOVUPS 2*SIZE(A_PTR), X5          \
	MOVUPS (A_PTR)(LDA*1), X6         \
	MOVUPS 2*SIZE(A_PTR)(LDA*1), X7   \
	MOVUPS (A_PTR)(LDA*2), X8         \
	MOVUPS 2*SIZE(A_PTR)(LDA*2), X9   \
	MOVUPS (A_PTR)(LDA3*1), X10       \
	MOVUPS 2*SIZE(A_PTR)(LDA3*1), X11 \
	MULPD  X12, X4                    \
	MULPD  X13, X5                    \
	MULPD  X12, X6                    \
	MULPD  X13, X7                    \
	MULPD  X12, X8                    \
	MULPD  X13, X9                    \
	MULPD  X12, X10                   \
	MULPD  X13, X11                   \
	ADDPD  X4, X0                     \
	ADDPD  X5, X0                     \
	ADDPD  X6, X1                     \
	ADDPD  X7, X1                     \
	ADDPD  X8, X2                     \
	ADDPD  X9, X2                     \
	ADDPD  X10, X3                    \
	ADDPD  X11, X3                    \
	ADDQ   $4*SIZE, A_PTR

#define KERNEL_4x2 \
	MOVUPS (A_PTR), X4         \
	MOVUPS (A_PTR)(LDA*1), X5  \
	MOVUPS (A_PTR)(LDA*2), X6  \
	MOVUPS (A_PTR)(LDA3*1), X7 \
	MULPD  X12, X4             \
	MULPD  X12, X5             \
	MULPD  X12, X6             \
	MULPD  X12, X7             \
	ADDPD  X4, X0              \
	ADDPD  X5, X1              \
	ADDPD  X6, X2              \
	ADDPD  X7, X3              \
	ADDQ   $2*SIZE, A_PTR

#define KERNEL_4x1 \
	MOVDDUP (X_PTR), X12        \
	MOVSD   (A_PTR), X4         \
	MOVHPD  (A_PTR)(LDA*1), X4  \
	MOVSD   (A_PTR)(LDA*2), X5  \
	MOVHPD  (A_PTR)(LDA3*1), X5 \
	MULPD   X12, X4             \
	MULPD   X12, X5             \
	ADDPD   X4, X0              \
	ADDPD   X5, X2              \
	ADDQ    $SIZE, A_PTR

#define STORE4 \
	MOVUPS (Y_PTR), X4       \
	MOVUPS 2*SIZE(Y_PTR), X5 \
	MULPD  ALPHA, X0         \
	MULPD  ALPHA, X2         \
	MULPD  BETA, X4          \
	MULPD  BETA, X5          \
	ADDPD  X0, X4            \
	ADDPD  X2, X5            \
	MOVUPS X4, (Y_PTR)       \
	MOVUPS X5, 2*SIZE(Y_PTR)

#define STORE4_INC \
	MOVSD  (Y_PTR), X4           \
	MOVHPD (Y_PTR)(INC_Y*1), X4  \
	MOVSD  (Y_PTR)(INC_Y*2), X5  \
	MOVHPD (Y_PTR)(INC3_Y*1), X5 \
	MULPD  ALPHA, X0             \
	MULPD  ALPHA, X2             \
	MULPD  BETA, X4              \
	MULPD  BETA, X5              \
	ADDPD  X0, X4                \
	ADDPD  X2, X5                \
	MOVLPD X4, (Y_PTR)           \
	MOVHPD X4, (Y_PTR)(INC_Y*1)  \
	MOVLPD X5, (Y_PTR)(INC_Y*2)  \
	MOVHPD X5, (Y_PTR)(INC3_Y*1)

#define KERNEL_2x4 \
	MOVUPS (A_PTR), X8               \
	MOVUPS 2*SIZE(A_PTR), X9         \
	MOVUPS (A_PTR)(LDA*1), X10       \
	MOVUPS 2*SIZE(A_PTR)(LDA*1), X11 \
	MULPD  X12, X8                   \
	MULPD  X13, X9                   \
	MULPD  X12, X10                  \
	MULPD  X13, X11                  \
	ADDPD  X8, X0                    \
	ADDPD  X10, X1                   \
	ADDPD  X9, X0                    \
	ADDPD  X11, X1                   \
	ADDQ   $4*SIZE, A_PTR

#define KERNEL_2x2 \
	MOVUPS (A_PTR), X8        \
	MOVUPS (A_PTR)(LDA*1), X9 \
	MULPD  X12, X8            \
	MULPD  X12, X9            \
	ADDPD  X8, X0             \
	ADDPD  X9, X1             \
	ADDQ   $2*SIZE, A_PTR

#define KERNEL_2x1 \
	MOVDDUP (X_PTR), X12       \
	MOVSD   (A_PTR), X8        \
	MOVHPD  (A_PTR)(LDA*1), X8 \
	MULPD   X12, X8            \
	ADDPD   X8, X0             \
	ADDQ    $SIZE, A_PTR

#define STORE2 \
	MOVUPS (Y_PTR), X4 \
	MULPD  ALPHA, X0   \
	MULPD  BETA, X4    \
	ADDPD  X0, X4      \
	MOVUPS X4, (Y_PTR)

#define STORE2_INC \
	MOVSD  (Y_PTR), X4          \
	MOVHPD (Y_PTR)(INC_Y*1), X4 \
	MULPD  ALPHA, X0            \
	MULPD  BETA, X4             \
	ADDPD  X0, X4               \
	MOVSD  X4, (Y_PTR)          \
	MOVHPD X4, (Y_PTR)(INC_Y*1)

#define KERNEL_1x4 \
	MOVUPS (A_PTR), X8       \
	MOVUPS 2*SIZE(A_PTR), X9 \
	MULPD  X12, X8           \
	MULPD  X13, X9           \
	ADDPD  X8, X0            \
	ADDPD  X9, X0            \
	ADDQ   $4*SIZE, A_PTR

#define KERNEL_1x2 \
	MOVUPS (A_PTR), X8    \
	MULPD  X12, X8        \
	ADDPD  X8, X0         \
	ADDQ   $2*SIZE, A_PTR

#define KERNEL_1x1 \
	MOVSD (X_PTR), X12 \
	MOVSD (A_PTR), X8  \
	MULSD X12, X8      \
	ADDSD X8, X0       \
	ADDQ  $SIZE, A_PTR

#define STORE1 \
	HADDPD X0, X0      \
	MOVSD  (Y_PTR), X4 \
	MULSD  ALPHA, X0   \
	MULSD  BETA, X4    \
	ADDSD  X0, X4      \
	MOVSD  X4, (Y_PTR)

// func GemvN(m, n int,
//	alpha float64,
//	a []float64, lda int,
//	x []float64, incX int,
//	beta float64,
//	y []float64, incY int)
TEXT ·GemvN(SB), NOSPLIT, $32-128
	MOVQ M_DIM, M
	MOVQ N_DIM, N
	CMPQ M, $0
	JE   end
	CMPQ N, $0
	JE   end

	MOVDDUP alpha+16(FP), ALPHA
	MOVDDUP beta+88(FP), BETA

	MOVQ x_base+56(FP), X_PTR
	MOVQ y_base+96(FP), Y_PTR
	MOVQ a_base+24(FP), A_ROW
	MOVQ incY+120(FP), INC_Y
	MOVQ lda+48(FP), LDA      // LDA = LDA * sizeof(float64)
	SHLQ $3, LDA
	LEAQ (LDA)(LDA*2), LDA3   // LDA3 = LDA * 3
	MOVQ A_ROW, A_PTR

	XORQ    TMP2, TMP2
	MOVQ    M, TMP1
	SUBQ    $1, TMP1
	IMULQ   INC_Y, TMP1
	NEGQ    TMP1
	CMPQ    INC_Y, $0
	CMOVQLT TMP1, TMP2
	LEAQ    (Y_PTR)(TMP2*SIZE), Y_PTR
	MOVQ    Y_PTR, Y

	SHLQ $3, INC_Y                // INC_Y = incY * sizeof(float64)
	LEAQ (INC_Y)(INC_Y*2), INC3_Y // INC3_Y = INC_Y * 3

	MOVSD  $0.0, X0
	COMISD BETA, X0
	JNE    gemv_start // if beta != 0 { goto gemv_start }

gemv_clear: // beta == 0 is special cased to clear memory (no nan handling)
	XORPS X0, X0
	XORPS X1, X1
	XORPS X2, X2
	XORPS X3, X3

	CMPQ incY+120(FP), $1 // Check for dense vector X (fast-path)
	JNE  inc_clear

	SHRQ $3, M
	JZ   clear4

clear8:
	MOVUPS X0, (Y_PTR)
	MOVUPS X1, 16(Y_PTR)
	MOVUPS X2, 32(Y_PTR)
	MOVUPS X3, 48(Y_PTR)
	ADDQ   $8*SIZE, Y_PTR
	DECQ   M
	JNZ    clear8

clear4:
	TESTQ  $4, M_DIM
	JZ     clear2
	MOVUPS X0, (Y_PTR)
	MOVUPS X1, 16(Y_PTR)
	ADDQ   $4*SIZE, Y_PTR

clear2:
	TESTQ  $2, M_DIM
	JZ     clear1
	MOVUPS X0, (Y_PTR)
	ADDQ   $2*SIZE, Y_PTR

clear1:
	TESTQ $1, M_DIM
	JZ    prep_end
	MOVSD X0, (Y_PTR)

	JMP prep_end

inc_clear:
	SHRQ $2, M
	JZ   inc_clear2

inc_clear4:
	MOVSD X0, (Y_PTR)
	MOVSD X1, (Y_PTR)(INC_Y*1)
	MOVSD X2, (Y_PTR)(INC_Y*2)
	MOVSD X3, (Y_PTR)(INC3_Y*1)
	LEAQ  (Y_PTR)(INC_Y*4), Y_PTR
	DECQ  M
	JNZ   inc_clear4

inc_clear2:
	TESTQ $2, M_DIM
	JZ    inc_clear1
	MOVSD X0, (Y_PTR)
	MOVSD X1, (Y_PTR)(INC_Y*1)
	LEAQ  (Y_PTR)(INC_Y*2), Y_PTR

inc_clear1:
	TESTQ $1, M_DIM
	JZ    prep_end
	MOVSD X0, (Y_PTR)

prep_end:
	MOVQ Y, Y_PTR
	MOVQ M_DIM, M

gemv_start:
	CMPQ incX+80(FP), $1 // Check for dense vector X (fast-path)
	JNE  inc

	SHRQ $2, M
	JZ   r2

r4:
	// LOAD 4
	INIT4

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   r4c2

r4c4:
	// 4x4 KERNEL
	KERNEL_LOAD4
	KERNEL_4x4

	ADDQ $4*SIZE, X_PTR

	DECQ N
	JNZ  r4c4

r4c2:
	TESTQ $2, N_DIM
	JZ    r4c1

	// 4x2 KERNEL
	KERNEL_LOAD2
	KERNEL_4x2

	ADDQ $2*SIZE, X_PTR

r4c1:
	HADDPD X1, X0
	HADDPD X3, X2
	TESTQ  $1, N_DIM
	JZ     r4end

	// 4x1 KERNEL
	KERNEL_4x1

	ADDQ $SIZE, X_PTR

r4end:
	CMPQ INC_Y, $SIZE
	JNZ  r4st_inc

	STORE4
	ADDQ $4*SIZE, Y_PTR
	JMP  r4inc

r4st_inc:
	STORE4_INC
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR

r4inc:
	MOVQ X, X_PTR
	LEAQ (A_ROW)(LDA*4), A_ROW
	MOVQ A_ROW, A_PTR

	DECQ M
	JNZ  r4

r2:
	TESTQ $2, M_DIM
	JZ    r1

	// LOAD 2
	INIT2

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   r2c2

r2c4:
	// 2x4 KERNEL
	KERNEL_LOAD4
	KERNEL_2x4

	ADDQ $4*SIZE, X_PTR

	DECQ N
	JNZ  r2c4

r2c2:
	TESTQ $2, N_DIM
	JZ    r2c1

	// 2x2 KERNEL
	KERNEL_LOAD2
	KERNEL_2x2

	ADDQ $2*SIZE, X_PTR

r2c1:
	HADDPD X1, X0
	TESTQ  $1, N_DIM
	JZ     r2end

	// 2x1 KERNEL
	KERNEL_2x1

	ADDQ $SIZE, X_PTR

r2end:
	CMPQ INC_Y, $SIZE
	JNE  r2st_inc

	STORE2
	ADDQ $2*SIZE, Y_PTR
	JMP  r2inc

r2st_inc:
	STORE2_INC
	LEAQ (Y_PTR)(INC_Y*2), Y_PTR

r2inc:
	MOVQ X, X_PTR
	LEAQ (A_ROW)(LDA*2), A_ROW
	MOVQ A_ROW, A_PTR

r1:
	TESTQ $1, M_DIM
	JZ    end

	// LOAD 1
	INIT1

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   r1c2

r1c4:
	// 1x4 KERNEL
	KERNEL_LOAD4
	KERNEL_1x4

	ADDQ $4*SIZE, X_PTR

	DECQ N
	JNZ  r1c4

r1c2:
	TESTQ $2, N_DIM
	JZ    r1c1

	// 1x2 KERNEL
	KERNEL_LOAD2
	KERNEL_1x2

	ADDQ $2*SIZE, X_PTR

r1c1:

	TESTQ $1, N_DIM
	JZ    r1end

	// 1x1 KERNEL
	KERNEL_1x1

r1end:
	STORE1

end:
	RET

inc:  // Algorithm for incX != 1 ( split loads in kernel )
	MOVQ incX+80(FP), INC_X // INC_X = incX

	XORQ    TMP2, TMP2                // TMP2  = 0
	MOVQ    N, TMP1                   // TMP1 = N
	SUBQ    $1, TMP1                  // TMP1 -= 1
	NEGQ    TMP1                      // TMP1 = -TMP1
	IMULQ   INC_X, TMP1               // TMP1 *= INC_X
	CMPQ    INC_X, $0                 // if INC_X < 0 { TMP2 = TMP1 }
	CMOVQLT TMP1, TMP2
	LEAQ    (X_PTR)(TMP2*SIZE), X_PTR // X_PTR = X_PTR[TMP2]
	MOVQ    X_PTR, X                  // X = X_PTR

	SHLQ $3, INC_X
	LEAQ (INC_X)(INC_X*2), INC3_X // INC3_X = INC_X * 3

	SHRQ $2, M
	JZ   inc_r2

inc_r4:
	// LOAD 4
	INIT4

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   inc_r4c2

inc_r4c4:
	// 4x4 KERNEL
	KERNEL_LOAD4_INC
	KERNEL_4x4

	LEAQ (X_PTR)(INC_X*4), X_PTR

	DECQ N
	JNZ  inc_r4c4

inc_r4c2:
	TESTQ $2, N_DIM
	JZ    inc_r4c1

	// 4x2 KERNEL
	KERNEL_LOAD2_INC
	KERNEL_4x2

	LEAQ (X_PTR)(INC_X*2), X_PTR

inc_r4c1:
	HADDPD X1, X0
	HADDPD X3, X2
	TESTQ  $1, N_DIM
	JZ     inc_r4end

	// 4x1 KERNEL
	KERNEL_4x1

	ADDQ INC_X, X_PTR

inc_r4end:
	CMPQ INC_Y, $SIZE
	JNE  inc_r4st_inc

	STORE4
	ADDQ $4*SIZE, Y_PTR
	JMP  inc_r4inc

inc_r4st_inc:
	STORE4_INC
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR

inc_r4inc:
	MOVQ X, X_PTR
	LEAQ (A_ROW)(LDA*4), A_ROW
	MOVQ A_ROW, A_PTR

	DECQ M
	JNZ  inc_r4

inc_r2:
	TESTQ $2, M_DIM
	JZ    inc_r1

	// LOAD 2
	INIT2

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   inc_r2c2

inc_r2c4:
	// 2x4 KERNEL
	KERNEL_LOAD4_INC
	KERNEL_2x4

	LEAQ (X_PTR)(INC_X*4), X_PTR
	DECQ N
	JNZ  inc_r2c4

inc_r2c2:
	TESTQ $2, N_DIM
	JZ    inc_r2c1

	// 2x2 KERNEL
	KERNEL_LOAD2_INC
	KERNEL_2x2

	LEAQ (X_PTR)(INC_X*2), X_PTR

inc_r2c1:
	HADDPD X1, X0
	TESTQ  $1, N_DIM
	JZ     inc_r2end

	// 2x1 KERNEL
	KERNEL_2x1

	ADDQ INC_X, X_PTR

inc_r2end:
	CMPQ INC_Y, $SIZE
	JNE  inc_r2st_inc

	STORE2
	ADDQ $2*SIZE, Y_PTR
	JMP  inc_r2inc

inc_r2st_inc:
	STORE2_INC
	LEAQ (Y_PTR)(INC_Y*2), Y_PTR

inc_r2inc:
	MOVQ X, X_PTR
	LEAQ (A_ROW)(LDA*2), A_ROW
	MOVQ A_ROW, A_PTR

inc_r1:
	TESTQ $1, M_DIM
	JZ    inc_end

	// LOAD 1
	INIT1

	MOVQ N_DIM, N
	SHRQ $2, N
	JZ   inc_r1c2

inc_r1c4:
	// 1x4 KERNEL
	KERNEL_LOAD4_INC
	KERNEL_1x4

	LEAQ (X_PTR)(INC_X*4), X_PTR
	DECQ N
	JNZ  inc_r1c4

inc_r1c2:
	TESTQ $2, N_DIM
	JZ    inc_r1c1

	// 1x2 KERNEL
	KERNEL_LOAD2_INC
	KERNEL_1x2

	LEAQ (X_PTR)(INC_X*2), X_PTR

inc_r1c1:
	TESTQ $1, N_DIM
	JZ    inc_r1end

	// 1x1 KERNEL
	KERNEL_1x1

inc_r1end:
	STORE1

inc_end:
	RET
