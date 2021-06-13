// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

#define SIZE 8

#define M_DIM n+8(FP)
#define M CX
#define N_DIM m+0(FP)
#define N BX

#define TMP1 R14
#define TMP2 R15

#define X_PTR SI
#define X x_base+56(FP)
#define Y_PTR DX
#define Y y_base+96(FP)
#define A_ROW AX
#define A_PTR DI

#define INC_X R8
#define INC3_X R9

#define INC_Y R10
#define INC3_Y R11

#define LDA R12
#define LDA3 R13

#define ALPHA X15
#define BETA X14

#define INIT4 \
	MOVDDUP (X_PTR), X8            \
	MOVDDUP (X_PTR)(INC_X*1), X9   \
	MOVDDUP (X_PTR)(INC_X*2), X10  \
	MOVDDUP (X_PTR)(INC3_X*1), X11 \
	MULPD   ALPHA, X8              \
	MULPD   ALPHA, X9              \
	MULPD   ALPHA, X10             \
	MULPD   ALPHA, X11

#define INIT2 \
	MOVDDUP (X_PTR), X8          \
	MOVDDUP (X_PTR)(INC_X*1), X9 \
	MULPD   ALPHA, X8            \
	MULPD   ALPHA, X9

#define INIT1 \
	MOVDDUP (X_PTR), X8 \
	MULPD   ALPHA, X8

#define KERNEL_LOAD4 \
	MOVUPS (Y_PTR), X0       \
	MOVUPS 2*SIZE(Y_PTR), X1

#define KERNEL_LOAD2 \
	MOVUPS (Y_PTR), X0

#define KERNEL_LOAD4_INC \
	MOVSD  (Y_PTR), X0           \
	MOVHPD (Y_PTR)(INC_Y*1), X0  \
	MOVSD  (Y_PTR)(INC_Y*2), X1  \
	MOVHPD (Y_PTR)(INC3_Y*1), X1

#define KERNEL_LOAD2_INC \
	MOVSD  (Y_PTR), X0          \
	MOVHPD (Y_PTR)(INC_Y*1), X0

#define KERNEL_4x4 \
	MOVUPS (A_PTR), X4               \
	MOVUPS 2*SIZE(A_PTR), X5         \
	MOVUPS (A_PTR)(LDA*1), X6        \
	MOVUPS 2*SIZE(A_PTR)(LDA*1), X7  \
	MULPD  X8, X4                    \
	MULPD  X8, X5                    \
	MULPD  X9, X6                    \
	MULPD  X9, X7                    \
	ADDPD  X4, X0                    \
	ADDPD  X5, X1                    \
	ADDPD  X6, X0                    \
	ADDPD  X7, X1                    \
	MOVUPS (A_PTR)(LDA*2), X4        \
	MOVUPS 2*SIZE(A_PTR)(LDA*2), X5  \
	MOVUPS (A_PTR)(LDA3*1), X6       \
	MOVUPS 2*SIZE(A_PTR)(LDA3*1), X7 \
	MULPD  X10, X4                   \
	MULPD  X10, X5                   \
	MULPD  X11, X6                   \
	MULPD  X11, X7                   \
	ADDPD  X4, X0                    \
	ADDPD  X5, X1                    \
	ADDPD  X6, X0                    \
	ADDPD  X7, X1                    \
	ADDQ   $4*SIZE, A_PTR

#define KERNEL_4x2 \
	MOVUPS (A_PTR), X4              \
	MOVUPS 2*SIZE(A_PTR), X5        \
	MOVUPS (A_PTR)(LDA*1), X6       \
	MOVUPS 2*SIZE(A_PTR)(LDA*1), X7 \
	MULPD  X8, X4                   \
	MULPD  X8, X5                   \
	MULPD  X9, X6                   \
	MULPD  X9, X7                   \
	ADDPD  X4, X0                   \
	ADDPD  X5, X1                   \
	ADDPD  X6, X0                   \
	ADDPD  X7, X1                   \
	ADDQ   $4*SIZE, A_PTR

#define KERNEL_4x1 \
	MOVUPS (A_PTR), X4       \
	MOVUPS 2*SIZE(A_PTR), X5 \
	MULPD  X8, X4            \
	MULPD  X8, X5            \
	ADDPD  X4, X0            \
	ADDPD  X5, X1            \
	ADDQ   $4*SIZE, A_PTR

#define STORE4 \
	MOVUPS X0, (Y_PTR)       \
	MOVUPS X1, 2*SIZE(Y_PTR)

#define STORE4_INC \
	MOVLPD X0, (Y_PTR)           \
	MOVHPD X0, (Y_PTR)(INC_Y*1)  \
	MOVLPD X1, (Y_PTR)(INC_Y*2)  \
	MOVHPD X1, (Y_PTR)(INC3_Y*1)

#define KERNEL_2x4 \
	MOVUPS (A_PTR), X4         \
	MOVUPS (A_PTR)(LDA*1), X5  \
	MOVUPS (A_PTR)(LDA*2), X6  \
	MOVUPS (A_PTR)(LDA3*1), X7 \
	MULPD  X8, X4              \
	MULPD  X9, X5              \
	MULPD  X10, X6             \
	MULPD  X11, X7             \
	ADDPD  X4, X0              \
	ADDPD  X5, X0              \
	ADDPD  X6, X0              \
	ADDPD  X7, X0              \
	ADDQ   $2*SIZE, A_PTR

#define KERNEL_2x2 \
	MOVUPS (A_PTR), X4        \
	MOVUPS (A_PTR)(LDA*1), X5 \
	MULPD  X8, X4             \
	MULPD  X9, X5             \
	ADDPD  X4, X0             \
	ADDPD  X5, X0             \
	ADDQ   $2*SIZE, A_PTR

#define KERNEL_2x1 \
	MOVUPS (A_PTR), X4    \
	MULPD  X8, X4         \
	ADDPD  X4, X0         \
	ADDQ   $2*SIZE, A_PTR

#define STORE2 \
	MOVUPS X0, (Y_PTR)

#define STORE2_INC \
	MOVLPD X0, (Y_PTR)          \
	MOVHPD X0, (Y_PTR)(INC_Y*1)

#define KERNEL_1x4 \
	MOVSD (Y_PTR), X0         \
	MOVSD (A_PTR), X4         \
	MOVSD (A_PTR)(LDA*1), X5  \
	MOVSD (A_PTR)(LDA*2), X6  \
	MOVSD (A_PTR)(LDA3*1), X7 \
	MULSD X8, X4              \
	MULSD X9, X5              \
	MULSD X10, X6             \
	MULSD X11, X7             \
	ADDSD X4, X0              \
	ADDSD X5, X0              \
	ADDSD X6, X0              \
	ADDSD X7, X0              \
	MOVSD X0, (Y_PTR)         \
	ADDQ  $SIZE, A_PTR

#define KERNEL_1x2 \
	MOVSD (Y_PTR), X0        \
	MOVSD (A_PTR), X4        \
	MOVSD (A_PTR)(LDA*1), X5 \
	MULSD X8, X4             \
	MULSD X9, X5             \
	ADDSD X4, X0             \
	ADDSD X5, X0             \
	MOVSD X0, (Y_PTR)        \
	ADDQ  $SIZE, A_PTR

#define KERNEL_1x1 \
	MOVSD (Y_PTR), X0  \
	MOVSD (A_PTR), X4  \
	MULSD X8, X4       \
	ADDSD X4, X0       \
	MOVSD X0, (Y_PTR)  \
	ADDQ  $SIZE, A_PTR

#define SCALE_8(PTR, SCAL) \
	MOVUPS (PTR), X0   \
	MOVUPS 16(PTR), X1 \
	MOVUPS 32(PTR), X2 \
	MOVUPS 48(PTR), X3 \
	MULPD  SCAL, X0    \
	MULPD  SCAL, X1    \
	MULPD  SCAL, X2    \
	MULPD  SCAL, X3    \
	MOVUPS X0, (PTR)   \
	MOVUPS X1, 16(PTR) \
	MOVUPS X2, 32(PTR) \
	MOVUPS X3, 48(PTR)

#define SCALE_4(PTR, SCAL) \
	MOVUPS (PTR), X0   \
	MOVUPS 16(PTR), X1 \
	MULPD  SCAL, X0    \
	MULPD  SCAL, X1    \
	MOVUPS X0, (PTR)   \
	MOVUPS X1, 16(PTR) \

#define SCALE_2(PTR, SCAL) \
	MOVUPS (PTR), X0 \
	MULPD  SCAL, X0  \
	MOVUPS X0, (PTR) \

#define SCALE_1(PTR, SCAL) \
	MOVSD (PTR), X0 \
	MULSD SCAL, X0  \
	MOVSD X0, (PTR) \

#define SCALEINC_4(PTR, INC, INC3, SCAL) \
	MOVSD (PTR), X0         \
	MOVSD (PTR)(INC*1), X1  \
	MOVSD (PTR)(INC*2), X2  \
	MOVSD (PTR)(INC3*1), X3 \
	MULSD SCAL, X0          \
	MULSD SCAL, X1          \
	MULSD SCAL, X2          \
	MULSD SCAL, X3          \
	MOVSD X0, (PTR)         \
	MOVSD X1, (PTR)(INC*1)  \
	MOVSD X2, (PTR)(INC*2)  \
	MOVSD X3, (PTR)(INC3*1)

#define SCALEINC_2(PTR, INC, SCAL) \
	MOVSD (PTR), X0        \
	MOVSD (PTR)(INC*1), X1 \
	MULSD SCAL, X0         \
	MULSD SCAL, X1         \
	MOVSD X0, (PTR)        \
	MOVSD X1, (PTR)(INC*1)

// func GemvT(m, n int,
//	alpha float64,
//	a []float64, lda int,
//	x []float64, incX int,
//	beta float64,
//	y []float64, incY int)
TEXT ·GemvT(SB), NOSPLIT, $32-128
	MOVQ M_DIM, M
	MOVQ N_DIM, N
	CMPQ M, $0
	JE   end
	CMPQ N, $0
	JE   end

	MOVDDUP alpha+16(FP), ALPHA

	MOVQ x_base+56(FP), X_PTR
	MOVQ y_base+96(FP), Y_PTR
	MOVQ a_base+24(FP), A_ROW
	MOVQ incY+120(FP), INC_Y  // INC_Y = incY * sizeof(float64)
	MOVQ lda+48(FP), LDA      // LDA = LDA * sizeof(float64)
	SHLQ $3, LDA
	LEAQ (LDA)(LDA*2), LDA3   // LDA3 = LDA * 3
	MOVQ A_ROW, A_PTR

	MOVQ incX+80(FP), INC_X // INC_X = incX * sizeof(float64)

	XORQ    TMP2, TMP2
	MOVQ    N, TMP1
	SUBQ    $1, TMP1
	NEGQ    TMP1
	IMULQ   INC_X, TMP1
	CMPQ    INC_X, $0
	CMOVQLT TMP1, TMP2
	LEAQ    (X_PTR)(TMP2*SIZE), X_PTR
	MOVQ    X_PTR, X

	SHLQ $3, INC_X
	LEAQ (INC_X)(INC_X*2), INC3_X // INC3_X = INC_X * 3

	CMPQ incY+120(FP), $1 // Check for dense vector Y (fast-path)
	JNE  inc

	MOVSD  $1.0, X0
	COMISD beta+88(FP), X0
	JE     gemv_start

	MOVSD  $0.0, X0
	COMISD beta+88(FP), X0
	JE     gemv_clear

	MOVDDUP beta+88(FP), BETA
	SHRQ    $3, M
	JZ      scal4

scal8:
	SCALE_8(Y_PTR, BETA)
	ADDQ $8*SIZE, Y_PTR
	DECQ M
	JNZ  scal8

scal4:
	TESTQ $4, M_DIM
	JZ    scal2
	SCALE_4(Y_PTR, BETA)
	ADDQ  $4*SIZE, Y_PTR

scal2:
	TESTQ $2, M_DIM
	JZ    scal1
	SCALE_2(Y_PTR, BETA)
	ADDQ  $2*SIZE, Y_PTR

scal1:
	TESTQ $1, M_DIM
	JZ    prep_end
	SCALE_1(Y_PTR, BETA)

	JMP prep_end

gemv_clear: // beta == 0 is special cased to clear memory (no nan handling)
	XORPS X0, X0
	XORPS X1, X1
	XORPS X2, X2
	XORPS X3, X3

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

prep_end:
	MOVQ Y, Y_PTR
	MOVQ M_DIM, M

gemv_start:
	SHRQ $2, N
	JZ   c2

c4:
	// LOAD 4
	INIT4

	MOVQ M_DIM, M
	SHRQ $2, M
	JZ   c4r2

c4r4:
	// 4x4 KERNEL
	KERNEL_LOAD4
	KERNEL_4x4
	STORE4

	ADDQ $4*SIZE, Y_PTR

	DECQ M
	JNZ  c4r4

c4r2:
	TESTQ $2, M_DIM
	JZ    c4r1

	// 4x2 KERNEL
	KERNEL_LOAD2
	KERNEL_2x4
	STORE2

	ADDQ $2*SIZE, Y_PTR

c4r1:
	TESTQ $1, M_DIM
	JZ    c4end

	// 4x1 KERNEL
	KERNEL_1x4

	ADDQ $SIZE, Y_PTR

c4end:
	LEAQ (X_PTR)(INC_X*4), X_PTR
	MOVQ Y, Y_PTR
	LEAQ (A_ROW)(LDA*4), A_ROW
	MOVQ A_ROW, A_PTR

	DECQ N
	JNZ  c4

c2:
	TESTQ $2, N_DIM
	JZ    c1

	// LOAD 2
	INIT2

	MOVQ M_DIM, M
	SHRQ $2, M
	JZ   c2r2

c2r4:
	// 2x4 KERNEL
	KERNEL_LOAD4
	KERNEL_4x2
	STORE4

	ADDQ $4*SIZE, Y_PTR

	DECQ M
	JNZ  c2r4

c2r2:
	TESTQ $2, M_DIM
	JZ    c2r1

	// 2x2 KERNEL
	KERNEL_LOAD2
	KERNEL_2x2
	STORE2

	ADDQ $2*SIZE, Y_PTR

c2r1:
	TESTQ $1, M_DIM
	JZ    c2end

	// 2x1 KERNEL
	KERNEL_1x2

	ADDQ $SIZE, Y_PTR

c2end:
	LEAQ (X_PTR)(INC_X*2), X_PTR
	MOVQ Y, Y_PTR
	LEAQ (A_ROW)(LDA*2), A_ROW
	MOVQ A_ROW, A_PTR

c1:
	TESTQ $1, N_DIM
	JZ    end

	// LOAD 1
	INIT1

	MOVQ M_DIM, M
	SHRQ $2, M
	JZ   c1r2

c1r4:
	// 1x4 KERNEL
	KERNEL_LOAD4
	KERNEL_4x1
	STORE4

	ADDQ $4*SIZE, Y_PTR

	DECQ M
	JNZ  c1r4

c1r2:
	TESTQ $2, M_DIM
	JZ    c1r1

	// 1x2 KERNEL
	KERNEL_LOAD2
	KERNEL_2x1
	STORE2

	ADDQ $2*SIZE, Y_PTR

c1r1:
	TESTQ $1, M_DIM
	JZ    end

	// 1x1 KERNEL
	KERNEL_1x1

end:
	RET

inc:  // Algorithm for incX != 0 ( split loads in kernel )
	XORQ    TMP2, TMP2
	MOVQ    M, TMP1
	SUBQ    $1, TMP1
	IMULQ   INC_Y, TMP1
	NEGQ    TMP1
	CMPQ    INC_Y, $0
	CMOVQLT TMP1, TMP2
	LEAQ    (Y_PTR)(TMP2*SIZE), Y_PTR
	MOVQ    Y_PTR, Y

	SHLQ $3, INC_Y
	LEAQ (INC_Y)(INC_Y*2), INC3_Y // INC3_Y = INC_Y * 3

	MOVSD  $1.0, X0
	COMISD beta+88(FP), X0
	JE     inc_gemv_start

	MOVSD  $0.0, X0
	COMISD beta+88(FP), X0
	JE     inc_gemv_clear

	MOVDDUP beta+88(FP), BETA
	SHRQ    $2, M
	JZ      inc_scal2

inc_scal4:
	SCALEINC_4(Y_PTR, INC_Y, INC3_Y, BETA)
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR
	DECQ M
	JNZ  inc_scal4

inc_scal2:
	TESTQ $2, M_DIM
	JZ    inc_scal1

	SCALEINC_2(Y_PTR, INC_Y, BETA)
	LEAQ (Y_PTR)(INC_Y*2), Y_PTR

inc_scal1:
	TESTQ $1, M_DIM
	JZ    inc_prep_end
	SCALE_1(Y_PTR, BETA)

	JMP inc_prep_end

inc_gemv_clear: // beta == 0 is special-cased to clear memory (no nan handling)
	XORPS X0, X0
	XORPS X1, X1
	XORPS X2, X2
	XORPS X3, X3

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
	JZ    inc_prep_end
	MOVSD X0, (Y_PTR)

inc_prep_end:
	MOVQ Y, Y_PTR
	MOVQ M_DIM, M

inc_gemv_start:
	SHRQ $2, N
	JZ   inc_c2

inc_c4:
	// LOAD 4
	INIT4

	MOVQ M_DIM, M
	SHRQ $2, M
	JZ   inc_c4r2

inc_c4r4:
	// 4x4 KERNEL
	KERNEL_LOAD4_INC
	KERNEL_4x4
	STORE4_INC

	LEAQ (Y_PTR)(INC_Y*4), Y_PTR

	DECQ M
	JNZ  inc_c4r4

inc_c4r2:
	TESTQ $2, M_DIM
	JZ    inc_c4r1

	// 4x2 KERNEL
	KERNEL_LOAD2_INC
	KERNEL_2x4
	STORE2_INC

	LEAQ (Y_PTR)(INC_Y*2), Y_PTR

inc_c4r1:
	TESTQ $1, M_DIM
	JZ    inc_c4end

	// 4x1 KERNEL
	KERNEL_1x4

	ADDQ INC_Y, Y_PTR

inc_c4end:
	LEAQ (X_PTR)(INC_X*4), X_PTR
	MOVQ Y, Y_PTR
	LEAQ (A_ROW)(LDA*4), A_ROW
	MOVQ A_ROW, A_PTR

	DECQ N
	JNZ  inc_c4

inc_c2:
	TESTQ $2, N_DIM
	JZ    inc_c1

	// LOAD 2
	INIT2

	MOVQ M_DIM, M
	SHRQ $2, M
	JZ   inc_c2r2

inc_c2r4:
	// 2x4 KERNEL
	KERNEL_LOAD4_INC
	KERNEL_4x2
	STORE4_INC

	LEAQ (Y_PTR)(INC_Y*4), Y_PTR
	DECQ M
	JNZ  inc_c2r4

inc_c2r2:
	TESTQ $2, M_DIM
	JZ    inc_c2r1

	// 2x2 KERNEL
	KERNEL_LOAD2_INC
	KERNEL_2x2
	STORE2_INC

	LEAQ (Y_PTR)(INC_Y*2), Y_PTR

inc_c2r1:
	TESTQ $1, M_DIM
	JZ    inc_c2end

	// 2x1 KERNEL
	KERNEL_1x2

	ADDQ INC_Y, Y_PTR

inc_c2end:
	LEAQ (X_PTR)(INC_X*2), X_PTR
	MOVQ Y, Y_PTR
	LEAQ (A_ROW)(LDA*2), A_ROW
	MOVQ A_ROW, A_PTR

inc_c1:
	TESTQ $1, N_DIM
	JZ    inc_end

	// LOAD 1
	INIT1

	MOVQ M_DIM, M
	SHRQ $2, M
	JZ   inc_c1r2

inc_c1r4:
	// 1x4 KERNEL
	KERNEL_LOAD4_INC
	KERNEL_4x1
	STORE4_INC

	LEAQ (Y_PTR)(INC_Y*4), Y_PTR
	DECQ M
	JNZ  inc_c1r4

inc_c1r2:
	TESTQ $2, M_DIM
	JZ    inc_c1r1

	// 1x2 KERNEL
	KERNEL_LOAD2_INC
	KERNEL_2x1
	STORE2_INC

	LEAQ (Y_PTR)(INC_Y*2), Y_PTR

inc_c1r1:
	TESTQ $1, M_DIM
	JZ    inc_end

	// 1x1 KERNEL
	KERNEL_1x1

inc_end:
	RET
