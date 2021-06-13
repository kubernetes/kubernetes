// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

#define MOVSHDUP_X3_X2    LONG $0xD3160FF3 // MOVSHDUP X3, X2
#define MOVSHDUP_X5_X4    LONG $0xE5160FF3 // MOVSHDUP X5, X4
#define MOVSHDUP_X7_X6    LONG $0xF7160FF3 // MOVSHDUP X7, X6
#define MOVSHDUP_X9_X8    LONG $0x160F45F3; BYTE $0xC1 // MOVSHDUP X9, X8

#define MOVSLDUP_X3_X3    LONG $0xDB120FF3 // MOVSLDUP X3, X3
#define MOVSLDUP_X5_X5    LONG $0xED120FF3 // MOVSLDUP X5, X5
#define MOVSLDUP_X7_X7    LONG $0xFF120FF3 // MOVSLDUP X7, X7
#define MOVSLDUP_X9_X9    LONG $0x120F45F3; BYTE $0xC9 // MOVSLDUP X9, X9

#define ADDSUBPS_X2_X3    LONG $0xDAD00FF2 // ADDSUBPS X2, X3
#define ADDSUBPS_X4_X5    LONG $0xECD00FF2 // ADDSUBPS X4, X5
#define ADDSUBPS_X6_X7    LONG $0xFED00FF2 // ADDSUBPS X6, X7
#define ADDSUBPS_X8_X9    LONG $0xD00F45F2; BYTE $0xC8 // ADDSUBPS X8, X9

#define X_PTR SI
#define Y_PTR DI
#define LEN CX
#define TAIL BX
#define SUM X0
#define P_SUM X1
#define INC_X R8
#define INCx3_X R9
#define INC_Y R10
#define INCx3_Y R11
#define NEG1 X15
#define P_NEG1 X14

// func DotcInc(x, y []complex64, n, incX, incY, ix, iy uintptr) (sum complex64)
TEXT ·DotcInc(SB), NOSPLIT, $0
	MOVQ   x_base+0(FP), X_PTR     // X_PTR = &x
	MOVQ   y_base+24(FP), Y_PTR    // Y_PTR = &y
	PXOR   SUM, SUM                // SUM = 0
	PXOR   P_SUM, P_SUM            // P_SUM = 0
	MOVQ   n+48(FP), LEN           // LEN = n
	CMPQ   LEN, $0                 // if LEN == 0 { return }
	JE     dotc_end
	MOVQ   ix+72(FP), INC_X
	MOVQ   iy+80(FP), INC_Y
	LEAQ   (X_PTR)(INC_X*8), X_PTR // X_PTR = &(X_PTR[ix])
	LEAQ   (Y_PTR)(INC_Y*8), Y_PTR // Y_PTR = &(Y_PTR[iy])
	MOVQ   incX+56(FP), INC_X      // INC_X = incX * sizeof(complex64)
	SHLQ   $3, INC_X
	MOVQ   incY+64(FP), INC_Y      // INC_Y = incY * sizeof(complex64)
	SHLQ   $3, INC_Y
	MOVSS  $(-1.0), NEG1
	SHUFPS $0, NEG1, NEG1          // { -1, -1, -1, -1 }

	MOVQ LEN, TAIL
	ANDQ $3, TAIL  // TAIL = LEN % 4
	SHRQ $2, LEN   // LEN = floor( LEN / 4 )
	JZ   dotc_tail // if LEN == 0 { goto dotc_tail }

	MOVUPS NEG1, P_NEG1              // Copy NEG1 for pipelining
	LEAQ   (INC_X)(INC_X*2), INCx3_X // INCx3_X = INC_X * 3
	LEAQ   (INC_Y)(INC_Y*2), INCx3_Y // INCx3_Y = INC_Y * 3

dotc_loop: // do {
	MOVSD (X_PTR), X3            // X_i = { imag(x[i]), real(x[i]) }
	MOVSD (X_PTR)(INC_X*1), X5
	MOVSD (X_PTR)(INC_X*2), X7
	MOVSD (X_PTR)(INCx3_X*1), X9

	// X_(i-1) = { imag(x[i]), imag(x[i]) }
	MOVSHDUP_X3_X2
	MOVSHDUP_X5_X4
	MOVSHDUP_X7_X6
	MOVSHDUP_X9_X8

	// X_i = { real(x[i]), real(x[i]) }
	MOVSLDUP_X3_X3
	MOVSLDUP_X5_X5
	MOVSLDUP_X7_X7
	MOVSLDUP_X9_X9

	// X_(i-1) = { -imag(x[i]), -imag(x[i]) }
	MULPS NEG1, X2
	MULPS P_NEG1, X4
	MULPS NEG1, X6
	MULPS P_NEG1, X8

	// X_j = { imag(y[i]), real(y[i]) }
	MOVSD (Y_PTR), X10
	MOVSD (Y_PTR)(INC_Y*1), X11
	MOVSD (Y_PTR)(INC_Y*2), X12
	MOVSD (Y_PTR)(INCx3_Y*1), X13

	// X_i     = { imag(y[i]) * real(x[i]), real(y[i]) * real(x[i]) }
	MULPS X10, X3
	MULPS X11, X5
	MULPS X12, X7
	MULPS X13, X9

	// X_j = { real(y[i]), imag(y[i]) }
	SHUFPS $0xB1, X10, X10
	SHUFPS $0xB1, X11, X11
	SHUFPS $0xB1, X12, X12
	SHUFPS $0xB1, X13, X13

	// X_(i-1) = { real(y[i]) * imag(x[i]), imag(y[i]) * imag(x[i]) }
	MULPS X10, X2
	MULPS X11, X4
	MULPS X12, X6
	MULPS X13, X8

	// X_i = {
	//	imag(result[i]):  imag(y[i]) * real(x[i]) + real(y[i]) * imag(x[i]),
	//	real(result[i]):  real(y[i]) * real(x[i]) - imag(y[i]) * imag(x[i])  }
	ADDSUBPS_X2_X3
	ADDSUBPS_X4_X5
	ADDSUBPS_X6_X7
	ADDSUBPS_X8_X9

	// SUM += X_i
	ADDPS X3, SUM
	ADDPS X5, P_SUM
	ADDPS X7, SUM
	ADDPS X9, P_SUM

	LEAQ (X_PTR)(INC_X*4), X_PTR // X_PTR = &(X_PTR[INC_X*4])
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR // Y_PTR = &(Y_PTR[INC_Y*4])

	DECQ LEN
	JNZ  dotc_loop // } while --LEN > 0

	ADDPS P_SUM, SUM // SUM = { P_SUM + SUM }
	CMPQ  TAIL, $0   // if TAIL == 0 { return }
	JE    dotc_end

dotc_tail: // do {
	MOVSD  (X_PTR), X3    // X_i = { imag(x[i]), real(x[i]) }
	MOVSHDUP_X3_X2        // X_(i-1) = { imag(x[i]), imag(x[i]) }
	MOVSLDUP_X3_X3        // X_i = { real(x[i]), real(x[i]) }
	MULPS  NEG1, X2       // X_(i-1) = { -imag(x[i]), imag(x[i]) }
	MOVUPS (Y_PTR), X10   // X_j = { imag(y[i]), real(y[i]) }
	MULPS  X10, X3        // X_i = { imag(y[i]) * real(x[i]), real(y[i]) * real(x[i]) }
	SHUFPS $0x1, X10, X10 // X_j = { real(y[i]), imag(y[i]) }
	MULPS  X10, X2        // X_(i-1) = { real(y[i]) * imag(x[i]), imag(y[i]) * imag(x[i]) }

	// X_i = {
	//	imag(result[i]):  imag(y[i])*real(x[i]) + real(y[i])*imag(x[i]),
	//	real(result[i]):  real(y[i])*real(x[i]) - imag(y[i])*imag(x[i]) }
	ADDSUBPS_X2_X3
	ADDPS X3, SUM      // SUM += X_i
	ADDQ  INC_X, X_PTR // X_PTR += INC_X
	ADDQ  INC_Y, Y_PTR // Y_PTR += INC_Y
	DECQ  TAIL
	JNZ   dotc_tail    // } while --TAIL > 0

dotc_end:
	MOVSD SUM, sum+88(FP) // return SUM
	RET
