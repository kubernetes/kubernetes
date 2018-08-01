// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !noasm,!appengine,!safe

#include "textflag.h"

#define X_PTR SI
#define Y_PTR DI
#define LEN CX
#define TAIL BX
#define INC_X R8
#define INCx3_X R10
#define INC_Y R9
#define INCx3_Y R11
#define SUM X0
#define P_SUM X1

// func DdotInc(x, y []float32, n, incX, incY, ix, iy uintptr) (sum float64)
TEXT ·DdotInc(SB), NOSPLIT, $0
	MOVQ x_base+0(FP), X_PTR  // X_PTR = &x
	MOVQ y_base+24(FP), Y_PTR // Y_PTR = &y
	MOVQ n+48(FP), LEN        // LEN = n
	PXOR SUM, SUM             // SUM = 0
	CMPQ LEN, $0
	JE   dot_end

	MOVQ ix+72(FP), INC_X        // INC_X = ix
	MOVQ iy+80(FP), INC_Y        // INC_Y = iy
	LEAQ (X_PTR)(INC_X*4), X_PTR // X_PTR = &(x[ix])
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR // Y_PTR = &(y[iy])

	MOVQ incX+56(FP), INC_X // INC_X = incX * sizeof(float32)
	SHLQ $2, INC_X
	MOVQ incY+64(FP), INC_Y // INC_Y = incY * sizeof(float32)
	SHLQ $2, INC_Y

	MOVQ LEN, TAIL
	ANDQ $3, TAIL  // TAIL = LEN % 4
	SHRQ $2, LEN   // LEN = floor( LEN / 4 )
	JZ   dot_tail  // if LEN == 0 { goto dot_tail }

	PXOR P_SUM, P_SUM              // P_SUM = 0  for pipelining
	LEAQ (INC_X)(INC_X*2), INCx3_X // INCx3_X = INC_X * 3
	LEAQ (INC_Y)(INC_Y*2), INCx3_Y // INCx3_Y = INC_Y * 3

dot_loop: // Loop unrolled 4x  do {
	CVTSS2SD (X_PTR), X2            // X_i = x[i:i+1]
	CVTSS2SD (X_PTR)(INC_X*1), X3
	CVTSS2SD (X_PTR)(INC_X*2), X4
	CVTSS2SD (X_PTR)(INCx3_X*1), X5

	CVTSS2SD (Y_PTR), X6            // X_j = y[i:i+1]
	CVTSS2SD (Y_PTR)(INC_Y*1), X7
	CVTSS2SD (Y_PTR)(INC_Y*2), X8
	CVTSS2SD (Y_PTR)(INCx3_Y*1), X9

	MULSD X6, X2 // X_i *= X_j
	MULSD X7, X3
	MULSD X8, X4
	MULSD X9, X5

	ADDSD X2, SUM   // SUM += X_i
	ADDSD X3, P_SUM
	ADDSD X4, SUM
	ADDSD X5, P_SUM

	LEAQ (X_PTR)(INC_X*4), X_PTR // X_PTR = &(X_PTR[INC_X * 4])
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR // Y_PTR = &(Y_PTR[INC_Y * 4])

	DECQ LEN
	JNZ  dot_loop // } while --LEN > 0

	ADDSD P_SUM, SUM // SUM += P_SUM
	CMPQ  TAIL, $0   // if TAIL == 0 { return }
	JE    dot_end

dot_tail: // do {
	CVTSS2SD (X_PTR), X2  // X2 = x[i]
	CVTSS2SD (Y_PTR), X3  // X2 *= y[i]
	MULSD    X3, X2
	ADDSD    X2, SUM      // SUM += X2
	ADDQ     INC_X, X_PTR // X_PTR += INC_X
	ADDQ     INC_Y, Y_PTR // Y_PTR += INC_Y
	DECQ     TAIL
	JNZ      dot_tail     // } while --TAIL > 0

dot_end:
	MOVSD SUM, sum+88(FP) // return SUM
	RET
