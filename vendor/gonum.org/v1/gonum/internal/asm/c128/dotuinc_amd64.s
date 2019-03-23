// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !noasm,!appengine,!safe

#include "textflag.h"

#define MOVDDUP_XPTR__X3    LONG $0x1E120FF2 // MOVDDUP (SI), X3
#define MOVDDUP_XPTR_INCX__X5    LONG $0x120F42F2; WORD $0x062C // MOVDDUP (SI)(R8*1), X5
#define MOVDDUP_XPTR_INCX_2__X7    LONG $0x120F42F2; WORD $0x463C // MOVDDUP (SI)(R8*2), X7
#define MOVDDUP_XPTR_INCx3X__X9    LONG $0x120F46F2; WORD $0x0E0C // MOVDDUP (SI)(R9*1), X9

#define MOVDDUP_8_XPTR__X2    LONG $0x56120FF2; BYTE $0x08 // MOVDDUP 8(SI), X2
#define MOVDDUP_8_XPTR_INCX__X4    LONG $0x120F42F2; WORD $0x0664; BYTE $0x08 // MOVDDUP 8(SI)(R8*1), X4
#define MOVDDUP_8_XPTR_INCX_2__X6    LONG $0x120F42F2; WORD $0x4674; BYTE $0x08 // MOVDDUP 8(SI)(R8*2), X6
#define MOVDDUP_8_XPTR_INCx3X__X8    LONG $0x120F46F2; WORD $0x0E44; BYTE $0x08 // MOVDDUP 8(SI)(R9*1), X8

#define ADDSUBPD_X2_X3    LONG $0xDAD00F66 // ADDSUBPD X2, X3
#define ADDSUBPD_X4_X5    LONG $0xECD00F66 // ADDSUBPD X4, X5
#define ADDSUBPD_X6_X7    LONG $0xFED00F66 // ADDSUBPD X6, X7
#define ADDSUBPD_X8_X9    LONG $0xD00F4566; BYTE $0xC8 // ADDSUBPD X8, X9

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

// func DotuInc(x, y []complex128, n, incX, incY, ix, iy uintptr) (sum complex128)
TEXT ·DotuInc(SB), NOSPLIT, $0
	MOVQ x_base+0(FP), X_PTR       // X_PTR = &x
	MOVQ y_base+24(FP), Y_PTR      // Y_PTR = &y
	MOVQ n+48(FP), LEN             // LEN = n
	PXOR SUM, SUM                  // sum = 0
	CMPQ LEN, $0                   // if LEN == 0 { return }
	JE   dot_end
	MOVQ ix+72(FP), INC_X          // INC_X = ix * sizeof(complex128)
	SHLQ $4, INC_X
	MOVQ iy+80(FP), INC_Y          // INC_Y = iy * sizeof(complex128)
	SHLQ $4, INC_Y
	LEAQ (X_PTR)(INC_X*1), X_PTR   // X_PTR = &(X_PTR[ix])
	LEAQ (Y_PTR)(INC_Y*1), Y_PTR   // Y_PTR = &(Y_PTR[iy])
	MOVQ incX+56(FP), INC_X        // INC_X = incX
	SHLQ $4, INC_X                 // INC_X *=  sizeof(complex128)
	MOVQ incY+64(FP), INC_Y        // INC_Y = incY
	SHLQ $4, INC_Y                 // INC_Y *=  sizeof(complex128)
	MOVQ LEN, TAIL
	ANDQ $3, TAIL                  // LEN = LEN % 4
	SHRQ $2, LEN                   // LEN = floor( LEN / 4 )
	JZ   dot_tail                  // if LEN <= 4 { goto dot_tail }
	PXOR P_SUM, P_SUM              // psum = 0
	LEAQ (INC_X)(INC_X*2), INCx3_X // INCx3_X = 3 * incX * sizeof(complex128)
	LEAQ (INC_Y)(INC_Y*2), INCx3_Y // INCx3_Y = 3 * incY * sizeof(complex128)

dot_loop: // do {
	MOVDDUP_XPTR__X3        // X_(i+1) = { real(x[i], real(x[i]) }
	MOVDDUP_XPTR_INCX__X5
	MOVDDUP_XPTR_INCX_2__X7
	MOVDDUP_XPTR_INCx3X__X9

	MOVDDUP_8_XPTR__X2        // X_i = { imag(x[i]), imag(x[i]) }
	MOVDDUP_8_XPTR_INCX__X4
	MOVDDUP_8_XPTR_INCX_2__X6
	MOVDDUP_8_XPTR_INCx3X__X8

	// X_j = { imag(y[i]), real(y[i]) }
	MOVUPS (Y_PTR), X10
	MOVUPS (Y_PTR)(INC_Y*1), X11
	MOVUPS (Y_PTR)(INC_Y*2), X12
	MOVUPS (Y_PTR)(INCx3_Y*1), X13

	// X_(i+1) = { imag(a) * real(x[i]), real(a) * real(x[i])  }
	MULPD X10, X3
	MULPD X11, X5
	MULPD X12, X7
	MULPD X13, X9

	// X_j     = { real(y[i]), imag(y[i]) }
	SHUFPD $0x1, X10, X10
	SHUFPD $0x1, X11, X11
	SHUFPD $0x1, X12, X12
	SHUFPD $0x1, X13, X13

	// X_i     = { real(a) * imag(x[i]), imag(a) * imag(x[i])  }
	MULPD X10, X2
	MULPD X11, X4
	MULPD X12, X6
	MULPD X13, X8

	// X_(i+1) = {
	//	imag(result[i]):  imag(a)*real(x[i]) + real(a)*imag(x[i]),
	//	real(result[i]):  real(a)*real(x[i]) - imag(a)*imag(x[i])
	//  }
	ADDSUBPD_X2_X3
	ADDSUBPD_X4_X5
	ADDSUBPD_X6_X7
	ADDSUBPD_X8_X9

	// psum += result[i]
	ADDPD X3, SUM
	ADDPD X5, P_SUM
	ADDPD X7, SUM
	ADDPD X9, P_SUM

	LEAQ (X_PTR)(INC_X*4), X_PTR // X_PTR = &(X_PTR[incX*4])
	LEAQ (Y_PTR)(INC_Y*4), Y_PTR // Y_PTR = &(Y_PTR[incY*4])

	DECQ  LEN
	JNZ   dot_loop   // } while --BX > 0
	ADDPD P_SUM, SUM // sum += psum
	CMPQ  TAIL, $0   // if TAIL == 0 { return }
	JE    dot_end

dot_tail: // do {
	MOVDDUP_XPTR__X3      // X_(i+1) = { real(x[i], real(x[i]) }
	MOVDDUP_8_XPTR__X2    // X_i = { imag(x[i]), imag(x[i]) }
	MOVUPS (Y_PTR), X10   // X_j     = {  imag(y[i])          ,  real(y[i])           }
	MULPD  X10, X3        // X_(i+1) = {  imag(a) * real(x[i]),  real(a) * real(x[i]) }
	SHUFPD $0x1, X10, X10 // X_j     = {  real(y[i])          ,  imag(y[i])           }
	MULPD  X10, X2        // X_i     = {  real(a) * imag(x[i]),  imag(a) * imag(x[i]) }

	// X_(i+1) = {
	//	imag(result[i]):  imag(a)*real(x[i]) + real(a)*imag(x[i]),
	//	real(result[i]):  real(a)*real(x[i]) - imag(a)*imag(x[i])
	//  }
	ADDSUBPD_X2_X3
	ADDPD X3, SUM      // sum += result[i]
	ADDQ  INC_X, X_PTR // X_PTR += incX
	ADDQ  INC_Y, Y_PTR // Y_PTR += incY
	DECQ  TAIL         // --TAIL
	JNZ   dot_tail     // }  while TAIL > 0

dot_end:
	MOVUPS SUM, sum+88(FP)
	RET
