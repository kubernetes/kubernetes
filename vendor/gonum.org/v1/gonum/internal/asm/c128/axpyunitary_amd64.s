// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

// MOVDDUP X2, X3
#define MOVDDUP_X2_X3 BYTE $0xF2; BYTE $0x0F; BYTE $0x12; BYTE $0xDA
// MOVDDUP X4, X5
#define MOVDDUP_X4_X5 BYTE $0xF2; BYTE $0x0F; BYTE $0x12; BYTE $0xEC
// MOVDDUP X6, X7
#define MOVDDUP_X6_X7 BYTE $0xF2; BYTE $0x0F; BYTE $0x12; BYTE $0xFE
// MOVDDUP X8, X9
#define MOVDDUP_X8_X9 BYTE $0xF2; BYTE $0x45; BYTE $0x0F; BYTE $0x12; BYTE $0xC8

// ADDSUBPD X2, X3
#define ADDSUBPD_X2_X3 BYTE $0x66; BYTE $0x0F; BYTE $0xD0; BYTE $0xDA
// ADDSUBPD X4, X5
#define ADDSUBPD_X4_X5 BYTE $0x66; BYTE $0x0F; BYTE $0xD0; BYTE $0xEC
// ADDSUBPD X6, X7
#define ADDSUBPD_X6_X7 BYTE $0x66; BYTE $0x0F; BYTE $0xD0; BYTE $0xFE
// ADDSUBPD X8, X9
#define ADDSUBPD_X8_X9 BYTE $0x66; BYTE $0x45; BYTE $0x0F; BYTE $0xD0; BYTE $0xC8

// func AxpyUnitary(alpha complex128, x, y []complex128)
TEXT ·AxpyUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+16(FP), SI // SI = &x
	MOVQ    y_base+40(FP), DI // DI = &y
	MOVQ    x_len+24(FP), CX  // CX = min( len(x), len(y) )
	CMPQ    y_len+48(FP), CX
	CMOVQLE y_len+48(FP), CX
	CMPQ    CX, $0            // if CX == 0 { return }
	JE      caxy_end
	PXOR    X0, X0            // Clear work registers and cache-align loop
	PXOR    X1, X1
	MOVUPS  alpha+0(FP), X0   // X0 = { imag(a), real(a) }
	MOVAPS  X0, X1
	SHUFPD  $0x1, X1, X1      // X1 = { real(a), imag(a) }
	XORQ    AX, AX            // i = 0
	MOVAPS  X0, X10           // Copy X0 and X1 for pipelining
	MOVAPS  X1, X11
	MOVQ    CX, BX
	ANDQ    $3, CX            // CX = n % 4
	SHRQ    $2, BX            // BX = floor( n / 4 )
	JZ      caxy_tail         // if BX == 0 { goto caxy_tail }

caxy_loop: // do {
	MOVUPS (SI)(AX*8), X2   // X_i = { imag(x[i]), real(x[i]) }
	MOVUPS 16(SI)(AX*8), X4
	MOVUPS 32(SI)(AX*8), X6
	MOVUPS 48(SI)(AX*8), X8

	// X_(i+1) = { real(x[i], real(x[i]) }
	MOVDDUP_X2_X3
	MOVDDUP_X4_X5
	MOVDDUP_X6_X7
	MOVDDUP_X8_X9

	// X_i = { imag(x[i]), imag(x[i]) }
	SHUFPD $0x3, X2, X2
	SHUFPD $0x3, X4, X4
	SHUFPD $0x3, X6, X6
	SHUFPD $0x3, X8, X8

	// X_i     = { real(a) * imag(x[i]), imag(a) * imag(x[i])  }
	// X_(i+1) = { imag(a) * real(x[i]), real(a) * real(x[i])  }
	MULPD X1, X2
	MULPD X0, X3
	MULPD X11, X4
	MULPD X10, X5
	MULPD X1, X6
	MULPD X0, X7
	MULPD X11, X8
	MULPD X10, X9

	// X_(i+1) = {
	//	imag(result[i]):  imag(a)*real(x[i]) + real(a)*imag(x[i]),
	//	real(result[i]):  real(a)*real(x[i]) - imag(a)*imag(x[i])
	//  }
	ADDSUBPD_X2_X3
	ADDSUBPD_X4_X5
	ADDSUBPD_X6_X7
	ADDSUBPD_X8_X9

	// X_(i+1) = { imag(result[i]) + imag(y[i]), real(result[i]) + real(y[i]) }
	ADDPD  (DI)(AX*8), X3
	ADDPD  16(DI)(AX*8), X5
	ADDPD  32(DI)(AX*8), X7
	ADDPD  48(DI)(AX*8), X9
	MOVUPS X3, (DI)(AX*8)   // y[i] = X_(i+1)
	MOVUPS X5, 16(DI)(AX*8)
	MOVUPS X7, 32(DI)(AX*8)
	MOVUPS X9, 48(DI)(AX*8)
	ADDQ   $8, AX           // i += 8
	DECQ   BX
	JNZ    caxy_loop        // } while --BX > 0
	CMPQ   CX, $0           // if CX == 0 { return }
	JE     caxy_end

caxy_tail: // do {
	MOVUPS (SI)(AX*8), X2 // X_i = { imag(x[i]), real(x[i]) }
	MOVDDUP_X2_X3         // X_(i+1) = { real(x[i], real(x[i]) }
	SHUFPD $0x3, X2, X2   // X_i = { imag(x[i]), imag(x[i]) }
	MULPD  X1, X2         // X_i     = { real(a) * imag(x[i]), imag(a) * imag(x[i])  }
	MULPD  X0, X3         // X_(i+1) = { imag(a) * real(x[i]), real(a) * real(x[i])  }

	// X_(i+1) = {
	//	imag(result[i]):  imag(a)*real(x[i]) + real(a)*imag(x[i]),
	//	real(result[i]):  real(a)*real(x[i]) - imag(a)*imag(x[i])
	//  }
	ADDSUBPD_X2_X3

	// X_(i+1) = { imag(result[i]) + imag(y[i]), real(result[i]) + real(y[i]) }
	ADDPD  (DI)(AX*8), X3
	MOVUPS X3, (DI)(AX*8) // y[i] = X_(i+1)
	ADDQ   $2, AX         // i += 2
	LOOP   caxy_tail      // }  while --CX > 0

caxy_end:
	RET
