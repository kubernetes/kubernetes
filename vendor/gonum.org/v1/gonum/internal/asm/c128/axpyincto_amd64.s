// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !noasm,!appengine,!safe

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

// func AxpyIncTo(dst []complex128, incDst, idst uintptr, alpha complex128, x, y []complex128, n, incX, incY, ix, iy uintptr)
TEXT ·AxpyIncTo(SB), NOSPLIT, $0
	MOVQ   dst_base+0(FP), DI // DI = &dst
	MOVQ   x_base+56(FP), SI  // SI = &x
	MOVQ   y_base+80(FP), DX  // DX = &y
	MOVQ   n+104(FP), CX      // CX = n
	CMPQ   CX, $0             // if n==0 { return }
	JE     axpyi_end
	MOVQ   ix+128(FP), R8     // R8 = ix  // Load the first index
	SHLQ   $4, R8             // R8 *= sizeof(complex128)
	MOVQ   iy+136(FP), R9     // R9 = iy
	SHLQ   $4, R9             // R9 *= sizeof(complex128)
	MOVQ   idst+32(FP), R10   // R10 = idst
	SHLQ   $4, R10            // R10 *= sizeof(complex128)
	LEAQ   (SI)(R8*1), SI     // SI = &(x[ix])
	LEAQ   (DX)(R9*1), DX     // DX = &(y[iy])
	LEAQ   (DI)(R10*1), DI    // DI = &(dst[idst])
	MOVQ   incX+112(FP), R8   // R8 = incX
	SHLQ   $4, R8             // R8 *= sizeof(complex128)
	MOVQ   incY+120(FP), R9   // R9 = incY
	SHLQ   $4, R9             // R9 *= sizeof(complex128)
	MOVQ   incDst+24(FP), R10 // R10 = incDst
	SHLQ   $4, R10            // R10 *= sizeof(complex128)
	MOVUPS alpha+40(FP), X0   // X0 = { imag(a), real(a) }
	MOVAPS X0, X1
	SHUFPD $0x1, X1, X1       // X1 = { real(a), imag(a) }
	MOVAPS X0, X10            // Copy X0 and X1 for pipelining
	MOVAPS X1, X11
	MOVQ   CX, BX
	ANDQ   $3, CX             // CX = n % 4
	SHRQ   $2, BX             // BX = floor( n / 4 )
	JZ     axpyi_tail         // if BX == 0 { goto axpyi_tail }

axpyi_loop: // do {
	MOVUPS (SI), X2       // X_i = { imag(x[i]), real(x[i]) }
	MOVUPS (SI)(R8*1), X4
	LEAQ   (SI)(R8*2), SI // SI = &(SI[incX*2])

	MOVUPS (SI), X6
	MOVUPS (SI)(R8*1), X8

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
	ADDPD  (DX), X3
	ADDPD  (DX)(R9*1), X5
	LEAQ   (DX)(R9*2), DX  // DX = &(DX[incY*2])
	ADDPD  (DX), X7
	ADDPD  (DX)(R9*1), X9
	MOVUPS X3, (DI)        // dst[i] = X_(i+1)
	MOVUPS X5, (DI)(R10*1)
	LEAQ   (DI)(R10*2), DI
	MOVUPS X7, (DI)
	MOVUPS X9, (DI)(R10*1)
	LEAQ   (SI)(R8*2), SI  // SI = &(SI[incX*2])
	LEAQ   (DX)(R9*2), DX  // DX = &(DX[incY*2])
	LEAQ   (DI)(R10*2), DI // DI = &(DI[incDst*2])
	DECQ   BX
	JNZ    axpyi_loop      // } while --BX > 0
	CMPQ   CX, $0          // if CX == 0 { return }
	JE     axpyi_end

axpyi_tail: // do {
	MOVUPS (SI), X2     // X_i = { imag(x[i]), real(x[i]) }
	MOVDDUP_X2_X3       // X_(i+1) = { real(x[i], real(x[i]) }
	SHUFPD $0x3, X2, X2 // X_i = { imag(x[i]), imag(x[i]) }
	MULPD  X1, X2       // X_i     = { real(a) * imag(x[i]), imag(a) * imag(x[i])  }
	MULPD  X0, X3       // X_(i+1) = { imag(a) * real(x[i]), real(a) * real(x[i])  }

	// X_(i+1) = {
	//	imag(result[i]):  imag(a)*real(x[i]) + real(a)*imag(x[i]),
	//	real(result[i]):  real(a)*real(x[i]) - imag(a)*imag(x[i])
	//  }
	ADDSUBPD_X2_X3

	// X_(i+1) = { imag(result[i]) + imag(y[i]), real(result[i]) + real(y[i]) }
	ADDPD  (DX), X3
	MOVUPS X3, (DI)   // y[i] X_(i+1)
	ADDQ   R8, SI     // SI += incX
	ADDQ   R9, DX     // DX += incY
	ADDQ   R10, DI    // DI += incDst
	LOOP   axpyi_tail // } while --CX > 0

axpyi_end:
	RET
