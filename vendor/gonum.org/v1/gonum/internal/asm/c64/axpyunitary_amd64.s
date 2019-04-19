// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !noasm,!appengine,!safe

#include "textflag.h"

// MOVSHDUP X3, X2
#define MOVSHDUP_X3_X2 BYTE $0xF3; BYTE $0x0F; BYTE $0x16; BYTE $0xD3
// MOVSLDUP X3, X3
#define MOVSLDUP_X3_X3 BYTE $0xF3; BYTE $0x0F; BYTE $0x12; BYTE $0xDB
// ADDSUBPS X2, X3
#define ADDSUBPS_X2_X3 BYTE $0xF2; BYTE $0x0F; BYTE $0xD0; BYTE $0xDA

// MOVSHDUP X5, X4
#define MOVSHDUP_X5_X4 BYTE $0xF3; BYTE $0x0F; BYTE $0x16; BYTE $0xE5
// MOVSLDUP X5, X5
#define MOVSLDUP_X5_X5 BYTE $0xF3; BYTE $0x0F; BYTE $0x12; BYTE $0xED
// ADDSUBPS X4, X5
#define ADDSUBPS_X4_X5 BYTE $0xF2; BYTE $0x0F; BYTE $0xD0; BYTE $0xEC

// MOVSHDUP X7, X6
#define MOVSHDUP_X7_X6 BYTE $0xF3; BYTE $0x0F; BYTE $0x16; BYTE $0xF7
// MOVSLDUP X7, X7
#define MOVSLDUP_X7_X7 BYTE $0xF3; BYTE $0x0F; BYTE $0x12; BYTE $0xFF
// ADDSUBPS X6, X7
#define ADDSUBPS_X6_X7 BYTE $0xF2; BYTE $0x0F; BYTE $0xD0; BYTE $0xFE

// MOVSHDUP X9, X8
#define MOVSHDUP_X9_X8 BYTE $0xF3; BYTE $0x45; BYTE $0x0F; BYTE $0x16; BYTE $0xC1
// MOVSLDUP X9, X9
#define MOVSLDUP_X9_X9 BYTE $0xF3; BYTE $0x45; BYTE $0x0F; BYTE $0x12; BYTE $0xC9
// ADDSUBPS X8, X9
#define ADDSUBPS_X8_X9 BYTE $0xF2; BYTE $0x45; BYTE $0x0F; BYTE $0xD0; BYTE $0xC8

// func AxpyUnitary(alpha complex64, x, y []complex64)
TEXT ·AxpyUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+8(FP), SI  // SI = &x
	MOVQ    y_base+32(FP), DI // DI = &y
	MOVQ    x_len+16(FP), CX  // CX = min( len(x), len(y) )
	CMPQ    y_len+40(FP), CX
	CMOVQLE y_len+40(FP), CX
	CMPQ    CX, $0            // if CX == 0 { return }
	JE      caxy_end
	PXOR    X0, X0            // Clear work registers and cache-align loop
	PXOR    X1, X1
	MOVSD   alpha+0(FP), X0   // X0 = { 0, 0, imag(a), real(a) }
	SHUFPD  $0, X0, X0        // X0  = { imag(a), real(a), imag(a), real(a) }
	MOVAPS  X0, X1
	SHUFPS  $0x11, X1, X1     // X1 = { real(a), imag(a), real(a), imag(a) }
	XORQ    AX, AX            // i = 0
	MOVQ    DI, BX            // Align on 16-byte boundary for ADDPS
	ANDQ    $15, BX           // BX = &y & 15
	JZ      caxy_no_trim      // if BX == 0 { goto caxy_no_trim }

	// Trim first value in unaligned buffer
	XORPS X2, X2         // Clear work registers and cache-align loop
	XORPS X3, X3
	XORPS X4, X4
	MOVSD (SI)(AX*8), X3 // X3 = { imag(x[i]), real(x[i]) }
	MOVSHDUP_X3_X2       // X2 = { imag(x[i]), imag(x[i]) }
	MOVSLDUP_X3_X3       // X3 = { real(x[i]), real(x[i]) }
	MULPS X1, X2         // X2 = { real(a) * imag(x[i]), imag(a) * imag(x[i]) }
	MULPS X0, X3         // X3 = { imag(a) * real(x[i]), real(a) * real(x[i]) }

	// X3 = { imag(a)*real(x[i]) + real(a)*imag(x[i]), real(a)*real(x[i]) - imag(a)*imag(x[i]) }
	ADDSUBPS_X2_X3
	MOVSD (DI)(AX*8), X4 // X3 += y[i]
	ADDPS X4, X3
	MOVSD X3, (DI)(AX*8) // y[i]  = X3
	INCQ  AX             // i++
	DECQ  CX             // --CX
	JZ    caxy_end       // if CX == 0 { return }

caxy_no_trim:
	MOVAPS X0, X10   // Copy X0 and X1 for pipelineing
	MOVAPS X1, X11
	MOVQ   CX, BX
	ANDQ   $7, CX    // CX = n % 8
	SHRQ   $3, BX    // BX = floor( n / 8 )
	JZ     caxy_tail // if BX == 0 { goto caxy_tail }

caxy_loop: // do {
	// X_i = { imag(x[i]), real(x[i]), imag(x[i+1]), real(x[i+1]) }
	MOVUPS (SI)(AX*8), X3
	MOVUPS 16(SI)(AX*8), X5
	MOVUPS 32(SI)(AX*8), X7
	MOVUPS 48(SI)(AX*8), X9

	// X_(i-1) = { imag(x[i]), imag(x[i]), imag(x[i]+1), imag(x[i]+1) }
	MOVSHDUP_X3_X2
	MOVSHDUP_X5_X4
	MOVSHDUP_X7_X6
	MOVSHDUP_X9_X8

	// X_i = { real(x[i]), real(x[i]), real(x[i+1]), real(x[i+1]) }
	MOVSLDUP_X3_X3
	MOVSLDUP_X5_X5
	MOVSLDUP_X7_X7
	MOVSLDUP_X9_X9

	// X_i     = {  imag(a) * real(x[i]),   real(a) * real(x[i]),
	// 		imag(a) * real(x[i+1]), real(a) * real(x[i+1])  }
	// X_(i-1) = {  real(a) * imag(x[i]),   imag(a) * imag(x[i]),
	//		real(a) * imag(x[i+1]), imag(a) * imag(x[i+1])  }
	MULPS X1, X2
	MULPS X0, X3
	MULPS X11, X4
	MULPS X10, X5
	MULPS X1, X6
	MULPS X0, X7
	MULPS X11, X8
	MULPS X10, X9

	// X_i = {
	//	imag(result[i]):   imag(a)*real(x[i]) + real(a)*imag(x[i]),
	//	real(result[i]):   real(a)*real(x[i]) - imag(a)*imag(x[i]),
	//	imag(result[i+1]): imag(a)*real(x[i+1]) + real(a)*imag(x[i+1]),
	//	real(result[i+1]): real(a)*real(x[i+1]) - imag(a)*imag(x[i+1]),
	//  }
	ADDSUBPS_X2_X3
	ADDSUBPS_X4_X5
	ADDSUBPS_X6_X7
	ADDSUBPS_X8_X9

	// X_i = { imag(result[i])   + imag(y[i]),   real(result[i])   + real(y[i]),
	//	   imag(result[i+1]) + imag(y[i+1]), real(result[i+1]) + real(y[i+1])  }
	ADDPS  (DI)(AX*8), X3
	ADDPS  16(DI)(AX*8), X5
	ADDPS  32(DI)(AX*8), X7
	ADDPS  48(DI)(AX*8), X9
	MOVUPS X3, (DI)(AX*8)   // y[i:i+1] = X_i
	MOVUPS X5, 16(DI)(AX*8)
	MOVUPS X7, 32(DI)(AX*8)
	MOVUPS X9, 48(DI)(AX*8)
	ADDQ   $8, AX           // i += 8
	DECQ   BX               // --BX
	JNZ    caxy_loop        // }  while BX > 0
	CMPQ   CX, $0           // if CX == 0  { return }
	JE     caxy_end

caxy_tail: // do {
	MOVSD (SI)(AX*8), X3 // X3 = { imag(x[i]), real(x[i]) }
	MOVSHDUP_X3_X2       // X2 = { imag(x[i]), imag(x[i]) }
	MOVSLDUP_X3_X3       // X3 = { real(x[i]), real(x[i]) }
	MULPS X1, X2         // X2 = { real(a) * imag(x[i]), imag(a) * imag(x[i]) }
	MULPS X0, X3         // X3 = { imag(a) * real(x[i]), real(a) * real(x[i]) }

	// X3 = { imag(a)*real(x[i]) + real(a)*imag(x[i]),
	//	  real(a)*real(x[i]) - imag(a)*imag(x[i])   }
	ADDSUBPS_X2_X3
	MOVSD (DI)(AX*8), X4 // X3 += y[i]
	ADDPS X4, X3
	MOVSD X3, (DI)(AX*8) // y[i]  = X3
	INCQ  AX             // ++i
	LOOP  caxy_tail      // } while --CX > 0

caxy_end:
	RET
