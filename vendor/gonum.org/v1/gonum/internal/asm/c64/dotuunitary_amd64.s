// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!appengine,!safe

#include "textflag.h"

#define MOVSLDUP_XPTR_IDX_8__X3    LONG $0x1C120FF3; BYTE $0xC6 // MOVSLDUP (SI)(AX*8), X3
#define MOVSLDUP_16_XPTR_IDX_8__X5    LONG $0x6C120FF3; WORD $0x10C6 // MOVSLDUP 16(SI)(AX*8), X5
#define MOVSLDUP_32_XPTR_IDX_8__X7    LONG $0x7C120FF3; WORD $0x20C6 // MOVSLDUP 32(SI)(AX*8), X7
#define MOVSLDUP_48_XPTR_IDX_8__X9    LONG $0x120F44F3; WORD $0xC64C; BYTE $0x30 // MOVSLDUP 48(SI)(AX*8), X9

#define MOVSHDUP_XPTR_IDX_8__X2    LONG $0x14160FF3; BYTE $0xC6 // MOVSHDUP (SI)(AX*8), X2
#define MOVSHDUP_16_XPTR_IDX_8__X4    LONG $0x64160FF3; WORD $0x10C6 // MOVSHDUP 16(SI)(AX*8), X4
#define MOVSHDUP_32_XPTR_IDX_8__X6    LONG $0x74160FF3; WORD $0x20C6 // MOVSHDUP 32(SI)(AX*8), X6
#define MOVSHDUP_48_XPTR_IDX_8__X8    LONG $0x160F44F3; WORD $0xC644; BYTE $0x30 // MOVSHDUP 48(SI)(AX*8), X8

#define MOVSHDUP_X3_X2    LONG $0xD3160FF3 // MOVSHDUP X3, X2
#define MOVSLDUP_X3_X3    LONG $0xDB120FF3 // MOVSLDUP X3, X3

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
#define IDX AX
#define I_IDX DX
#define NEG1 X15
#define P_NEG1 X14

// func DotuUnitary(x, y []complex64) (sum complex64)
TEXT ·DotuUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+0(FP), X_PTR  // X_PTR = &x
	MOVQ    y_base+24(FP), Y_PTR // Y_PTR = &y
	PXOR    SUM, SUM             // SUM = 0
	PXOR    P_SUM, P_SUM         // P_SUM = 0
	MOVQ    x_len+8(FP), LEN     // LEN = min( len(x), len(y) )
	CMPQ    y_len+32(FP), LEN
	CMOVQLE y_len+32(FP), LEN
	CMPQ    LEN, $0              // if LEN == 0 { return }
	JE      dotu_end
	XORQ    IDX, IDX             // IDX = 0

	MOVQ X_PTR, DX
	ANDQ $15, DX      // DX = &x & 15
	JZ   dotu_aligned // if DX == 0 { goto dotu_aligned }

	MOVSD  (X_PTR)(IDX*8), X3  // X_i     = { imag(x[i]), real(x[i]) }
	MOVSHDUP_X3_X2             // X_(i-1) = { imag(x[i]), imag(x[i]) }
	MOVSLDUP_X3_X3             // X_i     = { real(x[i]), real(x[i]) }
	MOVSD  (Y_PTR)(IDX*8), X10 // X_j     = { imag(y[i]), real(y[i]) }
	MULPS  X10, X3             // X_i     = { imag(y[i]) * real(x[i]), real(y[i]) * real(x[i]) }
	SHUFPS $0x1, X10, X10      // X_j     = { real(y[i]), imag(y[i]) }
	MULPS  X10, X2             // X_(i-1) = { real(y[i]) * imag(x[i]), imag(y[i]) * imag(x[i]) }

	// X_i = {
	//	imag(result[i]):  imag(y[i])*real(x[i]) + real(y[i])*imag(x[i]),
	//	real(result[i]):  real(y[i])*real(x[i]) - imag(y[i])*imag(x[i]) }
	ADDSUBPS_X2_X3

	MOVAPS X3, SUM  // SUM = X_i
	INCQ   IDX      // IDX++
	DECQ   LEN      // LEN--
	JZ     dotu_end // if LEN == 0 { goto dotu_end }

dotu_aligned:
	MOVQ LEN, TAIL
	ANDQ $7, TAIL     // TAIL = LEN % 8
	SHRQ $3, LEN      // LEN = floor( LEN / 8 )
	JZ   dotu_tail    // if LEN == 0 { goto dotu_tail }
	PXOR P_SUM, P_SUM

dotu_loop: // do {
	MOVSLDUP_XPTR_IDX_8__X3    // X_i = { real(x[i]), real(x[i]), real(x[i+1]), real(x[i+1]) }
	MOVSLDUP_16_XPTR_IDX_8__X5
	MOVSLDUP_32_XPTR_IDX_8__X7
	MOVSLDUP_48_XPTR_IDX_8__X9

	MOVSHDUP_XPTR_IDX_8__X2    // X_(i-1) = { imag(x[i]), imag(x[i]), imag(x[i]+1), imag(x[i]+1) }
	MOVSHDUP_16_XPTR_IDX_8__X4
	MOVSHDUP_32_XPTR_IDX_8__X6
	MOVSHDUP_48_XPTR_IDX_8__X8

	// X_j = { imag(y[i]), real(y[i]), imag(y[i+1]), real(y[i+1]) }
	MOVUPS (Y_PTR)(IDX*8), X10
	MOVUPS 16(Y_PTR)(IDX*8), X11
	MOVUPS 32(Y_PTR)(IDX*8), X12
	MOVUPS 48(Y_PTR)(IDX*8), X13

	// X_i     = {  imag(y[i])   * real(x[i]),   real(y[i])   * real(x[i]),
	// 		imag(y[i+1]) * real(x[i+1]), real(y[i+1]) * real(x[i+1])  }
	MULPS X10, X3
	MULPS X11, X5
	MULPS X12, X7
	MULPS X13, X9

	// X_j = { real(y[i]), imag(y[i]), real(y[i+1]), imag(y[i+1]) }
	SHUFPS $0xB1, X10, X10
	SHUFPS $0xB1, X11, X11
	SHUFPS $0xB1, X12, X12
	SHUFPS $0xB1, X13, X13

	// X_(i-1) = {  real(y[i])   * imag(x[i]),   imag(y[i])   * imag(x[i]),
	//		real(y[i+1]) * imag(x[i+1]), imag(y[i+1]) * imag(x[i+1])  }
	MULPS X10, X2
	MULPS X11, X4
	MULPS X12, X6
	MULPS X13, X8

	// X_i = {
	//	imag(result[i]):   imag(y[i])   * real(x[i])   + real(y[i])   * imag(x[i]),
	//	real(result[i]):   real(y[i])   * real(x[i])   - imag(y[i])   * imag(x[i]),
	//	imag(result[i+1]): imag(y[i+1]) * real(x[i+1]) + real(y[i+1]) * imag(x[i+1]),
	//	real(result[i+1]): real(y[i+1]) * real(x[i+1]) - imag(y[i+1]) * imag(x[i+1]),
	//  }
	ADDSUBPS_X2_X3
	ADDSUBPS_X4_X5
	ADDSUBPS_X6_X7
	ADDSUBPS_X8_X9

	// SUM += X_i
	ADDPS X3, SUM
	ADDPS X5, P_SUM
	ADDPS X7, SUM
	ADDPS X9, P_SUM

	ADDQ $8, IDX   // IDX += 8
	DECQ LEN
	JNZ  dotu_loop // } while --LEN > 0

	ADDPS SUM, P_SUM // P_SUM = { P_SUM[1] + SUM[1], P_SUM[0] + SUM[0] }
	XORPS SUM, SUM   // SUM = 0

	CMPQ TAIL, $0 // if TAIL == 0 { return }
	JE   dotu_end

dotu_tail:
	MOVQ TAIL, LEN
	SHRQ $1, LEN       // LEN = floor( LEN / 2 )
	JZ   dotu_tail_one // if LEN == 0 { goto dotc_tail_one }

dotu_tail_two: // do {
	MOVSLDUP_XPTR_IDX_8__X3    // X_i = { real(x[i]), real(x[i]), real(x[i+1]), real(x[i+1]) }
	MOVSHDUP_XPTR_IDX_8__X2    // X_(i-1) = { imag(x[i]), imag(x[i]), imag(x[i]+1), imag(x[i]+1) }
	MOVUPS (Y_PTR)(IDX*8), X10 // X_j = { imag(y[i]), real(y[i]) }
	MULPS  X10, X3             // X_i = { imag(y[i]) * real(x[i]), real(y[i]) * real(x[i]) }
	SHUFPS $0xB1, X10, X10     // X_j = { real(y[i]), imag(y[i]) }
	MULPS  X10, X2             // X_(i-1) = { real(y[i]) * imag(x[i]), imag(y[i]) * imag(x[i]) }

	// X_i = {
	//	imag(result[i]):  imag(y[i])*real(x[i]) + real(y[i])*imag(x[i]),
	//	real(result[i]):  real(y[i])*real(x[i]) - imag(y[i])*imag(x[i]) }
	ADDSUBPS_X2_X3

	ADDPS X3, SUM // SUM += X_i

	ADDQ $2, IDX       // IDX += 2
	DECQ LEN
	JNZ  dotu_tail_two // } while --LEN > 0

	ADDPS SUM, P_SUM // P_SUM = { P_SUM[1] + SUM[1], P_SUM[0] + SUM[0] }
	XORPS SUM, SUM   // SUM = 0

	ANDQ $1, TAIL
	JZ   dotu_end

dotu_tail_one:
	MOVSD  (X_PTR)(IDX*8), X3  // X_i = { imag(x[i]), real(x[i]) }
	MOVSHDUP_X3_X2             // X_(i-1) = { imag(x[i]), imag(x[i]) }
	MOVSLDUP_X3_X3             // X_i = { real(x[i]), real(x[i]) }
	MOVSD  (Y_PTR)(IDX*8), X10 // X_j = { imag(y[i]), real(y[i]) }
	MULPS  X10, X3             // X_i = { imag(y[i]) * real(x[i]), real(y[i]) * real(x[i]) }
	SHUFPS $0x1, X10, X10      // X_j = { real(y[i]), imag(y[i]) }
	MULPS  X10, X2             // X_(i-1) = { real(y[i]) * imag(x[i]), imag(y[i]) * imag(x[i]) }

	// X_i = {
	//	imag(result[i]):  imag(y[i])*real(x[i]) + real(y[i])*imag(x[i]),
	//	real(result[i]):  real(y[i])*real(x[i]) - imag(y[i])*imag(x[i]) }
	ADDSUBPS_X2_X3

	ADDPS X3, SUM // SUM += X_i

dotu_end:
	ADDPS   P_SUM, SUM   // SUM = { P_SUM[0] + SUM[0] }
	MOVHLPS P_SUM, P_SUM // P_SUM = { P_SUM[1], P_SUM[1] }
	ADDPS   P_SUM, SUM   // SUM = { P_SUM[1] + SUM[0] }

dotu_ret:
	MOVSD SUM, sum+48(FP) // return SUM
	RET
