// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

#define HADDPD_SUM_SUM    LONG $0xC07C0F66 // @ HADDPD X0, X0

#define X_PTR SI
#define Y_PTR DI
#define LEN CX
#define TAIL BX
#define IDX AX
#define SUM X0
#define P_SUM X1

// func DdotUnitary(x, y []float32) (sum float32)
TEXT ·DdotUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+0(FP), X_PTR  // X_PTR = &x
	MOVQ    y_base+24(FP), Y_PTR // Y_PTR = &y
	MOVQ    x_len+8(FP), LEN     // LEN = min( len(x), len(y) )
	CMPQ    y_len+32(FP), LEN
	CMOVQLE y_len+32(FP), LEN
	PXOR    SUM, SUM             // psum = 0
	CMPQ    LEN, $0
	JE      dot_end

	XORQ IDX, IDX
	MOVQ Y_PTR, DX
	ANDQ $0xF, DX    // Align on 16-byte boundary for ADDPS
	JZ   dot_no_trim // if DX == 0 { goto dot_no_trim }

	SUBQ $16, DX

dot_align: // Trim first value(s) in unaligned buffer  do {
	CVTSS2SD (X_PTR)(IDX*4), X2 // X2 = float64(x[i])
	CVTSS2SD (Y_PTR)(IDX*4), X3 // X3 = float64(y[i])
	MULSD    X3, X2
	ADDSD    X2, SUM            // SUM += X2
	INCQ     IDX                // IDX++
	DECQ     LEN
	JZ       dot_end            // if --TAIL == 0 { return }
	ADDQ     $4, DX
	JNZ      dot_align          // } while --LEN > 0

dot_no_trim:
	PXOR P_SUM, P_SUM   // P_SUM = 0  for pipelining
	MOVQ LEN, TAIL
	ANDQ $0x7, TAIL     // TAIL = LEN % 8
	SHRQ $3, LEN        // LEN = floor( LEN / 8 )
	JZ   dot_tail_start // if LEN == 0 { goto dot_tail_start }

dot_loop: // Loop unrolled 8x  do {
	CVTPS2PD (X_PTR)(IDX*4), X2   // X_i = x[i:i+1]
	CVTPS2PD 8(X_PTR)(IDX*4), X3
	CVTPS2PD 16(X_PTR)(IDX*4), X4
	CVTPS2PD 24(X_PTR)(IDX*4), X5

	CVTPS2PD (Y_PTR)(IDX*4), X6   // X_j = y[i:i+1]
	CVTPS2PD 8(Y_PTR)(IDX*4), X7
	CVTPS2PD 16(Y_PTR)(IDX*4), X8
	CVTPS2PD 24(Y_PTR)(IDX*4), X9

	MULPD X6, X2 // X_i *= X_j
	MULPD X7, X3
	MULPD X8, X4
	MULPD X9, X5

	ADDPD X2, SUM   // SUM += X_i
	ADDPD X3, P_SUM
	ADDPD X4, SUM
	ADDPD X5, P_SUM

	ADDQ $8, IDX  // IDX += 8
	DECQ LEN
	JNZ  dot_loop // } while --LEN > 0

	ADDPD P_SUM, SUM // SUM += P_SUM
	CMPQ  TAIL, $0   // if TAIL == 0 { return }
	JE    dot_end

dot_tail_start:
	MOVQ TAIL, LEN
	SHRQ $1, LEN
	JZ   dot_tail_one

dot_tail_two:
	CVTPS2PD (X_PTR)(IDX*4), X2 // X_i = x[i:i+1]
	CVTPS2PD (Y_PTR)(IDX*4), X6 // X_j = y[i:i+1]
	MULPD    X6, X2             // X_i *= X_j
	ADDPD    X2, SUM            // SUM += X_i
	ADDQ     $2, IDX            // IDX += 2
	DECQ     LEN
	JNZ      dot_tail_two       // } while --LEN > 0

	ANDQ $1, TAIL
	JZ   dot_end

dot_tail_one:
	CVTSS2SD (X_PTR)(IDX*4), X2 // X2 = float64(x[i])
	CVTSS2SD (Y_PTR)(IDX*4), X3 // X3 = float64(y[i])
	MULSD    X3, X2             // X2 *= X3
	ADDSD    X2, SUM            // SUM += X2

dot_end:
	HADDPD_SUM_SUM        // SUM = \sum{ SUM[i] }
	MOVSD SUM, sum+48(FP) // return SUM
	RET
