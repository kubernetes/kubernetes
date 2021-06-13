// Copyright ©2021 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

#define X_PTR SI
#define IDX AX
#define LEN CX
#define TAIL BX
#define SUM X0
#define SUM_1 X1
#define SUM_2 X2
#define SUM_3 X3

// func Sum(x []float32) float32
TEXT ·Sum(SB), NOSPLIT, $0
	MOVQ x_base+0(FP), X_PTR // X_PTR = &x
	MOVQ x_len+8(FP), LEN    // LEN = len(x)
	XORQ IDX, IDX            // i = 0
	PXOR SUM, SUM            // p_sum_i = 0
	CMPQ LEN, $0             // if LEN == 0 { return 0 }
	JE   sum_end

	PXOR SUM_1, SUM_1
	PXOR SUM_2, SUM_2
	PXOR SUM_3, SUM_3

	MOVQ X_PTR, TAIL // Check memory alignment
	ANDQ $15, TAIL   // TAIL = &x % 16
	JZ   no_trim     // if TAIL == 0 { goto no_trim }
	SUBQ $16, TAIL   // TAIL -= 16

sum_align: // Align on 16-byte boundary do {
	ADDSS (X_PTR)(IDX*4), SUM // SUM += x[0]
	INCQ  IDX                 // i++
	DECQ  LEN                 // LEN--
	JZ    sum_end             // if LEN == 0 { return }
	ADDQ  $4, TAIL            // TAIL += 4
	JNZ   sum_align           // } while TAIL < 0

no_trim:
	MOVQ LEN, TAIL
	SHRQ $4, LEN   // LEN = floor( n / 16 )
	JZ   sum_tail8 // if LEN == 0 { goto sum_tail8 }


sum_loop: // sum 16x wide do {
	ADDPS (X_PTR)(IDX*4), SUM     // sum_i += x[i:i+4]
	ADDPS 16(X_PTR)(IDX*4), SUM_1
	ADDPS 32(X_PTR)(IDX*4), SUM_2
	ADDPS 48(X_PTR)(IDX*4), SUM_3

	ADDQ  $16, IDX                // i += 16
	DECQ  LEN
	JNZ   sum_loop                // } while --LEN > 0

sum_tail8:
	ADDPS SUM_3, SUM
	ADDPS SUM_2, SUM_1

	TESTQ $8, TAIL
	JZ    sum_tail4

	ADDPS (X_PTR)(IDX*4), SUM // sum_i += x[i:i+4]
	ADDPS 16(X_PTR)(IDX*4), SUM_1
	ADDQ  $8, IDX

sum_tail4:
	ADDPS SUM_1, SUM

	TESTQ $4, TAIL
	JZ    sum_tail2

	ADDPS (X_PTR)(IDX*4), SUM // sum_i += x[i:i+4]
	ADDQ  $4, IDX

sum_tail2:
	HADDPS SUM, SUM            // sum_i[:2] += sum_i[2:4]

	TESTQ $2, TAIL
	JZ    sum_tail1

	MOVSD (X_PTR)(IDX*4), SUM_1 // reuse SUM_1
	ADDPS SUM_1, SUM            // sum_i += x[i:i+2]
	ADDQ  $2, IDX

sum_tail1:
	HADDPS SUM, SUM // sum_i[0] += sum_i[1]

	TESTQ $1, TAIL
	JZ    sum_end

	ADDSS (X_PTR)(IDX*4), SUM

sum_end: // return sum
	MOVSS SUM, ret+24(FP)
	RET
