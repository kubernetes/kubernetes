// Copyright ©2018 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!appengine,!safe

#include "textflag.h"

#define X_PTR SI
#define IDX AX
#define LEN CX
#define TAIL BX
#define SUM X0
#define SUM_1 X1
#define SUM_2 X2
#define SUM_3 X3

// func Sum(x []float64) float64
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
	ANDQ $15, TAIL   // TAIL = &y % 16
	JZ   no_trim     // if TAIL == 0 { goto no_trim }

	// Align on 16-byte boundary
	ADDSD (X_PTR), X0 // X0 += x[0]
	INCQ  IDX         // i++
	DECQ  LEN         // LEN--
	DECQ  TAIL        // TAIL--
	JZ    sum_end     // if TAIL == 0 { return }

no_trim:
	MOVQ LEN, TAIL
	SHRQ $4, LEN   // LEN = floor( n / 16 )
	JZ   sum_tail8 // if LEN == 0 { goto sum_tail8 }

sum_loop: // sum 16x wide do {
	ADDPD (SI)(AX*8), SUM      // sum_i += x[i:i+2]
	ADDPD 16(SI)(AX*8), SUM_1
	ADDPD 32(SI)(AX*8), SUM_2
	ADDPD 48(SI)(AX*8), SUM_3
	ADDPD 64(SI)(AX*8), SUM
	ADDPD 80(SI)(AX*8), SUM_1
	ADDPD 96(SI)(AX*8), SUM_2
	ADDPD 112(SI)(AX*8), SUM_3
	ADDQ  $16, IDX             // i += 16
	DECQ  LEN
	JNZ   sum_loop             // } while --CX > 0

sum_tail8:
	TESTQ $8, TAIL
	JZ    sum_tail4

	ADDPD (SI)(AX*8), SUM     // sum_i += x[i:i+2]
	ADDPD 16(SI)(AX*8), SUM_1
	ADDPD 32(SI)(AX*8), SUM_2
	ADDPD 48(SI)(AX*8), SUM_3
	ADDQ  $8, IDX

sum_tail4:
	ADDPD SUM_3, SUM
	ADDPD SUM_2, SUM_1

	TESTQ $4, TAIL
	JZ    sum_tail2

	ADDPD (SI)(AX*8), SUM     // sum_i += x[i:i+2]
	ADDPD 16(SI)(AX*8), SUM_1
	ADDQ  $4, IDX

sum_tail2:
	ADDPD SUM_1, SUM

	TESTQ $2, TAIL
	JZ    sum_tail1

	ADDPD (SI)(AX*8), SUM // sum_i += x[i:i+2]
	ADDQ  $2, IDX

sum_tail1:
	HADDPD SUM, SUM // sum_i[0] += sum_i[1]

	TESTQ $1, TAIL
	JZ    sum_end

	ADDSD (SI)(IDX*8), SUM

sum_end: // return sum
	MOVSD SUM, sum+24(FP)
	RET
