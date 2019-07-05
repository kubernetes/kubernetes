// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!appengine,!safe

#include "textflag.h"

// func L1Norm(x []float64) float64
TEXT ·L1Norm(SB), NOSPLIT, $0
	MOVQ x_base+0(FP), SI // SI = &x
	MOVQ x_len+8(FP), CX  // CX = len(x)
	XORQ AX, AX           // i = 0
	PXOR X0, X0           // p_sum_i = 0
	PXOR X1, X1
	PXOR X2, X2
	PXOR X3, X3
	PXOR X4, X4
	PXOR X5, X5
	PXOR X6, X6
	PXOR X7, X7
	CMPQ CX, $0           // if CX == 0 { return 0 }
	JE   absum_end
	MOVQ CX, BX
	ANDQ $7, BX           // BX = len(x) % 8
	SHRQ $3, CX           // CX = floor( len(x) / 8 )
	JZ   absum_tail_start // if CX == 0 { goto absum_tail_start }

absum_loop: // do {
	// p_sum += max( p_sum + x[i], p_sum - x[i] )
	MOVUPS (SI)(AX*8), X8    // X_i = x[i:i+1]
	MOVUPS 16(SI)(AX*8), X9
	MOVUPS 32(SI)(AX*8), X10
	MOVUPS 48(SI)(AX*8), X11
	ADDPD  X8, X0            // p_sum_i += X_i  ( positive values )
	ADDPD  X9, X2
	ADDPD  X10, X4
	ADDPD  X11, X6
	SUBPD  X8, X1            // p_sum_(i+1) -= X_i  ( negative values )
	SUBPD  X9, X3
	SUBPD  X10, X5
	SUBPD  X11, X7
	MAXPD  X1, X0            // p_sum_i = max( p_sum_i, p_sum_(i+1) )
	MAXPD  X3, X2
	MAXPD  X5, X4
	MAXPD  X7, X6
	MOVAPS X0, X1            // p_sum_(i+1) = p_sum_i
	MOVAPS X2, X3
	MOVAPS X4, X5
	MOVAPS X6, X7
	ADDQ   $8, AX            // i += 8
	LOOP   absum_loop        // } while --CX > 0

	// p_sum_0 = \sum_{i=1}^{3}( p_sum_(i*2) )
	ADDPD X3, X0
	ADDPD X5, X7
	ADDPD X7, X0

	// p_sum_0[0] = p_sum_0[0] + p_sum_0[1]
	MOVAPS X0, X1
	SHUFPD $0x3, X0, X0 // lower( p_sum_0 ) = upper( p_sum_0 )
	ADDSD  X1, X0
	CMPQ   BX, $0
	JE     absum_end    // if BX == 0 { goto absum_end }

absum_tail_start: // Reset loop registers
	MOVQ  BX, CX // Loop counter:  CX = BX
	XORPS X8, X8 // X_8 = 0

absum_tail: // do {
	// p_sum += max( p_sum + x[i], p_sum - x[i] )
	MOVSD (SI)(AX*8), X8 // X_8 = x[i]
	MOVSD X0, X1         // p_sum_1 = p_sum_0
	ADDSD X8, X0         // p_sum_0 += X_8
	SUBSD X8, X1         // p_sum_1 -= X_8
	MAXSD X1, X0         // p_sum_0 = max( p_sum_0, p_sum_1 )
	INCQ  AX             // i++
	LOOP  absum_tail     // } while --CX > 0

absum_end: // return p_sum_0
	MOVSD X0, sum+24(FP)
	RET
