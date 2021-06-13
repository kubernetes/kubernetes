// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

// func Addconst(alpha float64, x []float64)
TEXT ·AddConst(SB), NOSPLIT, $0
	MOVQ   x_base+8(FP), SI // SI = &x
	MOVQ   x_len+16(FP), CX // CX = len(x)
	CMPQ   CX, $0           // if len(x) == 0 { return }
	JE     ac_end
	MOVSD  alpha+0(FP), X4  // X4 = { a, a }
	SHUFPD $0, X4, X4
	MOVUPS X4, X5           // X5 = X4
	XORQ   AX, AX           // i = 0
	MOVQ   CX, BX
	ANDQ   $7, BX           // BX = len(x) % 8
	SHRQ   $3, CX           // CX = floor( len(x) / 8 )
	JZ     ac_tail_start    // if CX == 0 { goto ac_tail_start }

ac_loop: // Loop unrolled 8x   do {
	MOVUPS (SI)(AX*8), X0   // X_i = s[i:i+1]
	MOVUPS 16(SI)(AX*8), X1
	MOVUPS 32(SI)(AX*8), X2
	MOVUPS 48(SI)(AX*8), X3
	ADDPD  X4, X0           // X_i += a
	ADDPD  X5, X1
	ADDPD  X4, X2
	ADDPD  X5, X3
	MOVUPS X0, (SI)(AX*8)   // s[i:i+1] = X_i
	MOVUPS X1, 16(SI)(AX*8)
	MOVUPS X2, 32(SI)(AX*8)
	MOVUPS X3, 48(SI)(AX*8)
	ADDQ   $8, AX           // i += 8
	LOOP   ac_loop          // } while --CX > 0
	CMPQ   BX, $0           // if BX == 0 { return }
	JE     ac_end

ac_tail_start: // Reset loop counters
	MOVQ BX, CX // Loop counter: CX = BX

ac_tail: // do {
	MOVSD (SI)(AX*8), X0 // X0 = s[i]
	ADDSD X4, X0         // X0 += a
	MOVSD X0, (SI)(AX*8) // s[i] = X0
	INCQ  AX             // ++i
	LOOP  ac_tail        // } while --CX > 0

ac_end:
	RET
