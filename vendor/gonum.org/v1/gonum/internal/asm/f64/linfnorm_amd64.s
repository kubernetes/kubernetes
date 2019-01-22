// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!appengine,!safe

#include "textflag.h"

// func LinfDist(s, t []float64) float64
TEXT ·LinfDist(SB), NOSPLIT, $0
	MOVQ    s_base+0(FP), DI  // DI = &s
	MOVQ    t_base+24(FP), SI // SI = &t
	MOVQ    s_len+8(FP), CX   // CX = len(s)
	CMPQ    t_len+32(FP), CX  // CX = max( CX, len(t) )
	CMOVQLE t_len+32(FP), CX
	PXOR    X3, X3            // norm = 0
	CMPQ    CX, $0            // if CX == 0 { return 0 }
	JE      l1_end
	XORQ    AX, AX            // i = 0
	MOVQ    CX, BX
	ANDQ    $1, BX            // BX = CX % 2
	SHRQ    $1, CX            // CX = floor( CX / 2 )
	JZ      l1_tail_start     // if CX == 0 { return 0 }

l1_loop: // Loop unrolled 2x  do {
	MOVUPS (SI)(AX*8), X0 // X0 = t[i:i+1]
	MOVUPS (DI)(AX*8), X1 // X1 = s[i:i+1]
	MOVAPS X0, X2
	SUBPD  X1, X0
	SUBPD  X2, X1
	MAXPD  X1, X0         // X0 = max( X0 - X1, X1 - X0 )
	MAXPD  X0, X3         // norm = max( norm, X0 )
	ADDQ   $2, AX         // i += 2
	LOOP   l1_loop        // } while --CX > 0
	CMPQ   BX, $0         // if BX == 0 { return }
	JE     l1_end

l1_tail_start: // Reset loop registers
	MOVQ BX, CX // Loop counter: CX = BX
	PXOR X0, X0 // reset X0, X1 to break dependencies
	PXOR X1, X1

l1_tail:
	MOVSD  (SI)(AX*8), X0 // X0 = t[i]
	MOVSD  (DI)(AX*8), X1 // X1 = s[i]
	MOVAPD X0, X2
	SUBSD  X1, X0
	SUBSD  X2, X1
	MAXSD  X1, X0         // X0 = max( X0 - X1, X1 - X0 )
	MAXSD  X0, X3         // norm = max( norm, X0 )

l1_end:
	MOVAPS X3, X2
	SHUFPD $1, X2, X2
	MAXSD  X3, X2         // X2 = max( X3[1], X3[0] )
	MOVSD  X2, ret+48(FP) // return X2
	RET
