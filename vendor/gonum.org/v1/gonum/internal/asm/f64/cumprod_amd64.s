// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

TEXT ·CumProd(SB), NOSPLIT, $0
	MOVQ    dst_base+0(FP), DI // DI = &dst
	MOVQ    dst_len+8(FP), CX  // CX = len(dst)
	MOVQ    s_base+24(FP), SI  // SI = &s
	CMPQ    s_len+32(FP), CX   // CX = max( CX, len(s) )
	CMOVQLE s_len+32(FP), CX
	MOVQ    CX, ret_len+56(FP) // len(ret) = CX
	CMPQ    CX, $0             // if CX == 0 { return }
	JE      cp_end
	XORQ    AX, AX             // i = 0

	MOVSD  (SI), X5   // p_prod = { s[0], s[0] }
	SHUFPD $0, X5, X5
	MOVSD  X5, (DI)   // dst[0] = s[0]
	INCQ   AX         // ++i
	DECQ   CX         // -- CX
	JZ     cp_end     // if CX == 0 { return }

	MOVQ CX, BX
	ANDQ $3, BX        // BX = CX % 4
	SHRQ $2, CX        // CX = floor( CX / 4 )
	JZ   cp_tail_start // if CX == 0 { goto cp_tail_start }

cp_loop: // Loop unrolled 4x   do {
	MOVUPS (SI)(AX*8), X0   // X0 = s[i:i+1]
	MOVUPS 16(SI)(AX*8), X2
	MOVAPS X0, X1           // X1 = X0
	MOVAPS X2, X3
	SHUFPD $1, X1, X1       // { X1[0], X1[1] } = { X1[1], X1[0] }
	SHUFPD $1, X3, X3
	MULPD  X0, X1           // X1 *= X0
	MULPD  X2, X3
	SHUFPD $2, X1, X0       // { X0[0], X0[1] } = { X0[0], X1[1] }
	SHUFPD $3, X1, X1       // { X1[0], X1[1] } = { X1[1], X1[1] }
	SHUFPD $2, X3, X2
	SHUFPD $3, X3, X3
	MULPD  X5, X0           // X0 *= p_prod
	MULPD  X1, X5           // p_prod *= X1
	MULPD  X5, X2
	MOVUPS X0, (DI)(AX*8)   // dst[i] = X0
	MOVUPS X2, 16(DI)(AX*8)
	MULPD  X3, X5
	ADDQ   $4, AX           // i += 4
	LOOP   cp_loop          // } while --CX > 0

	// if BX == 0 { return }
	CMPQ BX, $0
	JE   cp_end

cp_tail_start: // Reset loop registers
	MOVQ BX, CX // Loop counter: CX = BX

cp_tail: // do {
	MULSD (SI)(AX*8), X5 // p_prod *= s[i]
	MOVSD X5, (DI)(AX*8) // dst[i] = p_prod
	INCQ  AX             // ++i
	LOOP  cp_tail        // } while --CX > 0

cp_end:
	MOVQ DI, ret_base+48(FP) // &ret = &dst
	MOVQ dst_cap+16(FP), SI  // cap(ret) = cap(dst)
	MOVQ SI, ret_cap+64(FP)
	RET
