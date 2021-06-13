// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

// func Div(dst, s []float64)
TEXT ·Div(SB), NOSPLIT, $0
	MOVQ    dst_base+0(FP), DI // DI = &dst
	MOVQ    dst_len+8(FP), CX  // CX = len(dst)
	MOVQ    s_base+24(FP), SI  // SI = &s
	CMPQ    s_len+32(FP), CX   // CX = max( CX, len(s) )
	CMOVQLE s_len+32(FP), CX
	CMPQ    CX, $0             // if CX == 0 { return }
	JE      div_end
	XORQ    AX, AX             // i = 0
	MOVQ    SI, BX
	ANDQ    $15, BX            // BX = &s & 15
	JZ      div_no_trim        // if BX == 0 { goto div_no_trim }

	// Align on 16-bit boundary
	MOVSD (DI)(AX*8), X0 // X0 = dst[i]
	DIVSD (SI)(AX*8), X0 // X0 /= s[i]
	MOVSD X0, (DI)(AX*8) // dst[i] = X0
	INCQ  AX             // ++i
	DECQ  CX             // --CX
	JZ    div_end        // if CX == 0 { return }

div_no_trim:
	MOVQ CX, BX
	ANDQ $7, BX         // BX = len(dst) % 8
	SHRQ $3, CX         // CX = floor( len(dst) / 8 )
	JZ   div_tail_start // if CX == 0 { goto div_tail_start }

div_loop: // Loop unrolled 8x   do {
	MOVUPS (DI)(AX*8), X0   // X0 = dst[i:i+1]
	MOVUPS 16(DI)(AX*8), X1
	MOVUPS 32(DI)(AX*8), X2
	MOVUPS 48(DI)(AX*8), X3
	DIVPD  (SI)(AX*8), X0   // X0 /= s[i:i+1]
	DIVPD  16(SI)(AX*8), X1
	DIVPD  32(SI)(AX*8), X2
	DIVPD  48(SI)(AX*8), X3
	MOVUPS X0, (DI)(AX*8)   // dst[i] = X0
	MOVUPS X1, 16(DI)(AX*8)
	MOVUPS X2, 32(DI)(AX*8)
	MOVUPS X3, 48(DI)(AX*8)
	ADDQ   $8, AX           // i += 8
	LOOP   div_loop         // } while --CX > 0
	CMPQ   BX, $0           // if BX == 0 { return }
	JE     div_end

div_tail_start: // Reset loop registers
	MOVQ BX, CX // Loop counter: CX = BX

div_tail: // do {
	MOVSD (DI)(AX*8), X0 // X0 = dst[i]
	DIVSD (SI)(AX*8), X0 // X0 /= s[i]
	MOVSD X0, (DI)(AX*8) // dst[i] = X0
	INCQ  AX             // ++i
	LOOP  div_tail       // } while --CX > 0

div_end:
	RET

