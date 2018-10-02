// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !noasm,!appengine,!safe

#include "textflag.h"

// func AxpyUnitary(alpha float32, x, y []float32)
TEXT ·AxpyUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+8(FP), SI  // SI = &x
	MOVQ    y_base+32(FP), DI // DI = &y
	MOVQ    x_len+16(FP), BX  // BX = min( len(x), len(y) )
	CMPQ    y_len+40(FP), BX
	CMOVQLE y_len+40(FP), BX
	CMPQ    BX, $0            // if BX == 0 { return }
	JE      axpy_end
	MOVSS   alpha+0(FP), X0
	SHUFPS  $0, X0, X0        // X0 = { a, a, a, a }
	XORQ    AX, AX            // i = 0
	PXOR    X2, X2            // 2 NOP instructions (PXOR) to align
	PXOR    X3, X3            // loop to cache line
	MOVQ    DI, CX
	ANDQ    $0xF, CX          // Align on 16-byte boundary for ADDPS
	JZ      axpy_no_trim      // if CX == 0 { goto axpy_no_trim }

	XORQ $0xF, CX // CX = 4 - floor( BX % 16 / 4 )
	INCQ CX
	SHRQ $2, CX

axpy_align: // Trim first value(s) in unaligned buffer  do {
	MOVSS (SI)(AX*4), X2 // X2 = x[i]
	MULSS X0, X2         // X2 *= a
	ADDSS (DI)(AX*4), X2 // X2 += y[i]
	MOVSS X2, (DI)(AX*4) // y[i] = X2
	INCQ  AX             // i++
	DECQ  BX
	JZ    axpy_end       // if --BX == 0 { return }
	LOOP  axpy_align     // } while --CX > 0

axpy_no_trim:
	MOVUPS X0, X1           // Copy X0 to X1 for pipelining
	MOVQ   BX, CX
	ANDQ   $0xF, BX         // BX = len % 16
	SHRQ   $4, CX           // CX = int( len / 16 )
	JZ     axpy_tail4_start // if CX == 0 { return }

axpy_loop: // Loop unrolled 16x   do {
	MOVUPS (SI)(AX*4), X2   // X2 = x[i:i+4]
	MOVUPS 16(SI)(AX*4), X3
	MOVUPS 32(SI)(AX*4), X4
	MOVUPS 48(SI)(AX*4), X5
	MULPS  X0, X2           // X2 *= a
	MULPS  X1, X3
	MULPS  X0, X4
	MULPS  X1, X5
	ADDPS  (DI)(AX*4), X2   // X2 += y[i:i+4]
	ADDPS  16(DI)(AX*4), X3
	ADDPS  32(DI)(AX*4), X4
	ADDPS  48(DI)(AX*4), X5
	MOVUPS X2, (DI)(AX*4)   // dst[i:i+4] = X2
	MOVUPS X3, 16(DI)(AX*4)
	MOVUPS X4, 32(DI)(AX*4)
	MOVUPS X5, 48(DI)(AX*4)
	ADDQ   $16, AX          // i += 16
	LOOP   axpy_loop        // while (--CX) > 0
	CMPQ   BX, $0           // if BX == 0 { return }
	JE     axpy_end

axpy_tail4_start: // Reset loop counter for 4-wide tail loop
	MOVQ BX, CX          // CX = floor( BX / 4 )
	SHRQ $2, CX
	JZ   axpy_tail_start // if CX == 0 { goto axpy_tail_start }

axpy_tail4: // Loop unrolled 4x   do {
	MOVUPS (SI)(AX*4), X2 // X2 = x[i]
	MULPS  X0, X2         // X2 *= a
	ADDPS  (DI)(AX*4), X2 // X2 += y[i]
	MOVUPS X2, (DI)(AX*4) // y[i] = X2
	ADDQ   $4, AX         // i += 4
	LOOP   axpy_tail4     // } while --CX > 0

axpy_tail_start: // Reset loop counter for 1-wide tail loop
	MOVQ BX, CX   // CX = BX % 4
	ANDQ $3, CX
	JZ   axpy_end // if CX == 0 { return }

axpy_tail:
	MOVSS (SI)(AX*4), X1 // X1 = x[i]
	MULSS X0, X1         // X1 *= a
	ADDSS (DI)(AX*4), X1 // X1 += y[i]
	MOVSS X1, (DI)(AX*4) // y[i] = X1
	INCQ  AX             // i++
	LOOP  axpy_tail      // } while --CX > 0

axpy_end:
	RET
