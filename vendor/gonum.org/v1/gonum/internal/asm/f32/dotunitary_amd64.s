// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !noasm,!appengine,!safe

#include "textflag.h"

#define HADDPS_SUM_SUM    LONG $0xC07C0FF2 // @ HADDPS X0, X0

#define X_PTR SI
#define Y_PTR DI
#define LEN CX
#define TAIL BX
#define IDX AX
#define SUM X0
#define P_SUM X1

// func DotUnitary(x, y []float32) (sum float32)
TEXT ·DotUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+0(FP), X_PTR  // X_PTR = &x
	MOVQ    y_base+24(FP), Y_PTR // Y_PTR = &y
	PXOR    SUM, SUM             // SUM = 0
	MOVQ    x_len+8(FP), LEN     // LEN = min( len(x), len(y) )
	CMPQ    y_len+32(FP), LEN
	CMOVQLE y_len+32(FP), LEN
	CMPQ    LEN, $0
	JE      dot_end

	XORQ IDX, IDX
	MOVQ Y_PTR, DX
	ANDQ $0xF, DX    // Align on 16-byte boundary for MULPS
	JZ   dot_no_trim // if DX == 0 { goto dot_no_trim }
	SUBQ $16, DX

dot_align: // Trim first value(s) in unaligned buffer  do {
	MOVSS (X_PTR)(IDX*4), X2 // X2 = x[i]
	MULSS (Y_PTR)(IDX*4), X2 // X2 *= y[i]
	ADDSS X2, SUM            // SUM += X2
	INCQ  IDX                // IDX++
	DECQ  LEN
	JZ    dot_end            // if --TAIL == 0 { return }
	ADDQ  $4, DX
	JNZ   dot_align          // } while --DX > 0

dot_no_trim:
	PXOR P_SUM, P_SUM    // P_SUM = 0  for pipelining
	MOVQ LEN, TAIL
	ANDQ $0xF, TAIL      // TAIL = LEN % 16
	SHRQ $4, LEN         // LEN = floor( LEN / 16 )
	JZ   dot_tail4_start // if LEN == 0 { goto dot_tail4_start }

dot_loop: // Loop unrolled 16x  do {
	MOVUPS (X_PTR)(IDX*4), X2   // X_i = x[i:i+1]
	MOVUPS 16(X_PTR)(IDX*4), X3
	MOVUPS 32(X_PTR)(IDX*4), X4
	MOVUPS 48(X_PTR)(IDX*4), X5

	MULPS (Y_PTR)(IDX*4), X2   // X_i *= y[i:i+1]
	MULPS 16(Y_PTR)(IDX*4), X3
	MULPS 32(Y_PTR)(IDX*4), X4
	MULPS 48(Y_PTR)(IDX*4), X5

	ADDPS X2, SUM   // SUM += X_i
	ADDPS X3, P_SUM
	ADDPS X4, SUM
	ADDPS X5, P_SUM

	ADDQ $16, IDX // IDX += 16
	DECQ LEN
	JNZ  dot_loop // } while --LEN > 0

	ADDPS P_SUM, SUM // SUM += P_SUM
	CMPQ  TAIL, $0   // if TAIL == 0 { return }
	JE    dot_end

dot_tail4_start: // Reset loop counter for 4-wide tail loop
	MOVQ TAIL, LEN      // LEN = floor( TAIL / 4 )
	SHRQ $2, LEN
	JZ   dot_tail_start // if LEN == 0 { goto dot_tail_start }

dot_tail4_loop: // Loop unrolled 4x  do {
	MOVUPS (X_PTR)(IDX*4), X2 // X_i = x[i:i+1]
	MULPS  (Y_PTR)(IDX*4), X2 // X_i *= y[i:i+1]
	ADDPS  X2, SUM            // SUM += X_i
	ADDQ   $4, IDX            // i += 4
	DECQ   LEN
	JNZ    dot_tail4_loop     // } while --LEN > 0

dot_tail_start: // Reset loop counter for 1-wide tail loop
	ANDQ $3, TAIL // TAIL = TAIL % 4
	JZ   dot_end  // if TAIL == 0 { return }

dot_tail: // do {
	MOVSS (X_PTR)(IDX*4), X2 // X2 = x[i]
	MULSS (Y_PTR)(IDX*4), X2 // X2 *= y[i]
	ADDSS X2, SUM            // psum += X2
	INCQ  IDX                // IDX++
	DECQ  TAIL
	JNZ   dot_tail           // } while --TAIL > 0

dot_end:
	HADDPS_SUM_SUM        // SUM = \sum{ SUM[i] }
	HADDPS_SUM_SUM
	MOVSS SUM, sum+48(FP) // return SUM
	RET
