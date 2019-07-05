// Copyright ©2017 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//+build !noasm,!appengine,!safe

#include "textflag.h"

#define SRC SI
#define DST SI
#define LEN CX
#define TAIL BX
#define INC R9
#define INC3 R10
#define ALPHA X0
#define ALPHA_2 X1

#define MOVDDUP_ALPHA    LONG $0x44120FF2; WORD $0x0824 // MOVDDUP 8(SP), X0

// func DscalInc(alpha float64, x []complex128, n, inc uintptr)
TEXT ·DscalInc(SB), NOSPLIT, $0
	MOVQ x_base+8(FP), SRC // SRC = &x
	MOVQ n+32(FP), LEN     // LEN = n
	CMPQ LEN, $0           // if LEN == 0 { return }
	JE   dscal_end

	MOVDDUP_ALPHA             // ALPHA = alpha
	MOVQ   inc+40(FP), INC    // INC = inc
	SHLQ   $4, INC            // INC = INC * sizeof(complex128)
	LEAQ   (INC)(INC*2), INC3 // INC3 = 3 * INC
	MOVUPS ALPHA, ALPHA_2     // Copy ALPHA and ALPHA_2 for pipelining
	MOVQ   LEN, TAIL          // TAIL = LEN
	SHRQ   $2, LEN            // LEN = floor( n / 4 )
	JZ     dscal_tail         // if LEN == 0 { goto dscal_tail }

dscal_loop: // do {
	MOVUPS (SRC), X2         // X_i = x[i]
	MOVUPS (SRC)(INC*1), X3
	MOVUPS (SRC)(INC*2), X4
	MOVUPS (SRC)(INC3*1), X5

	MULPD ALPHA, X2   // X_i *= ALPHA
	MULPD ALPHA_2, X3
	MULPD ALPHA, X4
	MULPD ALPHA_2, X5

	MOVUPS X2, (DST)         // x[i] = X_i
	MOVUPS X3, (DST)(INC*1)
	MOVUPS X4, (DST)(INC*2)
	MOVUPS X5, (DST)(INC3*1)

	LEAQ (SRC)(INC*4), SRC // SRC += INC*4
	DECQ LEN
	JNZ  dscal_loop        // } while --LEN > 0

dscal_tail:
	ANDQ $3, TAIL  // TAIL = TAIL % 4
	JE   dscal_end // if TAIL == 0 { return }

dscal_tail_loop: // do {
	MOVUPS (SRC), X2       // X_i = x[i]
	MULPD  ALPHA, X2       // X_i *= ALPHA
	MOVUPS X2, (DST)       // x[i] = X_i
	ADDQ   INC, SRC        // SRC += INC
	DECQ   TAIL
	JNZ    dscal_tail_loop // } while --TAIL > 0

dscal_end:
	RET
