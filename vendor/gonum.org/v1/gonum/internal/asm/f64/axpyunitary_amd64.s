// Copyright ©2015 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
//
// Some of the loop unrolling code is copied from:
// http://golang.org/src/math/big/arith_amd64.s
// which is distributed under these terms:
//
// Copyright (c) 2012 The Go Authors. All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
//    * Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//    * Redistributions in binary form must reproduce the above
// copyright notice, this list of conditions and the following disclaimer
// in the documentation and/or other materials provided with the
// distribution.
//    * Neither the name of Google Inc. nor the names of its
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
// "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
// LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
// A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
// OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
// SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
// LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
// DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
// THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
// (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

#define X_PTR SI
#define Y_PTR DI
#define DST_PTR DI
#define IDX AX
#define LEN CX
#define TAIL BX
#define ALPHA X0
#define ALPHA_2 X1

// func AxpyUnitary(alpha float64, x, y []float64)
TEXT ·AxpyUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+8(FP), X_PTR  // X_PTR := &x
	MOVQ    y_base+32(FP), Y_PTR // Y_PTR := &y
	MOVQ    x_len+16(FP), LEN    // LEN = min( len(x), len(y) )
	CMPQ    y_len+40(FP), LEN
	CMOVQLE y_len+40(FP), LEN
	CMPQ    LEN, $0              // if LEN == 0 { return }
	JE      end
	XORQ    IDX, IDX
	MOVSD   alpha+0(FP), ALPHA   // ALPHA := { alpha, alpha }
	SHUFPD  $0, ALPHA, ALPHA
	MOVUPS  ALPHA, ALPHA_2       // ALPHA_2 := ALPHA   for pipelining
	MOVQ    Y_PTR, TAIL          // Check memory alignment
	ANDQ    $15, TAIL            // TAIL = &y % 16
	JZ      no_trim              // if TAIL == 0 { goto no_trim }

	// Align on 16-byte boundary
	MOVSD (X_PTR), X2   // X2 := x[0]
	MULSD ALPHA, X2     // X2 *= a
	ADDSD (Y_PTR), X2   // X2 += y[0]
	MOVSD X2, (DST_PTR) // y[0] = X2
	INCQ  IDX           // i++
	DECQ  LEN           // LEN--
	JZ    end           // if LEN == 0 { return }

no_trim:
	MOVQ LEN, TAIL
	ANDQ $7, TAIL   // TAIL := n % 8
	SHRQ $3, LEN    // LEN = floor( n / 8 )
	JZ   tail_start // if LEN == 0 { goto tail2_start }

loop:  // do {
	// y[i] += alpha * x[i] unrolled 8x.
	MOVUPS (X_PTR)(IDX*8), X2   // X_i = x[i]
	MOVUPS 16(X_PTR)(IDX*8), X3
	MOVUPS 32(X_PTR)(IDX*8), X4
	MOVUPS 48(X_PTR)(IDX*8), X5

	MULPD ALPHA, X2   // X_i *= a
	MULPD ALPHA_2, X3
	MULPD ALPHA, X4
	MULPD ALPHA_2, X5

	ADDPD (Y_PTR)(IDX*8), X2   // X_i += y[i]
	ADDPD 16(Y_PTR)(IDX*8), X3
	ADDPD 32(Y_PTR)(IDX*8), X4
	ADDPD 48(Y_PTR)(IDX*8), X5

	MOVUPS X2, (DST_PTR)(IDX*8)   // y[i] = X_i
	MOVUPS X3, 16(DST_PTR)(IDX*8)
	MOVUPS X4, 32(DST_PTR)(IDX*8)
	MOVUPS X5, 48(DST_PTR)(IDX*8)

	ADDQ $8, IDX  // i += 8
	DECQ LEN
	JNZ  loop     // } while --LEN > 0
	CMPQ TAIL, $0 // if TAIL == 0 { return }
	JE   end

tail_start: // Reset loop registers
	MOVQ TAIL, LEN // Loop counter: LEN = TAIL
	SHRQ $1, LEN   // LEN = floor( TAIL / 2 )
	JZ   tail_one  // if TAIL == 0 { goto tail }

tail_two: // do {
	MOVUPS (X_PTR)(IDX*8), X2   // X2 = x[i]
	MULPD  ALPHA, X2            // X2 *= a
	ADDPD  (Y_PTR)(IDX*8), X2   // X2 += y[i]
	MOVUPS X2, (DST_PTR)(IDX*8) // y[i] = X2
	ADDQ   $2, IDX              // i += 2
	DECQ   LEN
	JNZ    tail_two             // } while --LEN > 0

	ANDQ $1, TAIL
	JZ   end      // if TAIL == 0 { goto end }

tail_one:
	MOVSD (X_PTR)(IDX*8), X2   // X2 = x[i]
	MULSD ALPHA, X2            // X2 *= a
	ADDSD (Y_PTR)(IDX*8), X2   // X2 += y[i]
	MOVSD X2, (DST_PTR)(IDX*8) // y[i] = X2

end:
	RET
