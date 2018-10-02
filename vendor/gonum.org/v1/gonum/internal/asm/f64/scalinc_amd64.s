// Copyright ©2016 The Gonum Authors. All rights reserved.
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

//+build !noasm,!appengine,!safe

#include "textflag.h"

#define X_PTR SI
#define LEN CX
#define TAIL BX
#define INC_X R8
#define INCx3_X R9
#define ALPHA X0
#define ALPHA_2 X1

// func ScalInc(alpha float64, x []float64, n, incX uintptr)
TEXT ·ScalInc(SB), NOSPLIT, $0
	MOVSD alpha+0(FP), ALPHA  // ALPHA = alpha
	MOVQ  x_base+8(FP), X_PTR // X_PTR = &x
	MOVQ  incX+40(FP), INC_X  // INC_X = incX
	SHLQ  $3, INC_X           // INC_X *= sizeof(float64)
	MOVQ  n+32(FP), LEN       // LEN = n
	CMPQ  LEN, $0
	JE    end                 // if LEN == 0 { return }

	MOVQ LEN, TAIL
	ANDQ $3, TAIL   // TAIL = LEN % 4
	SHRQ $2, LEN    // LEN = floor( LEN / 4 )
	JZ   tail_start // if LEN == 0 { goto tail_start }

	MOVUPS ALPHA, ALPHA_2            // ALPHA_2 = ALPHA for pipelining
	LEAQ   (INC_X)(INC_X*2), INCx3_X // INCx3_X = INC_X * 3

loop:  // do { // x[i] *= alpha unrolled 4x.
	MOVSD (X_PTR), X2            // X_i = x[i]
	MOVSD (X_PTR)(INC_X*1), X3
	MOVSD (X_PTR)(INC_X*2), X4
	MOVSD (X_PTR)(INCx3_X*1), X5

	MULSD ALPHA, X2   // X_i *= a
	MULSD ALPHA_2, X3
	MULSD ALPHA, X4
	MULSD ALPHA_2, X5

	MOVSD X2, (X_PTR)            // x[i] = X_i
	MOVSD X3, (X_PTR)(INC_X*1)
	MOVSD X4, (X_PTR)(INC_X*2)
	MOVSD X5, (X_PTR)(INCx3_X*1)

	LEAQ (X_PTR)(INC_X*4), X_PTR // X_PTR = &(X_PTR[incX*4])
	DECQ LEN
	JNZ  loop                    // } while --LEN > 0
	CMPQ TAIL, $0
	JE   end                     // if TAIL == 0 { return }

tail_start: // Reset loop registers
	MOVQ TAIL, LEN // Loop counter: LEN = TAIL
	SHRQ $1, LEN   // LEN = floor( LEN / 2 )
	JZ   tail_one

tail_two: // do {
	MOVSD (X_PTR), X2          // X_i = x[i]
	MOVSD (X_PTR)(INC_X*1), X3
	MULSD ALPHA, X2            // X_i *= a
	MULSD ALPHA, X3
	MOVSD X2, (X_PTR)          // x[i] = X_i
	MOVSD X3, (X_PTR)(INC_X*1)

	LEAQ (X_PTR)(INC_X*2), X_PTR // X_PTR = &(X_PTR[incX*2])

	ANDQ $1, TAIL
	JZ   end

tail_one:
	MOVSD (X_PTR), X2 // X_i = x[i]
	MULSD ALPHA, X2   // X_i *= ALPHA
	MOVSD X2, (X_PTR) // x[i] = X_i

end:
	RET
