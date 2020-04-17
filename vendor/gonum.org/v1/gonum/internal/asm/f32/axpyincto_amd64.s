// Copyright ©2016 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!appengine,!safe

#include "textflag.h"

// func AxpyIncTo(dst []float32, incDst, idst uintptr, alpha float32, x, y []float32, n, incX, incY, ix, iy uintptr)
TEXT ·AxpyIncTo(SB), NOSPLIT, $0
	MOVQ  n+96(FP), CX       // CX = n
	CMPQ  CX, $0             // if n==0 { return }
	JLE   axpyi_end
	MOVQ  dst_base+0(FP), DI // DI = &dst
	MOVQ  x_base+48(FP), SI  // SI = &x
	MOVQ  y_base+72(FP), DX  // DX = &y
	MOVQ  ix+120(FP), R8     // R8 = ix  // Load the first index
	MOVQ  iy+128(FP), R9     // R9 = iy
	MOVQ  idst+32(FP), R10   // R10 = idst
	LEAQ  (SI)(R8*4), SI     // SI = &(x[ix])
	LEAQ  (DX)(R9*4), DX     // DX = &(y[iy])
	LEAQ  (DI)(R10*4), DI    // DI = &(dst[idst])
	MOVQ  incX+104(FP), R8   // R8 = incX
	SHLQ  $2, R8             // R8 *= sizeof(float32)
	MOVQ  incY+112(FP), R9   // R9 = incY
	SHLQ  $2, R9             // R9 *= sizeof(float32)
	MOVQ  incDst+24(FP), R10 // R10 = incDst
	SHLQ  $2, R10            // R10 *= sizeof(float32)
	MOVSS alpha+40(FP), X0   // X0 = alpha
	MOVSS X0, X1             // X1 = X0  // for pipelining
	MOVQ  CX, BX
	ANDQ  $3, BX             // BX = n % 4
	SHRQ  $2, CX             // CX = floor( n / 4 )
	JZ    axpyi_tail_start   // if CX == 0 { goto axpyi_tail_start }

axpyi_loop: // Loop unrolled 4x   do {
	MOVSS (SI), X2        // X_i = x[i]
	MOVSS (SI)(R8*1), X3
	LEAQ  (SI)(R8*2), SI  // SI = &(SI[incX*2])
	MOVSS (SI), X4
	MOVSS (SI)(R8*1), X5
	MULSS X1, X2          // X_i *= a
	MULSS X0, X3
	MULSS X1, X4
	MULSS X0, X5
	ADDSS (DX), X2        // X_i += y[i]
	ADDSS (DX)(R9*1), X3
	LEAQ  (DX)(R9*2), DX  // DX = &(DX[incY*2])
	ADDSS (DX), X4
	ADDSS (DX)(R9*1), X5
	MOVSS X2, (DI)        // dst[i] = X_i
	MOVSS X3, (DI)(R10*1)
	LEAQ  (DI)(R10*2), DI // DI = &(DI[incDst*2])
	MOVSS X4, (DI)
	MOVSS X5, (DI)(R10*1)
	LEAQ  (SI)(R8*2), SI  // SI = &(SI[incX*2])  // Increment addresses
	LEAQ  (DX)(R9*2), DX  // DX = &(DX[incY*2])
	LEAQ  (DI)(R10*2), DI // DI = &(DI[incDst*2])
	LOOP  axpyi_loop      // } while --CX > 0
	CMPQ  BX, $0          // if BX == 0 { return }
	JE    axpyi_end

axpyi_tail_start: // Reset loop registers
	MOVQ BX, CX // Loop counter: CX = BX

axpyi_tail: // do {
	MOVSS (SI), X2   // X2 = x[i]
	MULSS X1, X2     // X2 *= a
	ADDSS (DX), X2   // X2 += y[i]
	MOVSS X2, (DI)   // dst[i] = X2
	ADDQ  R8, SI     // SI = &(SI[incX])
	ADDQ  R9, DX     // DX = &(DX[incY])
	ADDQ  R10, DI    // DI = &(DI[incY])
	LOOP  axpyi_tail // } while --CX > 0

axpyi_end:
	RET

