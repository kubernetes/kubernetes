// Copyright ©2019 The Gonum Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !noasm,!gccgo,!safe

#include "textflag.h"

#define SUMSQ X0
#define ABSX X1
#define SCALE X2
#define ZERO X3
#define TMP X4
#define ABSMASK X5
#define INF X7
#define INFMASK X11
#define NANMASK X12
#define IDX AX
#define X_ DI
#define Y_ BX
#define LEN SI

#define ABSMASK_DATA l2nrodata<>+0(SB)
#define INF_DATA l2nrodata<>+8(SB)
#define NAN_DATA l2nrodata<>+16(SB)
// AbsMask
DATA l2nrodata<>+0(SB)/8, $0x7FFFFFFFFFFFFFFF
// Inf
DATA l2nrodata<>+8(SB)/8, $0x7FF0000000000000
// NaN
DATA l2nrodata<>+16(SB)/8, $0xFFF8000000000000
GLOBL l2nrodata<>+0(SB), RODATA, $24

// L2DistanceUnitary returns the L2-norm of x-y.
// func L2DistanceUnitary(x,y []float64) (norm float64)
TEXT ·L2DistanceUnitary(SB), NOSPLIT, $0
	MOVQ    x_base+0(FP), X_
	MOVQ    y_base+24(FP), Y_
	PXOR    ZERO, ZERO
	MOVQ    x_len+8(FP), LEN  // LEN = min( len(x), len(y) )
	CMPQ    y_len+32(FP), LEN
	CMOVQLE y_len+32(FP), LEN
	CMPQ    LEN, $0           // if LEN == 0 { return 0 }
	JZ      retZero

	PXOR  INFMASK, INFMASK
	PXOR  NANMASK, NANMASK
	MOVSD $1.0, SUMSQ           // ssq = 1
	XORPS SCALE, SCALE
	MOVSD ABSMASK_DATA, ABSMASK
	MOVSD INF_DATA, INF
	XORQ  IDX, IDX              // idx == 0

initZero:  // for ;x[i]==0; i++ {}
	// Skip all leading zeros, to avoid divide by zero NaN
	MOVSD   (X_)(IDX*8), ABSX // absxi = x[i]
	SUBSD   (Y_)(IDX*8), ABSX // absxi = x[i]-y[i]
	UCOMISD ABSX, ZERO
	JP      retNaN            // if isNaN(absxi) { return NaN }
	JNE     loop              // if absxi != 0 { goto loop }
	INCQ    IDX               // i++
	CMPQ    IDX, LEN
	JE      retZero           // if i == LEN { return 0 }
	JMP     initZero

loop:
	MOVSD   (X_)(IDX*8), ABSX // absxi = x[i]
	SUBSD   (Y_)(IDX*8), ABSX // absxi = x[i]-y[i]
	MOVUPS  ABSX, TMP
	CMPSD   ABSX, TMP, $3
	ORPD    TMP, NANMASK      // NANMASK = NANMASK | IsNaN(absxi)
	MOVSD   INF, TMP
	ANDPD   ABSMASK, ABSX     // absxi == Abs(absxi)
	CMPSD   ABSX, TMP, $0
	ORPD    TMP, INFMASK      // INFMASK =  INFMASK | IsInf(absxi)
	UCOMISD SCALE, ABSX
	JA      adjScale          // IF SCALE > ABSXI { goto adjScale }

	DIVSD SCALE, ABSX // absxi = scale / absxi
	MULSD ABSX, ABSX  // absxi *= absxi
	ADDSD ABSX, SUMSQ // sumsq += absxi
	INCQ  IDX         // i++
	CMPQ  IDX, LEN
	JNE   loop        // if i < LEN { continue }
	JMP   retSum      // if i == LEN { goto retSum }

adjScale:  // Scale > Absxi
	DIVSD  ABSX, SCALE  // tmp = absxi / scale
	MULSD  SCALE, SUMSQ // sumsq *= tmp
	MULSD  SCALE, SUMSQ // sumsq *= tmp
	ADDSD  $1.0, SUMSQ  // sumsq += 1
	MOVUPS ABSX, SCALE  // scale = absxi
	INCQ   IDX          // i++
	CMPQ   IDX, LEN
	JNE    loop         // if i < LEN { continue }

retSum:  // Calculate return value
	SQRTSD  SUMSQ, SUMSQ     // sumsq = sqrt(sumsq)
	MULSD   SCALE, SUMSQ     // sumsq += scale
	MOVQ    SUMSQ, R10       // tmp = sumsq
	UCOMISD ZERO, INFMASK
	CMOVQPS INF_DATA, R10    // if INFMASK { tmp = INF }
	UCOMISD ZERO, NANMASK
	CMOVQPS NAN_DATA, R10    // if NANMASK { tmp = NaN }
	MOVQ    R10, norm+48(FP) // return tmp
	RET

retZero:
	MOVSD ZERO, norm+48(FP) // return 0
	RET

retNaN:
	MOVSD NAN_DATA, TMP    // return NaN
	MOVSD TMP, norm+48(FP)
	RET
