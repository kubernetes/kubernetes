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
#define LEN SI
#define INC BX
#define X_ DI

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

// func L2NormInc(x []float64, n, incX uintptr) (norm float64)
TEXT ·L2NormInc(SB), NOSPLIT, $0
	MOVQ n+24(FP), LEN    // LEN = len(x)
	MOVQ incX+32(FP), INC
	MOVQ x_base+0(FP), X_
	XORPS ZERO, ZERO
	CMPQ LEN, $0          // if LEN == 0 { return 0 }
	JZ   retZero

	XORPS INFMASK, INFMASK
	XORPS NANMASK, NANMASK
	MOVSD $1.0, SUMSQ           // ssq = 1
	XORPS SCALE, SCALE
	MOVSD ABSMASK_DATA, ABSMASK
	MOVSD INF_DATA, INF
	SHLQ  $3, INC               // INC *= sizeof(float64)

initZero:  // for ;x[i]==0; i++ {}
	// Skip all leading zeros, to avoid divide by zero NaN
	MOVSD   (X_), ABSX // absxi = x[i]
	UCOMISD ABSX, ZERO
	JP      retNaN     // if isNaN(x[i]) { return NaN }
	JNZ     loop       // if x[i] != 0 { goto loop }
	ADDQ    INC, X_    // i += INC
	DECQ    LEN        // LEN--
	JZ      retZero    // if LEN == 0 { return 0 }
	JMP     initZero

loop:
	MOVSD   (X_), ABSX    // absxi = x[i]
	MOVUPS  ABSX, TMP
	CMPSD   ABSX, TMP, $3
	ORPD    TMP, NANMASK  // NANMASK = NANMASK | IsNaN(absxi)
	MOVSD   INF, TMP
	ANDPD   ABSMASK, ABSX // absxi == Abs(absxi)
	CMPSD   ABSX, TMP, $0
	ORPD    TMP, INFMASK  // INFMASK =  INFMASK | IsInf(absxi)
	UCOMISD SCALE, ABSX
	JA      adjScale      // IF SCALE > ABSXI { goto adjScale }

	DIVSD SCALE, ABSX // absxi = scale / absxi
	MULSD ABSX, ABSX  // absxi *= absxi
	ADDSD ABSX, SUMSQ // sumsq += absxi
	ADDQ  INC, X_     // i += INC
	DECQ  LEN         // LEN--
	JNZ   loop        // if LEN > 0 { continue }
	JMP   retSum      // if LEN == 0 { goto retSum }

adjScale:  // Scale > Absxi
	DIVSD  ABSX, SCALE  // tmp = absxi / scale
	MULSD  SCALE, SUMSQ // sumsq *= tmp
	MULSD  SCALE, SUMSQ // sumsq *= tmp
	ADDSD  $1.0, SUMSQ  // sumsq += 1
	MOVUPS ABSX, SCALE  // scale = absxi
	ADDQ   INC, X_      // i += INC
	DECQ   LEN          // LEN--
	JNZ    loop         // if LEN > 0 { continue }

retSum:  // Calculate return value
	SQRTSD  SUMSQ, SUMSQ     // sumsq = sqrt(sumsq)
	MULSD   SCALE, SUMSQ     // sumsq += scale
	MOVQ    SUMSQ, R10       // tmp = sumsq
	UCOMISD ZERO, INFMASK
	CMOVQPS INF_DATA, R10    // if INFMASK { tmp = INF }
	UCOMISD ZERO, NANMASK
	CMOVQPS NAN_DATA, R10    // if NANMASK { tmp = NaN }
	MOVQ    R10, norm+40(FP) // return tmp
	RET

retZero:
	MOVSD ZERO, norm+40(FP) // return 0
	RET

retNaN:
	MOVSD NAN_DATA, TMP    // return NaN
	MOVSD TMP, norm+40(FP)
	RET
