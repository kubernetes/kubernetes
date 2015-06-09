
#define NOSPLIT 4

// func scanStringSSE(s []byte, j int) (int, byte)
TEXT scanStringSSE(SB),NOSPLIT,$0
	// TODO: http://www.strchr.com/strcmp_and_strlen_using_sse_4.2
	// Equal any, operand1 set to 
	RET

// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.
// func haveSSE42() bool
TEXT ·haveSSE42(SB),NOSPLIT,$0
	XORQ AX, AX
	INCL AX
	CPUID
	SHRQ $20, CX
	ANDQ $1, CX
	MOVB CX, ret+0(FP)
	RET
