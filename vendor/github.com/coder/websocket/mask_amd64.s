#include "textflag.h"

// func maskAsm(b *byte, len int, key uint32)
TEXT ·maskAsm(SB), NOSPLIT, $0-28
	// AX = b
	// CX = len (left length)
	// SI = key (uint32)
	// DI = uint64(SI) | uint64(SI)<<32
	MOVQ b+0(FP), AX
	MOVQ len+8(FP), CX
	MOVL key+16(FP), SI

	// calculate the DI
	// DI = SI<<32 | SI
	MOVL SI, DI
	MOVQ DI, DX
	SHLQ $32, DI
	ORQ  DX, DI

	CMPQ  CX, $15
	JLE   less_than_16
	CMPQ  CX, $63
	JLE   less_than_64
	CMPQ  CX, $128
	JLE   sse
	TESTQ $31, AX
	JNZ   unaligned

unaligned_loop_1byte:
	XORB  SI, (AX)
	INCQ  AX
	DECQ  CX
	ROLL  $24, SI
	TESTQ $7, AX
	JNZ   unaligned_loop_1byte

	// calculate DI again since SI was modified
	// DI = SI<<32 | SI
	MOVL SI, DI
	MOVQ DI, DX
	SHLQ $32, DI
	ORQ  DX, DI

	TESTQ $31, AX
	JZ    sse

unaligned:
	TESTQ $7, AX               // AND $7 & len, if not zero jump to loop_1b.
	JNZ   unaligned_loop_1byte

unaligned_loop:
	// we don't need to check the CX since we know it's above 128
	XORQ  DI, (AX)
	ADDQ  $8, AX
	SUBQ  $8, CX
	TESTQ $31, AX
	JNZ   unaligned_loop
	JMP   sse

sse:
	CMPQ       CX, $0x40
	JL         less_than_64
	MOVQ       DI, X0
	PUNPCKLQDQ X0, X0

sse_loop:
	MOVOU 0*16(AX), X1
	MOVOU 1*16(AX), X2
	MOVOU 2*16(AX), X3
	MOVOU 3*16(AX), X4
	PXOR  X0, X1
	PXOR  X0, X2
	PXOR  X0, X3
	PXOR  X0, X4
	MOVOU X1, 0*16(AX)
	MOVOU X2, 1*16(AX)
	MOVOU X3, 2*16(AX)
	MOVOU X4, 3*16(AX)
	ADDQ  $0x40, AX
	SUBQ  $0x40, CX
	CMPQ  CX, $0x40
	JAE   sse_loop

less_than_64:
	TESTQ $32, CX
	JZ    less_than_32
	XORQ  DI, (AX)
	XORQ  DI, 8(AX)
	XORQ  DI, 16(AX)
	XORQ  DI, 24(AX)
	ADDQ  $32, AX

less_than_32:
	TESTQ $16, CX
	JZ    less_than_16
	XORQ  DI, (AX)
	XORQ  DI, 8(AX)
	ADDQ  $16, AX

less_than_16:
	TESTQ $8, CX
	JZ    less_than_8
	XORQ  DI, (AX)
	ADDQ  $8, AX

less_than_8:
	TESTQ $4, CX
	JZ    less_than_4
	XORL  SI, (AX)
	ADDQ  $4, AX

less_than_4:
	TESTQ $2, CX
	JZ    less_than_2
	XORW  SI, (AX)
	ROLL  $16, SI
	ADDQ  $2, AX

less_than_2:
	TESTQ $1, CX
	JZ    done
	XORB  SI, (AX)
	ROLL  $24, SI

done:
	MOVL SI, ret+24(FP)
	RET
