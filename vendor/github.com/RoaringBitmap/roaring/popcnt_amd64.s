// +build amd64,!appengine,!go1.9

TEXT ·hasAsm(SB),4,$0-1
MOVQ $1, AX
CPUID
SHRQ $23, CX
ANDQ $1, CX
MOVB CX, ret+0(FP)
RET

#define POPCNTQ_DX_DX BYTE $0xf3; BYTE $0x48; BYTE $0x0f; BYTE $0xb8; BYTE $0xd2

TEXT ·popcntSliceAsm(SB),4,$0-32
XORQ	AX, AX
MOVQ	s+0(FP), SI
MOVQ	s_len+8(FP), CX
TESTQ	CX, CX
JZ		popcntSliceEnd
popcntSliceLoop:
BYTE $0xf3; BYTE $0x48; BYTE $0x0f; BYTE $0xb8; BYTE $0x16 // POPCNTQ (SI), DX
ADDQ	DX, AX
ADDQ	$8, SI
LOOP	popcntSliceLoop
popcntSliceEnd:
MOVQ	AX, ret+24(FP)
RET

TEXT ·popcntMaskSliceAsm(SB),4,$0-56
XORQ	AX, AX
MOVQ	s+0(FP), SI
MOVQ	s_len+8(FP), CX
TESTQ	CX, CX
JZ		popcntMaskSliceEnd
MOVQ	m+24(FP), DI
popcntMaskSliceLoop:
MOVQ	(DI), DX
NOTQ	DX
ANDQ	(SI), DX
POPCNTQ_DX_DX
ADDQ	DX, AX
ADDQ	$8, SI
ADDQ	$8, DI
LOOP	popcntMaskSliceLoop
popcntMaskSliceEnd:
MOVQ	AX, ret+48(FP)
RET

TEXT ·popcntAndSliceAsm(SB),4,$0-56
XORQ	AX, AX
MOVQ	s+0(FP), SI
MOVQ	s_len+8(FP), CX
TESTQ	CX, CX
JZ		popcntAndSliceEnd
MOVQ	m+24(FP), DI
popcntAndSliceLoop:
MOVQ	(DI), DX
ANDQ	(SI), DX
POPCNTQ_DX_DX
ADDQ	DX, AX
ADDQ	$8, SI
ADDQ	$8, DI
LOOP	popcntAndSliceLoop
popcntAndSliceEnd:
MOVQ	AX, ret+48(FP)
RET

TEXT ·popcntOrSliceAsm(SB),4,$0-56
XORQ	AX, AX
MOVQ	s+0(FP), SI
MOVQ	s_len+8(FP), CX
TESTQ	CX, CX
JZ		popcntOrSliceEnd
MOVQ	m+24(FP), DI
popcntOrSliceLoop:
MOVQ	(DI), DX
ORQ		(SI), DX
POPCNTQ_DX_DX
ADDQ	DX, AX
ADDQ	$8, SI
ADDQ	$8, DI
LOOP	popcntOrSliceLoop
popcntOrSliceEnd:
MOVQ	AX, ret+48(FP)
RET

TEXT ·popcntXorSliceAsm(SB),4,$0-56
XORQ	AX, AX
MOVQ	s+0(FP), SI
MOVQ	s_len+8(FP), CX
TESTQ	CX, CX
JZ		popcntXorSliceEnd
MOVQ	m+24(FP), DI
popcntXorSliceLoop:
MOVQ	(DI), DX
XORQ	(SI), DX
POPCNTQ_DX_DX
ADDQ	DX, AX
ADDQ	$8, SI
ADDQ	$8, DI
LOOP	popcntXorSliceLoop
popcntXorSliceEnd:
MOVQ	AX, ret+48(FP)
RET
