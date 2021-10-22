// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build 386,gc,!purego

#include "textflag.h"

DATA iv0<>+0x00(SB)/4, $0x6a09e667
DATA iv0<>+0x04(SB)/4, $0xbb67ae85
DATA iv0<>+0x08(SB)/4, $0x3c6ef372
DATA iv0<>+0x0c(SB)/4, $0xa54ff53a
GLOBL iv0<>(SB), (NOPTR+RODATA), $16

DATA iv1<>+0x00(SB)/4, $0x510e527f
DATA iv1<>+0x04(SB)/4, $0x9b05688c
DATA iv1<>+0x08(SB)/4, $0x1f83d9ab
DATA iv1<>+0x0c(SB)/4, $0x5be0cd19
GLOBL iv1<>(SB), (NOPTR+RODATA), $16

DATA rol16<>+0x00(SB)/8, $0x0504070601000302
DATA rol16<>+0x08(SB)/8, $0x0D0C0F0E09080B0A
GLOBL rol16<>(SB), (NOPTR+RODATA), $16

DATA rol8<>+0x00(SB)/8, $0x0407060500030201
DATA rol8<>+0x08(SB)/8, $0x0C0F0E0D080B0A09
GLOBL rol8<>(SB), (NOPTR+RODATA), $16

DATA counter<>+0x00(SB)/8, $0x40
DATA counter<>+0x08(SB)/8, $0x0
GLOBL counter<>(SB), (NOPTR+RODATA), $16

#define ROTL_SSE2(n, t, v) \
	MOVO  v, t;       \
	PSLLL $n, t;      \
	PSRLL $(32-n), v; \
	PXOR  t, v

#define ROTL_SSSE3(c, v) \
	PSHUFB c, v

#define ROUND_SSE2(v0, v1, v2, v3, m0, m1, m2, m3, t) \
	PADDL  m0, v0;        \
	PADDL  v1, v0;        \
	PXOR   v0, v3;        \
	ROTL_SSE2(16, t, v3); \
	PADDL  v3, v2;        \
	PXOR   v2, v1;        \
	ROTL_SSE2(20, t, v1); \
	PADDL  m1, v0;        \
	PADDL  v1, v0;        \
	PXOR   v0, v3;        \
	ROTL_SSE2(24, t, v3); \
	PADDL  v3, v2;        \
	PXOR   v2, v1;        \
	ROTL_SSE2(25, t, v1); \
	PSHUFL $0x39, v1, v1; \
	PSHUFL $0x4E, v2, v2; \
	PSHUFL $0x93, v3, v3; \
	PADDL  m2, v0;        \
	PADDL  v1, v0;        \
	PXOR   v0, v3;        \
	ROTL_SSE2(16, t, v3); \
	PADDL  v3, v2;        \
	PXOR   v2, v1;        \
	ROTL_SSE2(20, t, v1); \
	PADDL  m3, v0;        \
	PADDL  v1, v0;        \
	PXOR   v0, v3;        \
	ROTL_SSE2(24, t, v3); \
	PADDL  v3, v2;        \
	PXOR   v2, v1;        \
	ROTL_SSE2(25, t, v1); \
	PSHUFL $0x39, v3, v3; \
	PSHUFL $0x4E, v2, v2; \
	PSHUFL $0x93, v1, v1

#define ROUND_SSSE3(v0, v1, v2, v3, m0, m1, m2, m3, t, c16, c8) \
	PADDL  m0, v0;        \
	PADDL  v1, v0;        \
	PXOR   v0, v3;        \
	ROTL_SSSE3(c16, v3);  \
	PADDL  v3, v2;        \
	PXOR   v2, v1;        \
	ROTL_SSE2(20, t, v1); \
	PADDL  m1, v0;        \
	PADDL  v1, v0;        \
	PXOR   v0, v3;        \
	ROTL_SSSE3(c8, v3);   \
	PADDL  v3, v2;        \
	PXOR   v2, v1;        \
	ROTL_SSE2(25, t, v1); \
	PSHUFL $0x39, v1, v1; \
	PSHUFL $0x4E, v2, v2; \
	PSHUFL $0x93, v3, v3; \
	PADDL  m2, v0;        \
	PADDL  v1, v0;        \
	PXOR   v0, v3;        \
	ROTL_SSSE3(c16, v3);  \
	PADDL  v3, v2;        \
	PXOR   v2, v1;        \
	ROTL_SSE2(20, t, v1); \
	PADDL  m3, v0;        \
	PADDL  v1, v0;        \
	PXOR   v0, v3;        \
	ROTL_SSSE3(c8, v3);   \
	PADDL  v3, v2;        \
	PXOR   v2, v1;        \
	ROTL_SSE2(25, t, v1); \
	PSHUFL $0x39, v3, v3; \
	PSHUFL $0x4E, v2, v2; \
	PSHUFL $0x93, v1, v1

#define PRECOMPUTE(dst, off, src, t) \
	MOVL 0*4(src), t;          \
	MOVL t, 0*4+off+0(dst);    \
	MOVL t, 9*4+off+64(dst);   \
	MOVL t, 5*4+off+128(dst);  \
	MOVL t, 14*4+off+192(dst); \
	MOVL t, 4*4+off+256(dst);  \
	MOVL t, 2*4+off+320(dst);  \
	MOVL t, 8*4+off+384(dst);  \
	MOVL t, 12*4+off+448(dst); \
	MOVL t, 3*4+off+512(dst);  \
	MOVL t, 15*4+off+576(dst); \
	MOVL 1*4(src), t;          \
	MOVL t, 4*4+off+0(dst);    \
	MOVL t, 8*4+off+64(dst);   \
	MOVL t, 14*4+off+128(dst); \
	MOVL t, 5*4+off+192(dst);  \
	MOVL t, 12*4+off+256(dst); \
	MOVL t, 11*4+off+320(dst); \
	MOVL t, 1*4+off+384(dst);  \
	MOVL t, 6*4+off+448(dst);  \
	MOVL t, 10*4+off+512(dst); \
	MOVL t, 3*4+off+576(dst);  \
	MOVL 2*4(src), t;          \
	MOVL t, 1*4+off+0(dst);    \
	MOVL t, 13*4+off+64(dst);  \
	MOVL t, 6*4+off+128(dst);  \
	MOVL t, 8*4+off+192(dst);  \
	MOVL t, 2*4+off+256(dst);  \
	MOVL t, 0*4+off+320(dst);  \
	MOVL t, 14*4+off+384(dst); \
	MOVL t, 11*4+off+448(dst); \
	MOVL t, 12*4+off+512(dst); \
	MOVL t, 4*4+off+576(dst);  \
	MOVL 3*4(src), t;          \
	MOVL t, 5*4+off+0(dst);    \
	MOVL t, 15*4+off+64(dst);  \
	MOVL t, 9*4+off+128(dst);  \
	MOVL t, 1*4+off+192(dst);  \
	MOVL t, 11*4+off+256(dst); \
	MOVL t, 7*4+off+320(dst);  \
	MOVL t, 13*4+off+384(dst); \
	MOVL t, 3*4+off+448(dst);  \
	MOVL t, 6*4+off+512(dst);  \
	MOVL t, 10*4+off+576(dst); \
	MOVL 4*4(src), t;          \
	MOVL t, 2*4+off+0(dst);    \
	MOVL t, 1*4+off+64(dst);   \
	MOVL t, 15*4+off+128(dst); \
	MOVL t, 10*4+off+192(dst); \
	MOVL t, 6*4+off+256(dst);  \
	MOVL t, 8*4+off+320(dst);  \
	MOVL t, 3*4+off+384(dst);  \
	MOVL t, 13*4+off+448(dst); \
	MOVL t, 14*4+off+512(dst); \
	MOVL t, 5*4+off+576(dst);  \
	MOVL 5*4(src), t;          \
	MOVL t, 6*4+off+0(dst);    \
	MOVL t, 11*4+off+64(dst);  \
	MOVL t, 2*4+off+128(dst);  \
	MOVL t, 9*4+off+192(dst);  \
	MOVL t, 1*4+off+256(dst);  \
	MOVL t, 13*4+off+320(dst); \
	MOVL t, 4*4+off+384(dst);  \
	MOVL t, 8*4+off+448(dst);  \
	MOVL t, 15*4+off+512(dst); \
	MOVL t, 7*4+off+576(dst);  \
	MOVL 6*4(src), t;          \
	MOVL t, 3*4+off+0(dst);    \
	MOVL t, 7*4+off+64(dst);   \
	MOVL t, 13*4+off+128(dst); \
	MOVL t, 12*4+off+192(dst); \
	MOVL t, 10*4+off+256(dst); \
	MOVL t, 1*4+off+320(dst);  \
	MOVL t, 9*4+off+384(dst);  \
	MOVL t, 14*4+off+448(dst); \
	MOVL t, 0*4+off+512(dst);  \
	MOVL t, 6*4+off+576(dst);  \
	MOVL 7*4(src), t;          \
	MOVL t, 7*4+off+0(dst);    \
	MOVL t, 14*4+off+64(dst);  \
	MOVL t, 10*4+off+128(dst); \
	MOVL t, 0*4+off+192(dst);  \
	MOVL t, 5*4+off+256(dst);  \
	MOVL t, 9*4+off+320(dst);  \
	MOVL t, 12*4+off+384(dst); \
	MOVL t, 1*4+off+448(dst);  \
	MOVL t, 13*4+off+512(dst); \
	MOVL t, 2*4+off+576(dst);  \
	MOVL 8*4(src), t;          \
	MOVL t, 8*4+off+0(dst);    \
	MOVL t, 5*4+off+64(dst);   \
	MOVL t, 4*4+off+128(dst);  \
	MOVL t, 15*4+off+192(dst); \
	MOVL t, 14*4+off+256(dst); \
	MOVL t, 3*4+off+320(dst);  \
	MOVL t, 11*4+off+384(dst); \
	MOVL t, 10*4+off+448(dst); \
	MOVL t, 7*4+off+512(dst);  \
	MOVL t, 1*4+off+576(dst);  \
	MOVL 9*4(src), t;          \
	MOVL t, 12*4+off+0(dst);   \
	MOVL t, 2*4+off+64(dst);   \
	MOVL t, 11*4+off+128(dst); \
	MOVL t, 4*4+off+192(dst);  \
	MOVL t, 0*4+off+256(dst);  \
	MOVL t, 15*4+off+320(dst); \
	MOVL t, 10*4+off+384(dst); \
	MOVL t, 7*4+off+448(dst);  \
	MOVL t, 5*4+off+512(dst);  \
	MOVL t, 9*4+off+576(dst);  \
	MOVL 10*4(src), t;         \
	MOVL t, 9*4+off+0(dst);    \
	MOVL t, 4*4+off+64(dst);   \
	MOVL t, 8*4+off+128(dst);  \
	MOVL t, 13*4+off+192(dst); \
	MOVL t, 3*4+off+256(dst);  \
	MOVL t, 5*4+off+320(dst);  \
	MOVL t, 7*4+off+384(dst);  \
	MOVL t, 15*4+off+448(dst); \
	MOVL t, 11*4+off+512(dst); \
	MOVL t, 0*4+off+576(dst);  \
	MOVL 11*4(src), t;         \
	MOVL t, 13*4+off+0(dst);   \
	MOVL t, 10*4+off+64(dst);  \
	MOVL t, 0*4+off+128(dst);  \
	MOVL t, 3*4+off+192(dst);  \
	MOVL t, 9*4+off+256(dst);  \
	MOVL t, 6*4+off+320(dst);  \
	MOVL t, 15*4+off+384(dst); \
	MOVL t, 4*4+off+448(dst);  \
	MOVL t, 2*4+off+512(dst);  \
	MOVL t, 12*4+off+576(dst); \
	MOVL 12*4(src), t;         \
	MOVL t, 10*4+off+0(dst);   \
	MOVL t, 12*4+off+64(dst);  \
	MOVL t, 1*4+off+128(dst);  \
	MOVL t, 6*4+off+192(dst);  \
	MOVL t, 13*4+off+256(dst); \
	MOVL t, 4*4+off+320(dst);  \
	MOVL t, 0*4+off+384(dst);  \
	MOVL t, 2*4+off+448(dst);  \
	MOVL t, 8*4+off+512(dst);  \
	MOVL t, 14*4+off+576(dst); \
	MOVL 13*4(src), t;         \
	MOVL t, 14*4+off+0(dst);   \
	MOVL t, 3*4+off+64(dst);   \
	MOVL t, 7*4+off+128(dst);  \
	MOVL t, 2*4+off+192(dst);  \
	MOVL t, 15*4+off+256(dst); \
	MOVL t, 12*4+off+320(dst); \
	MOVL t, 6*4+off+384(dst);  \
	MOVL t, 0*4+off+448(dst);  \
	MOVL t, 9*4+off+512(dst);  \
	MOVL t, 11*4+off+576(dst); \
	MOVL 14*4(src), t;         \
	MOVL t, 11*4+off+0(dst);   \
	MOVL t, 0*4+off+64(dst);   \
	MOVL t, 12*4+off+128(dst); \
	MOVL t, 7*4+off+192(dst);  \
	MOVL t, 8*4+off+256(dst);  \
	MOVL t, 14*4+off+320(dst); \
	MOVL t, 2*4+off+384(dst);  \
	MOVL t, 5*4+off+448(dst);  \
	MOVL t, 1*4+off+512(dst);  \
	MOVL t, 13*4+off+576(dst); \
	MOVL 15*4(src), t;         \
	MOVL t, 15*4+off+0(dst);   \
	MOVL t, 6*4+off+64(dst);   \
	MOVL t, 3*4+off+128(dst);  \
	MOVL t, 11*4+off+192(dst); \
	MOVL t, 7*4+off+256(dst);  \
	MOVL t, 10*4+off+320(dst); \
	MOVL t, 5*4+off+384(dst);  \
	MOVL t, 9*4+off+448(dst);  \
	MOVL t, 4*4+off+512(dst);  \
	MOVL t, 8*4+off+576(dst)

// func hashBlocksSSE2(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte)
TEXT ·hashBlocksSSE2(SB), 0, $672-24 // frame = 656 + 16 byte alignment
	MOVL h+0(FP), AX
	MOVL c+4(FP), BX
	MOVL flag+8(FP), CX
	MOVL blocks_base+12(FP), SI
	MOVL blocks_len+16(FP), DX

	MOVL SP, DI
	ADDL $15, DI
	ANDL $~15, DI

	MOVL CX, 8(DI)
	MOVL 0(BX), CX
	MOVL CX, 0(DI)
	MOVL 4(BX), CX
	MOVL CX, 4(DI)
	XORL CX, CX
	MOVL CX, 12(DI)

	MOVOU 0(AX), X0
	MOVOU 16(AX), X1
	MOVOU counter<>(SB), X2

loop:
	MOVO  X0, X4
	MOVO  X1, X5
	MOVOU iv0<>(SB), X6
	MOVOU iv1<>(SB), X7

	MOVO  0(DI), X3
	PADDQ X2, X3
	PXOR  X3, X7
	MOVO  X3, 0(DI)

	PRECOMPUTE(DI, 16, SI, CX)
	ROUND_SSE2(X4, X5, X6, X7, 16(DI), 32(DI), 48(DI), 64(DI), X3)
	ROUND_SSE2(X4, X5, X6, X7, 16+64(DI), 32+64(DI), 48+64(DI), 64+64(DI), X3)
	ROUND_SSE2(X4, X5, X6, X7, 16+128(DI), 32+128(DI), 48+128(DI), 64+128(DI), X3)
	ROUND_SSE2(X4, X5, X6, X7, 16+192(DI), 32+192(DI), 48+192(DI), 64+192(DI), X3)
	ROUND_SSE2(X4, X5, X6, X7, 16+256(DI), 32+256(DI), 48+256(DI), 64+256(DI), X3)
	ROUND_SSE2(X4, X5, X6, X7, 16+320(DI), 32+320(DI), 48+320(DI), 64+320(DI), X3)
	ROUND_SSE2(X4, X5, X6, X7, 16+384(DI), 32+384(DI), 48+384(DI), 64+384(DI), X3)
	ROUND_SSE2(X4, X5, X6, X7, 16+448(DI), 32+448(DI), 48+448(DI), 64+448(DI), X3)
	ROUND_SSE2(X4, X5, X6, X7, 16+512(DI), 32+512(DI), 48+512(DI), 64+512(DI), X3)
	ROUND_SSE2(X4, X5, X6, X7, 16+576(DI), 32+576(DI), 48+576(DI), 64+576(DI), X3)

	PXOR X4, X0
	PXOR X5, X1
	PXOR X6, X0
	PXOR X7, X1

	LEAL 64(SI), SI
	SUBL $64, DX
	JNE  loop

	MOVL 0(DI), CX
	MOVL CX, 0(BX)
	MOVL 4(DI), CX
	MOVL CX, 4(BX)

	MOVOU X0, 0(AX)
	MOVOU X1, 16(AX)

	RET

// func hashBlocksSSSE3(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte)
TEXT ·hashBlocksSSSE3(SB), 0, $704-24 // frame = 688 + 16 byte alignment
	MOVL h+0(FP), AX
	MOVL c+4(FP), BX
	MOVL flag+8(FP), CX
	MOVL blocks_base+12(FP), SI
	MOVL blocks_len+16(FP), DX

	MOVL SP, DI
	ADDL $15, DI
	ANDL $~15, DI

	MOVL CX, 8(DI)
	MOVL 0(BX), CX
	MOVL CX, 0(DI)
	MOVL 4(BX), CX
	MOVL CX, 4(DI)
	XORL CX, CX
	MOVL CX, 12(DI)

	MOVOU 0(AX), X0
	MOVOU 16(AX), X1
	MOVOU counter<>(SB), X2

loop:
	MOVO  X0, 656(DI)
	MOVO  X1, 672(DI)
	MOVO  X0, X4
	MOVO  X1, X5
	MOVOU iv0<>(SB), X6
	MOVOU iv1<>(SB), X7

	MOVO  0(DI), X3
	PADDQ X2, X3
	PXOR  X3, X7
	MOVO  X3, 0(DI)

	MOVOU rol16<>(SB), X0
	MOVOU rol8<>(SB), X1

	PRECOMPUTE(DI, 16, SI, CX)
	ROUND_SSSE3(X4, X5, X6, X7, 16(DI), 32(DI), 48(DI), 64(DI), X3, X0, X1)
	ROUND_SSSE3(X4, X5, X6, X7, 16+64(DI), 32+64(DI), 48+64(DI), 64+64(DI), X3, X0, X1)
	ROUND_SSSE3(X4, X5, X6, X7, 16+128(DI), 32+128(DI), 48+128(DI), 64+128(DI), X3, X0, X1)
	ROUND_SSSE3(X4, X5, X6, X7, 16+192(DI), 32+192(DI), 48+192(DI), 64+192(DI), X3, X0, X1)
	ROUND_SSSE3(X4, X5, X6, X7, 16+256(DI), 32+256(DI), 48+256(DI), 64+256(DI), X3, X0, X1)
	ROUND_SSSE3(X4, X5, X6, X7, 16+320(DI), 32+320(DI), 48+320(DI), 64+320(DI), X3, X0, X1)
	ROUND_SSSE3(X4, X5, X6, X7, 16+384(DI), 32+384(DI), 48+384(DI), 64+384(DI), X3, X0, X1)
	ROUND_SSSE3(X4, X5, X6, X7, 16+448(DI), 32+448(DI), 48+448(DI), 64+448(DI), X3, X0, X1)
	ROUND_SSSE3(X4, X5, X6, X7, 16+512(DI), 32+512(DI), 48+512(DI), 64+512(DI), X3, X0, X1)
	ROUND_SSSE3(X4, X5, X6, X7, 16+576(DI), 32+576(DI), 48+576(DI), 64+576(DI), X3, X0, X1)

	MOVO 656(DI), X0
	MOVO 672(DI), X1
	PXOR X4, X0
	PXOR X5, X1
	PXOR X6, X0
	PXOR X7, X1

	LEAL 64(SI), SI
	SUBL $64, DX
	JNE  loop

	MOVL 0(DI), CX
	MOVL CX, 0(BX)
	MOVL 4(DI), CX
	MOVL CX, 4(BX)

	MOVOU X0, 0(AX)
	MOVOU X1, 16(AX)

	RET
