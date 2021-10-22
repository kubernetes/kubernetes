// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,!gccgo,!appengine

#include "textflag.h"

DATA ·c40<>+0x00(SB)/8, $0x0201000706050403
DATA ·c40<>+0x08(SB)/8, $0x0a09080f0e0d0c0b
GLOBL ·c40<>(SB), (NOPTR+RODATA), $16

DATA ·c48<>+0x00(SB)/8, $0x0100070605040302
DATA ·c48<>+0x08(SB)/8, $0x09080f0e0d0c0b0a
GLOBL ·c48<>(SB), (NOPTR+RODATA), $16

#define SHUFFLE(v2, v3, v4, v5, v6, v7, t1, t2) \
	MOVO       v4, t1; \
	MOVO       v5, v4; \
	MOVO       t1, v5; \
	MOVO       v6, t1; \
	PUNPCKLQDQ v6, t2; \
	PUNPCKHQDQ v7, v6; \
	PUNPCKHQDQ t2, v6; \
	PUNPCKLQDQ v7, t2; \
	MOVO       t1, v7; \
	MOVO       v2, t1; \
	PUNPCKHQDQ t2, v7; \
	PUNPCKLQDQ v3, t2; \
	PUNPCKHQDQ t2, v2; \
	PUNPCKLQDQ t1, t2; \
	PUNPCKHQDQ t2, v3

#define SHUFFLE_INV(v2, v3, v4, v5, v6, v7, t1, t2) \
	MOVO       v4, t1; \
	MOVO       v5, v4; \
	MOVO       t1, v5; \
	MOVO       v2, t1; \
	PUNPCKLQDQ v2, t2; \
	PUNPCKHQDQ v3, v2; \
	PUNPCKHQDQ t2, v2; \
	PUNPCKLQDQ v3, t2; \
	MOVO       t1, v3; \
	MOVO       v6, t1; \
	PUNPCKHQDQ t2, v3; \
	PUNPCKLQDQ v7, t2; \
	PUNPCKHQDQ t2, v6; \
	PUNPCKLQDQ t1, t2; \
	PUNPCKHQDQ t2, v7

#define HALF_ROUND(v0, v1, v2, v3, v4, v5, v6, v7, t0, c40, c48) \
	MOVO    v0, t0;        \
	PMULULQ v2, t0;        \
	PADDQ   v2, v0;        \
	PADDQ   t0, v0;        \
	PADDQ   t0, v0;        \
	PXOR    v0, v6;        \
	PSHUFD  $0xB1, v6, v6; \
	MOVO    v4, t0;        \
	PMULULQ v6, t0;        \
	PADDQ   v6, v4;        \
	PADDQ   t0, v4;        \
	PADDQ   t0, v4;        \
	PXOR    v4, v2;        \
	PSHUFB  c40, v2;       \
	MOVO    v0, t0;        \
	PMULULQ v2, t0;        \
	PADDQ   v2, v0;        \
	PADDQ   t0, v0;        \
	PADDQ   t0, v0;        \
	PXOR    v0, v6;        \
	PSHUFB  c48, v6;       \
	MOVO    v4, t0;        \
	PMULULQ v6, t0;        \
	PADDQ   v6, v4;        \
	PADDQ   t0, v4;        \
	PADDQ   t0, v4;        \
	PXOR    v4, v2;        \
	MOVO    v2, t0;        \
	PADDQ   v2, t0;        \
	PSRLQ   $63, v2;       \
	PXOR    t0, v2;        \
	MOVO    v1, t0;        \
	PMULULQ v3, t0;        \
	PADDQ   v3, v1;        \
	PADDQ   t0, v1;        \
	PADDQ   t0, v1;        \
	PXOR    v1, v7;        \
	PSHUFD  $0xB1, v7, v7; \
	MOVO    v5, t0;        \
	PMULULQ v7, t0;        \
	PADDQ   v7, v5;        \
	PADDQ   t0, v5;        \
	PADDQ   t0, v5;        \
	PXOR    v5, v3;        \
	PSHUFB  c40, v3;       \
	MOVO    v1, t0;        \
	PMULULQ v3, t0;        \
	PADDQ   v3, v1;        \
	PADDQ   t0, v1;        \
	PADDQ   t0, v1;        \
	PXOR    v1, v7;        \
	PSHUFB  c48, v7;       \
	MOVO    v5, t0;        \
	PMULULQ v7, t0;        \
	PADDQ   v7, v5;        \
	PADDQ   t0, v5;        \
	PADDQ   t0, v5;        \
	PXOR    v5, v3;        \
	MOVO    v3, t0;        \
	PADDQ   v3, t0;        \
	PSRLQ   $63, v3;       \
	PXOR    t0, v3

#define LOAD_MSG_0(block, off) \
	MOVOU 8*(off+0)(block), X0;  \
	MOVOU 8*(off+2)(block), X1;  \
	MOVOU 8*(off+4)(block), X2;  \
	MOVOU 8*(off+6)(block), X3;  \
	MOVOU 8*(off+8)(block), X4;  \
	MOVOU 8*(off+10)(block), X5; \
	MOVOU 8*(off+12)(block), X6; \
	MOVOU 8*(off+14)(block), X7

#define STORE_MSG_0(block, off) \
	MOVOU X0, 8*(off+0)(block);  \
	MOVOU X1, 8*(off+2)(block);  \
	MOVOU X2, 8*(off+4)(block);  \
	MOVOU X3, 8*(off+6)(block);  \
	MOVOU X4, 8*(off+8)(block);  \
	MOVOU X5, 8*(off+10)(block); \
	MOVOU X6, 8*(off+12)(block); \
	MOVOU X7, 8*(off+14)(block)

#define LOAD_MSG_1(block, off) \
	MOVOU 8*off+0*8(block), X0;  \
	MOVOU 8*off+16*8(block), X1; \
	MOVOU 8*off+32*8(block), X2; \
	MOVOU 8*off+48*8(block), X3; \
	MOVOU 8*off+64*8(block), X4; \
	MOVOU 8*off+80*8(block), X5; \
	MOVOU 8*off+96*8(block), X6; \
	MOVOU 8*off+112*8(block), X7

#define STORE_MSG_1(block, off) \
	MOVOU X0, 8*off+0*8(block);  \
	MOVOU X1, 8*off+16*8(block); \
	MOVOU X2, 8*off+32*8(block); \
	MOVOU X3, 8*off+48*8(block); \
	MOVOU X4, 8*off+64*8(block); \
	MOVOU X5, 8*off+80*8(block); \
	MOVOU X6, 8*off+96*8(block); \
	MOVOU X7, 8*off+112*8(block)

#define BLAMKA_ROUND_0(block, off, t0, t1, c40, c48) \
	LOAD_MSG_0(block, off);                                   \
	HALF_ROUND(X0, X1, X2, X3, X4, X5, X6, X7, t0, c40, c48); \
	SHUFFLE(X2, X3, X4, X5, X6, X7, t0, t1);                  \
	HALF_ROUND(X0, X1, X2, X3, X4, X5, X6, X7, t0, c40, c48); \
	SHUFFLE_INV(X2, X3, X4, X5, X6, X7, t0, t1);              \
	STORE_MSG_0(block, off)

#define BLAMKA_ROUND_1(block, off, t0, t1, c40, c48) \
	LOAD_MSG_1(block, off);                                   \
	HALF_ROUND(X0, X1, X2, X3, X4, X5, X6, X7, t0, c40, c48); \
	SHUFFLE(X2, X3, X4, X5, X6, X7, t0, t1);                  \
	HALF_ROUND(X0, X1, X2, X3, X4, X5, X6, X7, t0, c40, c48); \
	SHUFFLE_INV(X2, X3, X4, X5, X6, X7, t0, t1);              \
	STORE_MSG_1(block, off)

// func blamkaSSE4(b *block)
TEXT ·blamkaSSE4(SB), 4, $0-8
	MOVQ b+0(FP), AX

	MOVOU ·c40<>(SB), X10
	MOVOU ·c48<>(SB), X11

	BLAMKA_ROUND_0(AX, 0, X8, X9, X10, X11)
	BLAMKA_ROUND_0(AX, 16, X8, X9, X10, X11)
	BLAMKA_ROUND_0(AX, 32, X8, X9, X10, X11)
	BLAMKA_ROUND_0(AX, 48, X8, X9, X10, X11)
	BLAMKA_ROUND_0(AX, 64, X8, X9, X10, X11)
	BLAMKA_ROUND_0(AX, 80, X8, X9, X10, X11)
	BLAMKA_ROUND_0(AX, 96, X8, X9, X10, X11)
	BLAMKA_ROUND_0(AX, 112, X8, X9, X10, X11)

	BLAMKA_ROUND_1(AX, 0, X8, X9, X10, X11)
	BLAMKA_ROUND_1(AX, 2, X8, X9, X10, X11)
	BLAMKA_ROUND_1(AX, 4, X8, X9, X10, X11)
	BLAMKA_ROUND_1(AX, 6, X8, X9, X10, X11)
	BLAMKA_ROUND_1(AX, 8, X8, X9, X10, X11)
	BLAMKA_ROUND_1(AX, 10, X8, X9, X10, X11)
	BLAMKA_ROUND_1(AX, 12, X8, X9, X10, X11)
	BLAMKA_ROUND_1(AX, 14, X8, X9, X10, X11)
	RET

// func mixBlocksSSE2(out, a, b, c *block)
TEXT ·mixBlocksSSE2(SB), 4, $0-32
	MOVQ out+0(FP), DX
	MOVQ a+8(FP), AX
	MOVQ b+16(FP), BX
	MOVQ a+24(FP), CX
	MOVQ $128, BP

loop:
	MOVOU 0(AX), X0
	MOVOU 0(BX), X1
	MOVOU 0(CX), X2
	PXOR  X1, X0
	PXOR  X2, X0
	MOVOU X0, 0(DX)
	ADDQ  $16, AX
	ADDQ  $16, BX
	ADDQ  $16, CX
	ADDQ  $16, DX
	SUBQ  $2, BP
	JA    loop
	RET

// func xorBlocksSSE2(out, a, b, c *block)
TEXT ·xorBlocksSSE2(SB), 4, $0-32
	MOVQ out+0(FP), DX
	MOVQ a+8(FP), AX
	MOVQ b+16(FP), BX
	MOVQ a+24(FP), CX
	MOVQ $128, BP

loop:
	MOVOU 0(AX), X0
	MOVOU 0(BX), X1
	MOVOU 0(CX), X2
	MOVOU 0(DX), X3
	PXOR  X1, X0
	PXOR  X2, X0
	PXOR  X3, X0
	MOVOU X0, 0(DX)
	ADDQ  $16, AX
	ADDQ  $16, BX
	ADDQ  $16, CX
	ADDQ  $16, DX
	SUBQ  $2, BP
	JA    loop
	RET
