// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64,!gccgo,!appengine

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


#define LOAD_MSG_SSE4(m0, m1, m2, m3, src, i0, i1, i2, i3, i4, i5, i6, i7, i8, i9, i10, i11, i12, i13, i14, i15) \
	MOVL   i0*4(src), m0;      \
	PINSRD $1, i1*4(src), m0;  \
	PINSRD $2, i2*4(src), m0;  \
	PINSRD $3, i3*4(src), m0;  \
	MOVL   i4*4(src), m1;      \
	PINSRD $1, i5*4(src), m1;  \
	PINSRD $2, i6*4(src), m1;  \
	PINSRD $3, i7*4(src), m1;  \
	MOVL   i8*4(src), m2;      \
	PINSRD $1, i9*4(src), m2;  \
	PINSRD $2, i10*4(src), m2; \
	PINSRD $3, i11*4(src), m2; \
	MOVL   i12*4(src), m3;     \
	PINSRD $1, i13*4(src), m3; \
	PINSRD $2, i14*4(src), m3; \
	PINSRD $3, i15*4(src), m3

#define PRECOMPUTE_MSG(dst, off, src, R8, R9, R10, R11, R12, R13, R14, R15) \
	MOVQ 0*4(src), R8;           \
	MOVQ 2*4(src), R9;           \
	MOVQ 4*4(src), R10;          \
	MOVQ 6*4(src), R11;          \
	MOVQ 8*4(src), R12;          \
	MOVQ 10*4(src), R13;         \
	MOVQ 12*4(src), R14;         \
	MOVQ 14*4(src), R15;         \
	                             \
	MOVL R8, 0*4+off+0(dst);     \
	MOVL R8, 9*4+off+64(dst);    \
	MOVL R8, 5*4+off+128(dst);   \
	MOVL R8, 14*4+off+192(dst);  \
	MOVL R8, 4*4+off+256(dst);   \
	MOVL R8, 2*4+off+320(dst);   \
	MOVL R8, 8*4+off+384(dst);   \
	MOVL R8, 12*4+off+448(dst);  \
	MOVL R8, 3*4+off+512(dst);   \
	MOVL R8, 15*4+off+576(dst);  \
	SHRQ $32, R8;                \
	MOVL R8, 4*4+off+0(dst);     \
	MOVL R8, 8*4+off+64(dst);    \
	MOVL R8, 14*4+off+128(dst);  \
	MOVL R8, 5*4+off+192(dst);   \
	MOVL R8, 12*4+off+256(dst);  \
	MOVL R8, 11*4+off+320(dst);  \
	MOVL R8, 1*4+off+384(dst);   \
	MOVL R8, 6*4+off+448(dst);   \
	MOVL R8, 10*4+off+512(dst);  \
	MOVL R8, 3*4+off+576(dst);   \
	                             \
	MOVL R9, 1*4+off+0(dst);     \
	MOVL R9, 13*4+off+64(dst);   \
	MOVL R9, 6*4+off+128(dst);   \
	MOVL R9, 8*4+off+192(dst);   \
	MOVL R9, 2*4+off+256(dst);   \
	MOVL R9, 0*4+off+320(dst);   \
	MOVL R9, 14*4+off+384(dst);  \
	MOVL R9, 11*4+off+448(dst);  \
	MOVL R9, 12*4+off+512(dst);  \
	MOVL R9, 4*4+off+576(dst);   \
	SHRQ $32, R9;                \
	MOVL R9, 5*4+off+0(dst);     \
	MOVL R9, 15*4+off+64(dst);   \
	MOVL R9, 9*4+off+128(dst);   \
	MOVL R9, 1*4+off+192(dst);   \
	MOVL R9, 11*4+off+256(dst);  \
	MOVL R9, 7*4+off+320(dst);   \
	MOVL R9, 13*4+off+384(dst);  \
	MOVL R9, 3*4+off+448(dst);   \
	MOVL R9, 6*4+off+512(dst);   \
	MOVL R9, 10*4+off+576(dst);  \
	                             \
	MOVL R10, 2*4+off+0(dst);    \
	MOVL R10, 1*4+off+64(dst);   \
	MOVL R10, 15*4+off+128(dst); \
	MOVL R10, 10*4+off+192(dst); \
	MOVL R10, 6*4+off+256(dst);  \
	MOVL R10, 8*4+off+320(dst);  \
	MOVL R10, 3*4+off+384(dst);  \
	MOVL R10, 13*4+off+448(dst); \
	MOVL R10, 14*4+off+512(dst); \
	MOVL R10, 5*4+off+576(dst);  \
	SHRQ $32, R10;               \
	MOVL R10, 6*4+off+0(dst);    \
	MOVL R10, 11*4+off+64(dst);  \
	MOVL R10, 2*4+off+128(dst);  \
	MOVL R10, 9*4+off+192(dst);  \
	MOVL R10, 1*4+off+256(dst);  \
	MOVL R10, 13*4+off+320(dst); \
	MOVL R10, 4*4+off+384(dst);  \
	MOVL R10, 8*4+off+448(dst);  \
	MOVL R10, 15*4+off+512(dst); \
	MOVL R10, 7*4+off+576(dst);  \
	                             \
	MOVL R11, 3*4+off+0(dst);    \
	MOVL R11, 7*4+off+64(dst);   \
	MOVL R11, 13*4+off+128(dst); \
	MOVL R11, 12*4+off+192(dst); \
	MOVL R11, 10*4+off+256(dst); \
	MOVL R11, 1*4+off+320(dst);  \
	MOVL R11, 9*4+off+384(dst);  \
	MOVL R11, 14*4+off+448(dst); \
	MOVL R11, 0*4+off+512(dst);  \
	MOVL R11, 6*4+off+576(dst);  \
	SHRQ $32, R11;               \
	MOVL R11, 7*4+off+0(dst);    \
	MOVL R11, 14*4+off+64(dst);  \
	MOVL R11, 10*4+off+128(dst); \
	MOVL R11, 0*4+off+192(dst);  \
	MOVL R11, 5*4+off+256(dst);  \
	MOVL R11, 9*4+off+320(dst);  \
	MOVL R11, 12*4+off+384(dst); \
	MOVL R11, 1*4+off+448(dst);  \
	MOVL R11, 13*4+off+512(dst); \
	MOVL R11, 2*4+off+576(dst);  \
	                             \
	MOVL R12, 8*4+off+0(dst);    \
	MOVL R12, 5*4+off+64(dst);   \
	MOVL R12, 4*4+off+128(dst);  \
	MOVL R12, 15*4+off+192(dst); \
	MOVL R12, 14*4+off+256(dst); \
	MOVL R12, 3*4+off+320(dst);  \
	MOVL R12, 11*4+off+384(dst); \
	MOVL R12, 10*4+off+448(dst); \
	MOVL R12, 7*4+off+512(dst);  \
	MOVL R12, 1*4+off+576(dst);  \
	SHRQ $32, R12;               \
	MOVL R12, 12*4+off+0(dst);   \
	MOVL R12, 2*4+off+64(dst);   \
	MOVL R12, 11*4+off+128(dst); \
	MOVL R12, 4*4+off+192(dst);  \
	MOVL R12, 0*4+off+256(dst);  \
	MOVL R12, 15*4+off+320(dst); \
	MOVL R12, 10*4+off+384(dst); \
	MOVL R12, 7*4+off+448(dst);  \
	MOVL R12, 5*4+off+512(dst);  \
	MOVL R12, 9*4+off+576(dst);  \
	                             \
	MOVL R13, 9*4+off+0(dst);    \
	MOVL R13, 4*4+off+64(dst);   \
	MOVL R13, 8*4+off+128(dst);  \
	MOVL R13, 13*4+off+192(dst); \
	MOVL R13, 3*4+off+256(dst);  \
	MOVL R13, 5*4+off+320(dst);  \
	MOVL R13, 7*4+off+384(dst);  \
	MOVL R13, 15*4+off+448(dst); \
	MOVL R13, 11*4+off+512(dst); \
	MOVL R13, 0*4+off+576(dst);  \
	SHRQ $32, R13;               \
	MOVL R13, 13*4+off+0(dst);   \
	MOVL R13, 10*4+off+64(dst);  \
	MOVL R13, 0*4+off+128(dst);  \
	MOVL R13, 3*4+off+192(dst);  \
	MOVL R13, 9*4+off+256(dst);  \
	MOVL R13, 6*4+off+320(dst);  \
	MOVL R13, 15*4+off+384(dst); \
	MOVL R13, 4*4+off+448(dst);  \
	MOVL R13, 2*4+off+512(dst);  \
	MOVL R13, 12*4+off+576(dst); \
	                             \
	MOVL R14, 10*4+off+0(dst);   \
	MOVL R14, 12*4+off+64(dst);  \
	MOVL R14, 1*4+off+128(dst);  \
	MOVL R14, 6*4+off+192(dst);  \
	MOVL R14, 13*4+off+256(dst); \
	MOVL R14, 4*4+off+320(dst);  \
	MOVL R14, 0*4+off+384(dst);  \
	MOVL R14, 2*4+off+448(dst);  \
	MOVL R14, 8*4+off+512(dst);  \
	MOVL R14, 14*4+off+576(dst); \
	SHRQ $32, R14;               \
	MOVL R14, 14*4+off+0(dst);   \
	MOVL R14, 3*4+off+64(dst);   \
	MOVL R14, 7*4+off+128(dst);  \
	MOVL R14, 2*4+off+192(dst);  \
	MOVL R14, 15*4+off+256(dst); \
	MOVL R14, 12*4+off+320(dst); \
	MOVL R14, 6*4+off+384(dst);  \
	MOVL R14, 0*4+off+448(dst);  \
	MOVL R14, 9*4+off+512(dst);  \
	MOVL R14, 11*4+off+576(dst); \
	                             \
	MOVL R15, 11*4+off+0(dst);   \
	MOVL R15, 0*4+off+64(dst);   \
	MOVL R15, 12*4+off+128(dst); \
	MOVL R15, 7*4+off+192(dst);  \
	MOVL R15, 8*4+off+256(dst);  \
	MOVL R15, 14*4+off+320(dst); \
	MOVL R15, 2*4+off+384(dst);  \
	MOVL R15, 5*4+off+448(dst);  \
	MOVL R15, 1*4+off+512(dst);  \
	MOVL R15, 13*4+off+576(dst); \
	SHRQ $32, R15;               \
	MOVL R15, 15*4+off+0(dst);   \
	MOVL R15, 6*4+off+64(dst);   \
	MOVL R15, 3*4+off+128(dst);  \
	MOVL R15, 11*4+off+192(dst); \
	MOVL R15, 7*4+off+256(dst);  \
	MOVL R15, 10*4+off+320(dst); \
	MOVL R15, 5*4+off+384(dst);  \
	MOVL R15, 9*4+off+448(dst);  \
	MOVL R15, 4*4+off+512(dst);  \
	MOVL R15, 8*4+off+576(dst)

#define BLAKE2s_SSE2() \
	PRECOMPUTE_MSG(SP, 16, SI, R8, R9, R10, R11, R12, R13, R14, R15);               \
	ROUND_SSE2(X4, X5, X6, X7, 16(SP), 32(SP), 48(SP), 64(SP), X8);                 \
	ROUND_SSE2(X4, X5, X6, X7, 16+64(SP), 32+64(SP), 48+64(SP), 64+64(SP), X8);     \
	ROUND_SSE2(X4, X5, X6, X7, 16+128(SP), 32+128(SP), 48+128(SP), 64+128(SP), X8); \
	ROUND_SSE2(X4, X5, X6, X7, 16+192(SP), 32+192(SP), 48+192(SP), 64+192(SP), X8); \
	ROUND_SSE2(X4, X5, X6, X7, 16+256(SP), 32+256(SP), 48+256(SP), 64+256(SP), X8); \
	ROUND_SSE2(X4, X5, X6, X7, 16+320(SP), 32+320(SP), 48+320(SP), 64+320(SP), X8); \
	ROUND_SSE2(X4, X5, X6, X7, 16+384(SP), 32+384(SP), 48+384(SP), 64+384(SP), X8); \
	ROUND_SSE2(X4, X5, X6, X7, 16+448(SP), 32+448(SP), 48+448(SP), 64+448(SP), X8); \
	ROUND_SSE2(X4, X5, X6, X7, 16+512(SP), 32+512(SP), 48+512(SP), 64+512(SP), X8); \
	ROUND_SSE2(X4, X5, X6, X7, 16+576(SP), 32+576(SP), 48+576(SP), 64+576(SP), X8)

#define BLAKE2s_SSSE3() \
	PRECOMPUTE_MSG(SP, 16, SI, R8, R9, R10, R11, R12, R13, R14, R15);                          \
	ROUND_SSSE3(X4, X5, X6, X7, 16(SP), 32(SP), 48(SP), 64(SP), X8, X13, X14);                 \
	ROUND_SSSE3(X4, X5, X6, X7, 16+64(SP), 32+64(SP), 48+64(SP), 64+64(SP), X8, X13, X14);     \
	ROUND_SSSE3(X4, X5, X6, X7, 16+128(SP), 32+128(SP), 48+128(SP), 64+128(SP), X8, X13, X14); \
	ROUND_SSSE3(X4, X5, X6, X7, 16+192(SP), 32+192(SP), 48+192(SP), 64+192(SP), X8, X13, X14); \
	ROUND_SSSE3(X4, X5, X6, X7, 16+256(SP), 32+256(SP), 48+256(SP), 64+256(SP), X8, X13, X14); \
	ROUND_SSSE3(X4, X5, X6, X7, 16+320(SP), 32+320(SP), 48+320(SP), 64+320(SP), X8, X13, X14); \
	ROUND_SSSE3(X4, X5, X6, X7, 16+384(SP), 32+384(SP), 48+384(SP), 64+384(SP), X8, X13, X14); \
	ROUND_SSSE3(X4, X5, X6, X7, 16+448(SP), 32+448(SP), 48+448(SP), 64+448(SP), X8, X13, X14); \
	ROUND_SSSE3(X4, X5, X6, X7, 16+512(SP), 32+512(SP), 48+512(SP), 64+512(SP), X8, X13, X14); \
	ROUND_SSSE3(X4, X5, X6, X7, 16+576(SP), 32+576(SP), 48+576(SP), 64+576(SP), X8, X13, X14)

#define BLAKE2s_SSE4() \
	LOAD_MSG_SSE4(X8, X9, X10, X11, SI, 0, 2, 4, 6, 1, 3, 5, 7, 8, 10, 12, 14, 9, 11, 13, 15); \
	ROUND_SSSE3(X4, X5, X6, X7, X8, X9, X10, X11, X8, X13, X14);                               \
	LOAD_MSG_SSE4(X8, X9, X10, X11, SI, 14, 4, 9, 13, 10, 8, 15, 6, 1, 0, 11, 5, 12, 2, 7, 3); \
	ROUND_SSSE3(X4, X5, X6, X7, X8, X9, X10, X11, X8, X13, X14);                               \
	LOAD_MSG_SSE4(X8, X9, X10, X11, SI, 11, 12, 5, 15, 8, 0, 2, 13, 10, 3, 7, 9, 14, 6, 1, 4); \
	ROUND_SSSE3(X4, X5, X6, X7, X8, X9, X10, X11, X8, X13, X14);                               \
	LOAD_MSG_SSE4(X8, X9, X10, X11, SI, 7, 3, 13, 11, 9, 1, 12, 14, 2, 5, 4, 15, 6, 10, 0, 8); \
	ROUND_SSSE3(X4, X5, X6, X7, X8, X9, X10, X11, X8, X13, X14);                               \
	LOAD_MSG_SSE4(X8, X9, X10, X11, SI, 9, 5, 2, 10, 0, 7, 4, 15, 14, 11, 6, 3, 1, 12, 8, 13); \
	ROUND_SSSE3(X4, X5, X6, X7, X8, X9, X10, X11, X8, X13, X14);                               \
	LOAD_MSG_SSE4(X8, X9, X10, X11, SI, 2, 6, 0, 8, 12, 10, 11, 3, 4, 7, 15, 1, 13, 5, 14, 9); \
	ROUND_SSSE3(X4, X5, X6, X7, X8, X9, X10, X11, X8, X13, X14);                               \
	LOAD_MSG_SSE4(X8, X9, X10, X11, SI, 12, 1, 14, 4, 5, 15, 13, 10, 0, 6, 9, 8, 7, 3, 2, 11); \
	ROUND_SSSE3(X4, X5, X6, X7, X8, X9, X10, X11, X8, X13, X14);                               \
	LOAD_MSG_SSE4(X8, X9, X10, X11, SI, 13, 7, 12, 3, 11, 14, 1, 9, 5, 15, 8, 2, 0, 4, 6, 10); \
	ROUND_SSSE3(X4, X5, X6, X7, X8, X9, X10, X11, X8, X13, X14);                               \
	LOAD_MSG_SSE4(X8, X9, X10, X11, SI, 6, 14, 11, 0, 15, 9, 3, 8, 12, 13, 1, 10, 2, 7, 4, 5); \
	ROUND_SSSE3(X4, X5, X6, X7, X8, X9, X10, X11, X8, X13, X14);                               \
	LOAD_MSG_SSE4(X8, X9, X10, X11, SI, 10, 8, 7, 1, 2, 4, 6, 5, 15, 9, 3, 13, 11, 14, 12, 0); \
	ROUND_SSSE3(X4, X5, X6, X7, X8, X9, X10, X11, X8, X13, X14)

#define HASH_BLOCKS(h, c, flag, blocks_base, blocks_len, BLAKE2s_FUNC) \
	MOVQ  h, AX;                   \
	MOVQ  c, BX;                   \
	MOVL  flag, CX;                \
	MOVQ  blocks_base, SI;         \
	MOVQ  blocks_len, DX;          \
	                               \
	MOVQ  SP, BP;                  \
	MOVQ  SP, R9;                  \
	ADDQ  $15, R9;                 \
	ANDQ  $~15, R9;                \
	MOVQ  R9, SP;                  \
	                               \
	MOVQ  0(BX), R9;               \
	MOVQ  R9, 0(SP);               \
	XORQ  R9, R9;                  \
	MOVQ  R9, 8(SP);               \
	MOVL  CX, 8(SP);               \
	                               \
	MOVOU 0(AX), X0;               \
	MOVOU 16(AX), X1;              \
	MOVOU iv0<>(SB), X2;           \
	MOVOU iv1<>(SB), X3            \
	                               \
	MOVOU counter<>(SB), X12;      \
	MOVOU rol16<>(SB), X13;        \
	MOVOU rol8<>(SB), X14;         \
	MOVO  0(SP), X15;              \
	                               \
	loop:                          \
	MOVO  X0, X4;                  \
	MOVO  X1, X5;                  \
	MOVO  X2, X6;                  \
	MOVO  X3, X7;                  \
	                               \
	PADDQ X12, X15;                \
	PXOR  X15, X7;                 \
	                               \
	BLAKE2s_FUNC();                \
	                               \
	PXOR  X4, X0;                  \
	PXOR  X5, X1;                  \
	PXOR  X6, X0;                  \
	PXOR  X7, X1;                  \
	                               \
	LEAQ  64(SI), SI;              \
	SUBQ  $64, DX;                 \
	JNE   loop;                    \
	                               \
	MOVO  X15, 0(SP);              \
	MOVQ  0(SP), R9;               \
	MOVQ  R9, 0(BX);               \
	                               \
	MOVOU X0, 0(AX);               \
	MOVOU X1, 16(AX);              \
	                               \
	MOVQ  BP, SP

// func hashBlocksSSE2(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte)
TEXT ·hashBlocksSSE2(SB), 0, $672-48 // frame = 656 + 16 byte alignment
	HASH_BLOCKS(h+0(FP), c+8(FP), flag+16(FP), blocks_base+24(FP), blocks_len+32(FP), BLAKE2s_SSE2)
	RET

// func hashBlocksSSSE3(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte)
TEXT ·hashBlocksSSSE3(SB), 0, $672-48 // frame = 656 + 16 byte alignment
	HASH_BLOCKS(h+0(FP), c+8(FP), flag+16(FP), blocks_base+24(FP), blocks_len+32(FP), BLAKE2s_SSSE3)
	RET

// func hashBlocksSSE4(h *[8]uint32, c *[2]uint32, flag uint32, blocks []byte)
TEXT ·hashBlocksSSE4(SB), 0, $32-48 // frame = 16 + 16 byte alignment
	HASH_BLOCKS(h+0(FP), c+8(FP), flag+16(FP), blocks_base+24(FP), blocks_len+32(FP), BLAKE2s_SSE4)
	RET

// func supportSSE4() bool
TEXT ·supportSSE4(SB), 4, $0-1
	MOVL $1, AX
	CPUID
	SHRL $19, CX       // Bit 19 indicates SSE4.1.
	ANDL $1, CX
	MOVB CX, ret+0(FP)
	RET

// func supportSSSE3() bool
TEXT ·supportSSSE3(SB), 4, $0-1
	MOVL $1, AX
	CPUID
	MOVL CX, BX
	ANDL $0x1, BX      // Bit zero indicates SSE3 support.
	JZ   FALSE
	ANDL $0x200, CX    // Bit nine indicates SSSE3 support.
	JZ   FALSE
	MOVB $1, ret+0(FP)
	RET

FALSE:
	MOVB $0, ret+0(FP)
	RET
