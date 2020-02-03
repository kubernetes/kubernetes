// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine
// +build gc
// +build !noasm

#include "textflag.h"

// The asm code generally follows the pure Go code in decode_other.go, except
// where marked with a "!!!".

// func decode(dst, src []byte) int
//
// All local variables fit into registers. The non-zero stack size is only to
// spill registers and push args when issuing a CALL. The register allocation:
//	- AX	scratch
//	- BX	scratch
//	- CX	length or x
//	- DX	offset
//	- SI	&src[s]
//	- DI	&dst[d]
//	+ R8	dst_base
//	+ R9	dst_len
//	+ R10	dst_base + dst_len
//	+ R11	src_base
//	+ R12	src_len
//	+ R13	src_base + src_len
//	- R14	used by doCopy
//	- R15	used by doCopy
//
// The registers R8-R13 (marked with a "+") are set at the start of the
// function, and after a CALL returns, and are not otherwise modified.
//
// The d variable is implicitly DI - R8,  and len(dst)-d is R10 - DI.
// The s variable is implicitly SI - R11, and len(src)-s is R13 - SI.
TEXT ·decode(SB), NOSPLIT, $48-56
	// Initialize SI, DI and R8-R13.
	MOVQ dst_base+0(FP), R8
	MOVQ dst_len+8(FP), R9
	MOVQ R8, DI
	MOVQ R8, R10
	ADDQ R9, R10
	MOVQ src_base+24(FP), R11
	MOVQ src_len+32(FP), R12
	MOVQ R11, SI
	MOVQ R11, R13
	ADDQ R12, R13

loop:
	// for s < len(src)
	CMPQ SI, R13
	JEQ  end

	// CX = uint32(src[s])
	//
	// switch src[s] & 0x03
	MOVBLZX (SI), CX
	MOVL    CX, BX
	ANDL    $3, BX
	CMPL    BX, $1
	JAE     tagCopy

	// ----------------------------------------
	// The code below handles literal tags.

	// case tagLiteral:
	// x := uint32(src[s] >> 2)
	// switch
	SHRL $2, CX
	CMPL CX, $60
	JAE  tagLit60Plus

	// case x < 60:
	// s++
	INCQ SI

doLit:
	// This is the end of the inner "switch", when we have a literal tag.
	//
	// We assume that CX == x and x fits in a uint32, where x is the variable
	// used in the pure Go decode_other.go code.

	// length = int(x) + 1
	//
	// Unlike the pure Go code, we don't need to check if length <= 0 because
	// CX can hold 64 bits, so the increment cannot overflow.
	INCQ CX

	// Prepare to check if copying length bytes will run past the end of dst or
	// src.
	//
	// AX = len(dst) - d
	// BX = len(src) - s
	MOVQ R10, AX
	SUBQ DI, AX
	MOVQ R13, BX
	SUBQ SI, BX

	// !!! Try a faster technique for short (16 or fewer bytes) copies.
	//
	// if length > 16 || len(dst)-d < 16 || len(src)-s < 16 {
	//   goto callMemmove // Fall back on calling runtime·memmove.
	// }
	//
	// The C++ snappy code calls this TryFastAppend. It also checks len(src)-s
	// against 21 instead of 16, because it cannot assume that all of its input
	// is contiguous in memory and so it needs to leave enough source bytes to
	// read the next tag without refilling buffers, but Go's Decode assumes
	// contiguousness (the src argument is a []byte).
	CMPQ CX, $16
	JGT  callMemmove
	CMPQ AX, $16
	JLT  callMemmove
	CMPQ BX, $16
	JLT  callMemmove

	// !!! Implement the copy from src to dst as a 16-byte load and store.
	// (Decode's documentation says that dst and src must not overlap.)
	//
	// This always copies 16 bytes, instead of only length bytes, but that's
	// OK. If the input is a valid Snappy encoding then subsequent iterations
	// will fix up the overrun. Otherwise, Decode returns a nil []byte (and a
	// non-nil error), so the overrun will be ignored.
	//
	// Note that on amd64, it is legal and cheap to issue unaligned 8-byte or
	// 16-byte loads and stores. This technique probably wouldn't be as
	// effective on architectures that are fussier about alignment.
	MOVOU 0(SI), X0
	MOVOU X0, 0(DI)

	// d += length
	// s += length
	ADDQ CX, DI
	ADDQ CX, SI
	JMP  loop

callMemmove:
	// if length > len(dst)-d || length > len(src)-s { etc }
	CMPQ CX, AX
	JGT  errCorrupt
	CMPQ CX, BX
	JGT  errCorrupt

	// copy(dst[d:], src[s:s+length])
	//
	// This means calling runtime·memmove(&dst[d], &src[s], length), so we push
	// DI, SI and CX as arguments. Coincidentally, we also need to spill those
	// three registers to the stack, to save local variables across the CALL.
	MOVQ DI, 0(SP)
	MOVQ SI, 8(SP)
	MOVQ CX, 16(SP)
	MOVQ DI, 24(SP)
	MOVQ SI, 32(SP)
	MOVQ CX, 40(SP)
	CALL runtime·memmove(SB)

	// Restore local variables: unspill registers from the stack and
	// re-calculate R8-R13.
	MOVQ 24(SP), DI
	MOVQ 32(SP), SI
	MOVQ 40(SP), CX
	MOVQ dst_base+0(FP), R8
	MOVQ dst_len+8(FP), R9
	MOVQ R8, R10
	ADDQ R9, R10
	MOVQ src_base+24(FP), R11
	MOVQ src_len+32(FP), R12
	MOVQ R11, R13
	ADDQ R12, R13

	// d += length
	// s += length
	ADDQ CX, DI
	ADDQ CX, SI
	JMP  loop

tagLit60Plus:
	// !!! This fragment does the
	//
	// s += x - 58; if uint(s) > uint(len(src)) { etc }
	//
	// checks. In the asm version, we code it once instead of once per switch case.
	ADDQ CX, SI
	SUBQ $58, SI
	MOVQ SI, BX
	SUBQ R11, BX
	CMPQ BX, R12
	JA   errCorrupt

	// case x == 60:
	CMPL CX, $61
	JEQ  tagLit61
	JA   tagLit62Plus

	// x = uint32(src[s-1])
	MOVBLZX -1(SI), CX
	JMP     doLit

tagLit61:
	// case x == 61:
	// x = uint32(src[s-2]) | uint32(src[s-1])<<8
	MOVWLZX -2(SI), CX
	JMP     doLit

tagLit62Plus:
	CMPL CX, $62
	JA   tagLit63

	// case x == 62:
	// x = uint32(src[s-3]) | uint32(src[s-2])<<8 | uint32(src[s-1])<<16
	MOVWLZX -3(SI), CX
	MOVBLZX -1(SI), BX
	SHLL    $16, BX
	ORL     BX, CX
	JMP     doLit

tagLit63:
	// case x == 63:
	// x = uint32(src[s-4]) | uint32(src[s-3])<<8 | uint32(src[s-2])<<16 | uint32(src[s-1])<<24
	MOVL -4(SI), CX
	JMP  doLit

// The code above handles literal tags.
// ----------------------------------------
// The code below handles copy tags.

tagCopy4:
	// case tagCopy4:
	// s += 5
	ADDQ $5, SI

	// if uint(s) > uint(len(src)) { etc }
	MOVQ SI, BX
	SUBQ R11, BX
	CMPQ BX, R12
	JA   errCorrupt

	// length = 1 + int(src[s-5])>>2
	SHRQ $2, CX
	INCQ CX

	// offset = int(uint32(src[s-4]) | uint32(src[s-3])<<8 | uint32(src[s-2])<<16 | uint32(src[s-1])<<24)
	MOVLQZX -4(SI), DX
	JMP     doCopy

tagCopy2:
	// case tagCopy2:
	// s += 3
	ADDQ $3, SI

	// if uint(s) > uint(len(src)) { etc }
	MOVQ SI, BX
	SUBQ R11, BX
	CMPQ BX, R12
	JA   errCorrupt

	// length = 1 + int(src[s-3])>>2
	SHRQ $2, CX
	INCQ CX

	// offset = int(uint32(src[s-2]) | uint32(src[s-1])<<8)
	MOVWQZX -2(SI), DX
	JMP     doCopy

tagCopy:
	// We have a copy tag. We assume that:
	//	- BX == src[s] & 0x03
	//	- CX == src[s]
	CMPQ BX, $2
	JEQ  tagCopy2
	JA   tagCopy4

	// case tagCopy1:
	// s += 2
	ADDQ $2, SI

	// if uint(s) > uint(len(src)) { etc }
	MOVQ SI, BX
	SUBQ R11, BX
	CMPQ BX, R12
	JA   errCorrupt

	// offset = int(uint32(src[s-2])&0xe0<<3 | uint32(src[s-1]))
	MOVQ    CX, DX
	ANDQ    $0xe0, DX
	SHLQ    $3, DX
	MOVBQZX -1(SI), BX
	ORQ     BX, DX

	// length = 4 + int(src[s-2])>>2&0x7
	SHRQ $2, CX
	ANDQ $7, CX
	ADDQ $4, CX

doCopy:
	// This is the end of the outer "switch", when we have a copy tag.
	//
	// We assume that:
	//	- CX == length && CX > 0
	//	- DX == offset

	// if offset <= 0 { etc }
	CMPQ DX, $0
	JLE  errCorrupt

	// if d < offset { etc }
	MOVQ DI, BX
	SUBQ R8, BX
	CMPQ BX, DX
	JLT  errCorrupt

	// if length > len(dst)-d { etc }
	MOVQ R10, BX
	SUBQ DI, BX
	CMPQ CX, BX
	JGT  errCorrupt

	// forwardCopy(dst[d:d+length], dst[d-offset:]); d += length
	//
	// Set:
	//	- R14 = len(dst)-d
	//	- R15 = &dst[d-offset]
	MOVQ R10, R14
	SUBQ DI, R14
	MOVQ DI, R15
	SUBQ DX, R15

	// !!! Try a faster technique for short (16 or fewer bytes) forward copies.
	//
	// First, try using two 8-byte load/stores, similar to the doLit technique
	// above. Even if dst[d:d+length] and dst[d-offset:] can overlap, this is
	// still OK if offset >= 8. Note that this has to be two 8-byte load/stores
	// and not one 16-byte load/store, and the first store has to be before the
	// second load, due to the overlap if offset is in the range [8, 16).
	//
	// if length > 16 || offset < 8 || len(dst)-d < 16 {
	//   goto slowForwardCopy
	// }
	// copy 16 bytes
	// d += length
	CMPQ CX, $16
	JGT  slowForwardCopy
	CMPQ DX, $8
	JLT  slowForwardCopy
	CMPQ R14, $16
	JLT  slowForwardCopy
	MOVQ 0(R15), AX
	MOVQ AX, 0(DI)
	MOVQ 8(R15), BX
	MOVQ BX, 8(DI)
	ADDQ CX, DI
	JMP  loop

slowForwardCopy:
	// !!! If the forward copy is longer than 16 bytes, or if offset < 8, we
	// can still try 8-byte load stores, provided we can overrun up to 10 extra
	// bytes. As above, the overrun will be fixed up by subsequent iterations
	// of the outermost loop.
	//
	// The C++ snappy code calls this technique IncrementalCopyFastPath. Its
	// commentary says:
	//
	// ----
	//
	// The main part of this loop is a simple copy of eight bytes at a time
	// until we've copied (at least) the requested amount of bytes.  However,
	// if d and d-offset are less than eight bytes apart (indicating a
	// repeating pattern of length < 8), we first need to expand the pattern in
	// order to get the correct results. For instance, if the buffer looks like
	// this, with the eight-byte <d-offset> and <d> patterns marked as
	// intervals:
	//
	//    abxxxxxxxxxxxx
	//    [------]           d-offset
	//      [------]         d
	//
	// a single eight-byte copy from <d-offset> to <d> will repeat the pattern
	// once, after which we can move <d> two bytes without moving <d-offset>:
	//
	//    ababxxxxxxxxxx
	//    [------]           d-offset
	//        [------]       d
	//
	// and repeat the exercise until the two no longer overlap.
	//
	// This allows us to do very well in the special case of one single byte
	// repeated many times, without taking a big hit for more general cases.
	//
	// The worst case of extra writing past the end of the match occurs when
	// offset == 1 and length == 1; the last copy will read from byte positions
	// [0..7] and write to [4..11], whereas it was only supposed to write to
	// position 1. Thus, ten excess bytes.
	//
	// ----
	//
	// That "10 byte overrun" worst case is confirmed by Go's
	// TestSlowForwardCopyOverrun, which also tests the fixUpSlowForwardCopy
	// and finishSlowForwardCopy algorithm.
	//
	// if length > len(dst)-d-10 {
	//   goto verySlowForwardCopy
	// }
	SUBQ $10, R14
	CMPQ CX, R14
	JGT  verySlowForwardCopy

makeOffsetAtLeast8:
	// !!! As above, expand the pattern so that offset >= 8 and we can use
	// 8-byte load/stores.
	//
	// for offset < 8 {
	//   copy 8 bytes from dst[d-offset:] to dst[d:]
	//   length -= offset
	//   d      += offset
	//   offset += offset
	//   // The two previous lines together means that d-offset, and therefore
	//   // R15, is unchanged.
	// }
	CMPQ DX, $8
	JGE  fixUpSlowForwardCopy
	MOVQ (R15), BX
	MOVQ BX, (DI)
	SUBQ DX, CX
	ADDQ DX, DI
	ADDQ DX, DX
	JMP  makeOffsetAtLeast8

fixUpSlowForwardCopy:
	// !!! Add length (which might be negative now) to d (implied by DI being
	// &dst[d]) so that d ends up at the right place when we jump back to the
	// top of the loop. Before we do that, though, we save DI to AX so that, if
	// length is positive, copying the remaining length bytes will write to the
	// right place.
	MOVQ DI, AX
	ADDQ CX, DI

finishSlowForwardCopy:
	// !!! Repeat 8-byte load/stores until length <= 0. Ending with a negative
	// length means that we overrun, but as above, that will be fixed up by
	// subsequent iterations of the outermost loop.
	CMPQ CX, $0
	JLE  loop
	MOVQ (R15), BX
	MOVQ BX, (AX)
	ADDQ $8, R15
	ADDQ $8, AX
	SUBQ $8, CX
	JMP  finishSlowForwardCopy

verySlowForwardCopy:
	// verySlowForwardCopy is a simple implementation of forward copy. In C
	// parlance, this is a do/while loop instead of a while loop, since we know
	// that length > 0. In Go syntax:
	//
	// for {
	//   dst[d] = dst[d - offset]
	//   d++
	//   length--
	//   if length == 0 {
	//     break
	//   }
	// }
	MOVB (R15), BX
	MOVB BX, (DI)
	INCQ R15
	INCQ DI
	DECQ CX
	JNZ  verySlowForwardCopy
	JMP  loop

// The code above handles copy tags.
// ----------------------------------------

end:
	// This is the end of the "for s < len(src)".
	//
	// if d != len(dst) { etc }
	CMPQ DI, R10
	JNE  errCorrupt

	// return 0
	MOVQ $0, ret+48(FP)
	RET

errCorrupt:
	// return decodeErrCodeCorrupt
	MOVQ $1, ret+48(FP)
	RET
