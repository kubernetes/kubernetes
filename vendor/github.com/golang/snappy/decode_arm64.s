// Copyright 2020 The Go Authors. All rights reserved.
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
//	- R2	scratch
//	- R3	scratch
//	- R4	length or x
//	- R5	offset
//	- R6	&src[s]
//	- R7	&dst[d]
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
// The d variable is implicitly R7 - R8,  and len(dst)-d is R10 - R7.
// The s variable is implicitly R6 - R11, and len(src)-s is R13 - R6.
TEXT ·decode(SB), NOSPLIT, $56-56
	// Initialize R6, R7 and R8-R13.
	MOVD dst_base+0(FP), R8
	MOVD dst_len+8(FP), R9
	MOVD R8, R7
	MOVD R8, R10
	ADD  R9, R10, R10
	MOVD src_base+24(FP), R11
	MOVD src_len+32(FP), R12
	MOVD R11, R6
	MOVD R11, R13
	ADD  R12, R13, R13

loop:
	// for s < len(src)
	CMP R13, R6
	BEQ end

	// R4 = uint32(src[s])
	//
	// switch src[s] & 0x03
	MOVBU (R6), R4
	MOVW  R4, R3
	ANDW  $3, R3
	MOVW  $1, R1
	CMPW  R1, R3
	BGE   tagCopy

	// ----------------------------------------
	// The code below handles literal tags.

	// case tagLiteral:
	// x := uint32(src[s] >> 2)
	// switch
	MOVW $60, R1
	LSRW $2, R4, R4
	CMPW R4, R1
	BLS  tagLit60Plus

	// case x < 60:
	// s++
	ADD $1, R6, R6

doLit:
	// This is the end of the inner "switch", when we have a literal tag.
	//
	// We assume that R4 == x and x fits in a uint32, where x is the variable
	// used in the pure Go decode_other.go code.

	// length = int(x) + 1
	//
	// Unlike the pure Go code, we don't need to check if length <= 0 because
	// R4 can hold 64 bits, so the increment cannot overflow.
	ADD $1, R4, R4

	// Prepare to check if copying length bytes will run past the end of dst or
	// src.
	//
	// R2 = len(dst) - d
	// R3 = len(src) - s
	MOVD R10, R2
	SUB  R7, R2, R2
	MOVD R13, R3
	SUB  R6, R3, R3

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
	CMP $16, R4
	BGT callMemmove
	CMP $16, R2
	BLT callMemmove
	CMP $16, R3
	BLT callMemmove

	// !!! Implement the copy from src to dst as a 16-byte load and store.
	// (Decode's documentation says that dst and src must not overlap.)
	//
	// This always copies 16 bytes, instead of only length bytes, but that's
	// OK. If the input is a valid Snappy encoding then subsequent iterations
	// will fix up the overrun. Otherwise, Decode returns a nil []byte (and a
	// non-nil error), so the overrun will be ignored.
	//
	// Note that on arm64, it is legal and cheap to issue unaligned 8-byte or
	// 16-byte loads and stores. This technique probably wouldn't be as
	// effective on architectures that are fussier about alignment.
	LDP 0(R6), (R14, R15)
	STP (R14, R15), 0(R7)

	// d += length
	// s += length
	ADD R4, R7, R7
	ADD R4, R6, R6
	B   loop

callMemmove:
	// if length > len(dst)-d || length > len(src)-s { etc }
	CMP R2, R4
	BGT errCorrupt
	CMP R3, R4
	BGT errCorrupt

	// copy(dst[d:], src[s:s+length])
	//
	// This means calling runtime·memmove(&dst[d], &src[s], length), so we push
	// R7, R6 and R4 as arguments. Coincidentally, we also need to spill those
	// three registers to the stack, to save local variables across the CALL.
	MOVD R7, 8(RSP)
	MOVD R6, 16(RSP)
	MOVD R4, 24(RSP)
	MOVD R7, 32(RSP)
	MOVD R6, 40(RSP)
	MOVD R4, 48(RSP)
	CALL runtime·memmove(SB)

	// Restore local variables: unspill registers from the stack and
	// re-calculate R8-R13.
	MOVD 32(RSP), R7
	MOVD 40(RSP), R6
	MOVD 48(RSP), R4
	MOVD dst_base+0(FP), R8
	MOVD dst_len+8(FP), R9
	MOVD R8, R10
	ADD  R9, R10, R10
	MOVD src_base+24(FP), R11
	MOVD src_len+32(FP), R12
	MOVD R11, R13
	ADD  R12, R13, R13

	// d += length
	// s += length
	ADD R4, R7, R7
	ADD R4, R6, R6
	B   loop

tagLit60Plus:
	// !!! This fragment does the
	//
	// s += x - 58; if uint(s) > uint(len(src)) { etc }
	//
	// checks. In the asm version, we code it once instead of once per switch case.
	ADD  R4, R6, R6
	SUB  $58, R6, R6
	MOVD R6, R3
	SUB  R11, R3, R3
	CMP  R12, R3
	BGT  errCorrupt

	// case x == 60:
	MOVW $61, R1
	CMPW R1, R4
	BEQ  tagLit61
	BGT  tagLit62Plus

	// x = uint32(src[s-1])
	MOVBU -1(R6), R4
	B     doLit

tagLit61:
	// case x == 61:
	// x = uint32(src[s-2]) | uint32(src[s-1])<<8
	MOVHU -2(R6), R4
	B     doLit

tagLit62Plus:
	CMPW $62, R4
	BHI  tagLit63

	// case x == 62:
	// x = uint32(src[s-3]) | uint32(src[s-2])<<8 | uint32(src[s-1])<<16
	MOVHU -3(R6), R4
	MOVBU -1(R6), R3
	ORR   R3<<16, R4
	B     doLit

tagLit63:
	// case x == 63:
	// x = uint32(src[s-4]) | uint32(src[s-3])<<8 | uint32(src[s-2])<<16 | uint32(src[s-1])<<24
	MOVWU -4(R6), R4
	B     doLit

	// The code above handles literal tags.
	// ----------------------------------------
	// The code below handles copy tags.

tagCopy4:
	// case tagCopy4:
	// s += 5
	ADD $5, R6, R6

	// if uint(s) > uint(len(src)) { etc }
	MOVD R6, R3
	SUB  R11, R3, R3
	CMP  R12, R3
	BGT  errCorrupt

	// length = 1 + int(src[s-5])>>2
	MOVD $1, R1
	ADD  R4>>2, R1, R4

	// offset = int(uint32(src[s-4]) | uint32(src[s-3])<<8 | uint32(src[s-2])<<16 | uint32(src[s-1])<<24)
	MOVWU -4(R6), R5
	B     doCopy

tagCopy2:
	// case tagCopy2:
	// s += 3
	ADD $3, R6, R6

	// if uint(s) > uint(len(src)) { etc }
	MOVD R6, R3
	SUB  R11, R3, R3
	CMP  R12, R3
	BGT  errCorrupt

	// length = 1 + int(src[s-3])>>2
	MOVD $1, R1
	ADD  R4>>2, R1, R4

	// offset = int(uint32(src[s-2]) | uint32(src[s-1])<<8)
	MOVHU -2(R6), R5
	B     doCopy

tagCopy:
	// We have a copy tag. We assume that:
	//	- R3 == src[s] & 0x03
	//	- R4 == src[s]
	CMP $2, R3
	BEQ tagCopy2
	BGT tagCopy4

	// case tagCopy1:
	// s += 2
	ADD $2, R6, R6

	// if uint(s) > uint(len(src)) { etc }
	MOVD R6, R3
	SUB  R11, R3, R3
	CMP  R12, R3
	BGT  errCorrupt

	// offset = int(uint32(src[s-2])&0xe0<<3 | uint32(src[s-1]))
	MOVD  R4, R5
	AND   $0xe0, R5
	MOVBU -1(R6), R3
	ORR   R5<<3, R3, R5

	// length = 4 + int(src[s-2])>>2&0x7
	MOVD $7, R1
	AND  R4>>2, R1, R4
	ADD  $4, R4, R4

doCopy:
	// This is the end of the outer "switch", when we have a copy tag.
	//
	// We assume that:
	//	- R4 == length && R4 > 0
	//	- R5 == offset

	// if offset <= 0 { etc }
	MOVD $0, R1
	CMP  R1, R5
	BLE  errCorrupt

	// if d < offset { etc }
	MOVD R7, R3
	SUB  R8, R3, R3
	CMP  R5, R3
	BLT  errCorrupt

	// if length > len(dst)-d { etc }
	MOVD R10, R3
	SUB  R7, R3, R3
	CMP  R3, R4
	BGT  errCorrupt

	// forwardCopy(dst[d:d+length], dst[d-offset:]); d += length
	//
	// Set:
	//	- R14 = len(dst)-d
	//	- R15 = &dst[d-offset]
	MOVD R10, R14
	SUB  R7, R14, R14
	MOVD R7, R15
	SUB  R5, R15, R15

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
	CMP  $16, R4
	BGT  slowForwardCopy
	CMP  $8, R5
	BLT  slowForwardCopy
	CMP  $16, R14
	BLT  slowForwardCopy
	MOVD 0(R15), R2
	MOVD R2, 0(R7)
	MOVD 8(R15), R3
	MOVD R3, 8(R7)
	ADD  R4, R7, R7
	B    loop

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
	SUB $10, R14, R14
	CMP R14, R4
	BGT verySlowForwardCopy

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
	CMP  $8, R5
	BGE  fixUpSlowForwardCopy
	MOVD (R15), R3
	MOVD R3, (R7)
	SUB  R5, R4, R4
	ADD  R5, R7, R7
	ADD  R5, R5, R5
	B    makeOffsetAtLeast8

fixUpSlowForwardCopy:
	// !!! Add length (which might be negative now) to d (implied by R7 being
	// &dst[d]) so that d ends up at the right place when we jump back to the
	// top of the loop. Before we do that, though, we save R7 to R2 so that, if
	// length is positive, copying the remaining length bytes will write to the
	// right place.
	MOVD R7, R2
	ADD  R4, R7, R7

finishSlowForwardCopy:
	// !!! Repeat 8-byte load/stores until length <= 0. Ending with a negative
	// length means that we overrun, but as above, that will be fixed up by
	// subsequent iterations of the outermost loop.
	MOVD $0, R1
	CMP  R1, R4
	BLE  loop
	MOVD (R15), R3
	MOVD R3, (R2)
	ADD  $8, R15, R15
	ADD  $8, R2, R2
	SUB  $8, R4, R4
	B    finishSlowForwardCopy

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
	MOVB (R15), R3
	MOVB R3, (R7)
	ADD  $1, R15, R15
	ADD  $1, R7, R7
	SUB  $1, R4, R4
	CBNZ R4, verySlowForwardCopy
	B    loop

	// The code above handles copy tags.
	// ----------------------------------------

end:
	// This is the end of the "for s < len(src)".
	//
	// if d != len(dst) { etc }
	CMP R10, R7
	BNE errCorrupt

	// return 0
	MOVD $0, ret+48(FP)
	RET

errCorrupt:
	// return decodeErrCodeCorrupt
	MOVD $1, R2
	MOVD R2, ret+48(FP)
	RET
