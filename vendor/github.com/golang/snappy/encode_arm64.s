// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine
// +build gc
// +build !noasm

#include "textflag.h"

// The asm code generally follows the pure Go code in encode_other.go, except
// where marked with a "!!!".

// ----------------------------------------------------------------------------

// func emitLiteral(dst, lit []byte) int
//
// All local variables fit into registers. The register allocation:
//	- R3	len(lit)
//	- R4	n
//	- R6	return value
//	- R8	&dst[i]
//	- R10	&lit[0]
//
// The 32 bytes of stack space is to call runtime·memmove.
//
// The unusual register allocation of local variables, such as R10 for the
// source pointer, matches the allocation used at the call site in encodeBlock,
// which makes it easier to manually inline this function.
TEXT ·emitLiteral(SB), NOSPLIT, $32-56
	MOVD dst_base+0(FP), R8
	MOVD lit_base+24(FP), R10
	MOVD lit_len+32(FP), R3
	MOVD R3, R6
	MOVW R3, R4
	SUBW $1, R4, R4

	CMPW $60, R4
	BLT  oneByte
	CMPW $256, R4
	BLT  twoBytes

threeBytes:
	MOVD $0xf4, R2
	MOVB R2, 0(R8)
	MOVW R4, 1(R8)
	ADD  $3, R8, R8
	ADD  $3, R6, R6
	B    memmove

twoBytes:
	MOVD $0xf0, R2
	MOVB R2, 0(R8)
	MOVB R4, 1(R8)
	ADD  $2, R8, R8
	ADD  $2, R6, R6
	B    memmove

oneByte:
	LSLW $2, R4, R4
	MOVB R4, 0(R8)
	ADD  $1, R8, R8
	ADD  $1, R6, R6

memmove:
	MOVD R6, ret+48(FP)

	// copy(dst[i:], lit)
	//
	// This means calling runtime·memmove(&dst[i], &lit[0], len(lit)), so we push
	// R8, R10 and R3 as arguments.
	MOVD R8, 8(RSP)
	MOVD R10, 16(RSP)
	MOVD R3, 24(RSP)
	CALL runtime·memmove(SB)
	RET

// ----------------------------------------------------------------------------

// func emitCopy(dst []byte, offset, length int) int
//
// All local variables fit into registers. The register allocation:
//	- R3	length
//	- R7	&dst[0]
//	- R8	&dst[i]
//	- R11	offset
//
// The unusual register allocation of local variables, such as R11 for the
// offset, matches the allocation used at the call site in encodeBlock, which
// makes it easier to manually inline this function.
TEXT ·emitCopy(SB), NOSPLIT, $0-48
	MOVD dst_base+0(FP), R8
	MOVD R8, R7
	MOVD offset+24(FP), R11
	MOVD length+32(FP), R3

loop0:
	// for length >= 68 { etc }
	CMPW $68, R3
	BLT  step1

	// Emit a length 64 copy, encoded as 3 bytes.
	MOVD $0xfe, R2
	MOVB R2, 0(R8)
	MOVW R11, 1(R8)
	ADD  $3, R8, R8
	SUB  $64, R3, R3
	B    loop0

step1:
	// if length > 64 { etc }
	CMP $64, R3
	BLE step2

	// Emit a length 60 copy, encoded as 3 bytes.
	MOVD $0xee, R2
	MOVB R2, 0(R8)
	MOVW R11, 1(R8)
	ADD  $3, R8, R8
	SUB  $60, R3, R3

step2:
	// if length >= 12 || offset >= 2048 { goto step3 }
	CMP  $12, R3
	BGE  step3
	CMPW $2048, R11
	BGE  step3

	// Emit the remaining copy, encoded as 2 bytes.
	MOVB R11, 1(R8)
	LSRW $3, R11, R11
	AND  $0xe0, R11, R11
	SUB  $4, R3, R3
	LSLW $2, R3
	AND  $0xff, R3, R3
	ORRW R3, R11, R11
	ORRW $1, R11, R11
	MOVB R11, 0(R8)
	ADD  $2, R8, R8

	// Return the number of bytes written.
	SUB  R7, R8, R8
	MOVD R8, ret+40(FP)
	RET

step3:
	// Emit the remaining copy, encoded as 3 bytes.
	SUB  $1, R3, R3
	AND  $0xff, R3, R3
	LSLW $2, R3, R3
	ORRW $2, R3, R3
	MOVB R3, 0(R8)
	MOVW R11, 1(R8)
	ADD  $3, R8, R8

	// Return the number of bytes written.
	SUB  R7, R8, R8
	MOVD R8, ret+40(FP)
	RET

// ----------------------------------------------------------------------------

// func extendMatch(src []byte, i, j int) int
//
// All local variables fit into registers. The register allocation:
//	- R6	&src[0]
//	- R7	&src[j]
//	- R13	&src[len(src) - 8]
//	- R14	&src[len(src)]
//	- R15	&src[i]
//
// The unusual register allocation of local variables, such as R15 for a source
// pointer, matches the allocation used at the call site in encodeBlock, which
// makes it easier to manually inline this function.
TEXT ·extendMatch(SB), NOSPLIT, $0-48
	MOVD src_base+0(FP), R6
	MOVD src_len+8(FP), R14
	MOVD i+24(FP), R15
	MOVD j+32(FP), R7
	ADD  R6, R14, R14
	ADD  R6, R15, R15
	ADD  R6, R7, R7
	MOVD R14, R13
	SUB  $8, R13, R13

cmp8:
	// As long as we are 8 or more bytes before the end of src, we can load and
	// compare 8 bytes at a time. If those 8 bytes are equal, repeat.
	CMP  R13, R7
	BHI  cmp1
	MOVD (R15), R3
	MOVD (R7), R4
	CMP  R4, R3
	BNE  bsf
	ADD  $8, R15, R15
	ADD  $8, R7, R7
	B    cmp8

bsf:
	// If those 8 bytes were not equal, XOR the two 8 byte values, and return
	// the index of the first byte that differs.
	// RBIT reverses the bit order, then CLZ counts the leading zeros, the
	// combination of which finds the least significant bit which is set.
	// The arm64 architecture is little-endian, and the shift by 3 converts
	// a bit index to a byte index.
	EOR  R3, R4, R4
	RBIT R4, R4
	CLZ  R4, R4
	ADD  R4>>3, R7, R7

	// Convert from &src[ret] to ret.
	SUB  R6, R7, R7
	MOVD R7, ret+40(FP)
	RET

cmp1:
	// In src's tail, compare 1 byte at a time.
	CMP  R7, R14
	BLS  extendMatchEnd
	MOVB (R15), R3
	MOVB (R7), R4
	CMP  R4, R3
	BNE  extendMatchEnd
	ADD  $1, R15, R15
	ADD  $1, R7, R7
	B    cmp1

extendMatchEnd:
	// Convert from &src[ret] to ret.
	SUB  R6, R7, R7
	MOVD R7, ret+40(FP)
	RET

// ----------------------------------------------------------------------------

// func encodeBlock(dst, src []byte) (d int)
//
// All local variables fit into registers, other than "var table". The register
// allocation:
//	- R3	.	.
//	- R4	.	.
//	- R5	64	shift
//	- R6	72	&src[0], tableSize
//	- R7	80	&src[s]
//	- R8	88	&dst[d]
//	- R9	96	sLimit
//	- R10	.	&src[nextEmit]
//	- R11	104	prevHash, currHash, nextHash, offset
//	- R12	112	&src[base], skip
//	- R13	.	&src[nextS], &src[len(src) - 8]
//	- R14	.	len(src), bytesBetweenHashLookups, &src[len(src)], x
//	- R15	120	candidate
//	- R16	.	hash constant, 0x1e35a7bd
//	- R17	.	&table
//	- .  	128	table
//
// The second column (64, 72, etc) is the stack offset to spill the registers
// when calling other functions. We could pack this slightly tighter, but it's
// simpler to have a dedicated spill map independent of the function called.
//
// "var table [maxTableSize]uint16" takes up 32768 bytes of stack space. An
// extra 64 bytes, to call other functions, and an extra 64 bytes, to spill
// local variables (registers) during calls gives 32768 + 64 + 64 = 32896.
TEXT ·encodeBlock(SB), 0, $32896-56
	MOVD dst_base+0(FP), R8
	MOVD src_base+24(FP), R7
	MOVD src_len+32(FP), R14

	// shift, tableSize := uint32(32-8), 1<<8
	MOVD  $24, R5
	MOVD  $256, R6
	MOVW  $0xa7bd, R16
	MOVKW $(0x1e35<<16), R16

calcShift:
	// for ; tableSize < maxTableSize && tableSize < len(src); tableSize *= 2 {
	//	shift--
	// }
	MOVD $16384, R2
	CMP  R2, R6
	BGE  varTable
	CMP  R14, R6
	BGE  varTable
	SUB  $1, R5, R5
	LSL  $1, R6, R6
	B    calcShift

varTable:
	// var table [maxTableSize]uint16
	//
	// In the asm code, unlike the Go code, we can zero-initialize only the
	// first tableSize elements. Each uint16 element is 2 bytes and each
	// iterations writes 64 bytes, so we can do only tableSize/32 writes
	// instead of the 2048 writes that would zero-initialize all of table's
	// 32768 bytes. This clear could overrun the first tableSize elements, but
	// it won't overrun the allocated stack size.
	ADD  $128, RSP, R17
	MOVD R17, R4

	// !!! R6 = &src[tableSize]
	ADD R6<<1, R17, R6

memclr:
	STP.P (ZR, ZR), 64(R4)
	STP   (ZR, ZR), -48(R4)
	STP   (ZR, ZR), -32(R4)
	STP   (ZR, ZR), -16(R4)
	CMP   R4, R6
	BHI   memclr

	// !!! R6 = &src[0]
	MOVD R7, R6

	// sLimit := len(src) - inputMargin
	MOVD R14, R9
	SUB  $15, R9, R9

	// !!! Pre-emptively spill R5, R6 and R9 to the stack. Their values don't
	// change for the rest of the function.
	MOVD R5, 64(RSP)
	MOVD R6, 72(RSP)
	MOVD R9, 96(RSP)

	// nextEmit := 0
	MOVD R6, R10

	// s := 1
	ADD $1, R7, R7

	// nextHash := hash(load32(src, s), shift)
	MOVW 0(R7), R11
	MULW R16, R11, R11
	LSRW R5, R11, R11

outer:
	// for { etc }

	// skip := 32
	MOVD $32, R12

	// nextS := s
	MOVD R7, R13

	// candidate := 0
	MOVD $0, R15

inner0:
	// for { etc }

	// s := nextS
	MOVD R13, R7

	// bytesBetweenHashLookups := skip >> 5
	MOVD R12, R14
	LSR  $5, R14, R14

	// nextS = s + bytesBetweenHashLookups
	ADD R14, R13, R13

	// skip += bytesBetweenHashLookups
	ADD R14, R12, R12

	// if nextS > sLimit { goto emitRemainder }
	MOVD R13, R3
	SUB  R6, R3, R3
	CMP  R9, R3
	BHI  emitRemainder

	// candidate = int(table[nextHash])
	MOVHU 0(R17)(R11<<1), R15

	// table[nextHash] = uint16(s)
	MOVD R7, R3
	SUB  R6, R3, R3

	MOVH R3, 0(R17)(R11<<1)

	// nextHash = hash(load32(src, nextS), shift)
	MOVW 0(R13), R11
	MULW R16, R11
	LSRW R5, R11, R11

	// if load32(src, s) != load32(src, candidate) { continue } break
	MOVW 0(R7), R3
	MOVW (R6)(R15*1), R4
	CMPW R4, R3
	BNE  inner0

fourByteMatch:
	// As per the encode_other.go code:
	//
	// A 4-byte match has been found. We'll later see etc.

	// !!! Jump to a fast path for short (<= 16 byte) literals. See the comment
	// on inputMargin in encode.go.
	MOVD R7, R3
	SUB  R10, R3, R3
	CMP  $16, R3
	BLE  emitLiteralFastPath

	// ----------------------------------------
	// Begin inline of the emitLiteral call.
	//
	// d += emitLiteral(dst[d:], src[nextEmit:s])

	MOVW R3, R4
	SUBW $1, R4, R4

	MOVW $60, R2
	CMPW R2, R4
	BLT  inlineEmitLiteralOneByte
	MOVW $256, R2
	CMPW R2, R4
	BLT  inlineEmitLiteralTwoBytes

inlineEmitLiteralThreeBytes:
	MOVD $0xf4, R1
	MOVB R1, 0(R8)
	MOVW R4, 1(R8)
	ADD  $3, R8, R8
	B    inlineEmitLiteralMemmove

inlineEmitLiteralTwoBytes:
	MOVD $0xf0, R1
	MOVB R1, 0(R8)
	MOVB R4, 1(R8)
	ADD  $2, R8, R8
	B    inlineEmitLiteralMemmove

inlineEmitLiteralOneByte:
	LSLW $2, R4, R4
	MOVB R4, 0(R8)
	ADD  $1, R8, R8

inlineEmitLiteralMemmove:
	// Spill local variables (registers) onto the stack; call; unspill.
	//
	// copy(dst[i:], lit)
	//
	// This means calling runtime·memmove(&dst[i], &lit[0], len(lit)), so we push
	// R8, R10 and R3 as arguments.
	MOVD R8, 8(RSP)
	MOVD R10, 16(RSP)
	MOVD R3, 24(RSP)

	// Finish the "d +=" part of "d += emitLiteral(etc)".
	ADD   R3, R8, R8
	MOVD  R7, 80(RSP)
	MOVD  R8, 88(RSP)
	MOVD  R15, 120(RSP)
	CALL  runtime·memmove(SB)
	MOVD  64(RSP), R5
	MOVD  72(RSP), R6
	MOVD  80(RSP), R7
	MOVD  88(RSP), R8
	MOVD  96(RSP), R9
	MOVD  120(RSP), R15
	ADD   $128, RSP, R17
	MOVW  $0xa7bd, R16
	MOVKW $(0x1e35<<16), R16
	B     inner1

inlineEmitLiteralEnd:
	// End inline of the emitLiteral call.
	// ----------------------------------------

emitLiteralFastPath:
	// !!! Emit the 1-byte encoding "uint8(len(lit)-1)<<2".
	MOVB R3, R4
	SUBW $1, R4, R4
	AND  $0xff, R4, R4
	LSLW $2, R4, R4
	MOVB R4, (R8)
	ADD  $1, R8, R8

	// !!! Implement the copy from lit to dst as a 16-byte load and store.
	// (Encode's documentation says that dst and src must not overlap.)
	//
	// This always copies 16 bytes, instead of only len(lit) bytes, but that's
	// OK. Subsequent iterations will fix up the overrun.
	//
	// Note that on arm64, it is legal and cheap to issue unaligned 8-byte or
	// 16-byte loads and stores. This technique probably wouldn't be as
	// effective on architectures that are fussier about alignment.
	LDP 0(R10), (R0, R1)
	STP (R0, R1), 0(R8)
	ADD R3, R8, R8

inner1:
	// for { etc }

	// base := s
	MOVD R7, R12

	// !!! offset := base - candidate
	MOVD R12, R11
	SUB  R15, R11, R11
	SUB  R6, R11, R11

	// ----------------------------------------
	// Begin inline of the extendMatch call.
	//
	// s = extendMatch(src, candidate+4, s+4)

	// !!! R14 = &src[len(src)]
	MOVD src_len+32(FP), R14
	ADD  R6, R14, R14

	// !!! R13 = &src[len(src) - 8]
	MOVD R14, R13
	SUB  $8, R13, R13

	// !!! R15 = &src[candidate + 4]
	ADD $4, R15, R15
	ADD R6, R15, R15

	// !!! s += 4
	ADD $4, R7, R7

inlineExtendMatchCmp8:
	// As long as we are 8 or more bytes before the end of src, we can load and
	// compare 8 bytes at a time. If those 8 bytes are equal, repeat.
	CMP  R13, R7
	BHI  inlineExtendMatchCmp1
	MOVD (R15), R3
	MOVD (R7), R4
	CMP  R4, R3
	BNE  inlineExtendMatchBSF
	ADD  $8, R15, R15
	ADD  $8, R7, R7
	B    inlineExtendMatchCmp8

inlineExtendMatchBSF:
	// If those 8 bytes were not equal, XOR the two 8 byte values, and return
	// the index of the first byte that differs.
	// RBIT reverses the bit order, then CLZ counts the leading zeros, the
	// combination of which finds the least significant bit which is set.
	// The arm64 architecture is little-endian, and the shift by 3 converts
	// a bit index to a byte index.
	EOR  R3, R4, R4
	RBIT R4, R4
	CLZ  R4, R4
	ADD  R4>>3, R7, R7
	B    inlineExtendMatchEnd

inlineExtendMatchCmp1:
	// In src's tail, compare 1 byte at a time.
	CMP  R7, R14
	BLS  inlineExtendMatchEnd
	MOVB (R15), R3
	MOVB (R7), R4
	CMP  R4, R3
	BNE  inlineExtendMatchEnd
	ADD  $1, R15, R15
	ADD  $1, R7, R7
	B    inlineExtendMatchCmp1

inlineExtendMatchEnd:
	// End inline of the extendMatch call.
	// ----------------------------------------

	// ----------------------------------------
	// Begin inline of the emitCopy call.
	//
	// d += emitCopy(dst[d:], base-candidate, s-base)

	// !!! length := s - base
	MOVD R7, R3
	SUB  R12, R3, R3

inlineEmitCopyLoop0:
	// for length >= 68 { etc }
	MOVW $68, R2
	CMPW R2, R3
	BLT  inlineEmitCopyStep1

	// Emit a length 64 copy, encoded as 3 bytes.
	MOVD $0xfe, R1
	MOVB R1, 0(R8)
	MOVW R11, 1(R8)
	ADD  $3, R8, R8
	SUBW $64, R3, R3
	B    inlineEmitCopyLoop0

inlineEmitCopyStep1:
	// if length > 64 { etc }
	MOVW $64, R2
	CMPW R2, R3
	BLE  inlineEmitCopyStep2

	// Emit a length 60 copy, encoded as 3 bytes.
	MOVD $0xee, R1
	MOVB R1, 0(R8)
	MOVW R11, 1(R8)
	ADD  $3, R8, R8
	SUBW $60, R3, R3

inlineEmitCopyStep2:
	// if length >= 12 || offset >= 2048 { goto inlineEmitCopyStep3 }
	MOVW $12, R2
	CMPW R2, R3
	BGE  inlineEmitCopyStep3
	MOVW $2048, R2
	CMPW R2, R11
	BGE  inlineEmitCopyStep3

	// Emit the remaining copy, encoded as 2 bytes.
	MOVB R11, 1(R8)
	LSRW $8, R11, R11
	LSLW $5, R11, R11
	SUBW $4, R3, R3
	AND  $0xff, R3, R3
	LSLW $2, R3, R3
	ORRW R3, R11, R11
	ORRW $1, R11, R11
	MOVB R11, 0(R8)
	ADD  $2, R8, R8
	B    inlineEmitCopyEnd

inlineEmitCopyStep3:
	// Emit the remaining copy, encoded as 3 bytes.
	SUBW $1, R3, R3
	LSLW $2, R3, R3
	ORRW $2, R3, R3
	MOVB R3, 0(R8)
	MOVW R11, 1(R8)
	ADD  $3, R8, R8

inlineEmitCopyEnd:
	// End inline of the emitCopy call.
	// ----------------------------------------

	// nextEmit = s
	MOVD R7, R10

	// if s >= sLimit { goto emitRemainder }
	MOVD R7, R3
	SUB  R6, R3, R3
	CMP  R3, R9
	BLS  emitRemainder

	// As per the encode_other.go code:
	//
	// We could immediately etc.

	// x := load64(src, s-1)
	MOVD -1(R7), R14

	// prevHash := hash(uint32(x>>0), shift)
	MOVW R14, R11
	MULW R16, R11, R11
	LSRW R5, R11, R11

	// table[prevHash] = uint16(s-1)
	MOVD R7, R3
	SUB  R6, R3, R3
	SUB  $1, R3, R3

	MOVHU R3, 0(R17)(R11<<1)

	// currHash := hash(uint32(x>>8), shift)
	LSR  $8, R14, R14
	MOVW R14, R11
	MULW R16, R11, R11
	LSRW R5, R11, R11

	// candidate = int(table[currHash])
	MOVHU 0(R17)(R11<<1), R15

	// table[currHash] = uint16(s)
	ADD   $1, R3, R3
	MOVHU R3, 0(R17)(R11<<1)

	// if uint32(x>>8) == load32(src, candidate) { continue }
	MOVW (R6)(R15*1), R4
	CMPW R4, R14
	BEQ  inner1

	// nextHash = hash(uint32(x>>16), shift)
	LSR  $8, R14, R14
	MOVW R14, R11
	MULW R16, R11, R11
	LSRW R5, R11, R11

	// s++
	ADD $1, R7, R7

	// break out of the inner1 for loop, i.e. continue the outer loop.
	B outer

emitRemainder:
	// if nextEmit < len(src) { etc }
	MOVD src_len+32(FP), R3
	ADD  R6, R3, R3
	CMP  R3, R10
	BEQ  encodeBlockEnd

	// d += emitLiteral(dst[d:], src[nextEmit:])
	//
	// Push args.
	MOVD R8, 8(RSP)
	MOVD $0, 16(RSP)  // Unnecessary, as the callee ignores it, but conservative.
	MOVD $0, 24(RSP)  // Unnecessary, as the callee ignores it, but conservative.
	MOVD R10, 32(RSP)
	SUB  R10, R3, R3
	MOVD R3, 40(RSP)
	MOVD R3, 48(RSP)  // Unnecessary, as the callee ignores it, but conservative.

	// Spill local variables (registers) onto the stack; call; unspill.
	MOVD R8, 88(RSP)
	CALL ·emitLiteral(SB)
	MOVD 88(RSP), R8

	// Finish the "d +=" part of "d += emitLiteral(etc)".
	MOVD 56(RSP), R1
	ADD  R1, R8, R8

encodeBlockEnd:
	MOVD dst_base+0(FP), R3
	SUB  R3, R8, R8
	MOVD R8, d+48(FP)
	RET
