// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build !appengine
// +build gc
// +build !noasm

#include "textflag.h"

// The XXX lines assemble on Go 1.4, 1.5 and 1.7, but not 1.6, due to a
// Go toolchain regression. See https://github.com/golang/go/issues/15426 and
// https://github.com/golang/snappy/issues/29
//
// As a workaround, the package was built with a known good assembler, and
// those instructions were disassembled by "objdump -d" to yield the
//	4e 0f b7 7c 5c 78       movzwq 0x78(%rsp,%r11,2),%r15
// style comments, in AT&T asm syntax. Note that rsp here is a physical
// register, not Go/asm's SP pseudo-register (see https://golang.org/doc/asm).
// The instructions were then encoded as "BYTE $0x.." sequences, which assemble
// fine on Go 1.6.

// The asm code generally follows the pure Go code in encode_other.go, except
// where marked with a "!!!".

// ----------------------------------------------------------------------------

// func emitLiteral(dst, lit []byte) int
//
// All local variables fit into registers. The register allocation:
//	- AX	len(lit)
//	- BX	n
//	- DX	return value
//	- DI	&dst[i]
//	- R10	&lit[0]
//
// The 24 bytes of stack space is to call runtime·memmove.
//
// The unusual register allocation of local variables, such as R10 for the
// source pointer, matches the allocation used at the call site in encodeBlock,
// which makes it easier to manually inline this function.
TEXT ·emitLiteral(SB), NOSPLIT, $24-56
	MOVQ dst_base+0(FP), DI
	MOVQ lit_base+24(FP), R10
	MOVQ lit_len+32(FP), AX
	MOVQ AX, DX
	MOVL AX, BX
	SUBL $1, BX

	CMPL BX, $60
	JLT  oneByte
	CMPL BX, $256
	JLT  twoBytes

threeBytes:
	MOVB $0xf4, 0(DI)
	MOVW BX, 1(DI)
	ADDQ $3, DI
	ADDQ $3, DX
	JMP  memmove

twoBytes:
	MOVB $0xf0, 0(DI)
	MOVB BX, 1(DI)
	ADDQ $2, DI
	ADDQ $2, DX
	JMP  memmove

oneByte:
	SHLB $2, BX
	MOVB BX, 0(DI)
	ADDQ $1, DI
	ADDQ $1, DX

memmove:
	MOVQ DX, ret+48(FP)

	// copy(dst[i:], lit)
	//
	// This means calling runtime·memmove(&dst[i], &lit[0], len(lit)), so we push
	// DI, R10 and AX as arguments.
	MOVQ DI, 0(SP)
	MOVQ R10, 8(SP)
	MOVQ AX, 16(SP)
	CALL runtime·memmove(SB)
	RET

// ----------------------------------------------------------------------------

// func emitCopy(dst []byte, offset, length int) int
//
// All local variables fit into registers. The register allocation:
//	- AX	length
//	- SI	&dst[0]
//	- DI	&dst[i]
//	- R11	offset
//
// The unusual register allocation of local variables, such as R11 for the
// offset, matches the allocation used at the call site in encodeBlock, which
// makes it easier to manually inline this function.
TEXT ·emitCopy(SB), NOSPLIT, $0-48
	MOVQ dst_base+0(FP), DI
	MOVQ DI, SI
	MOVQ offset+24(FP), R11
	MOVQ length+32(FP), AX

loop0:
	// for length >= 68 { etc }
	CMPL AX, $68
	JLT  step1

	// Emit a length 64 copy, encoded as 3 bytes.
	MOVB $0xfe, 0(DI)
	MOVW R11, 1(DI)
	ADDQ $3, DI
	SUBL $64, AX
	JMP  loop0

step1:
	// if length > 64 { etc }
	CMPL AX, $64
	JLE  step2

	// Emit a length 60 copy, encoded as 3 bytes.
	MOVB $0xee, 0(DI)
	MOVW R11, 1(DI)
	ADDQ $3, DI
	SUBL $60, AX

step2:
	// if length >= 12 || offset >= 2048 { goto step3 }
	CMPL AX, $12
	JGE  step3
	CMPL R11, $2048
	JGE  step3

	// Emit the remaining copy, encoded as 2 bytes.
	MOVB R11, 1(DI)
	SHRL $8, R11
	SHLB $5, R11
	SUBB $4, AX
	SHLB $2, AX
	ORB  AX, R11
	ORB  $1, R11
	MOVB R11, 0(DI)
	ADDQ $2, DI

	// Return the number of bytes written.
	SUBQ SI, DI
	MOVQ DI, ret+40(FP)
	RET

step3:
	// Emit the remaining copy, encoded as 3 bytes.
	SUBL $1, AX
	SHLB $2, AX
	ORB  $2, AX
	MOVB AX, 0(DI)
	MOVW R11, 1(DI)
	ADDQ $3, DI

	// Return the number of bytes written.
	SUBQ SI, DI
	MOVQ DI, ret+40(FP)
	RET

// ----------------------------------------------------------------------------

// func extendMatch(src []byte, i, j int) int
//
// All local variables fit into registers. The register allocation:
//	- DX	&src[0]
//	- SI	&src[j]
//	- R13	&src[len(src) - 8]
//	- R14	&src[len(src)]
//	- R15	&src[i]
//
// The unusual register allocation of local variables, such as R15 for a source
// pointer, matches the allocation used at the call site in encodeBlock, which
// makes it easier to manually inline this function.
TEXT ·extendMatch(SB), NOSPLIT, $0-48
	MOVQ src_base+0(FP), DX
	MOVQ src_len+8(FP), R14
	MOVQ i+24(FP), R15
	MOVQ j+32(FP), SI
	ADDQ DX, R14
	ADDQ DX, R15
	ADDQ DX, SI
	MOVQ R14, R13
	SUBQ $8, R13

cmp8:
	// As long as we are 8 or more bytes before the end of src, we can load and
	// compare 8 bytes at a time. If those 8 bytes are equal, repeat.
	CMPQ SI, R13
	JA   cmp1
	MOVQ (R15), AX
	MOVQ (SI), BX
	CMPQ AX, BX
	JNE  bsf
	ADDQ $8, R15
	ADDQ $8, SI
	JMP  cmp8

bsf:
	// If those 8 bytes were not equal, XOR the two 8 byte values, and return
	// the index of the first byte that differs. The BSF instruction finds the
	// least significant 1 bit, the amd64 architecture is little-endian, and
	// the shift by 3 converts a bit index to a byte index.
	XORQ AX, BX
	BSFQ BX, BX
	SHRQ $3, BX
	ADDQ BX, SI

	// Convert from &src[ret] to ret.
	SUBQ DX, SI
	MOVQ SI, ret+40(FP)
	RET

cmp1:
	// In src's tail, compare 1 byte at a time.
	CMPQ SI, R14
	JAE  extendMatchEnd
	MOVB (R15), AX
	MOVB (SI), BX
	CMPB AX, BX
	JNE  extendMatchEnd
	ADDQ $1, R15
	ADDQ $1, SI
	JMP  cmp1

extendMatchEnd:
	// Convert from &src[ret] to ret.
	SUBQ DX, SI
	MOVQ SI, ret+40(FP)
	RET

// ----------------------------------------------------------------------------

// func encodeBlock(dst, src []byte) (d int)
//
// All local variables fit into registers, other than "var table". The register
// allocation:
//	- AX	.	.
//	- BX	.	.
//	- CX	56	shift (note that amd64 shifts by non-immediates must use CX).
//	- DX	64	&src[0], tableSize
//	- SI	72	&src[s]
//	- DI	80	&dst[d]
//	- R9	88	sLimit
//	- R10	.	&src[nextEmit]
//	- R11	96	prevHash, currHash, nextHash, offset
//	- R12	104	&src[base], skip
//	- R13	.	&src[nextS], &src[len(src) - 8]
//	- R14	.	len(src), bytesBetweenHashLookups, &src[len(src)], x
//	- R15	112	candidate
//
// The second column (56, 64, etc) is the stack offset to spill the registers
// when calling other functions. We could pack this slightly tighter, but it's
// simpler to have a dedicated spill map independent of the function called.
//
// "var table [maxTableSize]uint16" takes up 32768 bytes of stack space. An
// extra 56 bytes, to call other functions, and an extra 64 bytes, to spill
// local variables (registers) during calls gives 32768 + 56 + 64 = 32888.
TEXT ·encodeBlock(SB), 0, $32888-56
	MOVQ dst_base+0(FP), DI
	MOVQ src_base+24(FP), SI
	MOVQ src_len+32(FP), R14

	// shift, tableSize := uint32(32-8), 1<<8
	MOVQ $24, CX
	MOVQ $256, DX

calcShift:
	// for ; tableSize < maxTableSize && tableSize < len(src); tableSize *= 2 {
	//	shift--
	// }
	CMPQ DX, $16384
	JGE  varTable
	CMPQ DX, R14
	JGE  varTable
	SUBQ $1, CX
	SHLQ $1, DX
	JMP  calcShift

varTable:
	// var table [maxTableSize]uint16
	//
	// In the asm code, unlike the Go code, we can zero-initialize only the
	// first tableSize elements. Each uint16 element is 2 bytes and each MOVOU
	// writes 16 bytes, so we can do only tableSize/8 writes instead of the
	// 2048 writes that would zero-initialize all of table's 32768 bytes.
	SHRQ $3, DX
	LEAQ table-32768(SP), BX
	PXOR X0, X0

memclr:
	MOVOU X0, 0(BX)
	ADDQ  $16, BX
	SUBQ  $1, DX
	JNZ   memclr

	// !!! DX = &src[0]
	MOVQ SI, DX

	// sLimit := len(src) - inputMargin
	MOVQ R14, R9
	SUBQ $15, R9

	// !!! Pre-emptively spill CX, DX and R9 to the stack. Their values don't
	// change for the rest of the function.
	MOVQ CX, 56(SP)
	MOVQ DX, 64(SP)
	MOVQ R9, 88(SP)

	// nextEmit := 0
	MOVQ DX, R10

	// s := 1
	ADDQ $1, SI

	// nextHash := hash(load32(src, s), shift)
	MOVL  0(SI), R11
	IMULL $0x1e35a7bd, R11
	SHRL  CX, R11

outer:
	// for { etc }

	// skip := 32
	MOVQ $32, R12

	// nextS := s
	MOVQ SI, R13

	// candidate := 0
	MOVQ $0, R15

inner0:
	// for { etc }

	// s := nextS
	MOVQ R13, SI

	// bytesBetweenHashLookups := skip >> 5
	MOVQ R12, R14
	SHRQ $5, R14

	// nextS = s + bytesBetweenHashLookups
	ADDQ R14, R13

	// skip += bytesBetweenHashLookups
	ADDQ R14, R12

	// if nextS > sLimit { goto emitRemainder }
	MOVQ R13, AX
	SUBQ DX, AX
	CMPQ AX, R9
	JA   emitRemainder

	// candidate = int(table[nextHash])
	// XXX: MOVWQZX table-32768(SP)(R11*2), R15
	// XXX: 4e 0f b7 7c 5c 78       movzwq 0x78(%rsp,%r11,2),%r15
	BYTE $0x4e
	BYTE $0x0f
	BYTE $0xb7
	BYTE $0x7c
	BYTE $0x5c
	BYTE $0x78

	// table[nextHash] = uint16(s)
	MOVQ SI, AX
	SUBQ DX, AX

	// XXX: MOVW AX, table-32768(SP)(R11*2)
	// XXX: 66 42 89 44 5c 78       mov    %ax,0x78(%rsp,%r11,2)
	BYTE $0x66
	BYTE $0x42
	BYTE $0x89
	BYTE $0x44
	BYTE $0x5c
	BYTE $0x78

	// nextHash = hash(load32(src, nextS), shift)
	MOVL  0(R13), R11
	IMULL $0x1e35a7bd, R11
	SHRL  CX, R11

	// if load32(src, s) != load32(src, candidate) { continue } break
	MOVL 0(SI), AX
	MOVL (DX)(R15*1), BX
	CMPL AX, BX
	JNE  inner0

fourByteMatch:
	// As per the encode_other.go code:
	//
	// A 4-byte match has been found. We'll later see etc.

	// !!! Jump to a fast path for short (<= 16 byte) literals. See the comment
	// on inputMargin in encode.go.
	MOVQ SI, AX
	SUBQ R10, AX
	CMPQ AX, $16
	JLE  emitLiteralFastPath

	// ----------------------------------------
	// Begin inline of the emitLiteral call.
	//
	// d += emitLiteral(dst[d:], src[nextEmit:s])

	MOVL AX, BX
	SUBL $1, BX

	CMPL BX, $60
	JLT  inlineEmitLiteralOneByte
	CMPL BX, $256
	JLT  inlineEmitLiteralTwoBytes

inlineEmitLiteralThreeBytes:
	MOVB $0xf4, 0(DI)
	MOVW BX, 1(DI)
	ADDQ $3, DI
	JMP  inlineEmitLiteralMemmove

inlineEmitLiteralTwoBytes:
	MOVB $0xf0, 0(DI)
	MOVB BX, 1(DI)
	ADDQ $2, DI
	JMP  inlineEmitLiteralMemmove

inlineEmitLiteralOneByte:
	SHLB $2, BX
	MOVB BX, 0(DI)
	ADDQ $1, DI

inlineEmitLiteralMemmove:
	// Spill local variables (registers) onto the stack; call; unspill.
	//
	// copy(dst[i:], lit)
	//
	// This means calling runtime·memmove(&dst[i], &lit[0], len(lit)), so we push
	// DI, R10 and AX as arguments.
	MOVQ DI, 0(SP)
	MOVQ R10, 8(SP)
	MOVQ AX, 16(SP)
	ADDQ AX, DI              // Finish the "d +=" part of "d += emitLiteral(etc)".
	MOVQ SI, 72(SP)
	MOVQ DI, 80(SP)
	MOVQ R15, 112(SP)
	CALL runtime·memmove(SB)
	MOVQ 56(SP), CX
	MOVQ 64(SP), DX
	MOVQ 72(SP), SI
	MOVQ 80(SP), DI
	MOVQ 88(SP), R9
	MOVQ 112(SP), R15
	JMP  inner1

inlineEmitLiteralEnd:
	// End inline of the emitLiteral call.
	// ----------------------------------------

emitLiteralFastPath:
	// !!! Emit the 1-byte encoding "uint8(len(lit)-1)<<2".
	MOVB AX, BX
	SUBB $1, BX
	SHLB $2, BX
	MOVB BX, (DI)
	ADDQ $1, DI

	// !!! Implement the copy from lit to dst as a 16-byte load and store.
	// (Encode's documentation says that dst and src must not overlap.)
	//
	// This always copies 16 bytes, instead of only len(lit) bytes, but that's
	// OK. Subsequent iterations will fix up the overrun.
	//
	// Note that on amd64, it is legal and cheap to issue unaligned 8-byte or
	// 16-byte loads and stores. This technique probably wouldn't be as
	// effective on architectures that are fussier about alignment.
	MOVOU 0(R10), X0
	MOVOU X0, 0(DI)
	ADDQ  AX, DI

inner1:
	// for { etc }

	// base := s
	MOVQ SI, R12

	// !!! offset := base - candidate
	MOVQ R12, R11
	SUBQ R15, R11
	SUBQ DX, R11

	// ----------------------------------------
	// Begin inline of the extendMatch call.
	//
	// s = extendMatch(src, candidate+4, s+4)

	// !!! R14 = &src[len(src)]
	MOVQ src_len+32(FP), R14
	ADDQ DX, R14

	// !!! R13 = &src[len(src) - 8]
	MOVQ R14, R13
	SUBQ $8, R13

	// !!! R15 = &src[candidate + 4]
	ADDQ $4, R15
	ADDQ DX, R15

	// !!! s += 4
	ADDQ $4, SI

inlineExtendMatchCmp8:
	// As long as we are 8 or more bytes before the end of src, we can load and
	// compare 8 bytes at a time. If those 8 bytes are equal, repeat.
	CMPQ SI, R13
	JA   inlineExtendMatchCmp1
	MOVQ (R15), AX
	MOVQ (SI), BX
	CMPQ AX, BX
	JNE  inlineExtendMatchBSF
	ADDQ $8, R15
	ADDQ $8, SI
	JMP  inlineExtendMatchCmp8

inlineExtendMatchBSF:
	// If those 8 bytes were not equal, XOR the two 8 byte values, and return
	// the index of the first byte that differs. The BSF instruction finds the
	// least significant 1 bit, the amd64 architecture is little-endian, and
	// the shift by 3 converts a bit index to a byte index.
	XORQ AX, BX
	BSFQ BX, BX
	SHRQ $3, BX
	ADDQ BX, SI
	JMP  inlineExtendMatchEnd

inlineExtendMatchCmp1:
	// In src's tail, compare 1 byte at a time.
	CMPQ SI, R14
	JAE  inlineExtendMatchEnd
	MOVB (R15), AX
	MOVB (SI), BX
	CMPB AX, BX
	JNE  inlineExtendMatchEnd
	ADDQ $1, R15
	ADDQ $1, SI
	JMP  inlineExtendMatchCmp1

inlineExtendMatchEnd:
	// End inline of the extendMatch call.
	// ----------------------------------------

	// ----------------------------------------
	// Begin inline of the emitCopy call.
	//
	// d += emitCopy(dst[d:], base-candidate, s-base)

	// !!! length := s - base
	MOVQ SI, AX
	SUBQ R12, AX

inlineEmitCopyLoop0:
	// for length >= 68 { etc }
	CMPL AX, $68
	JLT  inlineEmitCopyStep1

	// Emit a length 64 copy, encoded as 3 bytes.
	MOVB $0xfe, 0(DI)
	MOVW R11, 1(DI)
	ADDQ $3, DI
	SUBL $64, AX
	JMP  inlineEmitCopyLoop0

inlineEmitCopyStep1:
	// if length > 64 { etc }
	CMPL AX, $64
	JLE  inlineEmitCopyStep2

	// Emit a length 60 copy, encoded as 3 bytes.
	MOVB $0xee, 0(DI)
	MOVW R11, 1(DI)
	ADDQ $3, DI
	SUBL $60, AX

inlineEmitCopyStep2:
	// if length >= 12 || offset >= 2048 { goto inlineEmitCopyStep3 }
	CMPL AX, $12
	JGE  inlineEmitCopyStep3
	CMPL R11, $2048
	JGE  inlineEmitCopyStep3

	// Emit the remaining copy, encoded as 2 bytes.
	MOVB R11, 1(DI)
	SHRL $8, R11
	SHLB $5, R11
	SUBB $4, AX
	SHLB $2, AX
	ORB  AX, R11
	ORB  $1, R11
	MOVB R11, 0(DI)
	ADDQ $2, DI
	JMP  inlineEmitCopyEnd

inlineEmitCopyStep3:
	// Emit the remaining copy, encoded as 3 bytes.
	SUBL $1, AX
	SHLB $2, AX
	ORB  $2, AX
	MOVB AX, 0(DI)
	MOVW R11, 1(DI)
	ADDQ $3, DI

inlineEmitCopyEnd:
	// End inline of the emitCopy call.
	// ----------------------------------------

	// nextEmit = s
	MOVQ SI, R10

	// if s >= sLimit { goto emitRemainder }
	MOVQ SI, AX
	SUBQ DX, AX
	CMPQ AX, R9
	JAE  emitRemainder

	// As per the encode_other.go code:
	//
	// We could immediately etc.

	// x := load64(src, s-1)
	MOVQ -1(SI), R14

	// prevHash := hash(uint32(x>>0), shift)
	MOVL  R14, R11
	IMULL $0x1e35a7bd, R11
	SHRL  CX, R11

	// table[prevHash] = uint16(s-1)
	MOVQ SI, AX
	SUBQ DX, AX
	SUBQ $1, AX

	// XXX: MOVW AX, table-32768(SP)(R11*2)
	// XXX: 66 42 89 44 5c 78       mov    %ax,0x78(%rsp,%r11,2)
	BYTE $0x66
	BYTE $0x42
	BYTE $0x89
	BYTE $0x44
	BYTE $0x5c
	BYTE $0x78

	// currHash := hash(uint32(x>>8), shift)
	SHRQ  $8, R14
	MOVL  R14, R11
	IMULL $0x1e35a7bd, R11
	SHRL  CX, R11

	// candidate = int(table[currHash])
	// XXX: MOVWQZX table-32768(SP)(R11*2), R15
	// XXX: 4e 0f b7 7c 5c 78       movzwq 0x78(%rsp,%r11,2),%r15
	BYTE $0x4e
	BYTE $0x0f
	BYTE $0xb7
	BYTE $0x7c
	BYTE $0x5c
	BYTE $0x78

	// table[currHash] = uint16(s)
	ADDQ $1, AX

	// XXX: MOVW AX, table-32768(SP)(R11*2)
	// XXX: 66 42 89 44 5c 78       mov    %ax,0x78(%rsp,%r11,2)
	BYTE $0x66
	BYTE $0x42
	BYTE $0x89
	BYTE $0x44
	BYTE $0x5c
	BYTE $0x78

	// if uint32(x>>8) == load32(src, candidate) { continue }
	MOVL (DX)(R15*1), BX
	CMPL R14, BX
	JEQ  inner1

	// nextHash = hash(uint32(x>>16), shift)
	SHRQ  $8, R14
	MOVL  R14, R11
	IMULL $0x1e35a7bd, R11
	SHRL  CX, R11

	// s++
	ADDQ $1, SI

	// break out of the inner1 for loop, i.e. continue the outer loop.
	JMP outer

emitRemainder:
	// if nextEmit < len(src) { etc }
	MOVQ src_len+32(FP), AX
	ADDQ DX, AX
	CMPQ R10, AX
	JEQ  encodeBlockEnd

	// d += emitLiteral(dst[d:], src[nextEmit:])
	//
	// Push args.
	MOVQ DI, 0(SP)
	MOVQ $0, 8(SP)   // Unnecessary, as the callee ignores it, but conservative.
	MOVQ $0, 16(SP)  // Unnecessary, as the callee ignores it, but conservative.
	MOVQ R10, 24(SP)
	SUBQ R10, AX
	MOVQ AX, 32(SP)
	MOVQ AX, 40(SP)  // Unnecessary, as the callee ignores it, but conservative.

	// Spill local variables (registers) onto the stack; call; unspill.
	MOVQ DI, 80(SP)
	CALL ·emitLiteral(SB)
	MOVQ 80(SP), DI

	// Finish the "d +=" part of "d += emitLiteral(etc)".
	ADDQ 48(SP), DI

encodeBlockEnd:
	MOVQ dst_base+0(FP), AX
	SUBQ AX, DI
	MOVQ DI, d+48(FP)
	RET
