// +build !appengine
// +build gc
// +build !purego

#include "textflag.h"

// Register allocation:
// AX	h
// CX	pointer to advance through b
// DX	n
// BX	loop end
// R8	v1, k1
// R9	v2
// R10	v3
// R11	v4
// R12	tmp
// R13	prime1v
// R14	prime2v
// R15	prime4v

// round reads from and advances the buffer pointer in CX.
// It assumes that R13 has prime1v and R14 has prime2v.
#define round(r) \
	MOVQ  (CX), R12 \
	ADDQ  $8, CX    \
	IMULQ R14, R12  \
	ADDQ  R12, r    \
	ROLQ  $31, r    \
	IMULQ R13, r

// mergeRound applies a merge round on the two registers acc and val.
// It assumes that R13 has prime1v, R14 has prime2v, and R15 has prime4v.
#define mergeRound(acc, val) \
	IMULQ R14, val \
	ROLQ  $31, val \
	IMULQ R13, val \
	XORQ  val, acc \
	IMULQ R13, acc \
	ADDQ  R15, acc

// func Sum64(b []byte) uint64
TEXT ·Sum64(SB), NOSPLIT, $0-32
	// Load fixed primes.
	MOVQ ·prime1v(SB), R13
	MOVQ ·prime2v(SB), R14
	MOVQ ·prime4v(SB), R15

	// Load slice.
	MOVQ b_base+0(FP), CX
	MOVQ b_len+8(FP), DX
	LEAQ (CX)(DX*1), BX

	// The first loop limit will be len(b)-32.
	SUBQ $32, BX

	// Check whether we have at least one block.
	CMPQ DX, $32
	JLT  noBlocks

	// Set up initial state (v1, v2, v3, v4).
	MOVQ R13, R8
	ADDQ R14, R8
	MOVQ R14, R9
	XORQ R10, R10
	XORQ R11, R11
	SUBQ R13, R11

	// Loop until CX > BX.
blockLoop:
	round(R8)
	round(R9)
	round(R10)
	round(R11)

	CMPQ CX, BX
	JLE  blockLoop

	MOVQ R8, AX
	ROLQ $1, AX
	MOVQ R9, R12
	ROLQ $7, R12
	ADDQ R12, AX
	MOVQ R10, R12
	ROLQ $12, R12
	ADDQ R12, AX
	MOVQ R11, R12
	ROLQ $18, R12
	ADDQ R12, AX

	mergeRound(AX, R8)
	mergeRound(AX, R9)
	mergeRound(AX, R10)
	mergeRound(AX, R11)

	JMP afterBlocks

noBlocks:
	MOVQ ·prime5v(SB), AX

afterBlocks:
	ADDQ DX, AX

	// Right now BX has len(b)-32, and we want to loop until CX > len(b)-8.
	ADDQ $24, BX

	CMPQ CX, BX
	JG   fourByte

wordLoop:
	// Calculate k1.
	MOVQ  (CX), R8
	ADDQ  $8, CX
	IMULQ R14, R8
	ROLQ  $31, R8
	IMULQ R13, R8

	XORQ  R8, AX
	ROLQ  $27, AX
	IMULQ R13, AX
	ADDQ  R15, AX

	CMPQ CX, BX
	JLE  wordLoop

fourByte:
	ADDQ $4, BX
	CMPQ CX, BX
	JG   singles

	MOVL  (CX), R8
	ADDQ  $4, CX
	IMULQ R13, R8
	XORQ  R8, AX

	ROLQ  $23, AX
	IMULQ R14, AX
	ADDQ  ·prime3v(SB), AX

singles:
	ADDQ $4, BX
	CMPQ CX, BX
	JGE  finalize

singlesLoop:
	MOVBQZX (CX), R12
	ADDQ    $1, CX
	IMULQ   ·prime5v(SB), R12
	XORQ    R12, AX

	ROLQ  $11, AX
	IMULQ R13, AX

	CMPQ CX, BX
	JL   singlesLoop

finalize:
	MOVQ  AX, R12
	SHRQ  $33, R12
	XORQ  R12, AX
	IMULQ R14, AX
	MOVQ  AX, R12
	SHRQ  $29, R12
	XORQ  R12, AX
	IMULQ ·prime3v(SB), AX
	MOVQ  AX, R12
	SHRQ  $32, R12
	XORQ  R12, AX

	MOVQ AX, ret+24(FP)
	RET

// writeBlocks uses the same registers as above except that it uses AX to store
// the d pointer.

// func writeBlocks(d *Digest, b []byte) int
TEXT ·writeBlocks(SB), NOSPLIT, $0-40
	// Load fixed primes needed for round.
	MOVQ ·prime1v(SB), R13
	MOVQ ·prime2v(SB), R14

	// Load slice.
	MOVQ b_base+8(FP), CX
	MOVQ b_len+16(FP), DX
	LEAQ (CX)(DX*1), BX
	SUBQ $32, BX

	// Load vN from d.
	MOVQ d+0(FP), AX
	MOVQ 0(AX), R8   // v1
	MOVQ 8(AX), R9   // v2
	MOVQ 16(AX), R10 // v3
	MOVQ 24(AX), R11 // v4

	// We don't need to check the loop condition here; this function is
	// always called with at least one block of data to process.
blockLoop:
	round(R8)
	round(R9)
	round(R10)
	round(R11)

	CMPQ CX, BX
	JLE  blockLoop

	// Copy vN back to d.
	MOVQ R8, 0(AX)
	MOVQ R9, 8(AX)
	MOVQ R10, 16(AX)
	MOVQ R11, 24(AX)

	// The number of bytes written is CX minus the old base pointer.
	SUBQ b_base+8(FP), CX
	MOVQ CX, ret+32(FP)

	RET
