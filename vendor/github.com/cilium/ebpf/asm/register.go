package asm

import (
	"fmt"
)

// Register is the source or destination of most operations.
type Register uint8

// R0 contains return values.
const R0 Register = 0

// Registers for function arguments.
const (
	R1 Register = R0 + 1 + iota
	R2
	R3
	R4
	R5
)

// Callee saved registers preserved by function calls.
const (
	R6 Register = R5 + 1 + iota
	R7
	R8
	R9
)

// Read-only frame pointer to access stack.
const (
	R10 Register = R9 + 1
	RFP          = R10
)

// Pseudo registers used by 64bit loads and jumps
const (
	PseudoMapFD     = R1 // BPF_PSEUDO_MAP_FD
	PseudoMapValue  = R2 // BPF_PSEUDO_MAP_VALUE
	PseudoCall      = R1 // BPF_PSEUDO_CALL
	PseudoFunc      = R4 // BPF_PSEUDO_FUNC
	PseudoKfuncCall = R2 // BPF_PSEUDO_KFUNC_CALL
)

func (r Register) String() string {
	v := uint8(r)
	if v == 10 {
		return "rfp"
	}
	return fmt.Sprintf("r%d", v)
}
