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

func (r Register) String() string {
	v := uint8(r)
	if v == 10 {
		return "rfp"
	}
	return fmt.Sprintf("r%d", v)
}
