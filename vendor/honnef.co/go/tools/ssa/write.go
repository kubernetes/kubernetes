package ssa

func NewJump(parent *BasicBlock) *Jump {
	return &Jump{anInstruction{parent}}
}
