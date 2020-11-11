package ir

func NewJump(parent *BasicBlock) *Jump {
	return &Jump{anInstruction{block: parent}, ""}
}
