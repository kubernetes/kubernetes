package asm

//go:generate stringer -output jump_string.go -type=JumpOp

// JumpOp affect control flow.
//
//    msb      lsb
//    +----+-+---+
//    |OP  |s|cls|
//    +----+-+---+
type JumpOp uint8

const jumpMask OpCode = aluMask

const (
	// InvalidJumpOp is returned by getters when invoked
	// on non branch OpCodes
	InvalidJumpOp JumpOp = 0xff
	// Ja jumps by offset unconditionally
	Ja JumpOp = 0x00
	// JEq jumps by offset if r == imm
	JEq JumpOp = 0x10
	// JGT jumps by offset if r > imm
	JGT JumpOp = 0x20
	// JGE jumps by offset if r >= imm
	JGE JumpOp = 0x30
	// JSet jumps by offset if r & imm
	JSet JumpOp = 0x40
	// JNE jumps by offset if r != imm
	JNE JumpOp = 0x50
	// JSGT jumps by offset if signed r > signed imm
	JSGT JumpOp = 0x60
	// JSGE jumps by offset if signed r >= signed imm
	JSGE JumpOp = 0x70
	// Call builtin or user defined function from imm
	Call JumpOp = 0x80
	// Exit ends execution, with value in r0
	Exit JumpOp = 0x90
	// JLT jumps by offset if r < imm
	JLT JumpOp = 0xa0
	// JLE jumps by offset if r <= imm
	JLE JumpOp = 0xb0
	// JSLT jumps by offset if signed r < signed imm
	JSLT JumpOp = 0xc0
	// JSLE jumps by offset if signed r <= signed imm
	JSLE JumpOp = 0xd0
)

// Return emits an exit instruction.
//
// Requires a return value in R0.
func Return() Instruction {
	return Instruction{
		OpCode: OpCode(JumpClass).SetJumpOp(Exit),
	}
}

// Op returns the OpCode for a given jump source.
func (op JumpOp) Op(source Source) OpCode {
	return OpCode(JumpClass).SetJumpOp(op).SetSource(source)
}

// Imm compares dst to value, and adjusts PC by offset if the condition is fulfilled.
func (op JumpOp) Imm(dst Register, value int32, label string) Instruction {
	if op == Exit || op == Call || op == Ja {
		return Instruction{OpCode: InvalidOpCode}
	}

	return Instruction{
		OpCode:    OpCode(JumpClass).SetJumpOp(op).SetSource(ImmSource),
		Dst:       dst,
		Offset:    -1,
		Constant:  int64(value),
		Reference: label,
	}
}

// Reg compares dst to src, and adjusts PC by offset if the condition is fulfilled.
func (op JumpOp) Reg(dst, src Register, label string) Instruction {
	if op == Exit || op == Call || op == Ja {
		return Instruction{OpCode: InvalidOpCode}
	}

	return Instruction{
		OpCode:    OpCode(JumpClass).SetJumpOp(op).SetSource(RegSource),
		Dst:       dst,
		Src:       src,
		Offset:    -1,
		Reference: label,
	}
}

// Label adjusts PC to the address of the label.
func (op JumpOp) Label(label string) Instruction {
	if op == Call {
		return Instruction{
			OpCode:    OpCode(JumpClass).SetJumpOp(Call),
			Src:       PseudoCall,
			Constant:  -1,
			Reference: label,
		}
	}

	return Instruction{
		OpCode:    OpCode(JumpClass).SetJumpOp(op),
		Offset:    -1,
		Reference: label,
	}
}
