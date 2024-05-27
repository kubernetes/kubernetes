package asm

//go:generate go run golang.org/x/tools/cmd/stringer@latest -output jump_string.go -type=JumpOp

// JumpOp affect control flow.
//
//	msb      lsb
//	+----+-+---+
//	|OP  |s|cls|
//	+----+-+---+
type JumpOp uint8

const jumpMask OpCode = 0xf0

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

// Imm compares 64 bit dst to 64 bit value (sign extended), and adjusts PC by offset if the condition is fulfilled.
func (op JumpOp) Imm(dst Register, value int32, label string) Instruction {
	return Instruction{
		OpCode:   op.opCode(JumpClass, ImmSource),
		Dst:      dst,
		Offset:   -1,
		Constant: int64(value),
	}.WithReference(label)
}

// Imm32 compares 32 bit dst to 32 bit value, and adjusts PC by offset if the condition is fulfilled.
// Requires kernel 5.1.
func (op JumpOp) Imm32(dst Register, value int32, label string) Instruction {
	return Instruction{
		OpCode:   op.opCode(Jump32Class, ImmSource),
		Dst:      dst,
		Offset:   -1,
		Constant: int64(value),
	}.WithReference(label)
}

// Reg compares 64 bit dst to 64 bit src, and adjusts PC by offset if the condition is fulfilled.
func (op JumpOp) Reg(dst, src Register, label string) Instruction {
	return Instruction{
		OpCode: op.opCode(JumpClass, RegSource),
		Dst:    dst,
		Src:    src,
		Offset: -1,
	}.WithReference(label)
}

// Reg32 compares 32 bit dst to 32 bit src, and adjusts PC by offset if the condition is fulfilled.
// Requires kernel 5.1.
func (op JumpOp) Reg32(dst, src Register, label string) Instruction {
	return Instruction{
		OpCode: op.opCode(Jump32Class, RegSource),
		Dst:    dst,
		Src:    src,
		Offset: -1,
	}.WithReference(label)
}

func (op JumpOp) opCode(class Class, source Source) OpCode {
	if op == Exit || op == Call {
		return InvalidOpCode
	}

	return OpCode(class).SetJumpOp(op).SetSource(source)
}

// LongJump returns a jump always instruction with a range of [-2^31, 2^31 - 1].
func LongJump(label string) Instruction {
	return Instruction{
		OpCode:   Ja.opCode(Jump32Class, ImmSource),
		Constant: -1,
	}.WithReference(label)
}

// Label adjusts PC to the address of the label.
func (op JumpOp) Label(label string) Instruction {
	if op == Call {
		return Instruction{
			OpCode:   OpCode(JumpClass).SetJumpOp(Call),
			Src:      PseudoCall,
			Constant: -1,
		}.WithReference(label)
	}

	return Instruction{
		OpCode: OpCode(JumpClass).SetJumpOp(op),
		Offset: -1,
	}.WithReference(label)
}
