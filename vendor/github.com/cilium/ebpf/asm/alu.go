package asm

//go:generate stringer -output alu_string.go -type=Source,Endianness,ALUOp

// Source of ALU / ALU64 / Branch operations
//
//	msb      lsb
//	+----+-+---+
//	|op  |S|cls|
//	+----+-+---+
type Source uint8

const sourceMask OpCode = 0x08

// Source bitmask
const (
	// InvalidSource is returned by getters when invoked
	// on non ALU / branch OpCodes.
	InvalidSource Source = 0xff
	// ImmSource src is from constant
	ImmSource Source = 0x00
	// RegSource src is from register
	RegSource Source = 0x08
)

// The Endianness of a byte swap instruction.
type Endianness uint8

const endianMask = sourceMask

// Endian flags
const (
	InvalidEndian Endianness = 0xff
	// Convert to little endian
	LE Endianness = 0x00
	// Convert to big endian
	BE Endianness = 0x08
)

// ALUOp are ALU / ALU64 operations
//
//	msb      lsb
//	+----+-+---+
//	|OP  |s|cls|
//	+----+-+---+
type ALUOp uint8

const aluMask OpCode = 0xf0

const (
	// InvalidALUOp is returned by getters when invoked
	// on non ALU OpCodes
	InvalidALUOp ALUOp = 0xff
	// Add - addition
	Add ALUOp = 0x00
	// Sub - subtraction
	Sub ALUOp = 0x10
	// Mul - multiplication
	Mul ALUOp = 0x20
	// Div - division
	Div ALUOp = 0x30
	// Or - bitwise or
	Or ALUOp = 0x40
	// And - bitwise and
	And ALUOp = 0x50
	// LSh - bitwise shift left
	LSh ALUOp = 0x60
	// RSh - bitwise shift right
	RSh ALUOp = 0x70
	// Neg - sign/unsign signing bit
	Neg ALUOp = 0x80
	// Mod - modulo
	Mod ALUOp = 0x90
	// Xor - bitwise xor
	Xor ALUOp = 0xa0
	// Mov - move value from one place to another
	Mov ALUOp = 0xb0
	// ArSh - arithmatic shift
	ArSh ALUOp = 0xc0
	// Swap - endian conversions
	Swap ALUOp = 0xd0
)

// HostTo converts from host to another endianness.
func HostTo(endian Endianness, dst Register, size Size) Instruction {
	var imm int64
	switch size {
	case Half:
		imm = 16
	case Word:
		imm = 32
	case DWord:
		imm = 64
	default:
		return Instruction{OpCode: InvalidOpCode}
	}

	return Instruction{
		OpCode:   OpCode(ALUClass).SetALUOp(Swap).SetSource(Source(endian)),
		Dst:      dst,
		Constant: imm,
	}
}

// Op returns the OpCode for an ALU operation with a given source.
func (op ALUOp) Op(source Source) OpCode {
	return OpCode(ALU64Class).SetALUOp(op).SetSource(source)
}

// Reg emits `dst (op) src`.
func (op ALUOp) Reg(dst, src Register) Instruction {
	return Instruction{
		OpCode: op.Op(RegSource),
		Dst:    dst,
		Src:    src,
	}
}

// Imm emits `dst (op) value`.
func (op ALUOp) Imm(dst Register, value int32) Instruction {
	return Instruction{
		OpCode:   op.Op(ImmSource),
		Dst:      dst,
		Constant: int64(value),
	}
}

// Op32 returns the OpCode for a 32-bit ALU operation with a given source.
func (op ALUOp) Op32(source Source) OpCode {
	return OpCode(ALUClass).SetALUOp(op).SetSource(source)
}

// Reg32 emits `dst (op) src`, zeroing the upper 32 bit of dst.
func (op ALUOp) Reg32(dst, src Register) Instruction {
	return Instruction{
		OpCode: op.Op32(RegSource),
		Dst:    dst,
		Src:    src,
	}
}

// Imm32 emits `dst (op) value`, zeroing the upper 32 bit of dst.
func (op ALUOp) Imm32(dst Register, value int32) Instruction {
	return Instruction{
		OpCode:   op.Op32(ImmSource),
		Dst:      dst,
		Constant: int64(value),
	}
}
