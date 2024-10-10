package asm

//go:generate go run golang.org/x/tools/cmd/stringer@latest -output alu_string.go -type=Source,Endianness,ALUOp

// Source of ALU / ALU64 / Branch operations
//
//	msb              lsb
//	+------------+-+---+
//	|     op     |S|cls|
//	+------------+-+---+
type Source uint16

const sourceMask OpCode = 0x0008

// Source bitmask
const (
	// InvalidSource is returned by getters when invoked
	// on non ALU / branch OpCodes.
	InvalidSource Source = 0xffff
	// ImmSource src is from constant
	ImmSource Source = 0x0000
	// RegSource src is from register
	RegSource Source = 0x0008
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
//	msb              lsb
//	+-------+----+-+---+
//	|  EXT  | OP |s|cls|
//	+-------+----+-+---+
type ALUOp uint16

const aluMask OpCode = 0x3ff0

const (
	// InvalidALUOp is returned by getters when invoked
	// on non ALU OpCodes
	InvalidALUOp ALUOp = 0xffff
	// Add - addition
	Add ALUOp = 0x0000
	// Sub - subtraction
	Sub ALUOp = 0x0010
	// Mul - multiplication
	Mul ALUOp = 0x0020
	// Div - division
	Div ALUOp = 0x0030
	// SDiv - signed division
	SDiv ALUOp = Div + 0x0100
	// Or - bitwise or
	Or ALUOp = 0x0040
	// And - bitwise and
	And ALUOp = 0x0050
	// LSh - bitwise shift left
	LSh ALUOp = 0x0060
	// RSh - bitwise shift right
	RSh ALUOp = 0x0070
	// Neg - sign/unsign signing bit
	Neg ALUOp = 0x0080
	// Mod - modulo
	Mod ALUOp = 0x0090
	// SMod - signed modulo
	SMod ALUOp = Mod + 0x0100
	// Xor - bitwise xor
	Xor ALUOp = 0x00a0
	// Mov - move value from one place to another
	Mov ALUOp = 0x00b0
	// MovSX8 - move lower 8 bits, sign extended upper bits of target
	MovSX8 ALUOp = Mov + 0x0100
	// MovSX16 - move lower 16 bits, sign extended upper bits of target
	MovSX16 ALUOp = Mov + 0x0200
	// MovSX32 - move lower 32 bits, sign extended upper bits of target
	MovSX32 ALUOp = Mov + 0x0300
	// ArSh - arithmetic shift
	ArSh ALUOp = 0x00c0
	// Swap - endian conversions
	Swap ALUOp = 0x00d0
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

// BSwap unconditionally reverses the order of bytes in a register.
func BSwap(dst Register, size Size) Instruction {
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
		OpCode:   OpCode(ALU64Class).SetALUOp(Swap),
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
