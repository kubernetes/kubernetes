package asm

import (
	"fmt"
	"strings"
)

//go:generate stringer -output opcode_string.go -type=Class

// Class of operations
//
//	msb      lsb
//	+---+--+---+
//	|  ??  |CLS|
//	+---+--+---+
type Class uint8

const classMask OpCode = 0x07

const (
	// LdClass loads immediate values into registers.
	// Also used for non-standard load operations from cBPF.
	LdClass Class = 0x00
	// LdXClass loads memory into registers.
	LdXClass Class = 0x01
	// StClass stores immediate values to memory.
	StClass Class = 0x02
	// StXClass stores registers to memory.
	StXClass Class = 0x03
	// ALUClass describes arithmetic operators.
	ALUClass Class = 0x04
	// JumpClass describes jump operators.
	JumpClass Class = 0x05
	// Jump32Class describes jump operators with 32-bit comparisons.
	// Requires kernel 5.1.
	Jump32Class Class = 0x06
	// ALU64Class describes arithmetic operators in 64-bit mode.
	ALU64Class Class = 0x07
)

// IsLoad checks if this is either LdClass or LdXClass.
func (cls Class) IsLoad() bool {
	return cls == LdClass || cls == LdXClass
}

// IsStore checks if this is either StClass or StXClass.
func (cls Class) IsStore() bool {
	return cls == StClass || cls == StXClass
}

func (cls Class) isLoadOrStore() bool {
	return cls.IsLoad() || cls.IsStore()
}

// IsALU checks if this is either ALUClass or ALU64Class.
func (cls Class) IsALU() bool {
	return cls == ALUClass || cls == ALU64Class
}

// IsJump checks if this is either JumpClass or Jump32Class.
func (cls Class) IsJump() bool {
	return cls == JumpClass || cls == Jump32Class
}

func (cls Class) isJumpOrALU() bool {
	return cls.IsJump() || cls.IsALU()
}

// OpCode is a packed eBPF opcode.
//
// Its encoding is defined by a Class value:
//
//	msb      lsb
//	+----+-+---+
//	| ???? |CLS|
//	+----+-+---+
type OpCode uint8

// InvalidOpCode is returned by setters on OpCode
const InvalidOpCode OpCode = 0xff

// rawInstructions returns the number of BPF instructions required
// to encode this opcode.
func (op OpCode) rawInstructions() int {
	if op.IsDWordLoad() {
		return 2
	}
	return 1
}

func (op OpCode) IsDWordLoad() bool {
	return op == LoadImmOp(DWord)
}

// Class returns the class of operation.
func (op OpCode) Class() Class {
	return Class(op & classMask)
}

// Mode returns the mode for load and store operations.
func (op OpCode) Mode() Mode {
	if !op.Class().isLoadOrStore() {
		return InvalidMode
	}
	return Mode(op & modeMask)
}

// Size returns the size for load and store operations.
func (op OpCode) Size() Size {
	if !op.Class().isLoadOrStore() {
		return InvalidSize
	}
	return Size(op & sizeMask)
}

// Source returns the source for branch and ALU operations.
func (op OpCode) Source() Source {
	if !op.Class().isJumpOrALU() || op.ALUOp() == Swap {
		return InvalidSource
	}
	return Source(op & sourceMask)
}

// ALUOp returns the ALUOp.
func (op OpCode) ALUOp() ALUOp {
	if !op.Class().IsALU() {
		return InvalidALUOp
	}
	return ALUOp(op & aluMask)
}

// Endianness returns the Endianness for a byte swap instruction.
func (op OpCode) Endianness() Endianness {
	if op.ALUOp() != Swap {
		return InvalidEndian
	}
	return Endianness(op & endianMask)
}

// JumpOp returns the JumpOp.
// Returns InvalidJumpOp if it doesn't encode a jump.
func (op OpCode) JumpOp() JumpOp {
	if !op.Class().IsJump() {
		return InvalidJumpOp
	}

	jumpOp := JumpOp(op & jumpMask)

	// Some JumpOps are only supported by JumpClass, not Jump32Class.
	if op.Class() == Jump32Class && (jumpOp == Exit || jumpOp == Call || jumpOp == Ja) {
		return InvalidJumpOp
	}

	return jumpOp
}

// SetMode sets the mode on load and store operations.
//
// Returns InvalidOpCode if op is of the wrong class.
func (op OpCode) SetMode(mode Mode) OpCode {
	if !op.Class().isLoadOrStore() || !valid(OpCode(mode), modeMask) {
		return InvalidOpCode
	}
	return (op & ^modeMask) | OpCode(mode)
}

// SetSize sets the size on load and store operations.
//
// Returns InvalidOpCode if op is of the wrong class.
func (op OpCode) SetSize(size Size) OpCode {
	if !op.Class().isLoadOrStore() || !valid(OpCode(size), sizeMask) {
		return InvalidOpCode
	}
	return (op & ^sizeMask) | OpCode(size)
}

// SetSource sets the source on jump and ALU operations.
//
// Returns InvalidOpCode if op is of the wrong class.
func (op OpCode) SetSource(source Source) OpCode {
	if !op.Class().isJumpOrALU() || !valid(OpCode(source), sourceMask) {
		return InvalidOpCode
	}
	return (op & ^sourceMask) | OpCode(source)
}

// SetALUOp sets the ALUOp on ALU operations.
//
// Returns InvalidOpCode if op is of the wrong class.
func (op OpCode) SetALUOp(alu ALUOp) OpCode {
	if !op.Class().IsALU() || !valid(OpCode(alu), aluMask) {
		return InvalidOpCode
	}
	return (op & ^aluMask) | OpCode(alu)
}

// SetJumpOp sets the JumpOp on jump operations.
//
// Returns InvalidOpCode if op is of the wrong class.
func (op OpCode) SetJumpOp(jump JumpOp) OpCode {
	if !op.Class().IsJump() || !valid(OpCode(jump), jumpMask) {
		return InvalidOpCode
	}

	newOp := (op & ^jumpMask) | OpCode(jump)

	// Check newOp is legal.
	if newOp.JumpOp() == InvalidJumpOp {
		return InvalidOpCode
	}

	return newOp
}

func (op OpCode) String() string {
	var f strings.Builder

	switch class := op.Class(); {
	case class.isLoadOrStore():
		f.WriteString(strings.TrimSuffix(class.String(), "Class"))

		mode := op.Mode()
		f.WriteString(strings.TrimSuffix(mode.String(), "Mode"))

		switch op.Size() {
		case DWord:
			f.WriteString("DW")
		case Word:
			f.WriteString("W")
		case Half:
			f.WriteString("H")
		case Byte:
			f.WriteString("B")
		}

	case class.IsALU():
		f.WriteString(op.ALUOp().String())

		if op.ALUOp() == Swap {
			// Width for Endian is controlled by Constant
			f.WriteString(op.Endianness().String())
		} else {
			if class == ALUClass {
				f.WriteString("32")
			}

			f.WriteString(strings.TrimSuffix(op.Source().String(), "Source"))
		}

	case class.IsJump():
		f.WriteString(op.JumpOp().String())

		if class == Jump32Class {
			f.WriteString("32")
		}

		if jop := op.JumpOp(); jop != Exit && jop != Call {
			f.WriteString(strings.TrimSuffix(op.Source().String(), "Source"))
		}

	default:
		fmt.Fprintf(&f, "OpCode(%#x)", uint8(op))
	}

	return f.String()
}

// valid returns true if all bits in value are covered by mask.
func valid(value, mask OpCode) bool {
	return value & ^mask == 0
}
