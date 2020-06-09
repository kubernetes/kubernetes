package asm

import (
	"fmt"
	"strings"
)

//go:generate stringer -output opcode_string.go -type=Class

type encoding int

const (
	unknownEncoding encoding = iota
	loadOrStore
	jumpOrALU
)

// Class of operations
//
//    msb      lsb
//    +---+--+---+
//    |  ??  |CLS|
//    +---+--+---+
type Class uint8

const classMask OpCode = 0x07

const (
	// LdClass load memory
	LdClass Class = 0x00
	// LdXClass load memory from constant
	LdXClass Class = 0x01
	// StClass load register from memory
	StClass Class = 0x02
	// StXClass load register from constant
	StXClass Class = 0x03
	// ALUClass arithmetic operators
	ALUClass Class = 0x04
	// JumpClass jump operators
	JumpClass Class = 0x05
	// ALU64Class arithmetic in 64 bit mode
	ALU64Class Class = 0x07
)

func (cls Class) encoding() encoding {
	switch cls {
	case LdClass, LdXClass, StClass, StXClass:
		return loadOrStore
	case ALU64Class, ALUClass, JumpClass:
		return jumpOrALU
	default:
		return unknownEncoding
	}
}

// OpCode is a packed eBPF opcode.
//
// Its encoding is defined by a Class value:
//
//    msb      lsb
//    +----+-+---+
//    | ???? |CLS|
//    +----+-+---+
type OpCode uint8

// InvalidOpCode is returned by setters on OpCode
const InvalidOpCode OpCode = 0xff

// marshalledInstructions returns the number of BPF instructions required
// to encode this opcode.
func (op OpCode) marshalledInstructions() int {
	if op == LoadImmOp(DWord) {
		return 2
	}
	return 1
}

func (op OpCode) isDWordLoad() bool {
	return op == LoadImmOp(DWord)
}

// Class returns the class of operation.
func (op OpCode) Class() Class {
	return Class(op & classMask)
}

// Mode returns the mode for load and store operations.
func (op OpCode) Mode() Mode {
	if op.Class().encoding() != loadOrStore {
		return InvalidMode
	}
	return Mode(op & modeMask)
}

// Size returns the size for load and store operations.
func (op OpCode) Size() Size {
	if op.Class().encoding() != loadOrStore {
		return InvalidSize
	}
	return Size(op & sizeMask)
}

// Source returns the source for branch and ALU operations.
func (op OpCode) Source() Source {
	if op.Class().encoding() != jumpOrALU || op.ALUOp() == Swap {
		return InvalidSource
	}
	return Source(op & sourceMask)
}

// ALUOp returns the ALUOp.
func (op OpCode) ALUOp() ALUOp {
	if op.Class().encoding() != jumpOrALU {
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
func (op OpCode) JumpOp() JumpOp {
	if op.Class().encoding() != jumpOrALU {
		return InvalidJumpOp
	}
	return JumpOp(op & jumpMask)
}

// SetMode sets the mode on load and store operations.
//
// Returns InvalidOpCode if op is of the wrong class.
func (op OpCode) SetMode(mode Mode) OpCode {
	if op.Class().encoding() != loadOrStore || !valid(OpCode(mode), modeMask) {
		return InvalidOpCode
	}
	return (op & ^modeMask) | OpCode(mode)
}

// SetSize sets the size on load and store operations.
//
// Returns InvalidOpCode if op is of the wrong class.
func (op OpCode) SetSize(size Size) OpCode {
	if op.Class().encoding() != loadOrStore || !valid(OpCode(size), sizeMask) {
		return InvalidOpCode
	}
	return (op & ^sizeMask) | OpCode(size)
}

// SetSource sets the source on jump and ALU operations.
//
// Returns InvalidOpCode if op is of the wrong class.
func (op OpCode) SetSource(source Source) OpCode {
	if op.Class().encoding() != jumpOrALU || !valid(OpCode(source), sourceMask) {
		return InvalidOpCode
	}
	return (op & ^sourceMask) | OpCode(source)
}

// SetALUOp sets the ALUOp on ALU operations.
//
// Returns InvalidOpCode if op is of the wrong class.
func (op OpCode) SetALUOp(alu ALUOp) OpCode {
	class := op.Class()
	if (class != ALUClass && class != ALU64Class) || !valid(OpCode(alu), aluMask) {
		return InvalidOpCode
	}
	return (op & ^aluMask) | OpCode(alu)
}

// SetJumpOp sets the JumpOp on jump operations.
//
// Returns InvalidOpCode if op is of the wrong class.
func (op OpCode) SetJumpOp(jump JumpOp) OpCode {
	if op.Class() != JumpClass || !valid(OpCode(jump), jumpMask) {
		return InvalidOpCode
	}
	return (op & ^jumpMask) | OpCode(jump)
}

func (op OpCode) String() string {
	var f strings.Builder

	switch class := op.Class(); class {
	case LdClass, LdXClass, StClass, StXClass:
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

	case ALU64Class, ALUClass:
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

	case JumpClass:
		f.WriteString(op.JumpOp().String())
		if jop := op.JumpOp(); jop != Exit && jop != Call {
			f.WriteString(strings.TrimSuffix(op.Source().String(), "Source"))
		}

	default:
		fmt.Fprintf(&f, "%#x", op)
	}

	return f.String()
}

// valid returns true if all bits in value are covered by mask.
func valid(value, mask OpCode) bool {
	return value & ^mask == 0
}
