package asm

//go:generate go run golang.org/x/tools/cmd/stringer@latest -output load_store_string.go -type=Mode,Size

// Mode for load and store operations
//
//	msb      lsb
//	+---+--+---+
//	|MDE|sz|cls|
//	+---+--+---+
type Mode uint8

const modeMask OpCode = 0xe0

const (
	// InvalidMode is returned by getters when invoked
	// on non load / store OpCodes
	InvalidMode Mode = 0xff
	// ImmMode - immediate value
	ImmMode Mode = 0x00
	// AbsMode - immediate value + offset
	AbsMode Mode = 0x20
	// IndMode - indirect (imm+src)
	IndMode Mode = 0x40
	// MemMode - load from memory
	MemMode Mode = 0x60
	// MemSXMode - load from memory, sign extension
	MemSXMode Mode = 0x80
	// XAddMode - add atomically across processors.
	XAddMode Mode = 0xc0
)

// Size of load and store operations
//
//	msb      lsb
//	+---+--+---+
//	|mde|SZ|cls|
//	+---+--+---+
type Size uint8

const sizeMask OpCode = 0x18

const (
	// InvalidSize is returned by getters when invoked
	// on non load / store OpCodes
	InvalidSize Size = 0xff
	// DWord - double word; 64 bits
	DWord Size = 0x18
	// Word - word; 32 bits
	Word Size = 0x00
	// Half - half-word; 16 bits
	Half Size = 0x08
	// Byte - byte; 8 bits
	Byte Size = 0x10
)

// Sizeof returns the size in bytes.
func (s Size) Sizeof() int {
	switch s {
	case DWord:
		return 8
	case Word:
		return 4
	case Half:
		return 2
	case Byte:
		return 1
	default:
		return -1
	}
}

// LoadMemOp returns the OpCode to load a value of given size from memory.
func LoadMemOp(size Size) OpCode {
	return OpCode(LdXClass).SetMode(MemMode).SetSize(size)
}

// LoadMemSXOp returns the OpCode to load a value of given size from memory sign extended.
func LoadMemSXOp(size Size) OpCode {
	return OpCode(LdXClass).SetMode(MemSXMode).SetSize(size)
}

// LoadMem emits `dst = *(size *)(src + offset)`.
func LoadMem(dst, src Register, offset int16, size Size) Instruction {
	return Instruction{
		OpCode: LoadMemOp(size),
		Dst:    dst,
		Src:    src,
		Offset: offset,
	}
}

// LoadMemSX emits `dst = *(size *)(src + offset)` but sign extends dst.
func LoadMemSX(dst, src Register, offset int16, size Size) Instruction {
	if size == DWord {
		return Instruction{OpCode: InvalidOpCode}
	}

	return Instruction{
		OpCode: LoadMemSXOp(size),
		Dst:    dst,
		Src:    src,
		Offset: offset,
	}
}

// LoadImmOp returns the OpCode to load an immediate of given size.
//
// As of kernel 4.20, only DWord size is accepted.
func LoadImmOp(size Size) OpCode {
	return OpCode(LdClass).SetMode(ImmMode).SetSize(size)
}

// LoadImm emits `dst = (size)value`.
//
// As of kernel 4.20, only DWord size is accepted.
func LoadImm(dst Register, value int64, size Size) Instruction {
	return Instruction{
		OpCode:   LoadImmOp(size),
		Dst:      dst,
		Constant: value,
	}
}

// LoadMapPtr stores a pointer to a map in dst.
func LoadMapPtr(dst Register, fd int) Instruction {
	if fd < 0 {
		return Instruction{OpCode: InvalidOpCode}
	}

	return Instruction{
		OpCode:   LoadImmOp(DWord),
		Dst:      dst,
		Src:      PseudoMapFD,
		Constant: int64(uint32(fd)),
	}
}

// LoadMapValue stores a pointer to the value at a certain offset of a map.
func LoadMapValue(dst Register, fd int, offset uint32) Instruction {
	if fd < 0 {
		return Instruction{OpCode: InvalidOpCode}
	}

	fdAndOffset := (uint64(offset) << 32) | uint64(uint32(fd))
	return Instruction{
		OpCode:   LoadImmOp(DWord),
		Dst:      dst,
		Src:      PseudoMapValue,
		Constant: int64(fdAndOffset),
	}
}

// LoadIndOp returns the OpCode for loading a value of given size from an sk_buff.
func LoadIndOp(size Size) OpCode {
	return OpCode(LdClass).SetMode(IndMode).SetSize(size)
}

// LoadInd emits `dst = ntoh(*(size *)(((sk_buff *)R6)->data + src + offset))`.
func LoadInd(dst, src Register, offset int32, size Size) Instruction {
	return Instruction{
		OpCode:   LoadIndOp(size),
		Dst:      dst,
		Src:      src,
		Constant: int64(offset),
	}
}

// LoadAbsOp returns the OpCode for loading a value of given size from an sk_buff.
func LoadAbsOp(size Size) OpCode {
	return OpCode(LdClass).SetMode(AbsMode).SetSize(size)
}

// LoadAbs emits `r0 = ntoh(*(size *)(((sk_buff *)R6)->data + offset))`.
func LoadAbs(offset int32, size Size) Instruction {
	return Instruction{
		OpCode:   LoadAbsOp(size),
		Dst:      R0,
		Constant: int64(offset),
	}
}

// StoreMemOp returns the OpCode for storing a register of given size in memory.
func StoreMemOp(size Size) OpCode {
	return OpCode(StXClass).SetMode(MemMode).SetSize(size)
}

// StoreMem emits `*(size *)(dst + offset) = src`
func StoreMem(dst Register, offset int16, src Register, size Size) Instruction {
	return Instruction{
		OpCode: StoreMemOp(size),
		Dst:    dst,
		Src:    src,
		Offset: offset,
	}
}

// StoreImmOp returns the OpCode for storing an immediate of given size in memory.
func StoreImmOp(size Size) OpCode {
	return OpCode(StClass).SetMode(MemMode).SetSize(size)
}

// StoreImm emits `*(size *)(dst + offset) = value`.
func StoreImm(dst Register, offset int16, value int64, size Size) Instruction {
	return Instruction{
		OpCode:   StoreImmOp(size),
		Dst:      dst,
		Offset:   offset,
		Constant: value,
	}
}

// StoreXAddOp returns the OpCode to atomically add a register to a value in memory.
func StoreXAddOp(size Size) OpCode {
	return OpCode(StXClass).SetMode(XAddMode).SetSize(size)
}

// StoreXAdd atomically adds src to *dst.
func StoreXAdd(dst, src Register, size Size) Instruction {
	return Instruction{
		OpCode: StoreXAddOp(size),
		Dst:    dst,
		Src:    src,
	}
}
