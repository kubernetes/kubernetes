package asm

import (
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"strings"

	"github.com/cilium/ebpf/internal/unix"
)

// InstructionSize is the size of a BPF instruction in bytes
const InstructionSize = 8

// RawInstructionOffset is an offset in units of raw BPF instructions.
type RawInstructionOffset uint64

// Bytes returns the offset of an instruction in bytes.
func (rio RawInstructionOffset) Bytes() uint64 {
	return uint64(rio) * InstructionSize
}

// Instruction is a single eBPF instruction.
type Instruction struct {
	OpCode    OpCode
	Dst       Register
	Src       Register
	Offset    int16
	Constant  int64
	Reference string
	Symbol    string
}

// Sym creates a symbol.
func (ins Instruction) Sym(name string) Instruction {
	ins.Symbol = name
	return ins
}

// Unmarshal decodes a BPF instruction.
func (ins *Instruction) Unmarshal(r io.Reader, bo binary.ByteOrder) (uint64, error) {
	var bi bpfInstruction
	err := binary.Read(r, bo, &bi)
	if err != nil {
		return 0, err
	}

	ins.OpCode = bi.OpCode
	ins.Offset = bi.Offset
	ins.Constant = int64(bi.Constant)
	ins.Dst, ins.Src, err = bi.Registers.Unmarshal(bo)
	if err != nil {
		return 0, fmt.Errorf("can't unmarshal registers: %s", err)
	}

	if !bi.OpCode.IsDWordLoad() {
		return InstructionSize, nil
	}

	var bi2 bpfInstruction
	if err := binary.Read(r, bo, &bi2); err != nil {
		// No Wrap, to avoid io.EOF clash
		return 0, errors.New("64bit immediate is missing second half")
	}
	if bi2.OpCode != 0 || bi2.Offset != 0 || bi2.Registers != 0 {
		return 0, errors.New("64bit immediate has non-zero fields")
	}
	ins.Constant = int64(uint64(uint32(bi2.Constant))<<32 | uint64(uint32(bi.Constant)))

	return 2 * InstructionSize, nil
}

// Marshal encodes a BPF instruction.
func (ins Instruction) Marshal(w io.Writer, bo binary.ByteOrder) (uint64, error) {
	if ins.OpCode == InvalidOpCode {
		return 0, errors.New("invalid opcode")
	}

	isDWordLoad := ins.OpCode.IsDWordLoad()

	cons := int32(ins.Constant)
	if isDWordLoad {
		// Encode least significant 32bit first for 64bit operations.
		cons = int32(uint32(ins.Constant))
	}

	regs, err := newBPFRegisters(ins.Dst, ins.Src, bo)
	if err != nil {
		return 0, fmt.Errorf("can't marshal registers: %s", err)
	}

	bpfi := bpfInstruction{
		ins.OpCode,
		regs,
		ins.Offset,
		cons,
	}

	if err := binary.Write(w, bo, &bpfi); err != nil {
		return 0, err
	}

	if !isDWordLoad {
		return InstructionSize, nil
	}

	bpfi = bpfInstruction{
		Constant: int32(ins.Constant >> 32),
	}

	if err := binary.Write(w, bo, &bpfi); err != nil {
		return 0, err
	}

	return 2 * InstructionSize, nil
}

// RewriteMapPtr changes an instruction to use a new map fd.
//
// Returns an error if the instruction doesn't load a map.
func (ins *Instruction) RewriteMapPtr(fd int) error {
	if !ins.OpCode.IsDWordLoad() {
		return fmt.Errorf("%s is not a 64 bit load", ins.OpCode)
	}

	if ins.Src != PseudoMapFD && ins.Src != PseudoMapValue {
		return errors.New("not a load from a map")
	}

	// Preserve the offset value for direct map loads.
	offset := uint64(ins.Constant) & (math.MaxUint32 << 32)
	rawFd := uint64(uint32(fd))
	ins.Constant = int64(offset | rawFd)
	return nil
}

// MapPtr returns the map fd for this instruction.
//
// The result is undefined if the instruction is not a load from a map,
// see IsLoadFromMap.
func (ins *Instruction) MapPtr() int {
	return int(int32(uint64(ins.Constant) & math.MaxUint32))
}

// RewriteMapOffset changes the offset of a direct load from a map.
//
// Returns an error if the instruction is not a direct load.
func (ins *Instruction) RewriteMapOffset(offset uint32) error {
	if !ins.OpCode.IsDWordLoad() {
		return fmt.Errorf("%s is not a 64 bit load", ins.OpCode)
	}

	if ins.Src != PseudoMapValue {
		return errors.New("not a direct load from a map")
	}

	fd := uint64(ins.Constant) & math.MaxUint32
	ins.Constant = int64(uint64(offset)<<32 | fd)
	return nil
}

func (ins *Instruction) mapOffset() uint32 {
	return uint32(uint64(ins.Constant) >> 32)
}

// IsLoadFromMap returns true if the instruction loads from a map.
//
// This covers both loading the map pointer and direct map value loads.
func (ins *Instruction) IsLoadFromMap() bool {
	return ins.OpCode == LoadImmOp(DWord) && (ins.Src == PseudoMapFD || ins.Src == PseudoMapValue)
}

// IsFunctionCall returns true if the instruction calls another BPF function.
//
// This is not the same thing as a BPF helper call.
func (ins *Instruction) IsFunctionCall() bool {
	return ins.OpCode.JumpOp() == Call && ins.Src == PseudoCall
}

// IsConstantLoad returns true if the instruction loads a constant of the
// given size.
func (ins *Instruction) IsConstantLoad(size Size) bool {
	return ins.OpCode == LoadImmOp(size) && ins.Src == R0 && ins.Offset == 0
}

// Format implements fmt.Formatter.
func (ins Instruction) Format(f fmt.State, c rune) {
	if c != 'v' {
		fmt.Fprintf(f, "{UNRECOGNIZED: %c}", c)
		return
	}

	op := ins.OpCode

	if op == InvalidOpCode {
		fmt.Fprint(f, "INVALID")
		return
	}

	// Omit trailing space for Exit
	if op.JumpOp() == Exit {
		fmt.Fprint(f, op)
		return
	}

	if ins.IsLoadFromMap() {
		fd := ins.MapPtr()
		switch ins.Src {
		case PseudoMapFD:
			fmt.Fprintf(f, "LoadMapPtr dst: %s fd: %d", ins.Dst, fd)

		case PseudoMapValue:
			fmt.Fprintf(f, "LoadMapValue dst: %s, fd: %d off: %d", ins.Dst, fd, ins.mapOffset())
		}

		goto ref
	}

	fmt.Fprintf(f, "%v ", op)
	switch cls := op.Class(); cls {
	case LdClass, LdXClass, StClass, StXClass:
		switch op.Mode() {
		case ImmMode:
			fmt.Fprintf(f, "dst: %s imm: %d", ins.Dst, ins.Constant)
		case AbsMode:
			fmt.Fprintf(f, "imm: %d", ins.Constant)
		case IndMode:
			fmt.Fprintf(f, "dst: %s src: %s imm: %d", ins.Dst, ins.Src, ins.Constant)
		case MemMode:
			fmt.Fprintf(f, "dst: %s src: %s off: %d imm: %d", ins.Dst, ins.Src, ins.Offset, ins.Constant)
		case XAddMode:
			fmt.Fprintf(f, "dst: %s src: %s", ins.Dst, ins.Src)
		}

	case ALU64Class, ALUClass:
		fmt.Fprintf(f, "dst: %s ", ins.Dst)
		if op.ALUOp() == Swap || op.Source() == ImmSource {
			fmt.Fprintf(f, "imm: %d", ins.Constant)
		} else {
			fmt.Fprintf(f, "src: %s", ins.Src)
		}

	case JumpClass:
		switch jop := op.JumpOp(); jop {
		case Call:
			if ins.Src == PseudoCall {
				// bpf-to-bpf call
				fmt.Fprint(f, ins.Constant)
			} else {
				fmt.Fprint(f, BuiltinFunc(ins.Constant))
			}

		default:
			fmt.Fprintf(f, "dst: %s off: %d ", ins.Dst, ins.Offset)
			if op.Source() == ImmSource {
				fmt.Fprintf(f, "imm: %d", ins.Constant)
			} else {
				fmt.Fprintf(f, "src: %s", ins.Src)
			}
		}
	}

ref:
	if ins.Reference != "" {
		fmt.Fprintf(f, " <%s>", ins.Reference)
	}
}

// Instructions is an eBPF program.
type Instructions []Instruction

func (insns Instructions) String() string {
	return fmt.Sprint(insns)
}

// RewriteMapPtr rewrites all loads of a specific map pointer to a new fd.
//
// Returns an error if the symbol isn't used, see IsUnreferencedSymbol.
func (insns Instructions) RewriteMapPtr(symbol string, fd int) error {
	if symbol == "" {
		return errors.New("empty symbol")
	}

	found := false
	for i := range insns {
		ins := &insns[i]
		if ins.Reference != symbol {
			continue
		}

		if err := ins.RewriteMapPtr(fd); err != nil {
			return err
		}

		found = true
	}

	if !found {
		return &unreferencedSymbolError{symbol}
	}

	return nil
}

// SymbolOffsets returns the set of symbols and their offset in
// the instructions.
func (insns Instructions) SymbolOffsets() (map[string]int, error) {
	offsets := make(map[string]int)

	for i, ins := range insns {
		if ins.Symbol == "" {
			continue
		}

		if _, ok := offsets[ins.Symbol]; ok {
			return nil, fmt.Errorf("duplicate symbol %s", ins.Symbol)
		}

		offsets[ins.Symbol] = i
	}

	return offsets, nil
}

// ReferenceOffsets returns the set of references and their offset in
// the instructions.
func (insns Instructions) ReferenceOffsets() map[string][]int {
	offsets := make(map[string][]int)

	for i, ins := range insns {
		if ins.Reference == "" {
			continue
		}

		offsets[ins.Reference] = append(offsets[ins.Reference], i)
	}

	return offsets
}

// Format implements fmt.Formatter.
//
// You can control indentation of symbols by
// specifying a width. Setting a precision controls the indentation of
// instructions.
// The default character is a tab, which can be overridden by specifying
// the ' ' space flag.
func (insns Instructions) Format(f fmt.State, c rune) {
	if c != 's' && c != 'v' {
		fmt.Fprintf(f, "{UNKNOWN FORMAT '%c'}", c)
		return
	}

	// Precision is better in this case, because it allows
	// specifying 0 padding easily.
	padding, ok := f.Precision()
	if !ok {
		padding = 1
	}

	indent := strings.Repeat("\t", padding)
	if f.Flag(' ') {
		indent = strings.Repeat(" ", padding)
	}

	symPadding, ok := f.Width()
	if !ok {
		symPadding = padding - 1
	}
	if symPadding < 0 {
		symPadding = 0
	}

	symIndent := strings.Repeat("\t", symPadding)
	if f.Flag(' ') {
		symIndent = strings.Repeat(" ", symPadding)
	}

	// Guess how many digits we need at most, by assuming that all instructions
	// are double wide.
	highestOffset := len(insns) * 2
	offsetWidth := int(math.Ceil(math.Log10(float64(highestOffset))))

	iter := insns.Iterate()
	for iter.Next() {
		if iter.Ins.Symbol != "" {
			fmt.Fprintf(f, "%s%s:\n", symIndent, iter.Ins.Symbol)
		}
		fmt.Fprintf(f, "%s%*d: %v\n", indent, offsetWidth, iter.Offset, iter.Ins)
	}
}

// Marshal encodes a BPF program into the kernel format.
func (insns Instructions) Marshal(w io.Writer, bo binary.ByteOrder) error {
	for i, ins := range insns {
		_, err := ins.Marshal(w, bo)
		if err != nil {
			return fmt.Errorf("instruction %d: %w", i, err)
		}
	}
	return nil
}

// Tag calculates the kernel tag for a series of instructions.
//
// It mirrors bpf_prog_calc_tag in the kernel and so can be compared
// to ProgramInfo.Tag to figure out whether a loaded program matches
// certain instructions.
func (insns Instructions) Tag(bo binary.ByteOrder) (string, error) {
	h := sha1.New()
	for i, ins := range insns {
		if ins.IsLoadFromMap() {
			ins.Constant = 0
		}
		_, err := ins.Marshal(h, bo)
		if err != nil {
			return "", fmt.Errorf("instruction %d: %w", i, err)
		}
	}
	return hex.EncodeToString(h.Sum(nil)[:unix.BPF_TAG_SIZE]), nil
}

// Iterate allows iterating a BPF program while keeping track of
// various offsets.
//
// Modifying the instruction slice will lead to undefined behaviour.
func (insns Instructions) Iterate() *InstructionIterator {
	return &InstructionIterator{insns: insns}
}

// InstructionIterator iterates over a BPF program.
type InstructionIterator struct {
	insns Instructions
	// The instruction in question.
	Ins *Instruction
	// The index of the instruction in the original instruction slice.
	Index int
	// The offset of the instruction in raw BPF instructions. This accounts
	// for double-wide instructions.
	Offset RawInstructionOffset
}

// Next returns true as long as there are any instructions remaining.
func (iter *InstructionIterator) Next() bool {
	if len(iter.insns) == 0 {
		return false
	}

	if iter.Ins != nil {
		iter.Index++
		iter.Offset += RawInstructionOffset(iter.Ins.OpCode.rawInstructions())
	}
	iter.Ins = &iter.insns[0]
	iter.insns = iter.insns[1:]
	return true
}

type bpfInstruction struct {
	OpCode    OpCode
	Registers bpfRegisters
	Offset    int16
	Constant  int32
}

type bpfRegisters uint8

func newBPFRegisters(dst, src Register, bo binary.ByteOrder) (bpfRegisters, error) {
	switch bo {
	case binary.LittleEndian:
		return bpfRegisters((src << 4) | (dst & 0xF)), nil
	case binary.BigEndian:
		return bpfRegisters((dst << 4) | (src & 0xF)), nil
	default:
		return 0, fmt.Errorf("unrecognized ByteOrder %T", bo)
	}
}

func (r bpfRegisters) Unmarshal(bo binary.ByteOrder) (dst, src Register, err error) {
	switch bo {
	case binary.LittleEndian:
		return Register(r & 0xF), Register(r >> 4), nil
	case binary.BigEndian:
		return Register(r >> 4), Register(r & 0xf), nil
	default:
		return 0, 0, fmt.Errorf("unrecognized ByteOrder %T", bo)
	}
}

type unreferencedSymbolError struct {
	symbol string
}

func (use *unreferencedSymbolError) Error() string {
	return fmt.Sprintf("unreferenced symbol %s", use.symbol)
}

// IsUnreferencedSymbol returns true if err was caused by
// an unreferenced symbol.
func IsUnreferencedSymbol(err error) bool {
	_, ok := err.(*unreferencedSymbolError)
	return ok
}
