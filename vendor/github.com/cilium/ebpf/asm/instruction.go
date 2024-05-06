package asm

import (
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"io"
	"math"
	"sort"
	"strings"

	"github.com/cilium/ebpf/internal/sys"
	"github.com/cilium/ebpf/internal/unix"
)

// InstructionSize is the size of a BPF instruction in bytes
const InstructionSize = 8

// RawInstructionOffset is an offset in units of raw BPF instructions.
type RawInstructionOffset uint64

var ErrUnreferencedSymbol = errors.New("unreferenced symbol")
var ErrUnsatisfiedMapReference = errors.New("unsatisfied map reference")
var ErrUnsatisfiedProgramReference = errors.New("unsatisfied program reference")

// Bytes returns the offset of an instruction in bytes.
func (rio RawInstructionOffset) Bytes() uint64 {
	return uint64(rio) * InstructionSize
}

// Instruction is a single eBPF instruction.
type Instruction struct {
	OpCode   OpCode
	Dst      Register
	Src      Register
	Offset   int16
	Constant int64

	// Metadata contains optional metadata about this instruction.
	Metadata Metadata
}

// Unmarshal decodes a BPF instruction.
func (ins *Instruction) Unmarshal(r io.Reader, bo binary.ByteOrder) (uint64, error) {
	data := make([]byte, InstructionSize)
	if _, err := io.ReadFull(r, data); err != nil {
		return 0, err
	}

	ins.OpCode = OpCode(data[0])

	regs := data[1]
	switch bo {
	case binary.LittleEndian:
		ins.Dst, ins.Src = Register(regs&0xF), Register(regs>>4)
	case binary.BigEndian:
		ins.Dst, ins.Src = Register(regs>>4), Register(regs&0xf)
	}

	ins.Offset = int16(bo.Uint16(data[2:4]))
	// Convert to int32 before widening to int64
	// to ensure the signed bit is carried over.
	ins.Constant = int64(int32(bo.Uint32(data[4:8])))

	if !ins.OpCode.IsDWordLoad() {
		return InstructionSize, nil
	}

	// Pull another instruction from the stream to retrieve the second
	// half of the 64-bit immediate value.
	if _, err := io.ReadFull(r, data); err != nil {
		// No Wrap, to avoid io.EOF clash
		return 0, errors.New("64bit immediate is missing second half")
	}

	// Require that all fields other than the value are zero.
	if bo.Uint32(data[0:4]) != 0 {
		return 0, errors.New("64bit immediate has non-zero fields")
	}

	cons1 := uint32(ins.Constant)
	cons2 := int32(bo.Uint32(data[4:8]))
	ins.Constant = int64(cons2)<<32 | int64(cons1)

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

	data := make([]byte, InstructionSize)
	data[0] = byte(ins.OpCode)
	data[1] = byte(regs)
	bo.PutUint16(data[2:4], uint16(ins.Offset))
	bo.PutUint32(data[4:8], uint32(cons))
	if _, err := w.Write(data); err != nil {
		return 0, err
	}

	if !isDWordLoad {
		return InstructionSize, nil
	}

	// The first half of the second part of a double-wide instruction
	// must be zero. The second half carries the value.
	bo.PutUint32(data[0:4], 0)
	bo.PutUint32(data[4:8], uint32(ins.Constant>>32))
	if _, err := w.Write(data); err != nil {
		return 0, err
	}

	return 2 * InstructionSize, nil
}

// AssociateMap associates a Map with this Instruction.
//
// Implicitly clears the Instruction's Reference field.
//
// Returns an error if the Instruction is not a map load.
func (ins *Instruction) AssociateMap(m FDer) error {
	if !ins.IsLoadFromMap() {
		return errors.New("not a load from a map")
	}

	ins.Metadata.Set(referenceMeta{}, nil)
	ins.Metadata.Set(mapMeta{}, m)

	return nil
}

// RewriteMapPtr changes an instruction to use a new map fd.
//
// Returns an error if the instruction doesn't load a map.
//
// Deprecated: use AssociateMap instead. If you cannot provide a Map,
// wrap an fd in a type implementing FDer.
func (ins *Instruction) RewriteMapPtr(fd int) error {
	if !ins.IsLoadFromMap() {
		return errors.New("not a load from a map")
	}

	ins.encodeMapFD(fd)

	return nil
}

func (ins *Instruction) encodeMapFD(fd int) {
	// Preserve the offset value for direct map loads.
	offset := uint64(ins.Constant) & (math.MaxUint32 << 32)
	rawFd := uint64(uint32(fd))
	ins.Constant = int64(offset | rawFd)
}

// MapPtr returns the map fd for this instruction.
//
// The result is undefined if the instruction is not a load from a map,
// see IsLoadFromMap.
//
// Deprecated: use Map() instead.
func (ins *Instruction) MapPtr() int {
	// If there is a map associated with the instruction, return its FD.
	if fd := ins.Metadata.Get(mapMeta{}); fd != nil {
		return fd.(FDer).FD()
	}

	// Fall back to the fd stored in the Constant field
	return ins.mapFd()
}

// mapFd returns the map file descriptor stored in the 32 least significant
// bits of ins' Constant field.
func (ins *Instruction) mapFd() int {
	return int(int32(ins.Constant))
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

// IsLoadOfFunctionPointer returns true if the instruction loads a function pointer.
func (ins *Instruction) IsLoadOfFunctionPointer() bool {
	return ins.OpCode.IsDWordLoad() && ins.Src == PseudoFunc
}

// IsFunctionReference returns true if the instruction references another BPF
// function, either by invoking a Call jump operation or by loading a function
// pointer.
func (ins *Instruction) IsFunctionReference() bool {
	return ins.IsFunctionCall() || ins.IsLoadOfFunctionPointer()
}

// IsBuiltinCall returns true if the instruction is a built-in call, i.e. BPF helper call.
func (ins *Instruction) IsBuiltinCall() bool {
	return ins.OpCode.JumpOp() == Call && ins.Src == R0 && ins.Dst == R0
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
		fd := ins.mapFd()
		m := ins.Map()
		switch ins.Src {
		case PseudoMapFD:
			if m != nil {
				fmt.Fprintf(f, "LoadMapPtr dst: %s map: %s", ins.Dst, m)
			} else {
				fmt.Fprintf(f, "LoadMapPtr dst: %s fd: %d", ins.Dst, fd)
			}

		case PseudoMapValue:
			if m != nil {
				fmt.Fprintf(f, "LoadMapValue dst: %s, map: %s off: %d", ins.Dst, m, ins.mapOffset())
			} else {
				fmt.Fprintf(f, "LoadMapValue dst: %s, fd: %d off: %d", ins.Dst, fd, ins.mapOffset())
			}
		}

		goto ref
	}

	fmt.Fprintf(f, "%v ", op)
	switch cls := op.Class(); {
	case cls.isLoadOrStore():
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

	case cls.IsALU():
		fmt.Fprintf(f, "dst: %s ", ins.Dst)
		if op.ALUOp() == Swap || op.Source() == ImmSource {
			fmt.Fprintf(f, "imm: %d", ins.Constant)
		} else {
			fmt.Fprintf(f, "src: %s", ins.Src)
		}

	case cls.IsJump():
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
	if ins.Reference() != "" {
		fmt.Fprintf(f, " <%s>", ins.Reference())
	}
}

func (ins Instruction) equal(other Instruction) bool {
	return ins.OpCode == other.OpCode &&
		ins.Dst == other.Dst &&
		ins.Src == other.Src &&
		ins.Offset == other.Offset &&
		ins.Constant == other.Constant
}

// Size returns the amount of bytes ins would occupy in binary form.
func (ins Instruction) Size() uint64 {
	return uint64(InstructionSize * ins.OpCode.rawInstructions())
}

type symbolMeta struct{}

// WithSymbol marks the Instruction as a Symbol, which other Instructions
// can point to using corresponding calls to WithReference.
func (ins Instruction) WithSymbol(name string) Instruction {
	ins.Metadata.Set(symbolMeta{}, name)
	return ins
}

// Sym creates a symbol.
//
// Deprecated: use WithSymbol instead.
func (ins Instruction) Sym(name string) Instruction {
	return ins.WithSymbol(name)
}

// Symbol returns the value ins has been marked with using WithSymbol,
// otherwise returns an empty string. A symbol is often an Instruction
// at the start of a function body.
func (ins Instruction) Symbol() string {
	sym, _ := ins.Metadata.Get(symbolMeta{}).(string)
	return sym
}

type referenceMeta struct{}

// WithReference makes ins reference another Symbol or map by name.
func (ins Instruction) WithReference(ref string) Instruction {
	ins.Metadata.Set(referenceMeta{}, ref)
	return ins
}

// Reference returns the Symbol or map name referenced by ins, if any.
func (ins Instruction) Reference() string {
	ref, _ := ins.Metadata.Get(referenceMeta{}).(string)
	return ref
}

type mapMeta struct{}

// Map returns the Map referenced by ins, if any.
// An Instruction will contain a Map if e.g. it references an existing,
// pinned map that was opened during ELF loading.
func (ins Instruction) Map() FDer {
	fd, _ := ins.Metadata.Get(mapMeta{}).(FDer)
	return fd
}

type sourceMeta struct{}

// WithSource adds source information about the Instruction.
func (ins Instruction) WithSource(src fmt.Stringer) Instruction {
	ins.Metadata.Set(sourceMeta{}, src)
	return ins
}

// Source returns source information about the Instruction. The field is
// present when the compiler emits BTF line info about the Instruction and
// usually contains the line of source code responsible for it.
func (ins Instruction) Source() fmt.Stringer {
	str, _ := ins.Metadata.Get(sourceMeta{}).(fmt.Stringer)
	return str
}

// A Comment can be passed to Instruction.WithSource to add a comment
// to an instruction.
type Comment string

func (s Comment) String() string {
	return string(s)
}

// FDer represents a resource tied to an underlying file descriptor.
// Used as a stand-in for e.g. ebpf.Map since that type cannot be
// imported here and FD() is the only method we rely on.
type FDer interface {
	FD() int
}

// Instructions is an eBPF program.
type Instructions []Instruction

// Unmarshal unmarshals an Instructions from a binary instruction stream.
// All instructions in insns are replaced by instructions decoded from r.
func (insns *Instructions) Unmarshal(r io.Reader, bo binary.ByteOrder) error {
	if len(*insns) > 0 {
		*insns = nil
	}

	var offset uint64
	for {
		var ins Instruction
		n, err := ins.Unmarshal(r, bo)
		if errors.Is(err, io.EOF) {
			break
		}
		if err != nil {
			return fmt.Errorf("offset %d: %w", offset, err)
		}

		*insns = append(*insns, ins)
		offset += n
	}

	return nil
}

// Name returns the name of the function insns belongs to, if any.
func (insns Instructions) Name() string {
	if len(insns) == 0 {
		return ""
	}
	return insns[0].Symbol()
}

func (insns Instructions) String() string {
	return fmt.Sprint(insns)
}

// Size returns the amount of bytes insns would occupy in binary form.
func (insns Instructions) Size() uint64 {
	var sum uint64
	for _, ins := range insns {
		sum += ins.Size()
	}
	return sum
}

// AssociateMap updates all Instructions that Reference the given symbol
// to point to an existing Map m instead.
//
// Returns ErrUnreferencedSymbol error if no references to symbol are found
// in insns. If symbol is anything else than the symbol name of map (e.g.
// a bpf2bpf subprogram), an error is returned.
func (insns Instructions) AssociateMap(symbol string, m FDer) error {
	if symbol == "" {
		return errors.New("empty symbol")
	}

	var found bool
	for i := range insns {
		ins := &insns[i]
		if ins.Reference() != symbol {
			continue
		}

		if err := ins.AssociateMap(m); err != nil {
			return err
		}

		found = true
	}

	if !found {
		return fmt.Errorf("symbol %s: %w", symbol, ErrUnreferencedSymbol)
	}

	return nil
}

// RewriteMapPtr rewrites all loads of a specific map pointer to a new fd.
//
// Returns ErrUnreferencedSymbol if the symbol isn't used.
//
// Deprecated: use AssociateMap instead.
func (insns Instructions) RewriteMapPtr(symbol string, fd int) error {
	if symbol == "" {
		return errors.New("empty symbol")
	}

	var found bool
	for i := range insns {
		ins := &insns[i]
		if ins.Reference() != symbol {
			continue
		}

		if !ins.IsLoadFromMap() {
			return errors.New("not a load from a map")
		}

		ins.encodeMapFD(fd)

		found = true
	}

	if !found {
		return fmt.Errorf("symbol %s: %w", symbol, ErrUnreferencedSymbol)
	}

	return nil
}

// SymbolOffsets returns the set of symbols and their offset in
// the instructions.
func (insns Instructions) SymbolOffsets() (map[string]int, error) {
	offsets := make(map[string]int)

	for i, ins := range insns {
		if ins.Symbol() == "" {
			continue
		}

		if _, ok := offsets[ins.Symbol()]; ok {
			return nil, fmt.Errorf("duplicate symbol %s", ins.Symbol())
		}

		offsets[ins.Symbol()] = i
	}

	return offsets, nil
}

// FunctionReferences returns a set of symbol names these Instructions make
// bpf-to-bpf calls to.
func (insns Instructions) FunctionReferences() []string {
	calls := make(map[string]struct{})
	for _, ins := range insns {
		if ins.Constant != -1 {
			// BPF-to-BPF calls have -1 constants.
			continue
		}

		if ins.Reference() == "" {
			continue
		}

		if !ins.IsFunctionReference() {
			continue
		}

		calls[ins.Reference()] = struct{}{}
	}

	result := make([]string, 0, len(calls))
	for call := range calls {
		result = append(result, call)
	}

	sort.Strings(result)
	return result
}

// ReferenceOffsets returns the set of references and their offset in
// the instructions.
func (insns Instructions) ReferenceOffsets() map[string][]int {
	offsets := make(map[string][]int)

	for i, ins := range insns {
		if ins.Reference() == "" {
			continue
		}

		offsets[ins.Reference()] = append(offsets[ins.Reference()], i)
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
		if iter.Ins.Symbol() != "" {
			fmt.Fprintf(f, "%s%s:\n", symIndent, iter.Ins.Symbol())
		}
		if src := iter.Ins.Source(); src != nil {
			line := strings.TrimSpace(src.String())
			if line != "" {
				fmt.Fprintf(f, "%s%*s; %s\n", indent, offsetWidth, " ", line)
			}
		}
		fmt.Fprintf(f, "%s%*d: %v\n", indent, offsetWidth, iter.Offset, iter.Ins)
	}
}

// Marshal encodes a BPF program into the kernel format.
//
// insns may be modified if there are unresolved jumps or bpf2bpf calls.
//
// Returns ErrUnsatisfiedProgramReference if there is a Reference Instruction
// without a matching Symbol Instruction within insns.
func (insns Instructions) Marshal(w io.Writer, bo binary.ByteOrder) error {
	if err := insns.encodeFunctionReferences(); err != nil {
		return err
	}

	if err := insns.encodeMapPointers(); err != nil {
		return err
	}

	for i, ins := range insns {
		if _, err := ins.Marshal(w, bo); err != nil {
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

// encodeFunctionReferences populates the Offset (or Constant, depending on
// the instruction type) field of instructions with a Reference field to point
// to the offset of the corresponding instruction with a matching Symbol field.
//
// Only Reference Instructions that are either jumps or BPF function references
// (calls or function pointer loads) are populated.
//
// Returns ErrUnsatisfiedProgramReference if there is a Reference Instruction
// without at least one corresponding Symbol Instruction within insns.
func (insns Instructions) encodeFunctionReferences() error {
	// Index the offsets of instructions tagged as a symbol.
	symbolOffsets := make(map[string]RawInstructionOffset)
	iter := insns.Iterate()
	for iter.Next() {
		ins := iter.Ins

		if ins.Symbol() == "" {
			continue
		}

		if _, ok := symbolOffsets[ins.Symbol()]; ok {
			return fmt.Errorf("duplicate symbol %s", ins.Symbol())
		}

		symbolOffsets[ins.Symbol()] = iter.Offset
	}

	// Find all instructions tagged as references to other symbols.
	// Depending on the instruction type, populate their constant or offset
	// fields to point to the symbol they refer to within the insn stream.
	iter = insns.Iterate()
	for iter.Next() {
		i := iter.Index
		offset := iter.Offset
		ins := iter.Ins

		if ins.Reference() == "" {
			continue
		}

		switch {
		case ins.IsFunctionReference() && ins.Constant == -1:
			symOffset, ok := symbolOffsets[ins.Reference()]
			if !ok {
				return fmt.Errorf("%s at insn %d: symbol %q: %w", ins.OpCode, i, ins.Reference(), ErrUnsatisfiedProgramReference)
			}

			ins.Constant = int64(symOffset - offset - 1)

		case ins.OpCode.Class().IsJump() && ins.Offset == -1:
			symOffset, ok := symbolOffsets[ins.Reference()]
			if !ok {
				return fmt.Errorf("%s at insn %d: symbol %q: %w", ins.OpCode, i, ins.Reference(), ErrUnsatisfiedProgramReference)
			}

			ins.Offset = int16(symOffset - offset - 1)
		}
	}

	return nil
}

// encodeMapPointers finds all Map Instructions and encodes their FDs
// into their Constant fields.
func (insns Instructions) encodeMapPointers() error {
	iter := insns.Iterate()
	for iter.Next() {
		ins := iter.Ins

		if !ins.IsLoadFromMap() {
			continue
		}

		m := ins.Map()
		if m == nil {
			continue
		}

		fd := m.FD()
		if fd < 0 {
			return fmt.Errorf("map %s: %w", m, sys.ErrClosedFd)
		}

		ins.encodeMapFD(m.FD())
	}

	return nil
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

// IsUnreferencedSymbol returns true if err was caused by
// an unreferenced symbol.
//
// Deprecated: use errors.Is(err, asm.ErrUnreferencedSymbol).
func IsUnreferencedSymbol(err error) bool {
	return errors.Is(err, ErrUnreferencedSymbol)
}
