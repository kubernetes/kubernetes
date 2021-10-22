// Copyright 2019 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package dwarf

// This file implements the mapping from PC to lines.
// TODO: Find a way to test this properly.

// http://www.dwarfstd.org/doc/DWARF4.pdf Section 6.2 page 108

import (
	"fmt"
	"sort"
	"strings"
)

// PCToLine returns the file and line number corresponding to the PC value.
// It returns an error if a correspondence cannot be found.
func (d *Data) PCToLine(pc uint64) (file string, line uint64, err error) {
	c := d.pcToLineEntries
	if len(c) == 0 {
		return "", 0, fmt.Errorf("PCToLine: no line table")
	}
	i := sort.Search(len(c), func(i int) bool { return c[i].pc > pc }) - 1
	// c[i] is now the entry in pcToLineEntries with the largest pc that is not
	// larger than the query pc.
	// The search has failed if:
	// - All pcs in c were larger than the query pc (i == -1).
	// - c[i] marked the end of a sequence of instructions (c[i].file == 0).
	// - c[i] is the last element of c, and isn't the end of a sequence of
	//   instructions, and the search pc is much larger than c[i].pc.  In this
	//   case, we don't know the range of the last instruction, but the search
	//   pc is probably past it.
	if i == -1 || c[i].file == 0 || (i+1 == len(c) && pc-c[i].pc > 1024) {
		return "", 0, fmt.Errorf("no source line defined for PC %#x", pc)
	}
	if c[i].file >= uint64(len(d.sourceFiles)) {
		return "", 0, fmt.Errorf("invalid file number in DWARF data")
	}
	return d.sourceFiles[c[i].file], c[i].line, nil
}

// LineToBreakpointPCs returns the PCs that should be used as breakpoints
// corresponding to the given file and line number.
// It returns an empty slice if no PCs were found.
func (d *Data) LineToBreakpointPCs(file string, line uint64) ([]uint64, error) {
	compDir := d.compilationDirectory()

	// Find the closest match in the executable for the specified file.
	// We choose the file with the largest number of path components matching
	// at the end of the name. If there is a tie, we prefer files that are
	// under the compilation directory.  If there is still a tie, we choose
	// the file with the shortest name.
	// TODO: handle duplicate file names in the DWARF?
	var bestFile struct {
		fileNum    uint64 // Index of the file in the DWARF data.
		components int    // Number of matching path components.
		length     int    // Length of the filename.
		underComp  bool   // File is under the compilation directory.
	}
	for filenum, filename := range d.sourceFiles {
		c := matchingPathComponentSuffixSize(filename, file)
		underComp := strings.HasPrefix(filename, compDir)
		better := false
		if c != bestFile.components {
			better = c > bestFile.components
		} else if underComp != bestFile.underComp {
			better = underComp
		} else {
			better = len(filename) < bestFile.length
		}
		if better {
			bestFile.fileNum = uint64(filenum)
			bestFile.components = c
			bestFile.length = len(filename)
			bestFile.underComp = underComp
		}
	}
	if bestFile.components == 0 {
		return nil, fmt.Errorf("couldn't find file %q", file)
	}
	c := d.lineToPCEntries[bestFile.fileNum]
	// c contains all (pc, line) pairs for the appropriate file.
	start := sort.Search(len(c), func(i int) bool { return c[i].line >= line })
	end := sort.Search(len(c), func(i int) bool { return c[i].line > line })
	// c[i].line == line for all i in the range [start, end).
	pcs := make([]uint64, 0, end-start)
	for i := start; i < end; i++ {
		pcs = append(pcs, c[i].pc)
	}
	return pcs, nil
}

// compilationDirectory finds the first compilation unit entry in d and returns
// the compilation directory contained in it.
// If it fails, it returns the empty string.
func (d *Data) compilationDirectory() string {
	r := d.Reader()
	for {
		entry, err := r.Next()
		if entry == nil || err != nil {
			return ""
		}
		if entry.Tag == TagCompileUnit {
			name, _ := entry.Val(AttrCompDir).(string)
			return name
		}
	}
}

// matchingPathComponentSuffixSize returns the largest n such that the last n
// components of the paths p1 and p2 are equal.
// e.g. matchingPathComponentSuffixSize("a/b/x/y.go", "b/a/x/y.go") returns 2.
func matchingPathComponentSuffixSize(p1, p2 string) int {
	// TODO: deal with other path separators.
	c1 := strings.Split(p1, "/")
	c2 := strings.Split(p2, "/")
	min := len(c1)
	if len(c2) < min {
		min = len(c2)
	}
	var n int
	for n = 0; n < min; n++ {
		if c1[len(c1)-1-n] != c2[len(c2)-1-n] {
			break
		}
	}
	return n
}

// Standard opcodes. Figure 37, page 178.
// If an opcode >= lineMachine.prologue.opcodeBase, it is a special
// opcode rather than the opcode defined in this table.
const (
	lineStdCopy             = 0x01
	lineStdAdvancePC        = 0x02
	lineStdAdvanceLine      = 0x03
	lineStdSetFile          = 0x04
	lineStdSetColumn        = 0x05
	lineStdNegateStmt       = 0x06
	lineStdSetBasicBlock    = 0x07
	lineStdConstAddPC       = 0x08
	lineStdFixedAdvancePC   = 0x09
	lineStdSetPrologueEnd   = 0x0a
	lineStdSetEpilogueBegin = 0x0b
	lineStdSetISA           = 0x0c
)

// Extended opcodes. Figure 38, page 179.
const (
	lineStartExtendedOpcode = 0x00 // Not defined as a named constant in the spec.
	lineExtEndSequence      = 0x01
	lineExtSetAddress       = 0x02
	lineExtDefineFile       = 0x03
	lineExtSetDiscriminator = 0x04 // New in version 4.
	lineExtLoUser           = 0x80
	lineExtHiUser           = 0xff
)

// lineHeader holds the information stored in the header of the line table for a
// single compilation unit.
// Section 6.2.4, page 112.
type lineHeader struct {
	unitLength           int
	version              int
	headerLength         int
	minInstructionLength int
	maxOpsPerInstruction int
	defaultIsStmt        bool
	lineBase             int
	lineRange            int
	opcodeBase           byte
	stdOpcodeLengths     []byte
	include              []string   // entry 0 is empty; means current directory
	file                 []lineFile // entry 0 is empty.
}

// lineFile represents a file name stored in the PC/line table, usually in the header.
type lineFile struct {
	name   string
	index  int // index into include directories
	time   int // implementation-defined time of last modification
	length int // length in bytes, 0 if not available.
}

// lineMachine holds the registers evaluated during executing of the PC/line mapping engine.
// Section 6.2.2, page 109.
// A .debug_line section consists of multiple line number programs, one for each compilation unit.
type lineMachine struct {
	// The program-counter value corresponding to a machine instruction generated by the compiler.
	address uint64

	// An unsigned integer representing the index of an operation within a VLIW
	// instruction. The index of the first operation is 0. For non-VLIW
	// architectures, this register will always be 0.
	// The address and op_index registers, taken together, form an operation
	// pointer that can reference any individual operation with the instruction
	// stream.
	opIndex uint64

	// An unsigned integer indicating the identity of the source file corresponding to a machine instruction.
	file uint64

	// An unsigned integer indicating a source line number. Lines are numbered
	// beginning at 1. The compiler may emit the value 0 in cases where an
	// instruction cannot be attributed to any source line.
	line uint64

	// An unsigned integer indicating a column number within a source line.
	// Columns are numbered beginning at 1. The value 0 is reserved to indicate
	// that a statement begins at the “left edge” of the line.
	column uint64

	// A boolean indicating that the current instruction is a recommended
	// breakpoint location. A recommended breakpoint location is intended to
	// “represent” a line, a statement and/or a semantically distinct subpart of a
	// statement.
	isStmt bool

	// A boolean indicating that the current instruction is the beginning of a basic
	// block.
	basicBlock bool

	// A boolean indicating that the current address is that of the first byte after
	// the end of a sequence of target machine instructions. end_sequence
	// terminates a sequence of lines; therefore other information in the same
	// row is not meaningful.
	endSequence bool

	// A boolean indicating that the current address is one (of possibly many)
	// where execution should be suspended for an entry breakpoint of a
	// function.
	prologueEnd bool

	// A boolean indicating that the current address is one (of possibly many)
	// where execution should be suspended for an exit breakpoint of a function.
	epilogueBegin bool

	// An unsigned integer whose value encodes the applicable instruction set
	// architecture for the current instruction.
	// The encoding of instruction sets should be shared by all users of a given
	// architecture. It is recommended that this encoding be defined by the ABI
	// authoring committee for each architecture.
	isa uint64

	// An unsigned integer identifying the block to which the current instruction
	// belongs. Discriminator values are assigned arbitrarily by the DWARF
	// producer and serve to distinguish among multiple blocks that may all be
	// associated with the same source file, line, and column. Where only one
	// block exists for a given source position, the discriminator value should be
	// zero.
	discriminator uint64

	// The header for the current compilation unit.
	// Not an actual register, but stored here for cleanliness.
	header lineHeader

	// Offset in buf of the end of the line number program for the current unit.
	unitEndOff Offset
}

// parseHeader parses the header describing the compilation unit in the line
// table starting at the specified offset.
func (m *lineMachine) parseHeader(b *buf) error {
	m.header = lineHeader{}
	m.header.unitLength = int(b.uint32()) // Note: We are assuming 32-bit DWARF format.
	m.unitEndOff = b.off + Offset(m.header.unitLength)
	if m.header.unitLength > len(b.data) {
		return fmt.Errorf("DWARF: bad PC/line header length")
	}
	m.header.version = int(b.uint16())
	m.header.headerLength = int(b.uint32())
	m.header.minInstructionLength = int(b.uint8())
	if m.header.version >= 4 {
		m.header.maxOpsPerInstruction = int(b.uint8())
	} else {
		m.header.maxOpsPerInstruction = 1
	}
	m.header.defaultIsStmt = b.uint8() != 0
	m.header.lineBase = int(int8(b.uint8()))
	m.header.lineRange = int(b.uint8())
	m.header.opcodeBase = b.uint8()
	m.header.stdOpcodeLengths = make([]byte, m.header.opcodeBase-1)
	copy(m.header.stdOpcodeLengths, b.bytes(int(m.header.opcodeBase-1)))
	m.header.include = make([]string, 1) // First entry is empty; file index entries are 1-indexed.
	// Includes
	for {
		name := b.string()
		if name == "" {
			break
		}
		m.header.include = append(m.header.include, name)
	}
	// Files
	// Files are 1-indexed in line number program, but we'll deal with that in Data.buildLineCaches.
	// Here, just collect the filenames.
	for {
		name := b.string()
		if name == "" {
			break
		}
		index := b.uint()
		time := b.uint()
		length := b.uint()
		f := lineFile{
			name:   name,
			index:  int(index),
			time:   int(time),
			length: int(length),
		}
		m.header.file = append(m.header.file, f)
	}
	return nil
}

// Special opcodes, page 117.
// There are seven steps to processing special opcodes.  We break them up here
// because the caller needs to output a row between steps 2 and 4, and because
// we need to perform just step 2 for the opcode DW_LNS_const_add_pc.

func (m *lineMachine) specialOpcodeStep1(opcode byte) {
	adjustedOpcode := int(opcode - m.header.opcodeBase)
	lineAdvance := m.header.lineBase + (adjustedOpcode % m.header.lineRange)
	m.line += uint64(lineAdvance)
}

func (m *lineMachine) specialOpcodeStep2(opcode byte) {
	adjustedOpcode := int(opcode - m.header.opcodeBase)
	advance := adjustedOpcode / m.header.lineRange
	delta := (int(m.opIndex) + advance) / m.header.maxOpsPerInstruction
	m.address += uint64(m.header.minInstructionLength * delta)
	m.opIndex = (m.opIndex + uint64(advance)) % uint64(m.header.maxOpsPerInstruction)
}

func (m *lineMachine) specialOpcodeSteps4To7() {
	m.basicBlock = false
	m.prologueEnd = false
	m.epilogueBegin = false
	m.discriminator = 0
}

// evalCompilationUnit reads the next compilation unit and calls f at each output row.
// Line machine execution continues while f returns true.
func (m *lineMachine) evalCompilationUnit(b *buf, f func(m *lineMachine) (cont bool)) error {
	m.reset()
	for b.off < m.unitEndOff {
		op := b.uint8()
		if op >= m.header.opcodeBase {
			m.specialOpcodeStep1(op)
			m.specialOpcodeStep2(op)
			// Step 3 is to output a row, so we call f here.
			if !f(m) {
				return nil
			}
			m.specialOpcodeSteps4To7()
			continue
		}
		switch op {
		case lineStartExtendedOpcode:
			if len(b.data) == 0 {
				return fmt.Errorf("DWARF: short extended opcode (1)")
			}
			size := b.uint()
			if uint64(len(b.data)) < size {
				return fmt.Errorf("DWARF: short extended opcode (2)")
			}
			op = b.uint8()
			switch op {
			case lineExtEndSequence:
				m.endSequence = true
				if !f(m) {
					return nil
				}
				if len(b.data) == 0 {
					return nil
				}
				m.reset()
			case lineExtSetAddress:
				m.address = b.addr()
				m.opIndex = 0
			case lineExtDefineFile:
				return fmt.Errorf("DWARF: unimplemented define_file op")
			case lineExtSetDiscriminator:
				discriminator := b.uint()
				m.discriminator = discriminator
			default:
				return fmt.Errorf("DWARF: unknown extended opcode %#x", op)
			}
		case lineStdCopy:
			if !f(m) {
				return nil
			}
			m.discriminator = 0
			m.basicBlock = false
			m.prologueEnd = false
			m.epilogueBegin = false
		case lineStdAdvancePC:
			advance := b.uint()
			delta := (int(m.opIndex) + int(advance)) / m.header.maxOpsPerInstruction
			m.address += uint64(m.header.minInstructionLength * delta)
			m.opIndex = (m.opIndex + uint64(advance)) % uint64(m.header.maxOpsPerInstruction)
			m.basicBlock = false
			m.prologueEnd = false
			m.epilogueBegin = false
			m.discriminator = 0
		case lineStdAdvanceLine:
			advance := b.int()
			m.line = uint64(int64(m.line) + advance)
		case lineStdSetFile:
			index := b.uint()
			m.file = index
		case lineStdSetColumn:
			column := b.uint()
			m.column = column
		case lineStdNegateStmt:
			m.isStmt = !m.isStmt
		case lineStdSetBasicBlock:
			m.basicBlock = true
		case lineStdFixedAdvancePC:
			m.address += uint64(b.uint16())
			m.opIndex = 0
		case lineStdSetPrologueEnd:
			m.prologueEnd = true
		case lineStdSetEpilogueBegin:
			m.epilogueBegin = true
		case lineStdSetISA:
			m.isa = b.uint()
		case lineStdConstAddPC:
			// Update the address and op_index registers.
			m.specialOpcodeStep2(255)
		default:
			panic("not reached")
		}
	}
	return nil
}

// reset sets the machine's registers to the initial state. Page 111.
func (m *lineMachine) reset() {
	m.address = 0
	m.opIndex = 0
	m.file = 1
	m.line = 1
	m.column = 0
	m.isStmt = m.header.defaultIsStmt
	m.basicBlock = false
	m.endSequence = false
	m.prologueEnd = false
	m.epilogueBegin = false
	m.isa = 0
	m.discriminator = 0
}
