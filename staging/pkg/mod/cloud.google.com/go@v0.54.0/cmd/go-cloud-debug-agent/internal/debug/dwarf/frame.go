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

// Mapping from PC to SP offset (called CFA - Canonical Frame Address - in DWARF).
// This value is the offset from the stack pointer to the virtual frame pointer
// (address of zeroth argument) at each PC value in the program.

package dwarf

import "fmt"

// http://www.dwarfstd.org/doc/DWARF4.pdf Section 6.4 page 126
// We implement only the CFA column of the table, not the location
// information about other registers. In other words, we implement
// only what we need to understand Go programs compiled by gc.

// PCToSPOffset returns the offset, at the specified PC, to add to the
// SP to reach the virtual frame pointer, which corresponds to the
// address of the zeroth argument of the function, the word on the
// stack immediately above the return PC.
func (d *Data) PCToSPOffset(pc uint64) (offset int64, err error) {
	if len(d.frame) == 0 {
		return 0, fmt.Errorf("PCToSPOffset: no frame table")
	}
	var m frameMachine
	// Assume the first info unit is the same as us. Extremely likely. TODO?
	if len(d.unit) == 0 {
		return 0, fmt.Errorf("PCToSPOffset: no info section")
	}
	buf := makeBuf(d, &d.unit[0], "frame", 0, d.frame)
	for len(buf.data) > 0 {
		offset, err := m.evalCompilationUnit(&buf, pc)
		if err != nil {
			return 0, err
		}
		return offset, nil
	}
	return 0, fmt.Errorf("PCToSPOffset: no frame defined for PC %#x", pc)
}

// Call Frame instructions. Figure 40, page 181.
// Structure is high two bits plus low 6 bits specified by + in comment.
// Some take one or two operands.
const (
	frameNop              = 0<<6 + 0x00
	frameAdvanceLoc       = 1<<6 + 0x00 // + delta
	frameOffset           = 2<<6 + 0x00 // + register op: ULEB128 offset
	frameRestore          = 3<<6 + 0x00 // + register
	frameSetLoc           = 0<<6 + 0x01 // op: address
	frameAdvanceLoc1      = 0<<6 + 0x02 // op: 1-byte delta
	frameAdvanceLoc2      = 0<<6 + 0x03 // op: 2-byte delta
	frameAdvanceLoc4      = 0<<6 + 0x04 // op: 4-byte delta
	frameOffsetExtended   = 0<<6 + 0x05 // ops: ULEB128 register ULEB128 offset
	frameRestoreExtended  = 0<<6 + 0x06 // op: ULEB128 register
	frameUndefined        = 0<<6 + 0x07 // op: ULEB128 register
	frameSameValue        = 0<<6 + 0x08 // op: ULEB128 register
	frameRegister         = 0<<6 + 0x09 // op: ULEB128 register ULEB128 register
	frameRememberState    = 0<<6 + 0x0a
	frameRestoreState     = 0<<6 + 0x0b
	frameDefCFA           = 0<<6 + 0x0c // op: ULEB128 register ULEB128 offset
	frameDefCFARegister   = 0<<6 + 0x0d // op: ULEB128 register
	frameDefCFAOffset     = 0<<6 + 0x0e // op: ULEB128 offset
	frameDefCFAExpression = 0<<6 + 0x0f // op: BLOCK
	frameExpression       = 0<<6 + 0x10 // op: ULEB128 register BLOCK
	frameOffsetExtendedSf = 0<<6 + 0x11 // op: ULEB128 register SLEB128 offset
	frameDefCFASf         = 0<<6 + 0x12 // op: ULEB128 register SLEB128 offset
	frameDefCFAOffsetSf   = 0<<6 + 0x13 // op: SLEB128 offset
	frameValOffset        = 0<<6 + 0x14 // op: ULEB128 ULEB128
	frameValOffsetSf      = 0<<6 + 0x15 // op: ULEB128 SLEB128
	frameValExpression    = 0<<6 + 0x16 // op: ULEB128 BLOCK
	frameLoUser           = 0<<6 + 0x1c
	frameHiUser           = 0<<6 + 0x3f
)

// frameMachine represents the PC/SP engine.
// Section 6.4, page 129.
type frameMachine struct {
	// Initial values from CIE.
	version               uint8  // Version number, "independent of DWARF version"
	augmentation          string // Augmentation; treated as unexpected for now. TODO.
	addressSize           uint8  // In DWARF v4 and above. Size of a target address.
	segmentSize           uint8  // In DWARF v4 and above. Size of a segment selector.
	codeAlignmentFactor   uint64 // Unit of code size in advance instructions.
	dataAlignmentFactor   int64  // Unit of data size in certain offset instructions.
	returnAddressRegister int    // Pseudo-register (actually data column) representing return address.
	returnRegisterOffset  int64  // Offset to saved PC from CFA in bytes.
	// CFA definition.
	cfaRegister int   // Which register represents the SP.
	cfaOffset   int64 // CFA offset value.
	// Running machine.
	location uint64
}

// evalCompilationUnit scans the frame data for one compilation unit to retrieve
// the offset information for the specified pc.
func (m *frameMachine) evalCompilationUnit(b *buf, pc uint64) (int64, error) {
	err := m.parseCIE(b)
	if err != nil {
		return 0, err
	}
	for {
		offset, found, err := m.scanFDE(b, pc)
		if err != nil {
			return 0, err
		}
		if found {
			return offset, nil
		}
	}
}

// parseCIE assumes the incoming buffer starts with a CIE block and parses it
// to initialize a frameMachine.
func (m *frameMachine) parseCIE(allBuf *buf) error {
	length := int(allBuf.uint32())
	if len(allBuf.data) < length {
		return fmt.Errorf("CIE parse error: too short")
	}
	// Create buffer for just this section.
	b := allBuf.slice(length)
	cie := b.uint32()
	if cie != 0xFFFFFFFF {
		return fmt.Errorf("CIE parse error: not CIE: %x", cie)
	}
	m.version = b.uint8()
	if m.version != 3 && m.version != 4 {
		return fmt.Errorf("CIE parse error: unsupported version %d", m.version)
	}
	m.augmentation = b.string()
	if len(m.augmentation) > 0 {
		return fmt.Errorf("CIE: can't handled augmentation string %q", m.augmentation)
	}
	if m.version >= 4 {
		m.addressSize = b.uint8()
		m.segmentSize = b.uint8()
	} else {
		// Unused. Gc generates version 3, so these values will not be
		// set, but they are also not used so it's OK.
	}
	m.codeAlignmentFactor = b.uint()
	m.dataAlignmentFactor = b.int()
	m.returnAddressRegister = int(b.uint())

	// Initial instructions. At least for Go, establishes SP register number
	// and initial value of CFA offset at start of function.
	_, err := m.run(&b, ^uint64(0))
	if err != nil {
		return err
	}

	// There's padding, but we can ignore it.
	return nil
}

// scanFDE assumes the incoming buffer starts with a FDE block and parses it
// to run a frameMachine and, if the PC is represented in its range, return
// the CFA offset for that PC. The boolean returned reports whether the
// PC is in range for this FDE.
func (m *frameMachine) scanFDE(allBuf *buf, pc uint64) (int64, bool, error) {
	length := int(allBuf.uint32())
	if len(allBuf.data) < length {
		return 0, false, fmt.Errorf("FDE parse error: too short")
	}
	if length <= 0 {
		if length == 0 {
			// EOF.
			return 0, false, fmt.Errorf("PC %#x not found in PC/SP table", pc)
		}
		return 0, false, fmt.Errorf("bad FDE length %d", length)
	}
	// Create buffer for just this section.
	b := allBuf.slice(length)
	cieOffset := b.uint32() // TODO: assumes 32 bits.
	// Expect 0: first CIE in this segment. TODO.
	if cieOffset != 0 {
		return 0, false, fmt.Errorf("FDE parse error: bad CIE offset: %.2x", cieOffset)
	}
	// Initial location.
	m.location = b.addr()
	addressRange := b.addr()
	// If the PC is not in this function, there's no point in executing the instructions.
	if pc < m.location || m.location+addressRange <= pc {
		return 0, false, nil
	}
	// The PC appears in this FDE. Scan to find the location.
	offset, err := m.run(&b, pc)
	if err != nil {
		return 0, false, err
	}

	// There's padding, but we can ignore it.
	return offset, true, nil
}

// run executes the instructions in the buffer, which has been sliced to contain
// only the data for this block. When we run out of data, we return.
// Since we are only called when we know the PC is in this block, reaching
// EOF is not an error, it just means the final CFA definition matches the
// tail of the block that holds the PC.
// The return value is the CFA at the end of the block or the PC, whichever
// comes first.
func (m *frameMachine) run(b *buf, pc uint64) (int64, error) {
	// We run the machine at location == PC because if the PC is at the first
	// instruction of a block, the definition of its offset arrives as an
	// offset-defining operand after the PC is set to that location.
	for m.location <= pc && len(b.data) > 0 {
		op := b.uint8()
		// Ops with embedded operands
		switch op & 0xC0 {
		case frameAdvanceLoc: // (6.4.2.1)
			// delta in low bits
			m.location += uint64(op & 0x3F)
			continue
		case frameOffset: // (6.4.2.3)
			// Register in low bits; ULEB128 offset.
			// For Go binaries we only see this in the CIE for the return address register.
			if int(op&0x3F) != m.returnAddressRegister {
				return 0, fmt.Errorf("invalid frameOffset register R%d should be R%d", op&0x3f, m.returnAddressRegister)
			}
			m.returnRegisterOffset = int64(b.uint()) * m.dataAlignmentFactor
			continue
		case frameRestore: // (6.4.2.3)
			// register in low bits
			return 0, fmt.Errorf("unimplemented frameRestore(R%d)\n", op&0x3F)
		}

		// The remaining ops do not have embedded operands.

		switch op {
		// Row creation instructions (6.4.2.1)
		case frameNop:
		case frameSetLoc: // op: address
			return 0, fmt.Errorf("unimplemented setloc") // what size is operand?
		case frameAdvanceLoc1: // op: 1-byte delta
			m.location += uint64(b.uint8())
		case frameAdvanceLoc2: // op: 2-byte delta
			m.location += uint64(b.uint16())
		case frameAdvanceLoc4: // op: 4-byte delta
			m.location += uint64(b.uint32())

		// CFA definition instructions (6.4.2.2)
		case frameDefCFA: // op: ULEB128 register ULEB128 offset
			m.cfaRegister = int(b.int())
			m.cfaOffset = int64(b.uint())
		case frameDefCFASf: // op: ULEB128 register SLEB128 offset
			return 0, fmt.Errorf("unimplemented frameDefCFASf")
		case frameDefCFARegister: // op: ULEB128 register
			return 0, fmt.Errorf("unimplemented frameDefCFARegister")
		case frameDefCFAOffset: // op: ULEB128 offset
			return 0, fmt.Errorf("unimplemented frameDefCFAOffset")
		case frameDefCFAOffsetSf: // op: SLEB128 offset
			offset := b.int()
			m.cfaOffset = offset * m.dataAlignmentFactor
			// TODO: Verify we are using a factored offset.
		case frameDefCFAExpression: // op: BLOCK
			return 0, fmt.Errorf("unimplemented frameDefCFAExpression")

		// Register Rule instructions (6.4.2.3)
		case frameOffsetExtended: // ops: ULEB128 register ULEB128 offset
			// The same as frameOffset, but with the register specified in an operand.
			reg := b.uint()
			// For Go binaries we only see this in the CIE for the return address register.
			if reg != uint64(m.returnAddressRegister) {
				return 0, fmt.Errorf("invalid frameOffsetExtended: register R%d should be R%d", reg, m.returnAddressRegister)
			}
			m.returnRegisterOffset = int64(b.uint()) * m.dataAlignmentFactor
		case frameRestoreExtended: // op: ULEB128 register
			return 0, fmt.Errorf("unimplemented frameRestoreExtended")
		case frameUndefined: // op: ULEB128 register; unimplemented
			return 0, fmt.Errorf("unimplemented frameUndefined")
		case frameSameValue: // op: ULEB128 register
			return 0, fmt.Errorf("unimplemented frameSameValue")
		case frameRegister: // op: ULEB128 register ULEB128 register
			return 0, fmt.Errorf("unimplemented frameRegister")
		case frameRememberState:
			return 0, fmt.Errorf("unimplemented frameRememberState")
		case frameRestoreState:
			return 0, fmt.Errorf("unimplemented frameRestoreState")
		case frameExpression: // op: ULEB128 register BLOCK
			return 0, fmt.Errorf("unimplemented frameExpression")
		case frameOffsetExtendedSf: // op: ULEB128 register SLEB128 offset
			return 0, fmt.Errorf("unimplemented frameOffsetExtended_sf")
		case frameValOffset: // op: ULEB128 ULEB128
			return 0, fmt.Errorf("unimplemented frameValOffset")
		case frameValOffsetSf: // op: ULEB128 SLEB128
			return 0, fmt.Errorf("unimplemented frameValOffsetSf")
		case frameValExpression: // op: ULEB128 BLOCK
			return 0, fmt.Errorf("unimplemented frameValExpression")

		default:
			if frameLoUser <= op && op <= frameHiUser {
				return 0, fmt.Errorf("unknown user-defined frame op %#x", op)
			}
			return 0, fmt.Errorf("unknown frame op %#x", op)
		}
	}
	return m.cfaOffset, nil
}
