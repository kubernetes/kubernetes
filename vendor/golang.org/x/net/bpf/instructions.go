// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpf

import "fmt"

// An Instruction is one instruction executed by the BPF virtual
// machine.
type Instruction interface {
	// Assemble assembles the Instruction into a RawInstruction.
	Assemble() (RawInstruction, error)
}

// A RawInstruction is a raw BPF virtual machine instruction.
type RawInstruction struct {
	// Operation to execute.
	Op uint16
	// For conditional jump instructions, the number of instructions
	// to skip if the condition is true/false.
	Jt uint8
	Jf uint8
	// Constant parameter. The meaning depends on the Op.
	K uint32
}

// Assemble implements the Instruction Assemble method.
func (ri RawInstruction) Assemble() (RawInstruction, error) { return ri, nil }

// Disassemble parses ri into an Instruction and returns it. If ri is
// not recognized by this package, ri itself is returned.
func (ri RawInstruction) Disassemble() Instruction {
	switch ri.Op & opMaskCls {
	case opClsLoadA, opClsLoadX:
		reg := Register(ri.Op & opMaskLoadDest)
		sz := 0
		switch ri.Op & opMaskLoadWidth {
		case opLoadWidth4:
			sz = 4
		case opLoadWidth2:
			sz = 2
		case opLoadWidth1:
			sz = 1
		default:
			return ri
		}
		switch ri.Op & opMaskLoadMode {
		case opAddrModeImmediate:
			if sz != 4 {
				return ri
			}
			return LoadConstant{Dst: reg, Val: ri.K}
		case opAddrModeScratch:
			if sz != 4 || ri.K > 15 {
				return ri
			}
			return LoadScratch{Dst: reg, N: int(ri.K)}
		case opAddrModeAbsolute:
			if ri.K > extOffset+0xffffffff {
				return LoadExtension{Num: Extension(-extOffset + ri.K)}
			}
			return LoadAbsolute{Size: sz, Off: ri.K}
		case opAddrModeIndirect:
			return LoadIndirect{Size: sz, Off: ri.K}
		case opAddrModePacketLen:
			if sz != 4 {
				return ri
			}
			return LoadExtension{Num: ExtLen}
		case opAddrModeMemShift:
			return LoadMemShift{Off: ri.K}
		default:
			return ri
		}

	case opClsStoreA:
		if ri.Op != opClsStoreA || ri.K > 15 {
			return ri
		}
		return StoreScratch{Src: RegA, N: int(ri.K)}

	case opClsStoreX:
		if ri.Op != opClsStoreX || ri.K > 15 {
			return ri
		}
		return StoreScratch{Src: RegX, N: int(ri.K)}

	case opClsALU:
		switch op := ALUOp(ri.Op & opMaskOperator); op {
		case ALUOpAdd, ALUOpSub, ALUOpMul, ALUOpDiv, ALUOpOr, ALUOpAnd, ALUOpShiftLeft, ALUOpShiftRight, ALUOpMod, ALUOpXor:
			if ri.Op&opMaskOperandSrc != 0 {
				return ALUOpX{Op: op}
			}
			return ALUOpConstant{Op: op, Val: ri.K}
		case aluOpNeg:
			return NegateA{}
		default:
			return ri
		}

	case opClsJump:
		if ri.Op&opMaskJumpConst != opClsJump {
			return ri
		}
		switch ri.Op & opMaskJumpCond {
		case opJumpAlways:
			return Jump{Skip: ri.K}
		case opJumpEqual:
			if ri.Jt == 0 {
				return JumpIf{
					Cond:      JumpNotEqual,
					Val:       ri.K,
					SkipTrue:  ri.Jf,
					SkipFalse: 0,
				}
			}
			return JumpIf{
				Cond:      JumpEqual,
				Val:       ri.K,
				SkipTrue:  ri.Jt,
				SkipFalse: ri.Jf,
			}
		case opJumpGT:
			if ri.Jt == 0 {
				return JumpIf{
					Cond:      JumpLessOrEqual,
					Val:       ri.K,
					SkipTrue:  ri.Jf,
					SkipFalse: 0,
				}
			}
			return JumpIf{
				Cond:      JumpGreaterThan,
				Val:       ri.K,
				SkipTrue:  ri.Jt,
				SkipFalse: ri.Jf,
			}
		case opJumpGE:
			if ri.Jt == 0 {
				return JumpIf{
					Cond:      JumpLessThan,
					Val:       ri.K,
					SkipTrue:  ri.Jf,
					SkipFalse: 0,
				}
			}
			return JumpIf{
				Cond:      JumpGreaterOrEqual,
				Val:       ri.K,
				SkipTrue:  ri.Jt,
				SkipFalse: ri.Jf,
			}
		case opJumpSet:
			return JumpIf{
				Cond:      JumpBitsSet,
				Val:       ri.K,
				SkipTrue:  ri.Jt,
				SkipFalse: ri.Jf,
			}
		default:
			return ri
		}

	case opClsReturn:
		switch ri.Op {
		case opClsReturn | opRetSrcA:
			return RetA{}
		case opClsReturn | opRetSrcConstant:
			return RetConstant{Val: ri.K}
		default:
			return ri
		}

	case opClsMisc:
		switch ri.Op {
		case opClsMisc | opMiscTAX:
			return TAX{}
		case opClsMisc | opMiscTXA:
			return TXA{}
		default:
			return ri
		}

	default:
		panic("unreachable") // switch is exhaustive on the bit pattern
	}
}

// LoadConstant loads Val into register Dst.
type LoadConstant struct {
	Dst Register
	Val uint32
}

// Assemble implements the Instruction Assemble method.
func (a LoadConstant) Assemble() (RawInstruction, error) {
	return assembleLoad(a.Dst, 4, opAddrModeImmediate, a.Val)
}

// String returns the the instruction in assembler notation.
func (a LoadConstant) String() string {
	switch a.Dst {
	case RegA:
		return fmt.Sprintf("ld #%d", a.Val)
	case RegX:
		return fmt.Sprintf("ldx #%d", a.Val)
	default:
		return fmt.Sprintf("unknown instruction: %#v", a)
	}
}

// LoadScratch loads scratch[N] into register Dst.
type LoadScratch struct {
	Dst Register
	N   int // 0-15
}

// Assemble implements the Instruction Assemble method.
func (a LoadScratch) Assemble() (RawInstruction, error) {
	if a.N < 0 || a.N > 15 {
		return RawInstruction{}, fmt.Errorf("invalid scratch slot %d", a.N)
	}
	return assembleLoad(a.Dst, 4, opAddrModeScratch, uint32(a.N))
}

// String returns the the instruction in assembler notation.
func (a LoadScratch) String() string {
	switch a.Dst {
	case RegA:
		return fmt.Sprintf("ld M[%d]", a.N)
	case RegX:
		return fmt.Sprintf("ldx M[%d]", a.N)
	default:
		return fmt.Sprintf("unknown instruction: %#v", a)
	}
}

// LoadAbsolute loads packet[Off:Off+Size] as an integer value into
// register A.
type LoadAbsolute struct {
	Off  uint32
	Size int // 1, 2 or 4
}

// Assemble implements the Instruction Assemble method.
func (a LoadAbsolute) Assemble() (RawInstruction, error) {
	return assembleLoad(RegA, a.Size, opAddrModeAbsolute, a.Off)
}

// String returns the the instruction in assembler notation.
func (a LoadAbsolute) String() string {
	switch a.Size {
	case 1: // byte
		return fmt.Sprintf("ldb [%d]", a.Off)
	case 2: // half word
		return fmt.Sprintf("ldh [%d]", a.Off)
	case 4: // word
		if a.Off > extOffset+0xffffffff {
			return LoadExtension{Num: Extension(a.Off + 0x1000)}.String()
		}
		return fmt.Sprintf("ld [%d]", a.Off)
	default:
		return fmt.Sprintf("unknown instruction: %#v", a)
	}
}

// LoadIndirect loads packet[X+Off:X+Off+Size] as an integer value
// into register A.
type LoadIndirect struct {
	Off  uint32
	Size int // 1, 2 or 4
}

// Assemble implements the Instruction Assemble method.
func (a LoadIndirect) Assemble() (RawInstruction, error) {
	return assembleLoad(RegA, a.Size, opAddrModeIndirect, a.Off)
}

// String returns the the instruction in assembler notation.
func (a LoadIndirect) String() string {
	switch a.Size {
	case 1: // byte
		return fmt.Sprintf("ldb [x + %d]", a.Off)
	case 2: // half word
		return fmt.Sprintf("ldh [x + %d]", a.Off)
	case 4: // word
		return fmt.Sprintf("ld [x + %d]", a.Off)
	default:
		return fmt.Sprintf("unknown instruction: %#v", a)
	}
}

// LoadMemShift multiplies the first 4 bits of the byte at packet[Off]
// by 4 and stores the result in register X.
//
// This instruction is mainly useful to load into X the length of an
// IPv4 packet header in a single instruction, rather than have to do
// the arithmetic on the header's first byte by hand.
type LoadMemShift struct {
	Off uint32
}

// Assemble implements the Instruction Assemble method.
func (a LoadMemShift) Assemble() (RawInstruction, error) {
	return assembleLoad(RegX, 1, opAddrModeMemShift, a.Off)
}

// String returns the the instruction in assembler notation.
func (a LoadMemShift) String() string {
	return fmt.Sprintf("ldx 4*([%d]&0xf)", a.Off)
}

// LoadExtension invokes a linux-specific extension and stores the
// result in register A.
type LoadExtension struct {
	Num Extension
}

// Assemble implements the Instruction Assemble method.
func (a LoadExtension) Assemble() (RawInstruction, error) {
	if a.Num == ExtLen {
		return assembleLoad(RegA, 4, opAddrModePacketLen, 0)
	}
	return assembleLoad(RegA, 4, opAddrModeAbsolute, uint32(extOffset+a.Num))
}

// String returns the the instruction in assembler notation.
func (a LoadExtension) String() string {
	switch a.Num {
	case ExtLen:
		return "ld #len"
	case ExtProto:
		return "ld #proto"
	case ExtType:
		return "ld #type"
	case ExtPayloadOffset:
		return "ld #poff"
	case ExtInterfaceIndex:
		return "ld #ifidx"
	case ExtNetlinkAttr:
		return "ld #nla"
	case ExtNetlinkAttrNested:
		return "ld #nlan"
	case ExtMark:
		return "ld #mark"
	case ExtQueue:
		return "ld #queue"
	case ExtLinkLayerType:
		return "ld #hatype"
	case ExtRXHash:
		return "ld #rxhash"
	case ExtCPUID:
		return "ld #cpu"
	case ExtVLANTag:
		return "ld #vlan_tci"
	case ExtVLANTagPresent:
		return "ld #vlan_avail"
	case ExtVLANProto:
		return "ld #vlan_tpid"
	case ExtRand:
		return "ld #rand"
	default:
		return fmt.Sprintf("unknown instruction: %#v", a)
	}
}

// StoreScratch stores register Src into scratch[N].
type StoreScratch struct {
	Src Register
	N   int // 0-15
}

// Assemble implements the Instruction Assemble method.
func (a StoreScratch) Assemble() (RawInstruction, error) {
	if a.N < 0 || a.N > 15 {
		return RawInstruction{}, fmt.Errorf("invalid scratch slot %d", a.N)
	}
	var op uint16
	switch a.Src {
	case RegA:
		op = opClsStoreA
	case RegX:
		op = opClsStoreX
	default:
		return RawInstruction{}, fmt.Errorf("invalid source register %v", a.Src)
	}

	return RawInstruction{
		Op: op,
		K:  uint32(a.N),
	}, nil
}

// String returns the the instruction in assembler notation.
func (a StoreScratch) String() string {
	switch a.Src {
	case RegA:
		return fmt.Sprintf("st M[%d]", a.N)
	case RegX:
		return fmt.Sprintf("stx M[%d]", a.N)
	default:
		return fmt.Sprintf("unknown instruction: %#v", a)
	}
}

// ALUOpConstant executes A = A <Op> Val.
type ALUOpConstant struct {
	Op  ALUOp
	Val uint32
}

// Assemble implements the Instruction Assemble method.
func (a ALUOpConstant) Assemble() (RawInstruction, error) {
	return RawInstruction{
		Op: opClsALU | opALUSrcConstant | uint16(a.Op),
		K:  a.Val,
	}, nil
}

// String returns the the instruction in assembler notation.
func (a ALUOpConstant) String() string {
	switch a.Op {
	case ALUOpAdd:
		return fmt.Sprintf("add #%d", a.Val)
	case ALUOpSub:
		return fmt.Sprintf("sub #%d", a.Val)
	case ALUOpMul:
		return fmt.Sprintf("mul #%d", a.Val)
	case ALUOpDiv:
		return fmt.Sprintf("div #%d", a.Val)
	case ALUOpMod:
		return fmt.Sprintf("mod #%d", a.Val)
	case ALUOpAnd:
		return fmt.Sprintf("and #%d", a.Val)
	case ALUOpOr:
		return fmt.Sprintf("or #%d", a.Val)
	case ALUOpXor:
		return fmt.Sprintf("xor #%d", a.Val)
	case ALUOpShiftLeft:
		return fmt.Sprintf("lsh #%d", a.Val)
	case ALUOpShiftRight:
		return fmt.Sprintf("rsh #%d", a.Val)
	default:
		return fmt.Sprintf("unknown instruction: %#v", a)
	}
}

// ALUOpX executes A = A <Op> X
type ALUOpX struct {
	Op ALUOp
}

// Assemble implements the Instruction Assemble method.
func (a ALUOpX) Assemble() (RawInstruction, error) {
	return RawInstruction{
		Op: opClsALU | opALUSrcX | uint16(a.Op),
	}, nil
}

// String returns the the instruction in assembler notation.
func (a ALUOpX) String() string {
	switch a.Op {
	case ALUOpAdd:
		return "add x"
	case ALUOpSub:
		return "sub x"
	case ALUOpMul:
		return "mul x"
	case ALUOpDiv:
		return "div x"
	case ALUOpMod:
		return "mod x"
	case ALUOpAnd:
		return "and x"
	case ALUOpOr:
		return "or x"
	case ALUOpXor:
		return "xor x"
	case ALUOpShiftLeft:
		return "lsh x"
	case ALUOpShiftRight:
		return "rsh x"
	default:
		return fmt.Sprintf("unknown instruction: %#v", a)
	}
}

// NegateA executes A = -A.
type NegateA struct{}

// Assemble implements the Instruction Assemble method.
func (a NegateA) Assemble() (RawInstruction, error) {
	return RawInstruction{
		Op: opClsALU | uint16(aluOpNeg),
	}, nil
}

// String returns the the instruction in assembler notation.
func (a NegateA) String() string {
	return fmt.Sprintf("neg")
}

// Jump skips the following Skip instructions in the program.
type Jump struct {
	Skip uint32
}

// Assemble implements the Instruction Assemble method.
func (a Jump) Assemble() (RawInstruction, error) {
	return RawInstruction{
		Op: opClsJump | opJumpAlways,
		K:  a.Skip,
	}, nil
}

// String returns the the instruction in assembler notation.
func (a Jump) String() string {
	return fmt.Sprintf("ja %d", a.Skip)
}

// JumpIf skips the following Skip instructions in the program if A
// <Cond> Val is true.
type JumpIf struct {
	Cond      JumpTest
	Val       uint32
	SkipTrue  uint8
	SkipFalse uint8
}

// Assemble implements the Instruction Assemble method.
func (a JumpIf) Assemble() (RawInstruction, error) {
	var (
		cond uint16
		flip bool
	)
	switch a.Cond {
	case JumpEqual:
		cond = opJumpEqual
	case JumpNotEqual:
		cond, flip = opJumpEqual, true
	case JumpGreaterThan:
		cond = opJumpGT
	case JumpLessThan:
		cond, flip = opJumpGE, true
	case JumpGreaterOrEqual:
		cond = opJumpGE
	case JumpLessOrEqual:
		cond, flip = opJumpGT, true
	case JumpBitsSet:
		cond = opJumpSet
	case JumpBitsNotSet:
		cond, flip = opJumpSet, true
	default:
		return RawInstruction{}, fmt.Errorf("unknown JumpTest %v", a.Cond)
	}
	jt, jf := a.SkipTrue, a.SkipFalse
	if flip {
		jt, jf = jf, jt
	}
	return RawInstruction{
		Op: opClsJump | cond,
		Jt: jt,
		Jf: jf,
		K:  a.Val,
	}, nil
}

// String returns the the instruction in assembler notation.
func (a JumpIf) String() string {
	switch a.Cond {
	// K == A
	case JumpEqual:
		return conditionalJump(a, "jeq", "jneq")
	// K != A
	case JumpNotEqual:
		return fmt.Sprintf("jneq #%d,%d", a.Val, a.SkipTrue)
	// K > A
	case JumpGreaterThan:
		return conditionalJump(a, "jgt", "jle")
	// K < A
	case JumpLessThan:
		return fmt.Sprintf("jlt #%d,%d", a.Val, a.SkipTrue)
	// K >= A
	case JumpGreaterOrEqual:
		return conditionalJump(a, "jge", "jlt")
	// K <= A
	case JumpLessOrEqual:
		return fmt.Sprintf("jle #%d,%d", a.Val, a.SkipTrue)
	// K & A != 0
	case JumpBitsSet:
		if a.SkipFalse > 0 {
			return fmt.Sprintf("jset #%d,%d,%d", a.Val, a.SkipTrue, a.SkipFalse)
		}
		return fmt.Sprintf("jset #%d,%d", a.Val, a.SkipTrue)
	// K & A == 0, there is no assembler instruction for JumpBitNotSet, use JumpBitSet and invert skips
	case JumpBitsNotSet:
		return JumpIf{Cond: JumpBitsSet, SkipTrue: a.SkipFalse, SkipFalse: a.SkipTrue, Val: a.Val}.String()
	default:
		return fmt.Sprintf("unknown instruction: %#v", a)
	}
}

func conditionalJump(inst JumpIf, positiveJump, negativeJump string) string {
	if inst.SkipTrue > 0 {
		if inst.SkipFalse > 0 {
			return fmt.Sprintf("%s #%d,%d,%d", positiveJump, inst.Val, inst.SkipTrue, inst.SkipFalse)
		}
		return fmt.Sprintf("%s #%d,%d", positiveJump, inst.Val, inst.SkipTrue)
	}
	return fmt.Sprintf("%s #%d,%d", negativeJump, inst.Val, inst.SkipFalse)
}

// RetA exits the BPF program, returning the value of register A.
type RetA struct{}

// Assemble implements the Instruction Assemble method.
func (a RetA) Assemble() (RawInstruction, error) {
	return RawInstruction{
		Op: opClsReturn | opRetSrcA,
	}, nil
}

// String returns the the instruction in assembler notation.
func (a RetA) String() string {
	return fmt.Sprintf("ret a")
}

// RetConstant exits the BPF program, returning a constant value.
type RetConstant struct {
	Val uint32
}

// Assemble implements the Instruction Assemble method.
func (a RetConstant) Assemble() (RawInstruction, error) {
	return RawInstruction{
		Op: opClsReturn | opRetSrcConstant,
		K:  a.Val,
	}, nil
}

// String returns the the instruction in assembler notation.
func (a RetConstant) String() string {
	return fmt.Sprintf("ret #%d", a.Val)
}

// TXA copies the value of register X to register A.
type TXA struct{}

// Assemble implements the Instruction Assemble method.
func (a TXA) Assemble() (RawInstruction, error) {
	return RawInstruction{
		Op: opClsMisc | opMiscTXA,
	}, nil
}

// String returns the the instruction in assembler notation.
func (a TXA) String() string {
	return fmt.Sprintf("txa")
}

// TAX copies the value of register A to register X.
type TAX struct{}

// Assemble implements the Instruction Assemble method.
func (a TAX) Assemble() (RawInstruction, error) {
	return RawInstruction{
		Op: opClsMisc | opMiscTAX,
	}, nil
}

// String returns the the instruction in assembler notation.
func (a TAX) String() string {
	return fmt.Sprintf("tax")
}

func assembleLoad(dst Register, loadSize int, mode uint16, k uint32) (RawInstruction, error) {
	var (
		cls uint16
		sz  uint16
	)
	switch dst {
	case RegA:
		cls = opClsLoadA
	case RegX:
		cls = opClsLoadX
	default:
		return RawInstruction{}, fmt.Errorf("invalid target register %v", dst)
	}
	switch loadSize {
	case 1:
		sz = opLoadWidth1
	case 2:
		sz = opLoadWidth2
	case 4:
		sz = opLoadWidth4
	default:
		return RawInstruction{}, fmt.Errorf("invalid load byte length %d", sz)
	}
	return RawInstruction{
		Op: cls | sz | mode,
		K:  k,
	}, nil
}
