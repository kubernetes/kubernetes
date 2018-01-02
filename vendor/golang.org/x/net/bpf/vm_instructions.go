// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpf

import (
	"encoding/binary"
	"fmt"
)

func aluOpConstant(ins ALUOpConstant, regA uint32) uint32 {
	return aluOpCommon(ins.Op, regA, ins.Val)
}

func aluOpX(ins ALUOpX, regA uint32, regX uint32) (uint32, bool) {
	// Guard against division or modulus by zero by terminating
	// the program, as the OS BPF VM does
	if regX == 0 {
		switch ins.Op {
		case ALUOpDiv, ALUOpMod:
			return 0, false
		}
	}

	return aluOpCommon(ins.Op, regA, regX), true
}

func aluOpCommon(op ALUOp, regA uint32, value uint32) uint32 {
	switch op {
	case ALUOpAdd:
		return regA + value
	case ALUOpSub:
		return regA - value
	case ALUOpMul:
		return regA * value
	case ALUOpDiv:
		// Division by zero not permitted by NewVM and aluOpX checks
		return regA / value
	case ALUOpOr:
		return regA | value
	case ALUOpAnd:
		return regA & value
	case ALUOpShiftLeft:
		return regA << value
	case ALUOpShiftRight:
		return regA >> value
	case ALUOpMod:
		// Modulus by zero not permitted by NewVM and aluOpX checks
		return regA % value
	case ALUOpXor:
		return regA ^ value
	default:
		return regA
	}
}

func jumpIf(ins JumpIf, value uint32) int {
	var ok bool
	inV := uint32(ins.Val)

	switch ins.Cond {
	case JumpEqual:
		ok = value == inV
	case JumpNotEqual:
		ok = value != inV
	case JumpGreaterThan:
		ok = value > inV
	case JumpLessThan:
		ok = value < inV
	case JumpGreaterOrEqual:
		ok = value >= inV
	case JumpLessOrEqual:
		ok = value <= inV
	case JumpBitsSet:
		ok = (value & inV) != 0
	case JumpBitsNotSet:
		ok = (value & inV) == 0
	}

	if ok {
		return int(ins.SkipTrue)
	}

	return int(ins.SkipFalse)
}

func loadAbsolute(ins LoadAbsolute, in []byte) (uint32, bool) {
	offset := int(ins.Off)
	size := int(ins.Size)

	return loadCommon(in, offset, size)
}

func loadConstant(ins LoadConstant, regA uint32, regX uint32) (uint32, uint32) {
	switch ins.Dst {
	case RegA:
		regA = ins.Val
	case RegX:
		regX = ins.Val
	}

	return regA, regX
}

func loadExtension(ins LoadExtension, in []byte) uint32 {
	switch ins.Num {
	case ExtLen:
		return uint32(len(in))
	default:
		panic(fmt.Sprintf("unimplemented extension: %d", ins.Num))
	}
}

func loadIndirect(ins LoadIndirect, in []byte, regX uint32) (uint32, bool) {
	offset := int(ins.Off) + int(regX)
	size := int(ins.Size)

	return loadCommon(in, offset, size)
}

func loadMemShift(ins LoadMemShift, in []byte) (uint32, bool) {
	offset := int(ins.Off)

	if !inBounds(len(in), offset, 0) {
		return 0, false
	}

	// Mask off high 4 bits and multiply low 4 bits by 4
	return uint32(in[offset]&0x0f) * 4, true
}

func inBounds(inLen int, offset int, size int) bool {
	return offset+size <= inLen
}

func loadCommon(in []byte, offset int, size int) (uint32, bool) {
	if !inBounds(len(in), offset, size) {
		return 0, false
	}

	switch size {
	case 1:
		return uint32(in[offset]), true
	case 2:
		return uint32(binary.BigEndian.Uint16(in[offset : offset+size])), true
	case 4:
		return uint32(binary.BigEndian.Uint32(in[offset : offset+size])), true
	default:
		panic(fmt.Sprintf("invalid load size: %d", size))
	}
}

func loadScratch(ins LoadScratch, regScratch [16]uint32, regA uint32, regX uint32) (uint32, uint32) {
	switch ins.Dst {
	case RegA:
		regA = regScratch[ins.N]
	case RegX:
		regX = regScratch[ins.N]
	}

	return regA, regX
}

func storeScratch(ins StoreScratch, regScratch [16]uint32, regA uint32, regX uint32) [16]uint32 {
	switch ins.Src {
	case RegA:
		regScratch[ins.N] = regA
	case RegX:
		regScratch[ins.N] = regX
	}

	return regScratch
}
