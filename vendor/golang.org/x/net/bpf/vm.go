// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package bpf

import (
	"errors"
	"fmt"
)

// A VM is an emulated BPF virtual machine.
type VM struct {
	filter []Instruction
}

// NewVM returns a new VM using the input BPF program.
func NewVM(filter []Instruction) (*VM, error) {
	if len(filter) == 0 {
		return nil, errors.New("one or more Instructions must be specified")
	}

	for i, ins := range filter {
		check := len(filter) - (i + 1)
		switch ins := ins.(type) {
		// Check for out-of-bounds jumps in instructions
		case Jump:
			if check <= int(ins.Skip) {
				return nil, fmt.Errorf("cannot jump %d instructions; jumping past program bounds", ins.Skip)
			}
		case JumpIf:
			if check <= int(ins.SkipTrue) {
				return nil, fmt.Errorf("cannot jump %d instructions in true case; jumping past program bounds", ins.SkipTrue)
			}
			if check <= int(ins.SkipFalse) {
				return nil, fmt.Errorf("cannot jump %d instructions in false case; jumping past program bounds", ins.SkipFalse)
			}
		case JumpIfX:
			if check <= int(ins.SkipTrue) {
				return nil, fmt.Errorf("cannot jump %d instructions in true case; jumping past program bounds", ins.SkipTrue)
			}
			if check <= int(ins.SkipFalse) {
				return nil, fmt.Errorf("cannot jump %d instructions in false case; jumping past program bounds", ins.SkipFalse)
			}
		// Check for division or modulus by zero
		case ALUOpConstant:
			if ins.Val != 0 {
				break
			}

			switch ins.Op {
			case ALUOpDiv, ALUOpMod:
				return nil, errors.New("cannot divide by zero using ALUOpConstant")
			}
		// Check for unknown extensions
		case LoadExtension:
			switch ins.Num {
			case ExtLen:
			default:
				return nil, fmt.Errorf("extension %d not implemented", ins.Num)
			}
		}
	}

	// Make sure last instruction is a return instruction
	switch filter[len(filter)-1].(type) {
	case RetA, RetConstant:
	default:
		return nil, errors.New("BPF program must end with RetA or RetConstant")
	}

	// Though our VM works using disassembled instructions, we
	// attempt to assemble the input filter anyway to ensure it is compatible
	// with an operating system VM.
	_, err := Assemble(filter)

	return &VM{
		filter: filter,
	}, err
}

// Run runs the VM's BPF program against the input bytes.
// Run returns the number of bytes accepted by the BPF program, and any errors
// which occurred while processing the program.
func (v *VM) Run(in []byte) (int, error) {
	var (
		// Registers of the virtual machine
		regA       uint32
		regX       uint32
		regScratch [16]uint32

		// OK is true if the program should continue processing the next
		// instruction, or false if not, causing the loop to break
		ok = true
	)

	// TODO(mdlayher): implement:
	// - NegateA:
	//   - would require a change from uint32 registers to int32
	//     registers

	// TODO(mdlayher): add interop tests that check signedness of ALU
	// operations against kernel implementation, and make sure Go
	// implementation matches behavior

	for i := 0; i < len(v.filter) && ok; i++ {
		ins := v.filter[i]

		switch ins := ins.(type) {
		case ALUOpConstant:
			regA = aluOpConstant(ins, regA)
		case ALUOpX:
			regA, ok = aluOpX(ins, regA, regX)
		case Jump:
			i += int(ins.Skip)
		case JumpIf:
			jump := jumpIf(ins, regA)
			i += jump
		case JumpIfX:
			jump := jumpIfX(ins, regA, regX)
			i += jump
		case LoadAbsolute:
			regA, ok = loadAbsolute(ins, in)
		case LoadConstant:
			regA, regX = loadConstant(ins, regA, regX)
		case LoadExtension:
			regA = loadExtension(ins, in)
		case LoadIndirect:
			regA, ok = loadIndirect(ins, in, regX)
		case LoadMemShift:
			regX, ok = loadMemShift(ins, in)
		case LoadScratch:
			regA, regX = loadScratch(ins, regScratch, regA, regX)
		case RetA:
			return int(regA), nil
		case RetConstant:
			return int(ins.Val), nil
		case StoreScratch:
			regScratch = storeScratch(ins, regScratch, regA, regX)
		case TAX:
			regX = regA
		case TXA:
			regA = regX
		default:
			return 0, fmt.Errorf("unknown Instruction at index %d: %T", i, ins)
		}
	}

	return 0, nil
}
