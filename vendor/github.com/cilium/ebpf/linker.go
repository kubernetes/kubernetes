package ebpf

import (
	"fmt"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal/btf"
)

// link resolves bpf-to-bpf calls.
//
// Each library may contain multiple functions / labels, and is only linked
// if prog references one of these functions.
//
// Libraries also linked.
func link(prog *ProgramSpec, libs []*ProgramSpec) error {
	var (
		linked  = make(map[*ProgramSpec]bool)
		pending = []asm.Instructions{prog.Instructions}
		insns   asm.Instructions
	)
	for len(pending) > 0 {
		insns, pending = pending[0], pending[1:]
		for _, lib := range libs {
			if linked[lib] {
				continue
			}

			needed, err := needSection(insns, lib.Instructions)
			if err != nil {
				return fmt.Errorf("linking %s: %w", lib.Name, err)
			}

			if !needed {
				continue
			}

			linked[lib] = true
			prog.Instructions = append(prog.Instructions, lib.Instructions...)
			pending = append(pending, lib.Instructions)

			if prog.BTF != nil && lib.BTF != nil {
				if err := btf.ProgramAppend(prog.BTF, lib.BTF); err != nil {
					return fmt.Errorf("linking BTF of %s: %w", lib.Name, err)
				}
			}
		}
	}

	return nil
}

func needSection(insns, section asm.Instructions) (bool, error) {
	// A map of symbols to the libraries which contain them.
	symbols, err := section.SymbolOffsets()
	if err != nil {
		return false, err
	}

	for _, ins := range insns {
		if ins.Reference == "" {
			continue
		}

		if ins.OpCode.JumpOp() != asm.Call || ins.Src != asm.PseudoCall {
			continue
		}

		if ins.Constant != -1 {
			// This is already a valid call, no need to link again.
			continue
		}

		if _, ok := symbols[ins.Reference]; !ok {
			// Symbol isn't available in this section
			continue
		}

		// At this point we know that at least one function in the
		// library is called from insns, so we have to link it.
		return true, nil
	}

	// None of the functions in the section are called.
	return false, nil
}

func fixupJumpsAndCalls(insns asm.Instructions) error {
	symbolOffsets := make(map[string]asm.RawInstructionOffset)
	iter := insns.Iterate()
	for iter.Next() {
		ins := iter.Ins

		if ins.Symbol == "" {
			continue
		}

		if _, ok := symbolOffsets[ins.Symbol]; ok {
			return fmt.Errorf("duplicate symbol %s", ins.Symbol)
		}

		symbolOffsets[ins.Symbol] = iter.Offset
	}

	iter = insns.Iterate()
	for iter.Next() {
		i := iter.Index
		offset := iter.Offset
		ins := iter.Ins

		if ins.Reference == "" {
			continue
		}

		switch {
		case ins.IsFunctionCall() && ins.Constant == -1:
			// Rewrite bpf to bpf call
			callOffset, ok := symbolOffsets[ins.Reference]
			if !ok {
				return fmt.Errorf("call at %d: reference to missing symbol %q", i, ins.Reference)
			}

			ins.Constant = int64(callOffset - offset - 1)

		case ins.OpCode.Class() == asm.JumpClass && ins.Offset == -1:
			// Rewrite jump to label
			jumpOffset, ok := symbolOffsets[ins.Reference]
			if !ok {
				return fmt.Errorf("jump at %d: reference to missing symbol %q", i, ins.Reference)
			}

			ins.Offset = int16(jumpOffset - offset - 1)

		case ins.IsLoadFromMap() && ins.MapPtr() == -1:
			return fmt.Errorf("map %s: %w", ins.Reference, errUnsatisfiedReference)
		}
	}

	return nil
}
