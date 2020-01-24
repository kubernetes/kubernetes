package ebpf

import (
	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal/btf"
	"github.com/pkg/errors"
)

// link resolves bpf-to-bpf calls.
//
// Each library may contain multiple functions / labels, and is only linked
// if the program being edited references one of these functions.
//
// Libraries must not require linking themselves.
func link(prog *ProgramSpec, libs []*ProgramSpec) error {
	for _, lib := range libs {
		insns, err := linkSection(prog.Instructions, lib.Instructions)
		if err != nil {
			return errors.Wrapf(err, "linking %s", lib.Name)
		}

		if len(insns) == len(prog.Instructions) {
			continue
		}

		prog.Instructions = insns
		if prog.BTF != nil && lib.BTF != nil {
			if err := btf.ProgramAppend(prog.BTF, lib.BTF); err != nil {
				return errors.Wrapf(err, "linking BTF of %s", lib.Name)
			}
		}
	}
	return nil
}

func linkSection(insns, section asm.Instructions) (asm.Instructions, error) {
	// A map of symbols to the libraries which contain them.
	symbols, err := section.SymbolOffsets()
	if err != nil {
		return nil, err
	}

	for _, ins := range insns {
		if ins.Reference == "" {
			continue
		}

		if ins.OpCode.JumpOp() != asm.Call || ins.Src != asm.R1 {
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
		// library is called from insns. Merge the two sections.
		// The rewrite of ins.Constant happens in asm.Instruction.Marshal.
		return append(insns, section...), nil
	}

	// None of the functions in the section are called. Do nothing.
	return insns, nil
}
