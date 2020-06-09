package ebpf

import (
	"github.com/cilium/ebpf/asm"
)

// link resolves bpf-to-bpf calls.
//
// Each section may contain multiple functions / labels, and is only linked
// if the program being edited references one of these functions.
//
// Sections must not require linking themselves.
func link(insns asm.Instructions, sections ...asm.Instructions) (asm.Instructions, error) {
	for _, section := range sections {
		var err error
		insns, err = linkSection(insns, section)
		if err != nil {
			return nil, err
		}
	}
	return insns, nil
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
