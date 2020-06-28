package ebpf

import (
	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/internal/btf"

	"golang.org/x/xerrors"
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
				return xerrors.Errorf("linking %s: %w", lib.Name, err)
			}

			if !needed {
				continue
			}

			linked[lib] = true
			prog.Instructions = append(prog.Instructions, lib.Instructions...)
			pending = append(pending, lib.Instructions)

			if prog.BTF != nil && lib.BTF != nil {
				if err := btf.ProgramAppend(prog.BTF, lib.BTF); err != nil {
					return xerrors.Errorf("linking BTF of %s: %w", lib.Name, err)
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
