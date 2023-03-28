package ebpf

import (
	"encoding/binary"
	"errors"
	"fmt"

	"github.com/cilium/ebpf/asm"
	"github.com/cilium/ebpf/btf"
	"github.com/cilium/ebpf/internal"
)

// splitSymbols splits insns into subsections delimited by Symbol Instructions.
// insns cannot be empty and must start with a Symbol Instruction.
//
// The resulting map is indexed by Symbol name.
func splitSymbols(insns asm.Instructions) (map[string]asm.Instructions, error) {
	if len(insns) == 0 {
		return nil, errors.New("insns is empty")
	}

	if insns[0].Symbol() == "" {
		return nil, errors.New("insns must start with a Symbol")
	}

	var name string
	progs := make(map[string]asm.Instructions)
	for _, ins := range insns {
		if sym := ins.Symbol(); sym != "" {
			if progs[sym] != nil {
				return nil, fmt.Errorf("insns contains duplicate Symbol %s", sym)
			}
			name = sym
		}

		progs[name] = append(progs[name], ins)
	}

	return progs, nil
}

// The linker is responsible for resolving bpf-to-bpf calls between programs
// within an ELF. Each BPF program must be a self-contained binary blob,
// so when an instruction in one ELF program section wants to jump to
// a function in another, the linker needs to pull in the bytecode
// (and BTF info) of the target function and concatenate the instruction
// streams.
//
// Later on in the pipeline, all call sites are fixed up with relative jumps
// within this newly-created instruction stream to then finally hand off to
// the kernel with BPF_PROG_LOAD.
//
// Each function is denoted by an ELF symbol and the compiler takes care of
// register setup before each jump instruction.

// hasFunctionReferences returns true if insns contains one or more bpf2bpf
// function references.
func hasFunctionReferences(insns asm.Instructions) bool {
	for _, i := range insns {
		if i.IsFunctionReference() {
			return true
		}
	}
	return false
}

// applyRelocations collects and applies any CO-RE relocations in insns.
//
// Passing a nil target will relocate against the running kernel. insns are
// modified in place.
func applyRelocations(insns asm.Instructions, target *btf.Spec, bo binary.ByteOrder) error {
	var relos []*btf.CORERelocation
	var reloInsns []*asm.Instruction
	iter := insns.Iterate()
	for iter.Next() {
		if relo := btf.CORERelocationMetadata(iter.Ins); relo != nil {
			relos = append(relos, relo)
			reloInsns = append(reloInsns, iter.Ins)
		}
	}

	if len(relos) == 0 {
		return nil
	}

	if bo == nil {
		bo = internal.NativeEndian
	}

	if target == nil {
		var err error
		target, err = btf.LoadKernelSpec()
		if err != nil {
			return fmt.Errorf("load kernel spec: %w", err)
		}
	}

	fixups, err := btf.CORERelocate(relos, target, bo)
	if err != nil {
		return err
	}

	for i, fixup := range fixups {
		if err := fixup.Apply(reloInsns[i]); err != nil {
			return fmt.Errorf("apply fixup %s: %w", &fixup, err)
		}
	}

	return nil
}

// flattenPrograms resolves bpf-to-bpf calls for a set of programs.
//
// Links all programs in names by modifying their ProgramSpec in progs.
func flattenPrograms(progs map[string]*ProgramSpec, names []string) {
	// Pre-calculate all function references.
	refs := make(map[*ProgramSpec][]string)
	for _, prog := range progs {
		refs[prog] = prog.Instructions.FunctionReferences()
	}

	// Create a flattened instruction stream, but don't modify progs yet to
	// avoid linking multiple times.
	flattened := make([]asm.Instructions, 0, len(names))
	for _, name := range names {
		flattened = append(flattened, flattenInstructions(name, progs, refs))
	}

	// Finally, assign the flattened instructions.
	for i, name := range names {
		progs[name].Instructions = flattened[i]
	}
}

// flattenInstructions resolves bpf-to-bpf calls for a single program.
//
// Flattens the instructions of prog by concatenating the instructions of all
// direct and indirect dependencies.
//
// progs contains all referenceable programs, while refs contain the direct
// dependencies of each program.
func flattenInstructions(name string, progs map[string]*ProgramSpec, refs map[*ProgramSpec][]string) asm.Instructions {
	prog := progs[name]

	insns := make(asm.Instructions, len(prog.Instructions))
	copy(insns, prog.Instructions)

	// Add all direct references of prog to the list of to be linked programs.
	pending := make([]string, len(refs[prog]))
	copy(pending, refs[prog])

	// All references for which we've appended instructions.
	linked := make(map[string]bool)

	// Iterate all pending references. We can't use a range since pending is
	// modified in the body below.
	for len(pending) > 0 {
		var ref string
		ref, pending = pending[0], pending[1:]

		if linked[ref] {
			// We've already linked this ref, don't append instructions again.
			continue
		}

		progRef := progs[ref]
		if progRef == nil {
			// We don't have instructions that go with this reference. This
			// happens when calling extern functions.
			continue
		}

		insns = append(insns, progRef.Instructions...)
		linked[ref] = true

		// Make sure we link indirect references.
		pending = append(pending, refs[progRef]...)
	}

	return insns
}

// fixupAndValidate is called by the ELF reader right before marshaling the
// instruction stream. It performs last-minute adjustments to the program and
// runs some sanity checks before sending it off to the kernel.
func fixupAndValidate(insns asm.Instructions) error {
	iter := insns.Iterate()
	for iter.Next() {
		ins := iter.Ins

		// Map load was tagged with a Reference, but does not contain a Map pointer.
		if ins.IsLoadFromMap() && ins.Reference() != "" && ins.Map() == nil {
			return fmt.Errorf("instruction %d: map %s: %w", iter.Index, ins.Reference(), asm.ErrUnsatisfiedMapReference)
		}

		fixupProbeReadKernel(ins)
	}

	return nil
}

// fixupProbeReadKernel replaces calls to bpf_probe_read_{kernel,user}(_str)
// with bpf_probe_read(_str) on kernels that don't support it yet.
func fixupProbeReadKernel(ins *asm.Instruction) {
	if !ins.IsBuiltinCall() {
		return
	}

	// Kernel supports bpf_probe_read_kernel, nothing to do.
	if haveProbeReadKernel() == nil {
		return
	}

	switch asm.BuiltinFunc(ins.Constant) {
	case asm.FnProbeReadKernel, asm.FnProbeReadUser:
		ins.Constant = int64(asm.FnProbeRead)
	case asm.FnProbeReadKernelStr, asm.FnProbeReadUserStr:
		ins.Constant = int64(asm.FnProbeReadStr)
	}
}
