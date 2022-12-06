package link

import (
	"fmt"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal/btf"
)

type FreplaceLink struct {
	RawLink
}

// AttachFreplace attaches the given eBPF program to the function it replaces.
//
// The program and name can either be provided at link time, or can be provided
// at program load time. If they were provided at load time, they should be nil
// and empty respectively here, as they will be ignored by the kernel.
// Examples:
//
//	AttachFreplace(dispatcher, "function", replacement)
//	AttachFreplace(nil, "", replacement)
func AttachFreplace(targetProg *ebpf.Program, name string, prog *ebpf.Program) (*FreplaceLink, error) {
	if (name == "") != (targetProg == nil) {
		return nil, fmt.Errorf("must provide both or neither of name and targetProg: %w", errInvalidInput)
	}
	if prog == nil {
		return nil, fmt.Errorf("prog cannot be nil: %w", errInvalidInput)
	}
	if prog.Type() != ebpf.Extension {
		return nil, fmt.Errorf("eBPF program type %s is not an Extension: %w", prog.Type(), errInvalidInput)
	}

	var (
		target int
		typeID btf.TypeID
	)
	if targetProg != nil {
		info, err := targetProg.Info()
		if err != nil {
			return nil, err
		}
		btfID, ok := info.BTFID()
		if !ok {
			return nil, fmt.Errorf("could not get BTF ID for program %s: %w", info.Name, errInvalidInput)
		}
		btfHandle, err := btf.NewHandleFromID(btfID)
		if err != nil {
			return nil, err
		}
		defer btfHandle.Close()

		var function *btf.Func
		if err := btfHandle.Spec().FindType(name, &function); err != nil {
			return nil, err
		}

		target = targetProg.FD()
		typeID = function.ID()
	}

	link, err := AttachRawLink(RawLinkOptions{
		Target:  target,
		Program: prog,
		Attach:  ebpf.AttachNone,
		BTF:     typeID,
	})
	if err != nil {
		return nil, err
	}

	return &FreplaceLink{*link}, nil
}

// Update implements the Link interface.
func (f *FreplaceLink) Update(new *ebpf.Program) error {
	return fmt.Errorf("freplace update: %w", ErrNotSupported)
}

// LoadPinnedFreplace loads a pinned iterator from a bpffs.
func LoadPinnedFreplace(fileName string, opts *ebpf.LoadPinOptions) (*FreplaceLink, error) {
	link, err := LoadPinnedRawLink(fileName, TracingType, opts)
	if err != nil {
		return nil, err
	}

	return &FreplaceLink{*link}, err
}
