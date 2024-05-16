package link

import (
	"fmt"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/btf"
	"github.com/cilium/ebpf/internal/sys"
)

type tracing struct {
	RawLink
}

func (f *tracing) Update(new *ebpf.Program) error {
	return fmt.Errorf("tracing update: %w", ErrNotSupported)
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
func AttachFreplace(targetProg *ebpf.Program, name string, prog *ebpf.Program) (Link, error) {
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
		btfHandle, err := targetProg.Handle()
		if err != nil {
			return nil, err
		}
		defer btfHandle.Close()

		spec, err := btfHandle.Spec(nil)
		if err != nil {
			return nil, err
		}

		var function *btf.Func
		if err := spec.TypeByName(name, &function); err != nil {
			return nil, err
		}

		target = targetProg.FD()
		typeID, err = spec.TypeID(function)
		if err != nil {
			return nil, err
		}
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

	return &tracing{*link}, nil
}

type TracingOptions struct {
	// Program must be of type Tracing with attach type
	// AttachTraceFEntry/AttachTraceFExit/AttachModifyReturn or
	// AttachTraceRawTp.
	Program *ebpf.Program
}

type LSMOptions struct {
	// Program must be of type LSM with attach type
	// AttachLSMMac.
	Program *ebpf.Program
}

// attachBTFID links all BPF program types (Tracing/LSM) that they attach to a btf_id.
func attachBTFID(program *ebpf.Program) (Link, error) {
	if program.FD() < 0 {
		return nil, fmt.Errorf("invalid program %w", sys.ErrClosedFd)
	}

	fd, err := sys.RawTracepointOpen(&sys.RawTracepointOpenAttr{
		ProgFd: uint32(program.FD()),
	})
	if err != nil {
		return nil, err
	}

	raw := RawLink{fd: fd}
	info, err := raw.Info()
	if err != nil {
		raw.Close()
		return nil, err
	}

	if info.Type == RawTracepointType {
		// Sadness upon sadness: a Tracing program with AttachRawTp returns
		// a raw_tracepoint link. Other types return a tracing link.
		return &rawTracepoint{raw}, nil
	}

	return &tracing{RawLink: RawLink{fd: fd}}, nil
}

// AttachTracing links a tracing (fentry/fexit/fmod_ret) BPF program or
// a BTF-powered raw tracepoint (tp_btf) BPF Program to a BPF hook defined
// in kernel modules.
func AttachTracing(opts TracingOptions) (Link, error) {
	if t := opts.Program.Type(); t != ebpf.Tracing {
		return nil, fmt.Errorf("invalid program type %s, expected Tracing", t)
	}

	return attachBTFID(opts.Program)
}

// AttachLSM links a Linux security module (LSM) BPF Program to a BPF
// hook defined in kernel modules.
func AttachLSM(opts LSMOptions) (Link, error) {
	if t := opts.Program.Type(); t != ebpf.LSM {
		return nil, fmt.Errorf("invalid program type %s, expected LSM", t)
	}

	return attachBTFID(opts.Program)
}
