package link

import (
	"errors"
	"fmt"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal/sys"
)

type RawTracepointOptions struct {
	// Tracepoint name.
	Name string
	// Program must be of type RawTracepoint*
	Program *ebpf.Program
}

// AttachRawTracepoint links a BPF program to a raw_tracepoint.
//
// Requires at least Linux 4.17.
func AttachRawTracepoint(opts RawTracepointOptions) (Link, error) {
	if t := opts.Program.Type(); t != ebpf.RawTracepoint && t != ebpf.RawTracepointWritable {
		return nil, fmt.Errorf("invalid program type %s, expected RawTracepoint(Writable)", t)
	}
	if opts.Program.FD() < 0 {
		return nil, fmt.Errorf("invalid program: %w", sys.ErrClosedFd)
	}

	fd, err := sys.RawTracepointOpen(&sys.RawTracepointOpenAttr{
		Name:   sys.NewStringPointer(opts.Name),
		ProgFd: uint32(opts.Program.FD()),
	})
	if err != nil {
		return nil, err
	}

	err = haveBPFLink()
	if errors.Is(err, ErrNotSupported) {
		// Prior to commit 70ed506c3bbc ("bpf: Introduce pinnable bpf_link abstraction")
		// raw_tracepoints are just a plain fd.
		return &simpleRawTracepoint{fd}, nil
	}

	if err != nil {
		return nil, err
	}

	return &rawTracepoint{RawLink{fd: fd}}, nil
}

type simpleRawTracepoint struct {
	fd *sys.FD
}

var _ Link = (*simpleRawTracepoint)(nil)

func (frt *simpleRawTracepoint) isLink() {}

func (frt *simpleRawTracepoint) Close() error {
	return frt.fd.Close()
}

func (frt *simpleRawTracepoint) Update(_ *ebpf.Program) error {
	return fmt.Errorf("update raw_tracepoint: %w", ErrNotSupported)
}

func (frt *simpleRawTracepoint) Pin(string) error {
	return fmt.Errorf("pin raw_tracepoint: %w", ErrNotSupported)
}

func (frt *simpleRawTracepoint) Unpin() error {
	return fmt.Errorf("unpin raw_tracepoint: %w", ErrNotSupported)
}

func (frt *simpleRawTracepoint) Info() (*Info, error) {
	return nil, fmt.Errorf("can't get raw_tracepoint info: %w", ErrNotSupported)
}

type rawTracepoint struct {
	RawLink
}

var _ Link = (*rawTracepoint)(nil)

func (rt *rawTracepoint) Update(_ *ebpf.Program) error {
	return fmt.Errorf("update raw_tracepoint: %w", ErrNotSupported)
}
