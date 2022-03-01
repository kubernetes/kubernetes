package link

import (
	"fmt"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal"
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
		return nil, fmt.Errorf("invalid program: %w", internal.ErrClosedFd)
	}

	fd, err := bpfRawTracepointOpen(&bpfRawTracepointOpenAttr{
		name: internal.NewStringPointer(opts.Name),
		fd:   uint32(opts.Program.FD()),
	})
	if err != nil {
		return nil, err
	}

	return &progAttachRawTracepoint{fd: fd}, nil
}

type progAttachRawTracepoint struct {
	fd *internal.FD
}

var _ Link = (*progAttachRawTracepoint)(nil)

func (rt *progAttachRawTracepoint) isLink() {}

func (rt *progAttachRawTracepoint) Close() error {
	return rt.fd.Close()
}

func (rt *progAttachRawTracepoint) Update(_ *ebpf.Program) error {
	return fmt.Errorf("can't update raw_tracepoint: %w", ErrNotSupported)
}

func (rt *progAttachRawTracepoint) Pin(_ string) error {
	return fmt.Errorf("can't pin raw_tracepoint: %w", ErrNotSupported)
}

func (rt *progAttachRawTracepoint) Unpin() error {
	return fmt.Errorf("unpin raw_tracepoint: %w", ErrNotSupported)
}
