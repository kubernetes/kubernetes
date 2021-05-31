package link

import (
	"fmt"
	"io"

	"github.com/cilium/ebpf"
)

type IterOptions struct {
	// Program must be of type Tracing with attach type
	// AttachTraceIter. The kind of iterator to attach to is
	// determined at load time via the AttachTo field.
	//
	// AttachTo requires the kernel to include BTF of itself,
	// and it to be compiled with a recent pahole (>= 1.16).
	Program *ebpf.Program
}

// AttachIter attaches a BPF seq_file iterator.
func AttachIter(opts IterOptions) (*Iter, error) {
	link, err := AttachRawLink(RawLinkOptions{
		Program: opts.Program,
		Attach:  ebpf.AttachTraceIter,
	})
	if err != nil {
		return nil, fmt.Errorf("can't link iterator: %w", err)
	}

	return &Iter{*link}, err
}

// LoadPinnedIter loads a pinned iterator from a bpffs.
func LoadPinnedIter(fileName string, opts *ebpf.LoadPinOptions) (*Iter, error) {
	link, err := LoadPinnedRawLink(fileName, IterType, opts)
	if err != nil {
		return nil, err
	}

	return &Iter{*link}, err
}

// Iter represents an attached bpf_iter.
type Iter struct {
	RawLink
}

// Open creates a new instance of the iterator.
//
// Reading from the returned reader triggers the BPF program.
func (it *Iter) Open() (io.ReadCloser, error) {
	linkFd, err := it.fd.Value()
	if err != nil {
		return nil, err
	}

	attr := &bpfIterCreateAttr{
		linkFd: linkFd,
	}

	fd, err := bpfIterCreate(attr)
	if err != nil {
		return nil, fmt.Errorf("can't create iterator: %w", err)
	}

	return fd.File("bpf_iter"), nil
}
