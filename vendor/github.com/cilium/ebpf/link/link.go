package link

import (
	"fmt"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal"
)

var ErrNotSupported = internal.ErrNotSupported

// Link represents a Program attached to a BPF hook.
type Link interface {
	// Replace the current program with a new program.
	//
	// Passing a nil program is an error.
	Update(*ebpf.Program) error

	// Persist a link by pinning it into a bpffs.
	//
	// May return an error wrapping ErrNotSupported.
	Pin(string) error

	// Close frees resources.
	//
	// The link will be broken unless it has been pinned. A link
	// may continue past the lifetime of the process if Close is
	// not called.
	Close() error

	// Prevent external users from implementing this interface.
	isLink()
}

// RawLinkOptions control the creation of a raw link.
type RawLinkOptions struct {
	// File descriptor to attach to. This differs for each attach type.
	Target int
	// Program to attach.
	Program *ebpf.Program
	// Attach must match the attach type of Program.
	Attach ebpf.AttachType
}

// RawLink is the low-level API to bpf_link.
//
// You should consider using the higher level interfaces in this
// package instead.
type RawLink struct {
	fd *internal.FD
}

// AttachRawLink creates a raw link.
func AttachRawLink(opts RawLinkOptions) (*RawLink, error) {
	if err := haveBPFLink(); err != nil {
		return nil, err
	}

	if opts.Target < 0 {
		return nil, fmt.Errorf("invalid target: %s", internal.ErrClosedFd)
	}

	progFd := opts.Program.FD()
	if progFd < 0 {
		return nil, fmt.Errorf("invalid program: %s", internal.ErrClosedFd)
	}

	attr := bpfLinkCreateAttr{
		targetFd:   uint32(opts.Target),
		progFd:     uint32(progFd),
		attachType: opts.Attach,
	}
	fd, err := bpfLinkCreate(&attr)
	if err != nil {
		return nil, fmt.Errorf("can't create link: %s", err)
	}

	return &RawLink{fd}, nil
}

// LoadPinnedRawLink loads a persisted link from a bpffs.
func LoadPinnedRawLink(fileName string) (*RawLink, error) {
	fd, err := internal.BPFObjGet(fileName)
	if err != nil {
		return nil, fmt.Errorf("can't load pinned link: %s", err)
	}

	return &RawLink{fd}, nil
}

func (l *RawLink) isLink() {}

// Close breaks the link.
//
// Use Pin if you want to make the link persistent.
func (l *RawLink) Close() error {
	return l.fd.Close()
}

// Pin persists a link past the lifetime of the process.
//
// Calling Close on a pinned Link will not break the link
// until the pin is removed.
func (l *RawLink) Pin(fileName string) error {
	if err := internal.BPFObjPin(fileName, l.fd); err != nil {
		return fmt.Errorf("can't pin link: %s", err)
	}
	return nil
}

// Update implements Link.
func (l *RawLink) Update(new *ebpf.Program) error {
	return l.UpdateArgs(RawLinkUpdateOptions{
		New: new,
	})
}

// RawLinkUpdateOptions control the behaviour of RawLink.UpdateArgs.
type RawLinkUpdateOptions struct {
	New   *ebpf.Program
	Old   *ebpf.Program
	Flags uint32
}

// UpdateArgs updates a link based on args.
func (l *RawLink) UpdateArgs(opts RawLinkUpdateOptions) error {
	newFd := opts.New.FD()
	if newFd < 0 {
		return fmt.Errorf("invalid program: %s", internal.ErrClosedFd)
	}

	var oldFd int
	if opts.Old != nil {
		oldFd = opts.Old.FD()
		if oldFd < 0 {
			return fmt.Errorf("invalid replacement program: %s", internal.ErrClosedFd)
		}
	}

	linkFd, err := l.fd.Value()
	if err != nil {
		return fmt.Errorf("can't update link: %s", err)
	}

	attr := bpfLinkUpdateAttr{
		linkFd:    linkFd,
		newProgFd: uint32(newFd),
		oldProgFd: uint32(oldFd),
		flags:     opts.Flags,
	}
	return bpfLinkUpdate(&attr)
}
