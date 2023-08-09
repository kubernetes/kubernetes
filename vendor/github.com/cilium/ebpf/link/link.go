package link

import (
	"fmt"
	"unsafe"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/btf"
)

var ErrNotSupported = internal.ErrNotSupported

// Link represents a Program attached to a BPF hook.
type Link interface {
	// Replace the current program with a new program.
	//
	// Passing a nil program is an error. May return an error wrapping ErrNotSupported.
	Update(*ebpf.Program) error

	// Persist a link by pinning it into a bpffs.
	//
	// May return an error wrapping ErrNotSupported.
	Pin(string) error

	// Undo a previous call to Pin.
	//
	// May return an error wrapping ErrNotSupported.
	Unpin() error

	// Close frees resources.
	//
	// The link will be broken unless it has been successfully pinned.
	// A link may continue past the lifetime of the process if Close is
	// not called.
	Close() error

	// Prevent external users from implementing this interface.
	isLink()
}

// ID uniquely identifies a BPF link.
type ID uint32

// RawLinkOptions control the creation of a raw link.
type RawLinkOptions struct {
	// File descriptor to attach to. This differs for each attach type.
	Target int
	// Program to attach.
	Program *ebpf.Program
	// Attach must match the attach type of Program.
	Attach ebpf.AttachType
	// BTF is the BTF of the attachment target.
	BTF btf.TypeID
}

// RawLinkInfo contains metadata on a link.
type RawLinkInfo struct {
	Type    Type
	ID      ID
	Program ebpf.ProgramID
}

// RawLink is the low-level API to bpf_link.
//
// You should consider using the higher level interfaces in this
// package instead.
type RawLink struct {
	fd         *internal.FD
	pinnedPath string
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
		targetFd:    uint32(opts.Target),
		progFd:      uint32(progFd),
		attachType:  opts.Attach,
		targetBTFID: uint32(opts.BTF),
	}
	fd, err := bpfLinkCreate(&attr)
	if err != nil {
		return nil, fmt.Errorf("can't create link: %s", err)
	}

	return &RawLink{fd, ""}, nil
}

// LoadPinnedRawLink loads a persisted link from a bpffs.
//
// Returns an error if the pinned link type doesn't match linkType. Pass
// UnspecifiedType to disable this behaviour.
func LoadPinnedRawLink(fileName string, linkType Type, opts *ebpf.LoadPinOptions) (*RawLink, error) {
	fd, err := internal.BPFObjGet(fileName, opts.Marshal())
	if err != nil {
		return nil, fmt.Errorf("load pinned link: %w", err)
	}

	link := &RawLink{fd, fileName}
	if linkType == UnspecifiedType {
		return link, nil
	}

	info, err := link.Info()
	if err != nil {
		link.Close()
		return nil, fmt.Errorf("get pinned link info: %s", err)
	}

	if info.Type != linkType {
		link.Close()
		return nil, fmt.Errorf("link type %v doesn't match %v", info.Type, linkType)
	}

	return link, nil
}

func (l *RawLink) isLink() {}

// FD returns the raw file descriptor.
func (l *RawLink) FD() int {
	fd, err := l.fd.Value()
	if err != nil {
		return -1
	}
	return int(fd)
}

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
	if err := internal.Pin(l.pinnedPath, fileName, l.fd); err != nil {
		return err
	}
	l.pinnedPath = fileName
	return nil
}

// Unpin implements the Link interface.
func (l *RawLink) Unpin() error {
	if err := internal.Unpin(l.pinnedPath); err != nil {
		return err
	}
	l.pinnedPath = ""
	return nil
}

// Update implements the Link interface.
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

// struct bpf_link_info
type bpfLinkInfo struct {
	typ     uint32
	id      uint32
	prog_id uint32
}

// Info returns metadata about the link.
func (l *RawLink) Info() (*RawLinkInfo, error) {
	var info bpfLinkInfo
	err := internal.BPFObjGetInfoByFD(l.fd, unsafe.Pointer(&info), unsafe.Sizeof(info))
	if err != nil {
		return nil, fmt.Errorf("link info: %s", err)
	}

	return &RawLinkInfo{
		Type(info.typ),
		ID(info.id),
		ebpf.ProgramID(info.prog_id),
	}, nil
}
