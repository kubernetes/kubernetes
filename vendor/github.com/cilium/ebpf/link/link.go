package link

import (
	"bytes"
	"encoding/binary"
	"fmt"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/btf"
	"github.com/cilium/ebpf/internal"
	"github.com/cilium/ebpf/internal/sys"
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

	// Info returns metadata on a link.
	//
	// May return an error wrapping ErrNotSupported.
	Info() (*Info, error)

	// Prevent external users from implementing this interface.
	isLink()
}

// LoadPinnedLink loads a link that was persisted into a bpffs.
func LoadPinnedLink(fileName string, opts *ebpf.LoadPinOptions) (Link, error) {
	raw, err := loadPinnedRawLink(fileName, opts)
	if err != nil {
		return nil, err
	}

	return wrapRawLink(raw)
}

// wrap a RawLink in a more specific type if possible.
//
// The function takes ownership of raw and closes it on error.
func wrapRawLink(raw *RawLink) (Link, error) {
	info, err := raw.Info()
	if err != nil {
		raw.Close()
		return nil, err
	}

	switch info.Type {
	case RawTracepointType:
		return &rawTracepoint{*raw}, nil
	case TracingType:
		return &tracing{*raw}, nil
	case CgroupType:
		return &linkCgroup{*raw}, nil
	case IterType:
		return &Iter{*raw}, nil
	case NetNsType:
		return &NetNsLink{*raw}, nil
	default:
		return raw, nil
	}
}

// ID uniquely identifies a BPF link.
type ID = sys.LinkID

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
	// Flags control the attach behaviour.
	Flags uint32
}

// Info contains metadata on a link.
type Info struct {
	Type    Type
	ID      ID
	Program ebpf.ProgramID
	extra   interface{}
}

type TracingInfo sys.TracingLinkInfo
type CgroupInfo sys.CgroupLinkInfo
type NetNsInfo sys.NetNsLinkInfo
type XDPInfo sys.XDPLinkInfo

// Tracing returns tracing type-specific link info.
//
// Returns nil if the type-specific link info isn't available.
func (r Info) Tracing() *TracingInfo {
	e, _ := r.extra.(*TracingInfo)
	return e
}

// Cgroup returns cgroup type-specific link info.
//
// Returns nil if the type-specific link info isn't available.
func (r Info) Cgroup() *CgroupInfo {
	e, _ := r.extra.(*CgroupInfo)
	return e
}

// NetNs returns netns type-specific link info.
//
// Returns nil if the type-specific link info isn't available.
func (r Info) NetNs() *NetNsInfo {
	e, _ := r.extra.(*NetNsInfo)
	return e
}

// ExtraNetNs returns XDP type-specific link info.
//
// Returns nil if the type-specific link info isn't available.
func (r Info) XDP() *XDPInfo {
	e, _ := r.extra.(*XDPInfo)
	return e
}

// RawLink is the low-level API to bpf_link.
//
// You should consider using the higher level interfaces in this
// package instead.
type RawLink struct {
	fd         *sys.FD
	pinnedPath string
}

// AttachRawLink creates a raw link.
func AttachRawLink(opts RawLinkOptions) (*RawLink, error) {
	if err := haveBPFLink(); err != nil {
		return nil, err
	}

	if opts.Target < 0 {
		return nil, fmt.Errorf("invalid target: %s", sys.ErrClosedFd)
	}

	progFd := opts.Program.FD()
	if progFd < 0 {
		return nil, fmt.Errorf("invalid program: %s", sys.ErrClosedFd)
	}

	attr := sys.LinkCreateAttr{
		TargetFd:    uint32(opts.Target),
		ProgFd:      uint32(progFd),
		AttachType:  sys.AttachType(opts.Attach),
		TargetBtfId: uint32(opts.BTF),
		Flags:       opts.Flags,
	}
	fd, err := sys.LinkCreate(&attr)
	if err != nil {
		return nil, fmt.Errorf("can't create link: %s", err)
	}

	return &RawLink{fd, ""}, nil
}

func loadPinnedRawLink(fileName string, opts *ebpf.LoadPinOptions) (*RawLink, error) {
	fd, err := sys.ObjGet(&sys.ObjGetAttr{
		Pathname:  sys.NewStringPointer(fileName),
		FileFlags: opts.Marshal(),
	})
	if err != nil {
		return nil, fmt.Errorf("load pinned link: %w", err)
	}

	return &RawLink{fd, fileName}, nil
}

func (l *RawLink) isLink() {}

// FD returns the raw file descriptor.
func (l *RawLink) FD() int {
	return l.fd.Int()
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
		return fmt.Errorf("invalid program: %s", sys.ErrClosedFd)
	}

	var oldFd int
	if opts.Old != nil {
		oldFd = opts.Old.FD()
		if oldFd < 0 {
			return fmt.Errorf("invalid replacement program: %s", sys.ErrClosedFd)
		}
	}

	attr := sys.LinkUpdateAttr{
		LinkFd:    l.fd.Uint(),
		NewProgFd: uint32(newFd),
		OldProgFd: uint32(oldFd),
		Flags:     opts.Flags,
	}
	return sys.LinkUpdate(&attr)
}

// Info returns metadata about the link.
func (l *RawLink) Info() (*Info, error) {
	var info sys.LinkInfo

	if err := sys.ObjInfo(l.fd, &info); err != nil {
		return nil, fmt.Errorf("link info: %s", err)
	}

	var extra interface{}
	switch info.Type {
	case CgroupType:
		extra = &CgroupInfo{}
	case IterType:
		// not supported
	case NetNsType:
		extra = &NetNsInfo{}
	case RawTracepointType:
		// not supported
	case TracingType:
		extra = &TracingInfo{}
	case XDPType:
		extra = &XDPInfo{}
	case PerfEventType:
		// no extra
	default:
		return nil, fmt.Errorf("unknown link info type: %d", info.Type)
	}

	if info.Type != RawTracepointType && info.Type != IterType && info.Type != PerfEventType {
		buf := bytes.NewReader(info.Extra[:])
		err := binary.Read(buf, internal.NativeEndian, extra)
		if err != nil {
			return nil, fmt.Errorf("can not read extra link info: %w", err)
		}
	}

	return &Info{
		info.Type,
		info.Id,
		ebpf.ProgramID(info.ProgId),
		extra,
	}, nil
}
