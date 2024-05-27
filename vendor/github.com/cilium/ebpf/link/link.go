package link

import (
	"errors"
	"fmt"
	"os"

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

// NewLinkFromFD creates a link from a raw fd.
//
// Deprecated: use [NewFromFD] instead.
func NewLinkFromFD(fd int) (Link, error) {
	return NewFromFD(fd)
}

// NewFromFD creates a link from a raw fd.
//
// You should not use fd after calling this function.
func NewFromFD(fd int) (Link, error) {
	sysFD, err := sys.NewFD(fd)
	if err != nil {
		return nil, err
	}

	return wrapRawLink(&RawLink{fd: sysFD})
}

// NewFromID returns the link associated with the given id.
//
// Returns ErrNotExist if there is no link with the given id.
func NewFromID(id ID) (Link, error) {
	getFdAttr := &sys.LinkGetFdByIdAttr{Id: id}
	fd, err := sys.LinkGetFdById(getFdAttr)
	if err != nil {
		return nil, fmt.Errorf("get link fd from ID %d: %w", id, err)
	}

	return wrapRawLink(&RawLink{fd, ""})
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
func wrapRawLink(raw *RawLink) (_ Link, err error) {
	defer func() {
		if err != nil {
			raw.Close()
		}
	}()

	info, err := raw.Info()
	if err != nil {
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
	case KprobeMultiType:
		return &kprobeMultiLink{*raw}, nil
	case UprobeMultiType:
		return &uprobeMultiLink{*raw}, nil
	case PerfEventType:
		return nil, fmt.Errorf("recovering perf event fd: %w", ErrNotSupported)
	case TCXType:
		return &tcxLink{*raw}, nil
	case NetfilterType:
		return &netfilterLink{*raw}, nil
	case NetkitType:
		return &netkitLink{*raw}, nil
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

type TracingInfo struct {
	AttachType  sys.AttachType
	TargetObjId uint32
	TargetBtfId sys.TypeID
}

type CgroupInfo struct {
	CgroupId   uint64
	AttachType sys.AttachType
	_          [4]byte
}

type NetNsInfo struct {
	NetnsIno   uint32
	AttachType sys.AttachType
}

type TCXInfo struct {
	Ifindex    uint32
	AttachType sys.AttachType
}

type XDPInfo struct {
	Ifindex uint32
}

type NetfilterInfo struct {
	Pf       uint32
	Hooknum  uint32
	Priority int32
	Flags    uint32
}

type NetkitInfo struct {
	Ifindex    uint32
	AttachType sys.AttachType
}

type KprobeMultiInfo struct {
	count  uint32
	flags  uint32
	missed uint64
}

// AddressCount is the number of addresses hooked by the kprobe.
func (kpm *KprobeMultiInfo) AddressCount() (uint32, bool) {
	return kpm.count, kpm.count > 0
}

func (kpm *KprobeMultiInfo) Flags() (uint32, bool) {
	return kpm.flags, kpm.count > 0
}

func (kpm *KprobeMultiInfo) Missed() (uint64, bool) {
	return kpm.missed, kpm.count > 0
}

type PerfEventInfo struct {
	Type  sys.PerfEventType
	extra interface{}
}

func (r *PerfEventInfo) Kprobe() *KprobeInfo {
	e, _ := r.extra.(*KprobeInfo)
	return e
}

type KprobeInfo struct {
	address uint64
	missed  uint64
}

func (kp *KprobeInfo) Address() (uint64, bool) {
	return kp.address, kp.address > 0
}

func (kp *KprobeInfo) Missed() (uint64, bool) {
	return kp.missed, kp.address > 0
}

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

// XDP returns XDP type-specific link info.
//
// Returns nil if the type-specific link info isn't available.
func (r Info) XDP() *XDPInfo {
	e, _ := r.extra.(*XDPInfo)
	return e
}

// TCX returns TCX type-specific link info.
//
// Returns nil if the type-specific link info isn't available.
func (r Info) TCX() *TCXInfo {
	e, _ := r.extra.(*TCXInfo)
	return e
}

// Netfilter returns netfilter type-specific link info.
//
// Returns nil if the type-specific link info isn't available.
func (r Info) Netfilter() *NetfilterInfo {
	e, _ := r.extra.(*NetfilterInfo)
	return e
}

// Netkit returns netkit type-specific link info.
//
// Returns nil if the type-specific link info isn't available.
func (r Info) Netkit() *NetkitInfo {
	e, _ := r.extra.(*NetkitInfo)
	return e
}

// KprobeMulti returns kprobe-multi type-specific link info.
//
// Returns nil if the type-specific link info isn't available.
func (r Info) KprobeMulti() *KprobeMultiInfo {
	e, _ := r.extra.(*KprobeMultiInfo)
	return e
}

// PerfEvent returns perf-event type-specific link info.
//
// Returns nil if the type-specific link info isn't available.
func (r Info) PerfEvent() *PerfEventInfo {
	e, _ := r.extra.(*PerfEventInfo)
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
		TargetBtfId: opts.BTF,
		Flags:       opts.Flags,
	}
	fd, err := sys.LinkCreate(&attr)
	if err != nil {
		return nil, fmt.Errorf("create link: %w", err)
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

// IsPinned returns true if the Link has a non-empty pinned path.
func (l *RawLink) IsPinned() bool {
	return l.pinnedPath != ""
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
		var cgroupInfo sys.CgroupLinkInfo
		if err := sys.ObjInfo(l.fd, &cgroupInfo); err != nil {
			return nil, fmt.Errorf("cgroup link info: %s", err)
		}
		extra = &CgroupInfo{
			CgroupId:   cgroupInfo.CgroupId,
			AttachType: cgroupInfo.AttachType,
		}
	case NetNsType:
		var netnsInfo sys.NetNsLinkInfo
		if err := sys.ObjInfo(l.fd, &netnsInfo); err != nil {
			return nil, fmt.Errorf("netns link info: %s", err)
		}
		extra = &NetNsInfo{
			NetnsIno:   netnsInfo.NetnsIno,
			AttachType: netnsInfo.AttachType,
		}
	case TracingType:
		var tracingInfo sys.TracingLinkInfo
		if err := sys.ObjInfo(l.fd, &tracingInfo); err != nil {
			return nil, fmt.Errorf("tracing link info: %s", err)
		}
		extra = &TracingInfo{
			TargetObjId: tracingInfo.TargetObjId,
			TargetBtfId: tracingInfo.TargetBtfId,
			AttachType:  tracingInfo.AttachType,
		}
	case XDPType:
		var xdpInfo sys.XDPLinkInfo
		if err := sys.ObjInfo(l.fd, &xdpInfo); err != nil {
			return nil, fmt.Errorf("xdp link info: %s", err)
		}
		extra = &XDPInfo{
			Ifindex: xdpInfo.Ifindex,
		}
	case RawTracepointType, IterType, UprobeMultiType:
		// Extra metadata not supported.
	case TCXType:
		var tcxInfo sys.TcxLinkInfo
		if err := sys.ObjInfo(l.fd, &tcxInfo); err != nil {
			return nil, fmt.Errorf("tcx link info: %s", err)
		}
		extra = &TCXInfo{
			Ifindex:    tcxInfo.Ifindex,
			AttachType: tcxInfo.AttachType,
		}
	case NetfilterType:
		var netfilterInfo sys.NetfilterLinkInfo
		if err := sys.ObjInfo(l.fd, &netfilterInfo); err != nil {
			return nil, fmt.Errorf("netfilter link info: %s", err)
		}
		extra = &NetfilterInfo{
			Pf:       netfilterInfo.Pf,
			Hooknum:  netfilterInfo.Hooknum,
			Priority: netfilterInfo.Priority,
			Flags:    netfilterInfo.Flags,
		}
	case NetkitType:
		var netkitInfo sys.NetkitLinkInfo
		if err := sys.ObjInfo(l.fd, &netkitInfo); err != nil {
			return nil, fmt.Errorf("tcx link info: %s", err)
		}
		extra = &NetkitInfo{
			Ifindex:    netkitInfo.Ifindex,
			AttachType: netkitInfo.AttachType,
		}
	case KprobeMultiType:
		var kprobeMultiInfo sys.KprobeMultiLinkInfo
		if err := sys.ObjInfo(l.fd, &kprobeMultiInfo); err != nil {
			return nil, fmt.Errorf("kprobe multi link info: %s", err)
		}
		extra = &KprobeMultiInfo{
			count:  kprobeMultiInfo.Count,
			flags:  kprobeMultiInfo.Flags,
			missed: kprobeMultiInfo.Missed,
		}
	case PerfEventType:
		var perfEventInfo sys.PerfEventLinkInfo
		if err := sys.ObjInfo(l.fd, &perfEventInfo); err != nil {
			return nil, fmt.Errorf("perf event link info: %s", err)
		}

		var extra2 interface{}
		switch perfEventInfo.PerfEventType {
		case sys.BPF_PERF_EVENT_KPROBE, sys.BPF_PERF_EVENT_KRETPROBE:
			var kprobeInfo sys.KprobeLinkInfo
			if err := sys.ObjInfo(l.fd, &kprobeInfo); err != nil {
				return nil, fmt.Errorf("kprobe multi link info: %s", err)
			}
			extra2 = &KprobeInfo{
				address: kprobeInfo.Addr,
				missed:  kprobeInfo.Missed,
			}
		}

		extra = &PerfEventInfo{
			Type:  perfEventInfo.PerfEventType,
			extra: extra2,
		}
	default:
		return nil, fmt.Errorf("unknown link info type: %d", info.Type)
	}

	return &Info{
		info.Type,
		info.Id,
		ebpf.ProgramID(info.ProgId),
		extra,
	}, nil
}

// Iterator allows iterating over links attached into the kernel.
type Iterator struct {
	// The ID of the current link. Only valid after a call to Next
	ID ID
	// The current link. Only valid until a call to Next.
	// See Take if you want to retain the link.
	Link Link
	err  error
}

// Next retrieves the next link.
//
// Returns true if another link was found. Call [Iterator.Err] after the function returns false.
func (it *Iterator) Next() bool {
	id := it.ID
	for {
		getIdAttr := &sys.LinkGetNextIdAttr{Id: id}
		err := sys.LinkGetNextId(getIdAttr)
		if errors.Is(err, os.ErrNotExist) {
			// There are no more links.
			break
		} else if err != nil {
			it.err = fmt.Errorf("get next link ID: %w", err)
			break
		}

		id = getIdAttr.NextId
		l, err := NewFromID(id)
		if errors.Is(err, os.ErrNotExist) {
			// Couldn't load the link fast enough. Try next ID.
			continue
		} else if err != nil {
			it.err = fmt.Errorf("get link for ID %d: %w", id, err)
			break
		}

		if it.Link != nil {
			it.Link.Close()
		}
		it.ID, it.Link = id, l
		return true
	}

	// No more links or we encountered an error.
	if it.Link != nil {
		it.Link.Close()
	}
	it.Link = nil
	return false
}

// Take the ownership of the current link.
//
// It's the callers responsibility to close the link.
func (it *Iterator) Take() Link {
	l := it.Link
	it.Link = nil
	return l
}

// Err returns an error if iteration failed for some reason.
func (it *Iterator) Err() error {
	return it.err
}

func (it *Iterator) Close() {
	if it.Link != nil {
		it.Link.Close()
	}
}
