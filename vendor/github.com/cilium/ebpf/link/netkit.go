package link

import (
	"fmt"
	"runtime"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal/sys"
)

type NetkitOptions struct {
	// Index of the interface to attach to.
	Interface int
	// Program to attach.
	Program *ebpf.Program
	// One of the AttachNetkit* constants.
	Attach ebpf.AttachType
	// Attach relative to an anchor. Optional.
	Anchor Anchor
	// Only attach if the expected revision matches.
	ExpectedRevision uint64
	// Flags control the attach behaviour. Specify an Anchor instead of
	// F_LINK, F_ID, F_BEFORE, F_AFTER and R_REPLACE. Optional.
	Flags uint32
}

func AttachNetkit(opts NetkitOptions) (Link, error) {
	if opts.Interface < 0 {
		return nil, fmt.Errorf("interface %d is out of bounds", opts.Interface)
	}

	if opts.Flags&anchorFlags != 0 {
		return nil, fmt.Errorf("disallowed flags: use Anchor to specify attach target")
	}

	attr := sys.LinkCreateNetkitAttr{
		ProgFd:           uint32(opts.Program.FD()),
		AttachType:       sys.AttachType(opts.Attach),
		TargetIfindex:    uint32(opts.Interface),
		ExpectedRevision: opts.ExpectedRevision,
		Flags:            opts.Flags,
	}

	if opts.Anchor != nil {
		fdOrID, flags, err := opts.Anchor.anchor()
		if err != nil {
			return nil, fmt.Errorf("attach netkit link: %w", err)
		}

		attr.RelativeFdOrId = fdOrID
		attr.Flags |= flags
	}

	fd, err := sys.LinkCreateNetkit(&attr)
	runtime.KeepAlive(opts.Program)
	runtime.KeepAlive(opts.Anchor)
	if err != nil {
		if haveFeatErr := haveNetkit(); haveFeatErr != nil {
			return nil, haveFeatErr
		}
		return nil, fmt.Errorf("attach netkit link: %w", err)
	}

	return &netkitLink{RawLink{fd, ""}}, nil
}

type netkitLink struct {
	RawLink
}

var _ Link = (*netkitLink)(nil)

func (netkit *netkitLink) Info() (*Info, error) {
	var info sys.NetkitLinkInfo
	if err := sys.ObjInfo(netkit.fd, &info); err != nil {
		return nil, fmt.Errorf("netkit link info: %s", err)
	}
	extra := &NetkitInfo{
		Ifindex:    info.Ifindex,
		AttachType: info.AttachType,
	}

	return &Info{
		info.Type,
		info.Id,
		ebpf.ProgramID(info.ProgId),
		extra,
	}, nil
}
