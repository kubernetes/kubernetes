package link

import (
	"fmt"
	"runtime"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal/sys"
)

type TCXOptions struct {
	// Index of the interface to attach to.
	Interface int
	// Program to attach.
	Program *ebpf.Program
	// One of the AttachTCX* constants.
	Attach ebpf.AttachType
	// Attach relative to an anchor. Optional.
	Anchor Anchor
	// Only attach if the expected revision matches.
	ExpectedRevision uint64
	// Flags control the attach behaviour. Specify an Anchor instead of
	// F_LINK, F_ID, F_BEFORE, F_AFTER and R_REPLACE. Optional.
	Flags uint32
}

func AttachTCX(opts TCXOptions) (Link, error) {
	if opts.Interface < 0 {
		return nil, fmt.Errorf("interface %d is out of bounds", opts.Interface)
	}

	if opts.Flags&anchorFlags != 0 {
		return nil, fmt.Errorf("disallowed flags: use Anchor to specify attach target")
	}

	attr := sys.LinkCreateTcxAttr{
		ProgFd:           uint32(opts.Program.FD()),
		AttachType:       sys.AttachType(opts.Attach),
		TargetIfindex:    uint32(opts.Interface),
		ExpectedRevision: opts.ExpectedRevision,
		Flags:            opts.Flags,
	}

	if opts.Anchor != nil {
		fdOrID, flags, err := opts.Anchor.anchor()
		if err != nil {
			return nil, fmt.Errorf("attach tcx link: %w", err)
		}

		attr.RelativeFdOrId = fdOrID
		attr.Flags |= flags
	}

	fd, err := sys.LinkCreateTcx(&attr)
	runtime.KeepAlive(opts.Program)
	runtime.KeepAlive(opts.Anchor)
	if err != nil {
		if haveFeatErr := haveTCX(); haveFeatErr != nil {
			return nil, haveFeatErr
		}
		return nil, fmt.Errorf("attach tcx link: %w", err)
	}

	return &tcxLink{RawLink{fd, ""}}, nil
}

type tcxLink struct {
	RawLink
}

var _ Link = (*tcxLink)(nil)
