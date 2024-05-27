package link

import (
	"fmt"

	"github.com/cilium/ebpf"
	"github.com/cilium/ebpf/internal/sys"
)

const NetfilterIPDefrag NetfilterAttachFlags = 0 // Enable IP packet defragmentation

type NetfilterAttachFlags uint32

type NetfilterOptions struct {
	// Program must be a netfilter BPF program.
	Program *ebpf.Program
	// The protocol family.
	ProtocolFamily uint32
	// The number of the hook you are interested in.
	HookNumber uint32
	// Priority within hook
	Priority int32
	// Extra link flags
	Flags uint32
	// Netfilter flags
	NetfilterFlags NetfilterAttachFlags
}

type netfilterLink struct {
	RawLink
}

// AttachNetfilter links a netfilter BPF program to a netfilter hook.
func AttachNetfilter(opts NetfilterOptions) (Link, error) {
	if opts.Program == nil {
		return nil, fmt.Errorf("netfilter program is nil")
	}

	if t := opts.Program.Type(); t != ebpf.Netfilter {
		return nil, fmt.Errorf("invalid program type %s, expected netfilter", t)
	}

	progFd := opts.Program.FD()
	if progFd < 0 {
		return nil, fmt.Errorf("invalid program: %s", sys.ErrClosedFd)
	}

	attr := sys.LinkCreateNetfilterAttr{
		ProgFd:         uint32(opts.Program.FD()),
		AttachType:     sys.BPF_NETFILTER,
		Flags:          opts.Flags,
		Pf:             uint32(opts.ProtocolFamily),
		Hooknum:        uint32(opts.HookNumber),
		Priority:       opts.Priority,
		NetfilterFlags: uint32(opts.NetfilterFlags),
	}

	fd, err := sys.LinkCreateNetfilter(&attr)
	if err != nil {
		return nil, fmt.Errorf("attach netfilter link: %w", err)
	}

	return &netfilterLink{RawLink{fd, ""}}, nil
}

func (*netfilterLink) Update(new *ebpf.Program) error {
	return fmt.Errorf("netfilter update: %w", ErrNotSupported)
}

var _ Link = (*netfilterLink)(nil)
