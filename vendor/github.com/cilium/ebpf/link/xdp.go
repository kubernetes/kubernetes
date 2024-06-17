package link

import (
	"fmt"

	"github.com/cilium/ebpf"
)

// XDPAttachFlags represents how XDP program will be attached to interface.
type XDPAttachFlags uint32

const (
	// XDPGenericMode (SKB) links XDP BPF program for drivers which do
	// not yet support native XDP.
	XDPGenericMode XDPAttachFlags = 1 << (iota + 1)
	// XDPDriverMode links XDP BPF program into the driver’s receive path.
	XDPDriverMode
	// XDPOffloadMode offloads the entire XDP BPF program into hardware.
	XDPOffloadMode
)

type XDPOptions struct {
	// Program must be an XDP BPF program.
	Program *ebpf.Program

	// Interface is the interface index to attach program to.
	Interface int

	// Flags is one of XDPAttachFlags (optional).
	//
	// Only one XDP mode should be set, without flag defaults
	// to driver/generic mode (best effort).
	Flags XDPAttachFlags
}

// AttachXDP links an XDP BPF program to an XDP hook.
func AttachXDP(opts XDPOptions) (Link, error) {
	if t := opts.Program.Type(); t != ebpf.XDP {
		return nil, fmt.Errorf("invalid program type %s, expected XDP", t)
	}

	if opts.Interface < 1 {
		return nil, fmt.Errorf("invalid interface index: %d", opts.Interface)
	}

	rawLink, err := AttachRawLink(RawLinkOptions{
		Program: opts.Program,
		Attach:  ebpf.AttachXDP,
		Target:  opts.Interface,
		Flags:   uint32(opts.Flags),
	})

	return rawLink, err
}
