package link

import (
	"fmt"

	"github.com/cilium/ebpf"
)

// NetNsLink is a program attached to a network namespace.
type NetNsLink struct {
	RawLink
}

// AttachNetNs attaches a program to a network namespace.
func AttachNetNs(ns int, prog *ebpf.Program) (*NetNsLink, error) {
	var attach ebpf.AttachType
	switch t := prog.Type(); t {
	case ebpf.FlowDissector:
		attach = ebpf.AttachFlowDissector
	case ebpf.SkLookup:
		attach = ebpf.AttachSkLookup
	default:
		return nil, fmt.Errorf("can't attach %v to network namespace", t)
	}

	link, err := AttachRawLink(RawLinkOptions{
		Target:  ns,
		Program: prog,
		Attach:  attach,
	})
	if err != nil {
		return nil, err
	}

	return &NetNsLink{*link}, nil
}
