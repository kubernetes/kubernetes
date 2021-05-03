package link

import (
	"fmt"

	"github.com/cilium/ebpf"
)

// NetNsInfo contains metadata about a network namespace link.
type NetNsInfo struct {
	RawLinkInfo
}

// NetNsLink is a program attached to a network namespace.
type NetNsLink struct {
	*RawLink
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

	return &NetNsLink{link}, nil
}

// LoadPinnedNetNs loads a network namespace link from bpffs.
func LoadPinnedNetNs(fileName string) (*NetNsLink, error) {
	link, err := loadPinnedRawLink(fileName, NetNsType)
	if err != nil {
		return nil, err
	}

	return &NetNsLink{link}, nil
}

// Info returns information about the link.
func (nns *NetNsLink) Info() (*NetNsInfo, error) {
	info, err := nns.RawLink.Info()
	if err != nil {
		return nil, err
	}
	return &NetNsInfo{*info}, nil
}
