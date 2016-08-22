package netlink

import (
	"fmt"
)

type Filter interface {
	Attrs() *FilterAttrs
	Type() string
}

// Filter represents a netlink filter. A filter is associated with a link,
// has a handle and a parent. The root filter of a device should have a
// parent == HANDLE_ROOT.
type FilterAttrs struct {
	LinkIndex int
	Handle    uint32
	Parent    uint32
	Priority  uint16 // lower is higher priority
	Protocol  uint16 // syscall.ETH_P_*
}

func (q FilterAttrs) String() string {
	return fmt.Sprintf("{LinkIndex: %d, Handle: %s, Parent: %s, Priority: %d, Protocol: %d}", q.LinkIndex, HandleStr(q.Handle), HandleStr(q.Parent), q.Priority, q.Protocol)
}

// U32 filters on many packet related properties
type U32 struct {
	FilterAttrs
	// Currently only supports redirecting to another interface
	RedirIndex int
}

func (filter *U32) Attrs() *FilterAttrs {
	return &filter.FilterAttrs
}

func (filter *U32) Type() string {
	return "u32"
}

// GenericFilter filters represent types that are not currently understood
// by this netlink library.
type GenericFilter struct {
	FilterAttrs
	FilterType string
}

func (filter *GenericFilter) Attrs() *FilterAttrs {
	return &filter.FilterAttrs
}

func (filter *GenericFilter) Type() string {
	return filter.FilterType
}
