package netlink

import (
	"fmt"
	"net"
)

// Scope is an enum representing a route scope.
type Scope uint8

type NextHopFlag int

// Route represents a netlink route.
type Route struct {
	LinkIndex  int
	ILinkIndex int
	Scope      Scope
	Dst        *net.IPNet
	Src        net.IP
	Gw         net.IP
	MultiPath  []*NexthopInfo
	Protocol   int
	Priority   int
	Table      int
	Type       int
	Tos        int
	Flags      int
}

func (r Route) String() string {
	if len(r.MultiPath) > 0 {
		return fmt.Sprintf("{Dst: %s Src: %s Gw: %s Flags: %s Table: %d}", r.Dst,
			r.Src, r.MultiPath, r.ListFlags(), r.Table)
	}
	return fmt.Sprintf("{Ifindex: %d Dst: %s Src: %s Gw: %s Flags: %s Table: %d}", r.LinkIndex, r.Dst,
		r.Src, r.Gw, r.ListFlags(), r.Table)
}

func (r *Route) SetFlag(flag NextHopFlag) {
	r.Flags |= int(flag)
}

func (r *Route) ClearFlag(flag NextHopFlag) {
	r.Flags &^= int(flag)
}

type flagString struct {
	f NextHopFlag
	s string
}

// RouteUpdate is sent when a route changes - type is RTM_NEWROUTE or RTM_DELROUTE
type RouteUpdate struct {
	Type uint16
	Route
}

type NexthopInfo struct {
	LinkIndex int
	Hops      int
	Gw        net.IP
}

func (n *NexthopInfo) String() string {
	return fmt.Sprintf("{Ifindex: %d Weight: %d, Gw: %s}", n.LinkIndex, n.Hops+1, n.Gw)
}
