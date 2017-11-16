package netlink

import (
	"fmt"
	"net"
	"strings"
)

// Scope is an enum representing a route scope.
type Scope uint8

type NextHopFlag int

type Destination interface {
	Family() int
	Decode([]byte) error
	Encode() ([]byte, error)
	String() string
}

type Encap interface {
	Type() int
	Decode([]byte) error
	Encode() ([]byte, error)
	String() string
}

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
	MPLSDst    *int
	NewDst     Destination
	Encap      Encap
}

func (r Route) String() string {
	elems := []string{}
	if len(r.MultiPath) == 0 {
		elems = append(elems, fmt.Sprintf("Ifindex: %d", r.LinkIndex))
	}
	if r.MPLSDst != nil {
		elems = append(elems, fmt.Sprintf("Dst: %d", r.MPLSDst))
	} else {
		elems = append(elems, fmt.Sprintf("Dst: %s", r.Dst))
	}
	if r.NewDst != nil {
		elems = append(elems, fmt.Sprintf("NewDst: %s", r.NewDst))
	}
	if r.Encap != nil {
		elems = append(elems, fmt.Sprintf("Encap: %s", r.Encap))
	}
	elems = append(elems, fmt.Sprintf("Src: %s", r.Src))
	if len(r.MultiPath) > 0 {
		elems = append(elems, fmt.Sprintf("Gw: %s", r.MultiPath))
	} else {
		elems = append(elems, fmt.Sprintf("Gw: %s", r.Gw))
	}
	elems = append(elems, fmt.Sprintf("Flags: %s", r.ListFlags()))
	elems = append(elems, fmt.Sprintf("Table: %d", r.Table))
	return fmt.Sprintf("{%s}", strings.Join(elems, " "))
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
	Flags     int
	NewDst    Destination
	Encap     Encap
}

func (n *NexthopInfo) String() string {
	elems := []string{}
	elems = append(elems, fmt.Sprintf("Ifindex: %d", n.LinkIndex))
	if n.NewDst != nil {
		elems = append(elems, fmt.Sprintf("NewDst: %s", n.NewDst))
	}
	if n.Encap != nil {
		elems = append(elems, fmt.Sprintf("Encap: %s", n.Encap))
	}
	elems = append(elems, fmt.Sprintf("Weight: %d", n.Hops+1))
	elems = append(elems, fmt.Sprintf("Gw: %d", n.Gw))
	elems = append(elems, fmt.Sprintf("Flags: %s", n.ListFlags()))
	return fmt.Sprintf("{%s}", strings.Join(elems, " "))
}
