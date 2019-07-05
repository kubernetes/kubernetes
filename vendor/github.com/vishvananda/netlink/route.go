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
	Equal(Destination) bool
}

type Encap interface {
	Type() int
	Decode([]byte) error
	Encode() ([]byte, error)
	String() string
	Equal(Encap) bool
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

func (r Route) Equal(x Route) bool {
	return r.LinkIndex == x.LinkIndex &&
		r.ILinkIndex == x.ILinkIndex &&
		r.Scope == x.Scope &&
		ipNetEqual(r.Dst, x.Dst) &&
		r.Src.Equal(x.Src) &&
		r.Gw.Equal(x.Gw) &&
		nexthopInfoSlice(r.MultiPath).Equal(x.MultiPath) &&
		r.Protocol == x.Protocol &&
		r.Priority == x.Priority &&
		r.Table == x.Table &&
		r.Type == x.Type &&
		r.Tos == x.Tos &&
		r.Flags == x.Flags &&
		(r.MPLSDst == x.MPLSDst || (r.MPLSDst != nil && x.MPLSDst != nil && *r.MPLSDst == *x.MPLSDst)) &&
		(r.NewDst == x.NewDst || (r.NewDst != nil && r.NewDst.Equal(x.NewDst))) &&
		(r.Encap == x.Encap || (r.Encap != nil && r.Encap.Equal(x.Encap)))
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
	elems = append(elems, fmt.Sprintf("Gw: %s", n.Gw))
	elems = append(elems, fmt.Sprintf("Flags: %s", n.ListFlags()))
	return fmt.Sprintf("{%s}", strings.Join(elems, " "))
}

func (n NexthopInfo) Equal(x NexthopInfo) bool {
	return n.LinkIndex == x.LinkIndex &&
		n.Hops == x.Hops &&
		n.Gw.Equal(x.Gw) &&
		n.Flags == x.Flags &&
		(n.NewDst == x.NewDst || (n.NewDst != nil && n.NewDst.Equal(x.NewDst))) &&
		(n.Encap == x.Encap || (n.Encap != nil && n.Encap.Equal(x.Encap)))
}

type nexthopInfoSlice []*NexthopInfo

func (n nexthopInfoSlice) Equal(x []*NexthopInfo) bool {
	if len(n) != len(x) {
		return false
	}
	for i := range n {
		if n[i] == nil || x[i] == nil {
			return false
		}
		if !n[i].Equal(*x[i]) {
			return false
		}
	}
	return true
}

// ipNetEqual returns true iff both IPNet are equal
func ipNetEqual(ipn1 *net.IPNet, ipn2 *net.IPNet) bool {
	if ipn1 == ipn2 {
		return true
	}
	if ipn1 == nil || ipn2 == nil {
		return false
	}
	m1, _ := ipn1.Mask.Size()
	m2, _ := ipn2.Mask.Size()
	return m1 == m2 && ipn1.IP.Equal(ipn2.IP)
}
