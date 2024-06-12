package netlink

import (
	"fmt"
	"net"
)

// Dir is an enum representing an ipsec template direction.
type Dir uint8

const (
	XFRM_DIR_IN Dir = iota
	XFRM_DIR_OUT
	XFRM_DIR_FWD
	XFRM_SOCKET_IN
	XFRM_SOCKET_OUT
	XFRM_SOCKET_FWD
)

func (d Dir) String() string {
	switch d {
	case XFRM_DIR_IN:
		return "dir in"
	case XFRM_DIR_OUT:
		return "dir out"
	case XFRM_DIR_FWD:
		return "dir fwd"
	case XFRM_SOCKET_IN:
		return "socket in"
	case XFRM_SOCKET_OUT:
		return "socket out"
	case XFRM_SOCKET_FWD:
		return "socket fwd"
	}
	return fmt.Sprintf("socket %d", d-XFRM_SOCKET_IN)
}

// PolicyAction is an enum representing an ipsec policy action.
type PolicyAction uint8

const (
	XFRM_POLICY_ALLOW PolicyAction = 0
	XFRM_POLICY_BLOCK PolicyAction = 1
)

func (a PolicyAction) String() string {
	switch a {
	case XFRM_POLICY_ALLOW:
		return "allow"
	case XFRM_POLICY_BLOCK:
		return "block"
	default:
		return fmt.Sprintf("action %d", a)
	}
}

// XfrmPolicyTmpl encapsulates a rule for the base addresses of an ipsec
// policy. These rules are matched with XfrmState to determine encryption
// and authentication algorithms.
type XfrmPolicyTmpl struct {
	Dst      net.IP
	Src      net.IP
	Proto    Proto
	Mode     Mode
	Spi      int
	Reqid    int
	Optional int
}

func (t XfrmPolicyTmpl) String() string {
	return fmt.Sprintf("{Dst: %v, Src: %v, Proto: %s, Mode: %s, Spi: 0x%x, Reqid: 0x%x}",
		t.Dst, t.Src, t.Proto, t.Mode, t.Spi, t.Reqid)
}

// XfrmPolicy represents an ipsec policy. It represents the overlay network
// and has a list of XfrmPolicyTmpls representing the base addresses of
// the policy.
type XfrmPolicy struct {
	Dst      *net.IPNet
	Src      *net.IPNet
	Proto    Proto
	DstPort  int
	SrcPort  int
	Dir      Dir
	Priority int
	Index    int
	Action   PolicyAction
	Ifindex  int
	Ifid     int
	Mark     *XfrmMark
	Tmpls    []XfrmPolicyTmpl
}

func (p XfrmPolicy) String() string {
	return fmt.Sprintf("{Dst: %v, Src: %v, Proto: %s, DstPort: %d, SrcPort: %d, Dir: %s, Priority: %d, Index: %d, Action: %s, Ifindex: %d, Ifid: %d, Mark: %s, Tmpls: %s}",
		p.Dst, p.Src, p.Proto, p.DstPort, p.SrcPort, p.Dir, p.Priority, p.Index, p.Action, p.Ifindex, p.Ifid, p.Mark, p.Tmpls)
}
