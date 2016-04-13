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

// XfrmPolicyTmpl encapsulates a rule for the base addresses of an ipsec
// policy. These rules are matched with XfrmState to determine encryption
// and authentication algorithms.
type XfrmPolicyTmpl struct {
	Dst   net.IP
	Src   net.IP
	Proto Proto
	Mode  Mode
	Reqid int
}

// XfrmPolicy represents an ipsec policy. It represents the overlay network
// and has a list of XfrmPolicyTmpls representing the base addresses of
// the policy.
type XfrmPolicy struct {
	Dst      *net.IPNet
	Src      *net.IPNet
	Dir      Dir
	Priority int
	Index    int
	Tmpls    []XfrmPolicyTmpl
}
