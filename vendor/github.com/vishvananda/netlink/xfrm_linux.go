package netlink

import (
	"fmt"

	"golang.org/x/sys/unix"
)

// Proto is an enum representing an ipsec protocol.
type Proto uint8

const (
	XFRM_PROTO_ROUTE2    Proto = unix.IPPROTO_ROUTING
	XFRM_PROTO_ESP       Proto = unix.IPPROTO_ESP
	XFRM_PROTO_AH        Proto = unix.IPPROTO_AH
	XFRM_PROTO_HAO       Proto = unix.IPPROTO_DSTOPTS
	XFRM_PROTO_COMP      Proto = unix.IPPROTO_COMP
	XFRM_PROTO_IPSEC_ANY Proto = unix.IPPROTO_RAW
)

func (p Proto) String() string {
	switch p {
	case XFRM_PROTO_ROUTE2:
		return "route2"
	case XFRM_PROTO_ESP:
		return "esp"
	case XFRM_PROTO_AH:
		return "ah"
	case XFRM_PROTO_HAO:
		return "hao"
	case XFRM_PROTO_COMP:
		return "comp"
	case XFRM_PROTO_IPSEC_ANY:
		return "ipsec-any"
	}
	return fmt.Sprintf("%d", p)
}

// Mode is an enum representing an ipsec transport.
type Mode uint8

const (
	XFRM_MODE_TRANSPORT Mode = iota
	XFRM_MODE_TUNNEL
	XFRM_MODE_ROUTEOPTIMIZATION
	XFRM_MODE_IN_TRIGGER
	XFRM_MODE_BEET
	XFRM_MODE_MAX
)

// SADir is an enum representing an ipsec template direction.
type SADir uint8

const (
	XFRM_SA_DIR_IN SADir = iota + 1
	XFRM_SA_DIR_OUT
)

func (m Mode) String() string {
	switch m {
	case XFRM_MODE_TRANSPORT:
		return "transport"
	case XFRM_MODE_TUNNEL:
		return "tunnel"
	case XFRM_MODE_ROUTEOPTIMIZATION:
		return "ro"
	case XFRM_MODE_IN_TRIGGER:
		return "in_trigger"
	case XFRM_MODE_BEET:
		return "beet"
	}
	return fmt.Sprintf("%d", m)
}

// XfrmMark represents the mark associated to the state or policy
type XfrmMark struct {
	Value uint32
	Mask  uint32
}

func (m *XfrmMark) String() string {
	return fmt.Sprintf("(0x%x,0x%x)", m.Value, m.Mask)
}
