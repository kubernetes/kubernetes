package netlink

import (
	"fmt"
	"syscall"
)

// Proto is an enum representing an ipsec protocol.
type Proto uint8

const (
	XFRM_PROTO_ROUTE2    Proto = syscall.IPPROTO_ROUTING
	XFRM_PROTO_ESP       Proto = syscall.IPPROTO_ESP
	XFRM_PROTO_AH        Proto = syscall.IPPROTO_AH
	XFRM_PROTO_HAO       Proto = syscall.IPPROTO_DSTOPTS
	XFRM_PROTO_COMP      Proto = syscall.IPPROTO_COMP
	XFRM_PROTO_IPSEC_ANY Proto = syscall.IPPROTO_RAW
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
