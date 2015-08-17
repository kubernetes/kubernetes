package netlink

import (
	"fmt"
	"net"
	"syscall"
)

// Scope is an enum representing a route scope.
type Scope uint8

const (
	SCOPE_UNIVERSE Scope = syscall.RT_SCOPE_UNIVERSE
	SCOPE_SITE     Scope = syscall.RT_SCOPE_SITE
	SCOPE_LINK     Scope = syscall.RT_SCOPE_LINK
	SCOPE_HOST     Scope = syscall.RT_SCOPE_HOST
	SCOPE_NOWHERE  Scope = syscall.RT_SCOPE_NOWHERE
)

// Route represents a netlink route. A route is associated with a link,
// has a destination network, an optional source ip, and optional
// gateway. Advanced route parameters and non-main routing tables are
// currently not supported.
type Route struct {
	LinkIndex int
	Scope     Scope
	Dst       *net.IPNet
	Src       net.IP
	Gw        net.IP
}

func (r Route) String() string {
	return fmt.Sprintf("{Ifindex: %d Dst: %s Src: %s Gw: %s}", r.LinkIndex, r.Dst,
		r.Src, r.Gw)
}
