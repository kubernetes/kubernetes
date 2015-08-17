package netlink

import (
	"fmt"
	"net"
	"syscall"

	"github.com/vishvananda/netlink/nl"
)

// RtAttr is shared so it is in netlink_linux.go

// RouteAdd will add a route to the system.
// Equivalent to: `ip route add $route`
func RouteAdd(route *Route) error {
	req := nl.NewNetlinkRequest(syscall.RTM_NEWROUTE, syscall.NLM_F_CREATE|syscall.NLM_F_EXCL|syscall.NLM_F_ACK)
	return routeHandle(route, req, nl.NewRtMsg())
}

// RouteAdd will delete a route from the system.
// Equivalent to: `ip route del $route`
func RouteDel(route *Route) error {
	req := nl.NewNetlinkRequest(syscall.RTM_DELROUTE, syscall.NLM_F_ACK)
	return routeHandle(route, req, nl.NewRtDelMsg())
}

func routeHandle(route *Route, req *nl.NetlinkRequest, msg *nl.RtMsg) error {
	if (route.Dst == nil || route.Dst.IP == nil) && route.Src == nil && route.Gw == nil {
		return fmt.Errorf("one of Dst.IP, Src, or Gw must not be nil")
	}

	msg.Scope = uint8(route.Scope)
	family := -1
	var rtAttrs []*nl.RtAttr

	if route.Dst != nil && route.Dst.IP != nil {
		dstLen, _ := route.Dst.Mask.Size()
		msg.Dst_len = uint8(dstLen)
		dstFamily := nl.GetIPFamily(route.Dst.IP)
		family = dstFamily
		var dstData []byte
		if dstFamily == FAMILY_V4 {
			dstData = route.Dst.IP.To4()
		} else {
			dstData = route.Dst.IP.To16()
		}
		rtAttrs = append(rtAttrs, nl.NewRtAttr(syscall.RTA_DST, dstData))
	}

	if route.Src != nil {
		srcFamily := nl.GetIPFamily(route.Src)
		if family != -1 && family != srcFamily {
			return fmt.Errorf("source and destination ip are not the same IP family")
		}
		family = srcFamily
		var srcData []byte
		if srcFamily == FAMILY_V4 {
			srcData = route.Src.To4()
		} else {
			srcData = route.Src.To16()
		}
		// The commonly used src ip for routes is actually PREFSRC
		rtAttrs = append(rtAttrs, nl.NewRtAttr(syscall.RTA_PREFSRC, srcData))
	}

	if route.Gw != nil {
		gwFamily := nl.GetIPFamily(route.Gw)
		if family != -1 && family != gwFamily {
			return fmt.Errorf("gateway, source, and destination ip are not the same IP family")
		}
		family = gwFamily
		var gwData []byte
		if gwFamily == FAMILY_V4 {
			gwData = route.Gw.To4()
		} else {
			gwData = route.Gw.To16()
		}
		rtAttrs = append(rtAttrs, nl.NewRtAttr(syscall.RTA_GATEWAY, gwData))
	}

	msg.Family = uint8(family)

	req.AddData(msg)
	for _, attr := range rtAttrs {
		req.AddData(attr)
	}

	var (
		b      = make([]byte, 4)
		native = nl.NativeEndian()
	)
	native.PutUint32(b, uint32(route.LinkIndex))

	req.AddData(nl.NewRtAttr(syscall.RTA_OIF, b))

	_, err := req.Execute(syscall.NETLINK_ROUTE, 0)
	return err
}

// RouteList gets a list of routes in the system.
// Equivalent to: `ip route show`.
// The list can be filtered by link and ip family.
func RouteList(link Link, family int) ([]Route, error) {
	req := nl.NewNetlinkRequest(syscall.RTM_GETROUTE, syscall.NLM_F_DUMP)
	msg := nl.NewIfInfomsg(family)
	req.AddData(msg)

	msgs, err := req.Execute(syscall.NETLINK_ROUTE, syscall.RTM_NEWROUTE)
	if err != nil {
		return nil, err
	}

	index := 0
	if link != nil {
		base := link.Attrs()
		ensureIndex(base)
		index = base.Index
	}

	native := nl.NativeEndian()
	var res []Route
	for _, m := range msgs {
		msg := nl.DeserializeRtMsg(m)

		if msg.Flags&syscall.RTM_F_CLONED != 0 {
			// Ignore cloned routes
			continue
		}

		if msg.Table != syscall.RT_TABLE_MAIN {
			// Ignore non-main tables
			continue
		}

		attrs, err := nl.ParseRouteAttr(m[msg.Len():])
		if err != nil {
			return nil, err
		}

		route := Route{Scope: Scope(msg.Scope)}
		for _, attr := range attrs {
			switch attr.Attr.Type {
			case syscall.RTA_GATEWAY:
				route.Gw = net.IP(attr.Value)
			case syscall.RTA_PREFSRC:
				route.Src = net.IP(attr.Value)
			case syscall.RTA_DST:
				route.Dst = &net.IPNet{
					IP:   attr.Value,
					Mask: net.CIDRMask(int(msg.Dst_len), 8*len(attr.Value)),
				}
			case syscall.RTA_OIF:
				routeIndex := int(native.Uint32(attr.Value[0:4]))
				if link != nil && routeIndex != index {
					// Ignore routes from other interfaces
					continue
				}
				route.LinkIndex = routeIndex
			}
		}
		res = append(res, route)
	}

	return res, nil
}

// RouteGet gets a route to a specific destination from the host system.
// Equivalent to: 'ip route get'.
func RouteGet(destination net.IP) ([]Route, error) {
	req := nl.NewNetlinkRequest(syscall.RTM_GETROUTE, syscall.NLM_F_REQUEST)
	family := nl.GetIPFamily(destination)
	var destinationData []byte
	var bitlen uint8
	if family == FAMILY_V4 {
		destinationData = destination.To4()
		bitlen = 32
	} else {
		destinationData = destination.To16()
		bitlen = 128
	}
	msg := &nl.RtMsg{}
	msg.Family = uint8(family)
	msg.Dst_len = bitlen
	req.AddData(msg)

	rtaDst := nl.NewRtAttr(syscall.RTA_DST, destinationData)
	req.AddData(rtaDst)

	msgs, err := req.Execute(syscall.NETLINK_ROUTE, syscall.RTM_NEWROUTE)
	if err != nil {
		return nil, err
	}

	native := nl.NativeEndian()
	var res []Route
	for _, m := range msgs {
		msg := nl.DeserializeRtMsg(m)
		attrs, err := nl.ParseRouteAttr(m[msg.Len():])
		if err != nil {
			return nil, err
		}

		route := Route{}
		for _, attr := range attrs {
			switch attr.Attr.Type {
			case syscall.RTA_GATEWAY:
				route.Gw = net.IP(attr.Value)
			case syscall.RTA_PREFSRC:
				route.Src = net.IP(attr.Value)
			case syscall.RTA_DST:
				route.Dst = &net.IPNet{
					IP:   attr.Value,
					Mask: net.CIDRMask(int(msg.Dst_len), 8*len(attr.Value)),
				}
			case syscall.RTA_OIF:
				routeIndex := int(native.Uint32(attr.Value[0:4]))
				route.LinkIndex = routeIndex
			}
		}
		res = append(res, route)
	}
	return res, nil

}
