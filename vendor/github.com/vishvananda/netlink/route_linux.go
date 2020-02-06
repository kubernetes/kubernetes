package netlink

import (
	"fmt"
	"net"
	"strings"
	"syscall"

	"github.com/vishvananda/netlink/nl"
	"github.com/vishvananda/netns"
	"golang.org/x/sys/unix"
)

// RtAttr is shared so it is in netlink_linux.go

const (
	SCOPE_UNIVERSE Scope = unix.RT_SCOPE_UNIVERSE
	SCOPE_SITE     Scope = unix.RT_SCOPE_SITE
	SCOPE_LINK     Scope = unix.RT_SCOPE_LINK
	SCOPE_HOST     Scope = unix.RT_SCOPE_HOST
	SCOPE_NOWHERE  Scope = unix.RT_SCOPE_NOWHERE
)

const (
	RT_FILTER_PROTOCOL uint64 = 1 << (1 + iota)
	RT_FILTER_SCOPE
	RT_FILTER_TYPE
	RT_FILTER_TOS
	RT_FILTER_IIF
	RT_FILTER_OIF
	RT_FILTER_DST
	RT_FILTER_SRC
	RT_FILTER_GW
	RT_FILTER_TABLE
)

const (
	FLAG_ONLINK    NextHopFlag = unix.RTNH_F_ONLINK
	FLAG_PERVASIVE NextHopFlag = unix.RTNH_F_PERVASIVE
)

var testFlags = []flagString{
	{f: FLAG_ONLINK, s: "onlink"},
	{f: FLAG_PERVASIVE, s: "pervasive"},
}

func listFlags(flag int) []string {
	var flags []string
	for _, tf := range testFlags {
		if flag&int(tf.f) != 0 {
			flags = append(flags, tf.s)
		}
	}
	return flags
}

func (r *Route) ListFlags() []string {
	return listFlags(r.Flags)
}

func (n *NexthopInfo) ListFlags() []string {
	return listFlags(n.Flags)
}

type MPLSDestination struct {
	Labels []int
}

func (d *MPLSDestination) Family() int {
	return nl.FAMILY_MPLS
}

func (d *MPLSDestination) Decode(buf []byte) error {
	d.Labels = nl.DecodeMPLSStack(buf)
	return nil
}

func (d *MPLSDestination) Encode() ([]byte, error) {
	return nl.EncodeMPLSStack(d.Labels...), nil
}

func (d *MPLSDestination) String() string {
	s := make([]string, 0, len(d.Labels))
	for _, l := range d.Labels {
		s = append(s, fmt.Sprintf("%d", l))
	}
	return strings.Join(s, "/")
}

func (d *MPLSDestination) Equal(x Destination) bool {
	o, ok := x.(*MPLSDestination)
	if !ok {
		return false
	}
	if d == nil && o == nil {
		return true
	}
	if d == nil || o == nil {
		return false
	}
	if d.Labels == nil && o.Labels == nil {
		return true
	}
	if d.Labels == nil || o.Labels == nil {
		return false
	}
	if len(d.Labels) != len(o.Labels) {
		return false
	}
	for i := range d.Labels {
		if d.Labels[i] != o.Labels[i] {
			return false
		}
	}
	return true
}

type MPLSEncap struct {
	Labels []int
}

func (e *MPLSEncap) Type() int {
	return nl.LWTUNNEL_ENCAP_MPLS
}

func (e *MPLSEncap) Decode(buf []byte) error {
	if len(buf) < 4 {
		return fmt.Errorf("lack of bytes")
	}
	native := nl.NativeEndian()
	l := native.Uint16(buf)
	if len(buf) < int(l) {
		return fmt.Errorf("lack of bytes")
	}
	buf = buf[:l]
	typ := native.Uint16(buf[2:])
	if typ != nl.MPLS_IPTUNNEL_DST {
		return fmt.Errorf("unknown MPLS Encap Type: %d", typ)
	}
	e.Labels = nl.DecodeMPLSStack(buf[4:])
	return nil
}

func (e *MPLSEncap) Encode() ([]byte, error) {
	s := nl.EncodeMPLSStack(e.Labels...)
	native := nl.NativeEndian()
	hdr := make([]byte, 4)
	native.PutUint16(hdr, uint16(len(s)+4))
	native.PutUint16(hdr[2:], nl.MPLS_IPTUNNEL_DST)
	return append(hdr, s...), nil
}

func (e *MPLSEncap) String() string {
	s := make([]string, 0, len(e.Labels))
	for _, l := range e.Labels {
		s = append(s, fmt.Sprintf("%d", l))
	}
	return strings.Join(s, "/")
}

func (e *MPLSEncap) Equal(x Encap) bool {
	o, ok := x.(*MPLSEncap)
	if !ok {
		return false
	}
	if e == nil && o == nil {
		return true
	}
	if e == nil || o == nil {
		return false
	}
	if e.Labels == nil && o.Labels == nil {
		return true
	}
	if e.Labels == nil || o.Labels == nil {
		return false
	}
	if len(e.Labels) != len(o.Labels) {
		return false
	}
	for i := range e.Labels {
		if e.Labels[i] != o.Labels[i] {
			return false
		}
	}
	return true
}

// SEG6 definitions
type SEG6Encap struct {
	Mode     int
	Segments []net.IP
}

func (e *SEG6Encap) Type() int {
	return nl.LWTUNNEL_ENCAP_SEG6
}
func (e *SEG6Encap) Decode(buf []byte) error {
	if len(buf) < 4 {
		return fmt.Errorf("lack of bytes")
	}
	native := nl.NativeEndian()
	// Get Length(l) & Type(typ) : 2 + 2 bytes
	l := native.Uint16(buf)
	if len(buf) < int(l) {
		return fmt.Errorf("lack of bytes")
	}
	buf = buf[:l] // make sure buf size upper limit is Length
	typ := native.Uint16(buf[2:])
	if typ != nl.SEG6_IPTUNNEL_SRH {
		return fmt.Errorf("unknown SEG6 Type: %d", typ)
	}

	var err error
	e.Mode, e.Segments, err = nl.DecodeSEG6Encap(buf[4:])

	return err
}
func (e *SEG6Encap) Encode() ([]byte, error) {
	s, err := nl.EncodeSEG6Encap(e.Mode, e.Segments)
	native := nl.NativeEndian()
	hdr := make([]byte, 4)
	native.PutUint16(hdr, uint16(len(s)+4))
	native.PutUint16(hdr[2:], nl.SEG6_IPTUNNEL_SRH)
	return append(hdr, s...), err
}
func (e *SEG6Encap) String() string {
	segs := make([]string, 0, len(e.Segments))
	// append segment backwards (from n to 0) since seg#0 is the last segment.
	for i := len(e.Segments); i > 0; i-- {
		segs = append(segs, fmt.Sprintf("%s", e.Segments[i-1]))
	}
	str := fmt.Sprintf("mode %s segs %d [ %s ]", nl.SEG6EncapModeString(e.Mode),
		len(e.Segments), strings.Join(segs, " "))
	return str
}
func (e *SEG6Encap) Equal(x Encap) bool {
	o, ok := x.(*SEG6Encap)
	if !ok {
		return false
	}
	if e == o {
		return true
	}
	if e == nil || o == nil {
		return false
	}
	if e.Mode != o.Mode {
		return false
	}
	if len(e.Segments) != len(o.Segments) {
		return false
	}
	for i := range e.Segments {
		if !e.Segments[i].Equal(o.Segments[i]) {
			return false
		}
	}
	return true
}

// RouteAdd will add a route to the system.
// Equivalent to: `ip route add $route`
func RouteAdd(route *Route) error {
	return pkgHandle.RouteAdd(route)
}

// RouteAdd will add a route to the system.
// Equivalent to: `ip route add $route`
func (h *Handle) RouteAdd(route *Route) error {
	flags := unix.NLM_F_CREATE | unix.NLM_F_EXCL | unix.NLM_F_ACK
	req := h.newNetlinkRequest(unix.RTM_NEWROUTE, flags)
	return h.routeHandle(route, req, nl.NewRtMsg())
}

// RouteReplace will add a route to the system.
// Equivalent to: `ip route replace $route`
func RouteReplace(route *Route) error {
	return pkgHandle.RouteReplace(route)
}

// RouteReplace will add a route to the system.
// Equivalent to: `ip route replace $route`
func (h *Handle) RouteReplace(route *Route) error {
	flags := unix.NLM_F_CREATE | unix.NLM_F_REPLACE | unix.NLM_F_ACK
	req := h.newNetlinkRequest(unix.RTM_NEWROUTE, flags)
	return h.routeHandle(route, req, nl.NewRtMsg())
}

// RouteDel will delete a route from the system.
// Equivalent to: `ip route del $route`
func RouteDel(route *Route) error {
	return pkgHandle.RouteDel(route)
}

// RouteDel will delete a route from the system.
// Equivalent to: `ip route del $route`
func (h *Handle) RouteDel(route *Route) error {
	req := h.newNetlinkRequest(unix.RTM_DELROUTE, unix.NLM_F_ACK)
	return h.routeHandle(route, req, nl.NewRtDelMsg())
}

func (h *Handle) routeHandle(route *Route, req *nl.NetlinkRequest, msg *nl.RtMsg) error {
	if (route.Dst == nil || route.Dst.IP == nil) && route.Src == nil && route.Gw == nil && route.MPLSDst == nil {
		return fmt.Errorf("one of Dst.IP, Src, or Gw must not be nil")
	}

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
		rtAttrs = append(rtAttrs, nl.NewRtAttr(unix.RTA_DST, dstData))
	} else if route.MPLSDst != nil {
		family = nl.FAMILY_MPLS
		msg.Dst_len = uint8(20)
		msg.Type = unix.RTN_UNICAST
		rtAttrs = append(rtAttrs, nl.NewRtAttr(unix.RTA_DST, nl.EncodeMPLSStack(*route.MPLSDst)))
	}

	if route.NewDst != nil {
		if family != -1 && family != route.NewDst.Family() {
			return fmt.Errorf("new destination and destination are not the same address family")
		}
		buf, err := route.NewDst.Encode()
		if err != nil {
			return err
		}
		rtAttrs = append(rtAttrs, nl.NewRtAttr(nl.RTA_NEWDST, buf))
	}

	if route.Encap != nil {
		buf := make([]byte, 2)
		native.PutUint16(buf, uint16(route.Encap.Type()))
		rtAttrs = append(rtAttrs, nl.NewRtAttr(nl.RTA_ENCAP_TYPE, buf))
		buf, err := route.Encap.Encode()
		if err != nil {
			return err
		}
		rtAttrs = append(rtAttrs, nl.NewRtAttr(nl.RTA_ENCAP, buf))
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
		rtAttrs = append(rtAttrs, nl.NewRtAttr(unix.RTA_PREFSRC, srcData))
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
		rtAttrs = append(rtAttrs, nl.NewRtAttr(unix.RTA_GATEWAY, gwData))
	}

	if len(route.MultiPath) > 0 {
		buf := []byte{}
		for _, nh := range route.MultiPath {
			rtnh := &nl.RtNexthop{
				RtNexthop: unix.RtNexthop{
					Hops:    uint8(nh.Hops),
					Ifindex: int32(nh.LinkIndex),
					Flags:   uint8(nh.Flags),
				},
			}
			children := []nl.NetlinkRequestData{}
			if nh.Gw != nil {
				gwFamily := nl.GetIPFamily(nh.Gw)
				if family != -1 && family != gwFamily {
					return fmt.Errorf("gateway, source, and destination ip are not the same IP family")
				}
				if gwFamily == FAMILY_V4 {
					children = append(children, nl.NewRtAttr(unix.RTA_GATEWAY, []byte(nh.Gw.To4())))
				} else {
					children = append(children, nl.NewRtAttr(unix.RTA_GATEWAY, []byte(nh.Gw.To16())))
				}
			}
			if nh.NewDst != nil {
				if family != -1 && family != nh.NewDst.Family() {
					return fmt.Errorf("new destination and destination are not the same address family")
				}
				buf, err := nh.NewDst.Encode()
				if err != nil {
					return err
				}
				children = append(children, nl.NewRtAttr(nl.RTA_NEWDST, buf))
			}
			if nh.Encap != nil {
				buf := make([]byte, 2)
				native.PutUint16(buf, uint16(nh.Encap.Type()))
				rtAttrs = append(rtAttrs, nl.NewRtAttr(nl.RTA_ENCAP_TYPE, buf))
				buf, err := nh.Encap.Encode()
				if err != nil {
					return err
				}
				children = append(children, nl.NewRtAttr(nl.RTA_ENCAP, buf))
			}
			rtnh.Children = children
			buf = append(buf, rtnh.Serialize()...)
		}
		rtAttrs = append(rtAttrs, nl.NewRtAttr(unix.RTA_MULTIPATH, buf))
	}

	if route.Table > 0 {
		if route.Table >= 256 {
			msg.Table = unix.RT_TABLE_UNSPEC
			b := make([]byte, 4)
			native.PutUint32(b, uint32(route.Table))
			rtAttrs = append(rtAttrs, nl.NewRtAttr(unix.RTA_TABLE, b))
		} else {
			msg.Table = uint8(route.Table)
		}
	}

	if route.Priority > 0 {
		b := make([]byte, 4)
		native.PutUint32(b, uint32(route.Priority))
		rtAttrs = append(rtAttrs, nl.NewRtAttr(unix.RTA_PRIORITY, b))
	}
	if route.Tos > 0 {
		msg.Tos = uint8(route.Tos)
	}
	if route.Protocol > 0 {
		msg.Protocol = uint8(route.Protocol)
	}
	if route.Type > 0 {
		msg.Type = uint8(route.Type)
	}

	var metrics []*nl.RtAttr
	// TODO: support other rta_metric values
	if route.MTU > 0 {
		b := nl.Uint32Attr(uint32(route.MTU))
		metrics = append(metrics, nl.NewRtAttr(unix.RTAX_MTU, b))
	}
	if route.AdvMSS > 0 {
		b := nl.Uint32Attr(uint32(route.AdvMSS))
		metrics = append(metrics, nl.NewRtAttr(unix.RTAX_ADVMSS, b))
	}

	if metrics != nil {
		attr := nl.NewRtAttr(unix.RTA_METRICS, nil)
		for _, metric := range metrics {
			attr.AddChild(metric)
		}
		rtAttrs = append(rtAttrs, attr)
	}

	msg.Flags = uint32(route.Flags)
	msg.Scope = uint8(route.Scope)
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

	req.AddData(nl.NewRtAttr(unix.RTA_OIF, b))

	_, err := req.Execute(unix.NETLINK_ROUTE, 0)
	return err
}

// RouteList gets a list of routes in the system.
// Equivalent to: `ip route show`.
// The list can be filtered by link and ip family.
func RouteList(link Link, family int) ([]Route, error) {
	return pkgHandle.RouteList(link, family)
}

// RouteList gets a list of routes in the system.
// Equivalent to: `ip route show`.
// The list can be filtered by link and ip family.
func (h *Handle) RouteList(link Link, family int) ([]Route, error) {
	var routeFilter *Route
	if link != nil {
		routeFilter = &Route{
			LinkIndex: link.Attrs().Index,
		}
	}
	return h.RouteListFiltered(family, routeFilter, RT_FILTER_OIF)
}

// RouteListFiltered gets a list of routes in the system filtered with specified rules.
// All rules must be defined in RouteFilter struct
func RouteListFiltered(family int, filter *Route, filterMask uint64) ([]Route, error) {
	return pkgHandle.RouteListFiltered(family, filter, filterMask)
}

// RouteListFiltered gets a list of routes in the system filtered with specified rules.
// All rules must be defined in RouteFilter struct
func (h *Handle) RouteListFiltered(family int, filter *Route, filterMask uint64) ([]Route, error) {
	req := h.newNetlinkRequest(unix.RTM_GETROUTE, unix.NLM_F_DUMP)
	infmsg := nl.NewIfInfomsg(family)
	req.AddData(infmsg)

	msgs, err := req.Execute(unix.NETLINK_ROUTE, unix.RTM_NEWROUTE)
	if err != nil {
		return nil, err
	}

	var res []Route
	for _, m := range msgs {
		msg := nl.DeserializeRtMsg(m)
		if msg.Flags&unix.RTM_F_CLONED != 0 {
			// Ignore cloned routes
			continue
		}
		if msg.Table != unix.RT_TABLE_MAIN {
			if filter == nil || filter != nil && filterMask&RT_FILTER_TABLE == 0 {
				// Ignore non-main tables
				continue
			}
		}
		route, err := deserializeRoute(m)
		if err != nil {
			return nil, err
		}
		if filter != nil {
			switch {
			case filterMask&RT_FILTER_TABLE != 0 && filter.Table != unix.RT_TABLE_UNSPEC && route.Table != filter.Table:
				continue
			case filterMask&RT_FILTER_PROTOCOL != 0 && route.Protocol != filter.Protocol:
				continue
			case filterMask&RT_FILTER_SCOPE != 0 && route.Scope != filter.Scope:
				continue
			case filterMask&RT_FILTER_TYPE != 0 && route.Type != filter.Type:
				continue
			case filterMask&RT_FILTER_TOS != 0 && route.Tos != filter.Tos:
				continue
			case filterMask&RT_FILTER_OIF != 0 && route.LinkIndex != filter.LinkIndex:
				continue
			case filterMask&RT_FILTER_IIF != 0 && route.ILinkIndex != filter.ILinkIndex:
				continue
			case filterMask&RT_FILTER_GW != 0 && !route.Gw.Equal(filter.Gw):
				continue
			case filterMask&RT_FILTER_SRC != 0 && !route.Src.Equal(filter.Src):
				continue
			case filterMask&RT_FILTER_DST != 0:
				if filter.MPLSDst == nil || route.MPLSDst == nil || (*filter.MPLSDst) != (*route.MPLSDst) {
					if !ipNetEqual(route.Dst, filter.Dst) {
						continue
					}
				}
			}
		}
		res = append(res, route)
	}
	return res, nil
}

// deserializeRoute decodes a binary netlink message into a Route struct
func deserializeRoute(m []byte) (Route, error) {
	msg := nl.DeserializeRtMsg(m)
	attrs, err := nl.ParseRouteAttr(m[msg.Len():])
	if err != nil {
		return Route{}, err
	}
	route := Route{
		Scope:    Scope(msg.Scope),
		Protocol: int(msg.Protocol),
		Table:    int(msg.Table),
		Type:     int(msg.Type),
		Tos:      int(msg.Tos),
		Flags:    int(msg.Flags),
	}

	native := nl.NativeEndian()
	var encap, encapType syscall.NetlinkRouteAttr
	for _, attr := range attrs {
		switch attr.Attr.Type {
		case unix.RTA_GATEWAY:
			route.Gw = net.IP(attr.Value)
		case unix.RTA_PREFSRC:
			route.Src = net.IP(attr.Value)
		case unix.RTA_DST:
			if msg.Family == nl.FAMILY_MPLS {
				stack := nl.DecodeMPLSStack(attr.Value)
				if len(stack) == 0 || len(stack) > 1 {
					return route, fmt.Errorf("invalid MPLS RTA_DST")
				}
				route.MPLSDst = &stack[0]
			} else {
				route.Dst = &net.IPNet{
					IP:   attr.Value,
					Mask: net.CIDRMask(int(msg.Dst_len), 8*len(attr.Value)),
				}
			}
		case unix.RTA_OIF:
			route.LinkIndex = int(native.Uint32(attr.Value[0:4]))
		case unix.RTA_IIF:
			route.ILinkIndex = int(native.Uint32(attr.Value[0:4]))
		case unix.RTA_PRIORITY:
			route.Priority = int(native.Uint32(attr.Value[0:4]))
		case unix.RTA_TABLE:
			route.Table = int(native.Uint32(attr.Value[0:4]))
		case unix.RTA_MULTIPATH:
			parseRtNexthop := func(value []byte) (*NexthopInfo, []byte, error) {
				if len(value) < unix.SizeofRtNexthop {
					return nil, nil, fmt.Errorf("lack of bytes")
				}
				nh := nl.DeserializeRtNexthop(value)
				if len(value) < int(nh.RtNexthop.Len) {
					return nil, nil, fmt.Errorf("lack of bytes")
				}
				info := &NexthopInfo{
					LinkIndex: int(nh.RtNexthop.Ifindex),
					Hops:      int(nh.RtNexthop.Hops),
					Flags:     int(nh.RtNexthop.Flags),
				}
				attrs, err := nl.ParseRouteAttr(value[unix.SizeofRtNexthop:int(nh.RtNexthop.Len)])
				if err != nil {
					return nil, nil, err
				}
				var encap, encapType syscall.NetlinkRouteAttr
				for _, attr := range attrs {
					switch attr.Attr.Type {
					case unix.RTA_GATEWAY:
						info.Gw = net.IP(attr.Value)
					case nl.RTA_NEWDST:
						var d Destination
						switch msg.Family {
						case nl.FAMILY_MPLS:
							d = &MPLSDestination{}
						}
						if err := d.Decode(attr.Value); err != nil {
							return nil, nil, err
						}
						info.NewDst = d
					case nl.RTA_ENCAP_TYPE:
						encapType = attr
					case nl.RTA_ENCAP:
						encap = attr
					}
				}

				if len(encap.Value) != 0 && len(encapType.Value) != 0 {
					typ := int(native.Uint16(encapType.Value[0:2]))
					var e Encap
					switch typ {
					case nl.LWTUNNEL_ENCAP_MPLS:
						e = &MPLSEncap{}
						if err := e.Decode(encap.Value); err != nil {
							return nil, nil, err
						}
					}
					info.Encap = e
				}

				return info, value[int(nh.RtNexthop.Len):], nil
			}
			rest := attr.Value
			for len(rest) > 0 {
				info, buf, err := parseRtNexthop(rest)
				if err != nil {
					return route, err
				}
				route.MultiPath = append(route.MultiPath, info)
				rest = buf
			}
		case nl.RTA_NEWDST:
			var d Destination
			switch msg.Family {
			case nl.FAMILY_MPLS:
				d = &MPLSDestination{}
			}
			if err := d.Decode(attr.Value); err != nil {
				return route, err
			}
			route.NewDst = d
		case nl.RTA_ENCAP_TYPE:
			encapType = attr
		case nl.RTA_ENCAP:
			encap = attr
		case unix.RTA_METRICS:
			metrics, err := nl.ParseRouteAttr(attr.Value)
			if err != nil {
				return route, err
			}
			for _, metric := range metrics {
				switch metric.Attr.Type {
				case unix.RTAX_MTU:
					route.MTU = int(native.Uint32(metric.Value[0:4]))
				case unix.RTAX_ADVMSS:
					route.AdvMSS = int(native.Uint32(metric.Value[0:4]))
				}
			}
		}
	}

	if len(encap.Value) != 0 && len(encapType.Value) != 0 {
		typ := int(native.Uint16(encapType.Value[0:2]))
		var e Encap
		switch typ {
		case nl.LWTUNNEL_ENCAP_MPLS:
			e = &MPLSEncap{}
			if err := e.Decode(encap.Value); err != nil {
				return route, err
			}
		case nl.LWTUNNEL_ENCAP_SEG6:
			e = &SEG6Encap{}
			if err := e.Decode(encap.Value); err != nil {
				return route, err
			}
		}
		route.Encap = e
	}

	return route, nil
}

// RouteGet gets a route to a specific destination from the host system.
// Equivalent to: 'ip route get'.
func RouteGet(destination net.IP) ([]Route, error) {
	return pkgHandle.RouteGet(destination)
}

// RouteGet gets a route to a specific destination from the host system.
// Equivalent to: 'ip route get'.
func (h *Handle) RouteGet(destination net.IP) ([]Route, error) {
	req := h.newNetlinkRequest(unix.RTM_GETROUTE, unix.NLM_F_REQUEST)
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

	rtaDst := nl.NewRtAttr(unix.RTA_DST, destinationData)
	req.AddData(rtaDst)

	msgs, err := req.Execute(unix.NETLINK_ROUTE, unix.RTM_NEWROUTE)
	if err != nil {
		return nil, err
	}

	var res []Route
	for _, m := range msgs {
		route, err := deserializeRoute(m)
		if err != nil {
			return nil, err
		}
		res = append(res, route)
	}
	return res, nil

}

// RouteSubscribe takes a chan down which notifications will be sent
// when routes are added or deleted. Close the 'done' chan to stop subscription.
func RouteSubscribe(ch chan<- RouteUpdate, done <-chan struct{}) error {
	return routeSubscribeAt(netns.None(), netns.None(), ch, done, nil, false)
}

// RouteSubscribeAt works like RouteSubscribe plus it allows the caller
// to choose the network namespace in which to subscribe (ns).
func RouteSubscribeAt(ns netns.NsHandle, ch chan<- RouteUpdate, done <-chan struct{}) error {
	return routeSubscribeAt(ns, netns.None(), ch, done, nil, false)
}

// RouteSubscribeOptions contains a set of options to use with
// RouteSubscribeWithOptions.
type RouteSubscribeOptions struct {
	Namespace     *netns.NsHandle
	ErrorCallback func(error)
	ListExisting  bool
}

// RouteSubscribeWithOptions work like RouteSubscribe but enable to
// provide additional options to modify the behavior. Currently, the
// namespace can be provided as well as an error callback.
func RouteSubscribeWithOptions(ch chan<- RouteUpdate, done <-chan struct{}, options RouteSubscribeOptions) error {
	if options.Namespace == nil {
		none := netns.None()
		options.Namespace = &none
	}
	return routeSubscribeAt(*options.Namespace, netns.None(), ch, done, options.ErrorCallback, options.ListExisting)
}

func routeSubscribeAt(newNs, curNs netns.NsHandle, ch chan<- RouteUpdate, done <-chan struct{}, cberr func(error), listExisting bool) error {
	s, err := nl.SubscribeAt(newNs, curNs, unix.NETLINK_ROUTE, unix.RTNLGRP_IPV4_ROUTE, unix.RTNLGRP_IPV6_ROUTE)
	if err != nil {
		return err
	}
	if done != nil {
		go func() {
			<-done
			s.Close()
		}()
	}
	if listExisting {
		req := pkgHandle.newNetlinkRequest(unix.RTM_GETROUTE,
			unix.NLM_F_DUMP)
		infmsg := nl.NewIfInfomsg(unix.AF_UNSPEC)
		req.AddData(infmsg)
		if err := s.Send(req); err != nil {
			return err
		}
	}
	go func() {
		defer close(ch)
		for {
			msgs, err := s.Receive()
			if err != nil {
				if cberr != nil {
					cberr(err)
				}
				return
			}
			for _, m := range msgs {
				if m.Header.Type == unix.NLMSG_DONE {
					continue
				}
				if m.Header.Type == unix.NLMSG_ERROR {
					native := nl.NativeEndian()
					error := int32(native.Uint32(m.Data[0:4]))
					if error == 0 {
						continue
					}
					if cberr != nil {
						cberr(syscall.Errno(-error))
					}
					return
				}
				route, err := deserializeRoute(m.Data)
				if err != nil {
					if cberr != nil {
						cberr(err)
					}
					return
				}
				ch <- RouteUpdate{Type: m.Header.Type, Route: route}
			}
		}
	}()

	return nil
}
