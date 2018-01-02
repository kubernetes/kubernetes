// +build linux

package netlink

import (
	"net"
	"strconv"
	"testing"
	"time"

	"github.com/vishvananda/netlink/nl"
	"github.com/vishvananda/netns"
	"golang.org/x/sys/unix"
)

func TestRouteAddDel(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	// get loopback interface
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}

	// bring the interface up
	if err := LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	// add a gateway route
	dst := &net.IPNet{
		IP:   net.IPv4(192, 168, 0, 0),
		Mask: net.CIDRMask(24, 32),
	}

	ip := net.IPv4(127, 1, 1, 1)
	route := Route{LinkIndex: link.Attrs().Index, Dst: dst, Src: ip}
	if err := RouteAdd(&route); err != nil {
		t.Fatal(err)
	}
	routes, err := RouteList(link, FAMILY_V4)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 1 {
		t.Fatal("Route not added properly")
	}

	dstIP := net.IPv4(192, 168, 0, 42)
	routeToDstIP, err := RouteGet(dstIP)
	if err != nil {
		t.Fatal(err)
	}

	if len(routeToDstIP) == 0 {
		t.Fatal("Default route not present")
	}
	if err := RouteDel(&route); err != nil {
		t.Fatal(err)
	}
	routes, err = RouteList(link, FAMILY_V4)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 0 {
		t.Fatal("Route not removed properly")
	}

}

func TestRouteReplace(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	// get loopback interface
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}

	// bring the interface up
	if err := LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	// add a gateway route
	dst := &net.IPNet{
		IP:   net.IPv4(192, 168, 0, 0),
		Mask: net.CIDRMask(24, 32),
	}

	ip := net.IPv4(127, 1, 1, 1)
	route := Route{LinkIndex: link.Attrs().Index, Dst: dst, Src: ip}
	if err := RouteAdd(&route); err != nil {
		t.Fatal(err)
	}
	routes, err := RouteList(link, FAMILY_V4)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 1 {
		t.Fatal("Route not added properly")
	}

	ip = net.IPv4(127, 1, 1, 2)
	route = Route{LinkIndex: link.Attrs().Index, Dst: dst, Src: ip}
	if err := RouteReplace(&route); err != nil {
		t.Fatal(err)
	}

	routes, err = RouteList(link, FAMILY_V4)
	if err != nil {
		t.Fatal(err)
	}

	if len(routes) != 1 || !routes[0].Src.Equal(ip) {
		t.Fatal("Route not replaced properly")
	}

	if err := RouteDel(&route); err != nil {
		t.Fatal(err)
	}
	routes, err = RouteList(link, FAMILY_V4)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 0 {
		t.Fatal("Route not removed properly")
	}

}

func TestRouteAddIncomplete(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	// get loopback interface
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}

	// bring the interface up
	if err = LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	route := Route{LinkIndex: link.Attrs().Index}
	if err := RouteAdd(&route); err == nil {
		t.Fatal("Adding incomplete route should fail")
	}
}

func expectRouteUpdate(ch <-chan RouteUpdate, t uint16, dst net.IP) bool {
	for {
		timeout := time.After(time.Minute)
		select {
		case update := <-ch:
			if update.Type == t && update.Route.Dst.IP.Equal(dst) {
				return true
			}
		case <-timeout:
			return false
		}
	}
}

func TestRouteSubscribe(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	ch := make(chan RouteUpdate)
	done := make(chan struct{})
	defer close(done)
	if err := RouteSubscribe(ch, done); err != nil {
		t.Fatal(err)
	}

	// get loopback interface
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}

	// bring the interface up
	if err = LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	// add a gateway route
	dst := &net.IPNet{
		IP:   net.IPv4(192, 168, 0, 0),
		Mask: net.CIDRMask(24, 32),
	}

	ip := net.IPv4(127, 1, 1, 1)
	route := Route{LinkIndex: link.Attrs().Index, Dst: dst, Src: ip}
	if err := RouteAdd(&route); err != nil {
		t.Fatal(err)
	}

	if !expectRouteUpdate(ch, unix.RTM_NEWROUTE, dst.IP) {
		t.Fatal("Add update not received as expected")
	}
	if err := RouteDel(&route); err != nil {
		t.Fatal(err)
	}
	if !expectRouteUpdate(ch, unix.RTM_DELROUTE, dst.IP) {
		t.Fatal("Del update not received as expected")
	}
}

func TestRouteSubscribeWithOptions(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	ch := make(chan RouteUpdate)
	done := make(chan struct{})
	defer close(done)
	var lastError error
	defer func() {
		if lastError != nil {
			t.Fatalf("Fatal error received during subscription: %v", lastError)
		}
	}()
	if err := RouteSubscribeWithOptions(ch, done, RouteSubscribeOptions{
		ErrorCallback: func(err error) {
			lastError = err
		},
	}); err != nil {
		t.Fatal(err)
	}

	// get loopback interface
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}

	// bring the interface up
	if err = LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	// add a gateway route
	dst := &net.IPNet{
		IP:   net.IPv4(192, 168, 0, 0),
		Mask: net.CIDRMask(24, 32),
	}

	ip := net.IPv4(127, 1, 1, 1)
	route := Route{LinkIndex: link.Attrs().Index, Dst: dst, Src: ip}
	if err := RouteAdd(&route); err != nil {
		t.Fatal(err)
	}

	if !expectRouteUpdate(ch, unix.RTM_NEWROUTE, dst.IP) {
		t.Fatal("Add update not received as expected")
	}
}

func TestRouteSubscribeAt(t *testing.T) {
	skipUnlessRoot(t)

	// Create an handle on a custom netns
	newNs, err := netns.New()
	if err != nil {
		t.Fatal(err)
	}
	defer newNs.Close()

	nh, err := NewHandleAt(newNs)
	if err != nil {
		t.Fatal(err)
	}
	defer nh.Delete()

	// Subscribe for Route events on the custom netns
	ch := make(chan RouteUpdate)
	done := make(chan struct{})
	defer close(done)
	if err := RouteSubscribeAt(newNs, ch, done); err != nil {
		t.Fatal(err)
	}

	// get loopback interface
	link, err := nh.LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}

	// bring the interface up
	if err = nh.LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	// add a gateway route
	dst := &net.IPNet{
		IP:   net.IPv4(192, 169, 0, 0),
		Mask: net.CIDRMask(24, 32),
	}

	ip := net.IPv4(127, 100, 1, 1)
	route := Route{LinkIndex: link.Attrs().Index, Dst: dst, Src: ip}
	if err := nh.RouteAdd(&route); err != nil {
		t.Fatal(err)
	}

	if !expectRouteUpdate(ch, unix.RTM_NEWROUTE, dst.IP) {
		t.Fatal("Add update not received as expected")
	}
	if err := nh.RouteDel(&route); err != nil {
		t.Fatal(err)
	}
	if !expectRouteUpdate(ch, unix.RTM_DELROUTE, dst.IP) {
		t.Fatal("Del update not received as expected")
	}
}

func TestRouteFilterAllTables(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	// get loopback interface
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}
	// bring the interface up
	if err = LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	// add a gateway route
	dst := &net.IPNet{
		IP:   net.IPv4(1, 1, 1, 1),
		Mask: net.CIDRMask(32, 32),
	}

	tables := []int{1000, 1001, 1002}
	src := net.IPv4(127, 3, 3, 3)
	for _, table := range tables {
		route := Route{
			LinkIndex: link.Attrs().Index,
			Dst:       dst,
			Src:       src,
			Scope:     unix.RT_SCOPE_LINK,
			Priority:  13,
			Table:     table,
			Type:      unix.RTN_UNICAST,
			Tos:       14,
		}
		if err := RouteAdd(&route); err != nil {
			t.Fatal(err)
		}
	}
	routes, err := RouteListFiltered(FAMILY_V4, &Route{
		Dst:   dst,
		Src:   src,
		Scope: unix.RT_SCOPE_LINK,
		Table: unix.RT_TABLE_UNSPEC,
		Type:  unix.RTN_UNICAST,
		Tos:   14,
	}, RT_FILTER_DST|RT_FILTER_SRC|RT_FILTER_SCOPE|RT_FILTER_TABLE|RT_FILTER_TYPE|RT_FILTER_TOS)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 3 {
		t.Fatal("Routes not added properly")
	}

	for _, route := range routes {
		if route.Scope != unix.RT_SCOPE_LINK {
			t.Fatal("Invalid Scope. Route not added properly")
		}
		if route.Priority != 13 {
			t.Fatal("Invalid Priority. Route not added properly")
		}
		if !tableIDIn(tables, route.Table) {
			t.Fatalf("Invalid Table %d. Route not added properly", route.Table)
		}
		if route.Type != unix.RTN_UNICAST {
			t.Fatal("Invalid Type. Route not added properly")
		}
		if route.Tos != 14 {
			t.Fatal("Invalid Tos. Route not added properly")
		}
	}
}

func tableIDIn(ids []int, id int) bool {
	for _, v := range ids {
		if v == id {
			return true
		}
	}
	return false
}

func TestRouteExtraFields(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	// get loopback interface
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}
	// bring the interface up
	if err = LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	// add a gateway route
	dst := &net.IPNet{
		IP:   net.IPv4(1, 1, 1, 1),
		Mask: net.CIDRMask(32, 32),
	}

	src := net.IPv4(127, 3, 3, 3)
	route := Route{
		LinkIndex: link.Attrs().Index,
		Dst:       dst,
		Src:       src,
		Scope:     unix.RT_SCOPE_LINK,
		Priority:  13,
		Table:     unix.RT_TABLE_MAIN,
		Type:      unix.RTN_UNICAST,
		Tos:       14,
	}
	if err := RouteAdd(&route); err != nil {
		t.Fatal(err)
	}
	routes, err := RouteListFiltered(FAMILY_V4, &Route{
		Dst:   dst,
		Src:   src,
		Scope: unix.RT_SCOPE_LINK,
		Table: unix.RT_TABLE_MAIN,
		Type:  unix.RTN_UNICAST,
		Tos:   14,
	}, RT_FILTER_DST|RT_FILTER_SRC|RT_FILTER_SCOPE|RT_FILTER_TABLE|RT_FILTER_TYPE|RT_FILTER_TOS)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 1 {
		t.Fatal("Route not added properly")
	}

	if routes[0].Scope != unix.RT_SCOPE_LINK {
		t.Fatal("Invalid Scope. Route not added properly")
	}
	if routes[0].Priority != 13 {
		t.Fatal("Invalid Priority. Route not added properly")
	}
	if routes[0].Table != unix.RT_TABLE_MAIN {
		t.Fatal("Invalid Scope. Route not added properly")
	}
	if routes[0].Type != unix.RTN_UNICAST {
		t.Fatal("Invalid Type. Route not added properly")
	}
	if routes[0].Tos != 14 {
		t.Fatal("Invalid Tos. Route not added properly")
	}
}

func TestRouteMultiPath(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	// get loopback interface
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}
	// bring the interface up
	if err = LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	// add a gateway route
	dst := &net.IPNet{
		IP:   net.IPv4(192, 168, 0, 0),
		Mask: net.CIDRMask(24, 32),
	}

	idx := link.Attrs().Index
	route := Route{Dst: dst, MultiPath: []*NexthopInfo{&NexthopInfo{LinkIndex: idx}, &NexthopInfo{LinkIndex: idx}}}
	if err := RouteAdd(&route); err != nil {
		t.Fatal(err)
	}
	routes, err := RouteList(nil, FAMILY_V4)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 1 {
		t.Fatal("MultiPath Route not added properly")
	}
	if len(routes[0].MultiPath) != 2 {
		t.Fatal("MultiPath Route not added properly")
	}
}

func TestFilterDefaultRoute(t *testing.T) {
	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	// get loopback interface
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}
	// bring the interface up
	if err = LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	address := &Addr{
		IPNet: &net.IPNet{
			IP:   net.IPv4(127, 0, 0, 2),
			Mask: net.CIDRMask(24, 32),
		},
	}
	if err = AddrAdd(link, address); err != nil {
		t.Fatal(err)
	}

	// Add default route
	gw := net.IPv4(127, 0, 0, 2)

	defaultRoute := Route{
		Dst: nil,
		Gw:  gw,
	}

	if err := RouteAdd(&defaultRoute); err != nil {
		t.Fatal(err)
	}

	// add an extra route
	dst := &net.IPNet{
		IP:   net.IPv4(192, 168, 0, 0),
		Mask: net.CIDRMask(24, 32),
	}

	extraRoute := Route{
		Dst: dst,
		Gw:  gw,
	}

	if err := RouteAdd(&extraRoute); err != nil {
		t.Fatal(err)
	}
	var filterTests = []struct {
		filter   *Route
		mask     uint64
		expected net.IP
	}{
		{
			&Route{Dst: nil},
			RT_FILTER_DST,
			gw,
		},
		{
			&Route{Dst: dst},
			RT_FILTER_DST,
			gw,
		},
	}

	for _, f := range filterTests {
		routes, err := RouteListFiltered(FAMILY_V4, f.filter, f.mask)
		if err != nil {
			t.Fatal(err)
		}
		if len(routes) != 1 {
			t.Fatal("Route not filtered properly")
		}
		if !routes[0].Gw.Equal(gw) {
			t.Fatal("Unexpected Gateway")
		}
	}

}

func TestMPLSRouteAddDel(t *testing.T) {
	tearDown := setUpMPLSNetlinkTest(t)
	defer tearDown()

	// get loopback interface
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}

	// bring the interface up
	if err := LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	mplsDst := 100
	route := Route{
		LinkIndex: link.Attrs().Index,
		MPLSDst:   &mplsDst,
		NewDst: &MPLSDestination{
			Labels: []int{200, 300},
		},
	}
	if err := RouteAdd(&route); err != nil {
		t.Fatal(err)
	}
	routes, err := RouteList(link, FAMILY_MPLS)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 1 {
		t.Fatal("Route not added properly")
	}

	if err := RouteDel(&route); err != nil {
		t.Fatal(err)
	}
	routes, err = RouteList(link, FAMILY_MPLS)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 0 {
		t.Fatal("Route not removed properly")
	}

}

func TestRouteEqual(t *testing.T) {
	mplsDst := 100
	seg6encap := &SEG6Encap{Mode: nl.SEG6_IPTUN_MODE_ENCAP}
	seg6encap.Segments = []net.IP{net.ParseIP("fc00:a000::11")}
	cases := []Route{
		Route{
			Dst: nil,
			Gw:  net.IPv4(1, 1, 1, 1),
		},
		Route{
			LinkIndex: 20,
			Dst:       nil,
			Gw:        net.IPv4(1, 1, 1, 1),
		},
		Route{
			ILinkIndex: 21,
			LinkIndex:  20,
			Dst:        nil,
			Gw:         net.IPv4(1, 1, 1, 1),
		},
		Route{
			LinkIndex: 20,
			Dst:       nil,
			Protocol:  20,
			Gw:        net.IPv4(1, 1, 1, 1),
		},
		Route{
			LinkIndex: 20,
			Dst:       nil,
			Priority:  20,
			Gw:        net.IPv4(1, 1, 1, 1),
		},
		Route{
			LinkIndex: 20,
			Dst:       nil,
			Type:      20,
			Gw:        net.IPv4(1, 1, 1, 1),
		},
		Route{
			LinkIndex: 20,
			Dst:       nil,
			Table:     200,
			Gw:        net.IPv4(1, 1, 1, 1),
		},
		Route{
			LinkIndex: 20,
			Dst:       nil,
			Tos:       1,
			Gw:        net.IPv4(1, 1, 1, 1),
		},
		Route{
			LinkIndex: 20,
			Dst:       nil,
			Flags:     int(FLAG_ONLINK),
			Gw:        net.IPv4(1, 1, 1, 1),
		},
		Route{
			LinkIndex: 10,
			Dst: &net.IPNet{
				IP:   net.IPv4(192, 168, 0, 0),
				Mask: net.CIDRMask(24, 32),
			},
			Src: net.IPv4(127, 1, 1, 1),
		},
		Route{
			LinkIndex: 10,
			Scope:     unix.RT_SCOPE_LINK,
			Dst: &net.IPNet{
				IP:   net.IPv4(192, 168, 0, 0),
				Mask: net.CIDRMask(24, 32),
			},
			Src: net.IPv4(127, 1, 1, 1),
		},
		Route{
			LinkIndex: 3,
			Dst: &net.IPNet{
				IP:   net.IPv4(1, 1, 1, 1),
				Mask: net.CIDRMask(32, 32),
			},
			Src:      net.IPv4(127, 3, 3, 3),
			Scope:    unix.RT_SCOPE_LINK,
			Priority: 13,
			Table:    unix.RT_TABLE_MAIN,
			Type:     unix.RTN_UNICAST,
			Tos:      14,
		},
		Route{
			LinkIndex: 10,
			MPLSDst:   &mplsDst,
			NewDst: &MPLSDestination{
				Labels: []int{200, 300},
			},
		},
		Route{
			Dst: nil,
			Gw:  net.IPv4(1, 1, 1, 1),
			Encap: &MPLSEncap{
				Labels: []int{100},
			},
		},
		Route{
			LinkIndex: 10,
			Dst: &net.IPNet{
				IP:   net.IPv4(10, 0, 0, 102),
				Mask: net.CIDRMask(32, 32),
			},
			Encap: seg6encap,
		},
		Route{
			Dst:       nil,
			MultiPath: []*NexthopInfo{&NexthopInfo{LinkIndex: 10}, &NexthopInfo{LinkIndex: 20}},
		},
		Route{
			Dst: nil,
			MultiPath: []*NexthopInfo{&NexthopInfo{
				LinkIndex: 10,
				Gw:        net.IPv4(1, 1, 1, 1),
			}, &NexthopInfo{LinkIndex: 20}},
		},
		Route{
			Dst: nil,
			MultiPath: []*NexthopInfo{&NexthopInfo{
				LinkIndex: 10,
				Gw:        net.IPv4(1, 1, 1, 1),
				Encap: &MPLSEncap{
					Labels: []int{100},
				},
			}, &NexthopInfo{LinkIndex: 20}},
		},
		Route{
			Dst: nil,
			MultiPath: []*NexthopInfo{&NexthopInfo{
				LinkIndex: 10,
				NewDst: &MPLSDestination{
					Labels: []int{200, 300},
				},
			}, &NexthopInfo{LinkIndex: 20}},
		},
		Route{
			Dst: nil,
			MultiPath: []*NexthopInfo{&NexthopInfo{
				LinkIndex: 10,
				Encap:     seg6encap,
			}, &NexthopInfo{LinkIndex: 20}},
		},
	}
	for i1 := range cases {
		for i2 := range cases {
			got := cases[i1].Equal(cases[i2])
			expected := i1 == i2
			if got != expected {
				t.Errorf("Equal(%q,%q) == %s but expected %s",
					cases[i1], cases[i2],
					strconv.FormatBool(got),
					strconv.FormatBool(expected))
			}
		}
	}
}

func TestIPNetEqual(t *testing.T) {
	cases := []string{
		"1.1.1.1/24", "1.1.1.0/24", "1.1.1.1/32",
		"0.0.0.0/0", "0.0.0.0/14",
		"2001:db8::/32", "2001:db8::/128",
		"2001:db8::caff/32", "2001:db8::caff/128",
		"",
	}
	for _, c1 := range cases {
		var n1 *net.IPNet
		if c1 != "" {
			var i1 net.IP
			var err1 error
			i1, n1, err1 = net.ParseCIDR(c1)
			if err1 != nil {
				panic(err1)
			}
			n1.IP = i1
		}
		for _, c2 := range cases {
			var n2 *net.IPNet
			if c2 != "" {
				var i2 net.IP
				var err2 error
				i2, n2, err2 = net.ParseCIDR(c2)
				if err2 != nil {
					panic(err2)
				}
				n2.IP = i2
			}

			got := ipNetEqual(n1, n2)
			expected := c1 == c2
			if got != expected {
				t.Errorf("IPNetEqual(%q,%q) == %s but expected %s",
					c1, c2,
					strconv.FormatBool(got),
					strconv.FormatBool(expected))
			}
		}
	}
}

func TestSEG6RouteAddDel(t *testing.T) {
	// add/del IPv4 routes with LWTUNNEL_SEG6 to/from loopback interface.
	// Test both seg6 modes: encap & inline.
	tearDown := setUpSEG6NetlinkTest(t)
	defer tearDown()

	// get loopback interface and bring it up
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}
	if err := LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	dst1 := &net.IPNet{
		IP:   net.IPv4(10, 0, 0, 101),
		Mask: net.CIDRMask(32, 32),
	}
	dst2 := &net.IPNet{
		IP:   net.IPv4(10, 0, 0, 102),
		Mask: net.CIDRMask(32, 32),
	}
	var s1, s2 []net.IP
	s1 = append(s1, net.ParseIP("::")) // inline requires "::"
	s1 = append(s1, net.ParseIP("fc00:a000::12"))
	s1 = append(s1, net.ParseIP("fc00:a000::11"))
	s2 = append(s2, net.ParseIP("fc00:a000::22"))
	s2 = append(s2, net.ParseIP("fc00:a000::21"))
	e1 := &SEG6Encap{Mode: nl.SEG6_IPTUN_MODE_INLINE}
	e2 := &SEG6Encap{Mode: nl.SEG6_IPTUN_MODE_ENCAP}
	e1.Segments = s1
	e2.Segments = s2
	route1 := Route{LinkIndex: link.Attrs().Index, Dst: dst1, Encap: e1}
	route2 := Route{LinkIndex: link.Attrs().Index, Dst: dst2, Encap: e2}

	// Add SEG6 routes
	if err := RouteAdd(&route1); err != nil {
		t.Fatal(err)
	}
	if err := RouteAdd(&route2); err != nil {
		t.Fatal(err)
	}
	routes, err := RouteList(link, FAMILY_V4)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 2 {
		t.Fatal("SEG6 routes not added properly")
	}
	for _, route := range routes {
		if route.Encap.Type() != nl.LWTUNNEL_ENCAP_SEG6 {
			t.Fatal("Invalid Type. SEG6 routes not added properly")
		}
	}

	// Del (remove) SEG6 routes
	if err := RouteDel(&route1); err != nil {
		t.Fatal(err)
	}
	if err := RouteDel(&route2); err != nil {
		t.Fatal(err)
	}
	routes, err = RouteList(link, FAMILY_V4)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 0 {
		t.Fatal("SEG6 routes not removed properly")
	}
}

func TestMTURouteAddDel(t *testing.T) {
	_, err := RouteList(nil, FAMILY_V4)
	if err != nil {
		t.Fatal(err)
	}

	tearDown := setUpNetlinkTest(t)
	defer tearDown()

	// get loopback interface
	link, err := LinkByName("lo")
	if err != nil {
		t.Fatal(err)
	}

	// bring the interface up
	if err := LinkSetUp(link); err != nil {
		t.Fatal(err)
	}

	// add a gateway route
	dst := &net.IPNet{
		IP:   net.IPv4(192, 168, 0, 0),
		Mask: net.CIDRMask(24, 32),
	}

	route := Route{LinkIndex: link.Attrs().Index, Dst: dst, MTU: 500}
	if err := RouteAdd(&route); err != nil {
		t.Fatal(err)
	}
	routes, err := RouteList(link, FAMILY_V4)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 1 {
		t.Fatal("Route not added properly")
	}

	if route.MTU != routes[0].MTU {
		t.Fatal("Route mtu not set properly")
	}

	if err := RouteDel(&route); err != nil {
		t.Fatal(err)
	}
	routes, err = RouteList(link, FAMILY_V4)
	if err != nil {
		t.Fatal(err)
	}
	if len(routes) != 0 {
		t.Fatal("Route not removed properly")
	}
}
