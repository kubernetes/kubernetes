// +build linux

package netlink

import (
	"fmt"
	"net"
	"runtime"
	"testing"

	"github.com/vishvananda/netns"
	"golang.org/x/sys/unix"
)

func CheckErrorFail(t *testing.T, err error) {
	if err != nil {
		t.Fatalf("Fatal Error: %s", err)
	}
}
func CheckError(t *testing.T, err error) {
	if err != nil {
		t.Errorf("Error: %s", err)
	}
}

func udpFlowCreateProg(t *testing.T, flows, srcPort int, dstIP string, dstPort int) {
	for i := 0; i < flows; i++ {
		ServerAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("%s:%d", dstIP, dstPort))
		CheckError(t, err)

		LocalAddr, err := net.ResolveUDPAddr("udp", fmt.Sprintf("127.0.0.1:%d", srcPort+i))
		CheckError(t, err)

		Conn, err := net.DialUDP("udp", LocalAddr, ServerAddr)
		CheckError(t, err)

		Conn.Write([]byte("Hello World"))
		Conn.Close()
	}
}

func nsCreateAndEnter(t *testing.T) (*netns.NsHandle, *netns.NsHandle, *Handle) {
	// Lock the OS Thread so we don't accidentally switch namespaces
	runtime.LockOSThread()

	// Save the current network namespace
	origns, _ := netns.Get()

	ns, err := netns.New()
	CheckErrorFail(t, err)

	h, err := NewHandleAt(ns)
	CheckErrorFail(t, err)

	// Enter the new namespace
	netns.Set(ns)

	// Bing up loopback
	link, _ := h.LinkByName("lo")
	h.LinkSetUp(link)

	return &origns, &ns, h
}

func applyFilter(flowList []ConntrackFlow, ipv4Filter *ConntrackFilter, ipv6Filter *ConntrackFilter) (ipv4Match, ipv6Match uint) {
	for _, flow := range flowList {
		if ipv4Filter.MatchConntrackFlow(&flow) == true {
			ipv4Match++
		}
		if ipv6Filter.MatchConntrackFlow(&flow) == true {
			ipv6Match++
		}
	}
	return ipv4Match, ipv6Match
}

// TestConntrackSocket test the opening of a NETFILTER family socket
func TestConntrackSocket(t *testing.T) {
	skipUnlessRoot(t)
	setUpNetlinkTestWithKModule(t, "nf_conntrack")
	setUpNetlinkTestWithKModule(t, "nf_conntrack_netlink")

	h, err := NewHandle(unix.NETLINK_NETFILTER)
	CheckErrorFail(t, err)

	if h.SupportsNetlinkFamily(unix.NETLINK_NETFILTER) != true {
		t.Fatal("ERROR not supporting the NETFILTER family")
	}
}

// TestConntrackTableList test the conntrack table list
// Creates some flows and checks that they are correctly fetched from the conntrack table
func TestConntrackTableList(t *testing.T) {
	skipUnlessRoot(t)
	setUpNetlinkTestWithKModule(t, "nf_conntrack")
	setUpNetlinkTestWithKModule(t, "nf_conntrack_netlink")
	setUpNetlinkTestWithKModule(t, "nf_conntrack_ipv4")
	setUpNetlinkTestWithKModule(t, "nf_conntrack_ipv6")

	// Creates a new namespace and bring up the loopback interface
	origns, ns, h := nsCreateAndEnter(t)
	defer netns.Set(*origns)
	defer origns.Close()
	defer ns.Close()
	defer runtime.UnlockOSThread()

	// Flush the table to start fresh
	err := h.ConntrackTableFlush(ConntrackTable)
	CheckErrorFail(t, err)

	// Create 5 udp
	udpFlowCreateProg(t, 5, 2000, "127.0.0.10", 3000)

	// Fetch the conntrack table
	flows, err := h.ConntrackTableList(ConntrackTable, unix.AF_INET)
	CheckErrorFail(t, err)

	// Check that it is able to find the 5 flows created
	var found int
	for _, flow := range flows {
		if flow.Forward.Protocol == 17 &&
			flow.Forward.DstIP.Equal(net.ParseIP("127.0.0.10")) &&
			flow.Forward.DstPort == 3000 &&
			(flow.Forward.SrcPort >= 2000 && flow.Forward.SrcPort <= 2005) {
			found++
		}
	}
	if found != 5 {
		t.Fatalf("Found only %d flows over 5", found)
	}

	// Give a try also to the IPv6 version
	_, err = h.ConntrackTableList(ConntrackTable, unix.AF_INET6)
	CheckErrorFail(t, err)

	// Switch back to the original namespace
	netns.Set(*origns)
}

// TestConntrackTableFlush test the conntrack table flushing
// Creates some flows and then call the table flush
func TestConntrackTableFlush(t *testing.T) {
	skipUnlessRoot(t)
	setUpNetlinkTestWithKModule(t, "nf_conntrack")
	setUpNetlinkTestWithKModule(t, "nf_conntrack_netlink")
	setUpNetlinkTestWithKModule(t, "nf_conntrack_ipv4")

	// Creates a new namespace and bring up the loopback interface
	origns, ns, h := nsCreateAndEnter(t)
	defer netns.Set(*origns)
	defer origns.Close()
	defer ns.Close()
	defer runtime.UnlockOSThread()

	// Create 5 udp flows using netcat
	udpFlowCreateProg(t, 5, 3000, "127.0.0.10", 4000)

	// Fetch the conntrack table
	flows, err := h.ConntrackTableList(ConntrackTable, unix.AF_INET)
	CheckErrorFail(t, err)

	// Check that it is able to find the 5 flows created
	var found int
	for _, flow := range flows {
		if flow.Forward.Protocol == 17 &&
			flow.Forward.DstIP.Equal(net.ParseIP("127.0.0.10")) &&
			flow.Forward.DstPort == 4000 &&
			(flow.Forward.SrcPort >= 3000 && flow.Forward.SrcPort <= 3005) {
			found++
		}
	}
	if found != 5 {
		t.Fatalf("Found only %d flows over 5", found)
	}

	// Flush the table
	err = h.ConntrackTableFlush(ConntrackTable)
	CheckErrorFail(t, err)

	// Fetch again the flows to validate the flush
	flows, err = h.ConntrackTableList(ConntrackTable, unix.AF_INET)
	CheckErrorFail(t, err)

	// Check if it is still able to find the 5 flows created
	found = 0
	for _, flow := range flows {
		if flow.Forward.Protocol == 17 &&
			flow.Forward.DstIP.Equal(net.ParseIP("127.0.0.10")) &&
			flow.Forward.DstPort == 4000 &&
			(flow.Forward.SrcPort >= 3000 && flow.Forward.SrcPort <= 3005) {
			found++
		}
	}
	if found > 0 {
		t.Fatalf("Found %d flows, they should had been flushed", found)
	}

	// Switch back to the original namespace
	netns.Set(*origns)
}

// TestConntrackTableDelete tests the deletion with filter
// Creates 2 group of flows then deletes only one group and validates the result
func TestConntrackTableDelete(t *testing.T) {
	skipUnlessRoot(t)
	setUpNetlinkTestWithKModule(t, "nf_conntrack")
	setUpNetlinkTestWithKModule(t, "nf_conntrack_netlink")
	setUpNetlinkTestWithKModule(t, "nf_conntrack_ipv4")

	// Creates a new namespace and bring up the loopback interface
	origns, ns, h := nsCreateAndEnter(t)
	defer netns.Set(*origns)
	defer origns.Close()
	defer ns.Close()
	defer runtime.UnlockOSThread()

	// Create 10 udp flows
	udpFlowCreateProg(t, 5, 5000, "127.0.0.10", 6000)
	udpFlowCreateProg(t, 5, 7000, "127.0.0.20", 8000)

	// Fetch the conntrack table
	flows, err := h.ConntrackTableList(ConntrackTable, unix.AF_INET)
	CheckErrorFail(t, err)

	// Check that it is able to find the 5 flows created for each group
	var groupA int
	var groupB int
	for _, flow := range flows {
		if flow.Forward.Protocol == 17 &&
			flow.Forward.DstIP.Equal(net.ParseIP("127.0.0.10")) &&
			flow.Forward.DstPort == 6000 &&
			(flow.Forward.SrcPort >= 5000 && flow.Forward.SrcPort <= 5005) {
			groupA++
		}
		if flow.Forward.Protocol == 17 &&
			flow.Forward.DstIP.Equal(net.ParseIP("127.0.0.20")) &&
			flow.Forward.DstPort == 8000 &&
			(flow.Forward.SrcPort >= 7000 && flow.Forward.SrcPort <= 7005) {
			groupB++
		}
	}
	if groupA != 5 || groupB != 5 {
		t.Fatalf("Flow creation issue groupA:%d, groupB:%d", groupA, groupB)
	}

	// Create a filter to erase groupB flows
	filter := &ConntrackFilter{}
	filter.AddIP(ConntrackOrigDstIP, net.ParseIP("127.0.0.20"))

	// Flush entries of groupB
	var deleted uint
	if deleted, err = h.ConntrackDeleteFilter(ConntrackTable, unix.AF_INET, filter); err != nil {
		t.Fatalf("Error during the erase: %s", err)
	}
	if deleted != 5 {
		t.Fatalf("Error deleted a wrong number of flows:%d instead of 5", deleted)
	}

	// Check again the table to verify that are gone
	flows, err = h.ConntrackTableList(ConntrackTable, unix.AF_INET)
	CheckErrorFail(t, err)

	// Check if it is able to find the 5 flows of groupA but none of groupB
	groupA = 0
	groupB = 0
	for _, flow := range flows {
		if flow.Forward.Protocol == 17 &&
			flow.Forward.DstIP.Equal(net.ParseIP("127.0.0.10")) &&
			flow.Forward.DstPort == 6000 &&
			(flow.Forward.SrcPort >= 5000 && flow.Forward.SrcPort <= 5005) {
			groupA++
		}
		if flow.Forward.Protocol == 17 &&
			flow.Forward.DstIP.Equal(net.ParseIP("127.0.0.20")) &&
			flow.Forward.DstPort == 8000 &&
			(flow.Forward.SrcPort >= 7000 && flow.Forward.SrcPort <= 7005) {
			groupB++
		}
	}
	if groupA != 5 || groupB > 0 {
		t.Fatalf("Error during the erase groupA:%d, groupB:%d", groupA, groupB)
	}

	// Switch back to the original namespace
	netns.Set(*origns)
}

func TestConntrackFilter(t *testing.T) {
	var flowList []ConntrackFlow
	flowList = append(flowList, ConntrackFlow{
		FamilyType: unix.AF_INET,
		Forward: ipTuple{
			SrcIP:   net.ParseIP("10.0.0.1"),
			DstIP:   net.ParseIP("20.0.0.1"),
			SrcPort: 1000,
			DstPort: 2000,
		},
		Reverse: ipTuple{
			SrcIP:   net.ParseIP("20.0.0.1"),
			DstIP:   net.ParseIP("192.168.1.1"),
			SrcPort: 2000,
			DstPort: 1000,
		},
	},
		ConntrackFlow{
			FamilyType: unix.AF_INET,
			Forward: ipTuple{
				SrcIP:   net.ParseIP("10.0.0.2"),
				DstIP:   net.ParseIP("20.0.0.2"),
				SrcPort: 5000,
				DstPort: 6000,
			},
			Reverse: ipTuple{
				SrcIP:   net.ParseIP("20.0.0.2"),
				DstIP:   net.ParseIP("192.168.1.1"),
				SrcPort: 6000,
				DstPort: 5000,
			},
		},
		ConntrackFlow{
			FamilyType: unix.AF_INET6,
			Forward: ipTuple{
				SrcIP:   net.ParseIP("eeee:eeee:eeee:eeee:eeee:eeee:eeee:eeee"),
				DstIP:   net.ParseIP("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd"),
				SrcPort: 1000,
				DstPort: 2000,
			},
			Reverse: ipTuple{
				SrcIP:   net.ParseIP("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd"),
				DstIP:   net.ParseIP("eeee:eeee:eeee:eeee:eeee:eeee:eeee:eeee"),
				SrcPort: 2000,
				DstPort: 1000,
			},
		})

	// Empty filter
	v4Match, v6Match := applyFilter(flowList, &ConntrackFilter{}, &ConntrackFilter{})
	if v4Match > 0 || v6Match > 0 {
		t.Fatalf("Error, empty filter cannot match, v4:%d, v6:%d", v4Match, v6Match)
	}

	// SrcIP filter
	filterV4 := &ConntrackFilter{}
	filterV4.AddIP(ConntrackOrigSrcIP, net.ParseIP("10.0.0.1"))

	filterV6 := &ConntrackFilter{}
	filterV6.AddIP(ConntrackOrigSrcIP, net.ParseIP("eeee:eeee:eeee:eeee:eeee:eeee:eeee:eeee"))

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 1 || v6Match != 1 {
		t.Fatalf("Error, there should be only 1 match, v4:%d, v6:%d", v4Match, v6Match)
	}

	// DstIp filter
	filterV4 = &ConntrackFilter{}
	filterV4.AddIP(ConntrackOrigDstIP, net.ParseIP("20.0.0.1"))

	filterV6 = &ConntrackFilter{}
	filterV6.AddIP(ConntrackOrigDstIP, net.ParseIP("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd"))

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 1 || v6Match != 1 {
		t.Fatalf("Error, there should be only 1 match, v4:%d, v6:%d", v4Match, v6Match)
	}

	// SrcIP for NAT
	filterV4 = &ConntrackFilter{}
	filterV4.AddIP(ConntrackNatSrcIP, net.ParseIP("20.0.0.1"))

	filterV6 = &ConntrackFilter{}
	filterV6.AddIP(ConntrackNatSrcIP, net.ParseIP("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd"))

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 1 || v6Match != 1 {
		t.Fatalf("Error, there should be only 1 match, v4:%d, v6:%d", v4Match, v6Match)
	}

	// DstIP for NAT
	filterV4 = &ConntrackFilter{}
	filterV4.AddIP(ConntrackNatDstIP, net.ParseIP("192.168.1.1"))

	filterV6 = &ConntrackFilter{}
	filterV6.AddIP(ConntrackNatDstIP, net.ParseIP("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd"))

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 2 || v6Match != 0 {
		t.Fatalf("Error, there should be an exact match, v4:%d, v6:%d", v4Match, v6Match)
	}

	// AnyIp for Nat
	filterV4 = &ConntrackFilter{}
	filterV4.AddIP(ConntrackNatAnyIP, net.ParseIP("192.168.1.1"))

	filterV6 = &ConntrackFilter{}
	filterV6.AddIP(ConntrackNatAnyIP, net.ParseIP("eeee:eeee:eeee:eeee:eeee:eeee:eeee:eeee"))

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 2 || v6Match != 1 {
		t.Fatalf("Error, there should be an exact match, v4:%d, v6:%d", v4Match, v6Match)
	}
}
