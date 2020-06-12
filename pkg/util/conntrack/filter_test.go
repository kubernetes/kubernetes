/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package conntrack

import (
	"net"
	"testing"

	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"
)

func applyFilter(flowList []netlink.ConntrackFlow, ipv4Filter *conntrackFilter, ipv6Filter *conntrackFilter) (ipv4Match, ipv6Match uint) {
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

func TestConntrackFilter(t *testing.T) {
	var flowList []netlink.ConntrackFlow

	flow1 := netlink.ConntrackFlow{}
	flow1.FamilyType = unix.AF_INET
	flow1.Forward.SrcIP = net.ParseIP("10.0.0.1")
	flow1.Forward.DstIP = net.ParseIP("20.0.0.1")
	flow1.Forward.SrcPort = 1000
	flow1.Forward.DstPort = 2000
	flow1.Forward.Protocol = 17
	flow1.Reverse.SrcIP = net.ParseIP("20.0.0.1")
	flow1.Reverse.DstIP = net.ParseIP("192.168.1.1")
	flow1.Reverse.SrcPort = 2000
	flow1.Reverse.DstPort = 1000
	flow1.Reverse.Protocol = 17

	flow2 := netlink.ConntrackFlow{}
	flow2.FamilyType = unix.AF_INET
	flow2.Forward.SrcIP = net.ParseIP("10.0.0.2")
	flow2.Forward.DstIP = net.ParseIP("20.0.0.2")
	flow2.Forward.SrcPort = 5000
	flow2.Forward.DstPort = 6000
	flow2.Forward.Protocol = 6
	flow2.Reverse.SrcIP = net.ParseIP("20.0.0.2")
	flow2.Reverse.DstIP = net.ParseIP("192.168.1.1")
	flow2.Reverse.SrcPort = 6000
	flow2.Reverse.DstPort = 5000
	flow2.Reverse.Protocol = 6

	flow3 := netlink.ConntrackFlow{}
	flow3.FamilyType = unix.AF_INET
	flow3.Forward.SrcIP = net.ParseIP("eeee:eeee:eeee:eeee:eeee:eeee:eeee:eeee")
	flow3.Forward.DstIP = net.ParseIP("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd")
	flow3.Forward.SrcPort = 1000
	flow3.Forward.DstPort = 2000
	flow3.Forward.Protocol = 132
	flow3.Reverse.SrcIP = net.ParseIP("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd")
	flow3.Reverse.DstIP = net.ParseIP("eeee:eeee:eeee:eeee:eeee:eeee:eeee:eeee")
	flow3.Reverse.SrcPort = 2000
	flow3.Reverse.DstPort = 1000
	flow3.Reverse.Protocol = 132

	flowList = append(flowList, flow1, flow2, flow3)

	// Empty filter
	v4Match, v6Match := applyFilter(flowList, &conntrackFilter{}, &conntrackFilter{})
	if v4Match > 0 || v6Match > 0 {
		t.Fatalf("Error, empty filter cannot match, v4:%d, v6:%d", v4Match, v6Match)
	}

	// Filter errors

	// Adding same attribute should fail
	filter := &conntrackFilter{}
	filter.addIP(conntrackOrigSrcIP, net.ParseIP("10.0.0.1"))
	if err := filter.addIP(conntrackOrigSrcIP, net.ParseIP("10.0.0.1")); err == nil {
		t.Fatalf("Error, it should fail adding same attribute to the filter")
	}
	filter.addProtocol(6)
	if err := filter.addProtocol(17); err == nil {
		t.Fatalf("Error, it should fail adding same attribute to the filter")
	}
	filter.addPort(conntrackOrigSrcPort, 80)
	if err := filter.addPort(conntrackOrigSrcPort, 80); err == nil {
		t.Fatalf("Error, it should fail adding same attribute to the filter")
	}

	// Can not add a Port filter without Layer 4 protocol
	filter = &conntrackFilter{}
	if err := filter.addPort(conntrackOrigSrcPort, 80); err == nil {
		t.Fatalf("Error, it should fail adding a port filter without a protocol")
	}

	// Can not add a Port filter if the Layer 4 protocol does not support it
	filter = &conntrackFilter{}
	filter.addProtocol(47)
	if err := filter.addPort(conntrackOrigSrcPort, 80); err == nil {
		t.Fatalf("Error, it should fail adding a port filter with a wrong protocol")
	}

	// Proto filter
	filterV4 := &conntrackFilter{}
	filterV4.addProtocol(6)

	filterV6 := &conntrackFilter{}
	filterV6.addProtocol(132)

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 1 || v6Match != 1 {
		t.Fatalf("Error, there should be only 1 match for TCP:%d, UDP:%d", v4Match, v6Match)
	}

	// SrcIP filter
	filterV4 = &conntrackFilter{}
	filterV4.addIP(conntrackOrigSrcIP, net.ParseIP("10.0.0.1"))

	filterV6 = &conntrackFilter{}
	filterV6.addIP(conntrackOrigSrcIP, net.ParseIP("eeee:eeee:eeee:eeee:eeee:eeee:eeee:eeee"))

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 1 || v6Match != 1 {
		t.Fatalf("Error, there should be only 1 match, v4:%d, v6:%d", v4Match, v6Match)
	}

	// DstIp filter
	filterV4 = &conntrackFilter{}
	filterV4.addIP(conntrackOrigDstIP, net.ParseIP("20.0.0.1"))

	filterV6 = &conntrackFilter{}
	filterV6.addIP(conntrackOrigDstIP, net.ParseIP("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd"))

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 1 || v6Match != 1 {
		t.Fatalf("Error, there should be only 1 match, v4:%d, v6:%d", v4Match, v6Match)
	}

	// SrcIP for NAT
	filterV4 = &conntrackFilter{}
	filterV4.addIP(conntrackReplySrcIP, net.ParseIP("20.0.0.1"))

	filterV6 = &conntrackFilter{}
	filterV6.addIP(conntrackReplySrcIP, net.ParseIP("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd"))

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 1 || v6Match != 1 {
		t.Fatalf("Error, there should be only 1 match, v4:%d, v6:%d", v4Match, v6Match)
	}

	// DstIP for NAT
	filterV4 = &conntrackFilter{}
	filterV4.addIP(conntrackReplyDstIP, net.ParseIP("192.168.1.1"))

	filterV6 = &conntrackFilter{}
	filterV6.addIP(conntrackReplyDstIP, net.ParseIP("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd"))

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 2 || v6Match != 0 {
		t.Fatalf("Error, there should be an exact match, v4:%d, v6:%d", v4Match, v6Match)
	}

	// AnyIp for Nat
	filterV4 = &conntrackFilter{}
	filterV4.addIP(conntrackReplyAnyIP, net.ParseIP("192.168.1.1"))

	filterV6 = &conntrackFilter{}
	filterV6.addIP(conntrackReplyAnyIP, net.ParseIP("eeee:eeee:eeee:eeee:eeee:eeee:eeee:eeee"))

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 2 || v6Match != 1 {
		t.Fatalf("Error, there should be an exact match, v4:%d, v6:%d", v4Match, v6Match)
	}

	// SrcPort filter
	filterV4 = &conntrackFilter{}
	filterV4.addProtocol(6)
	filterV4.addPort(conntrackOrigSrcPort, 5000)

	filterV6 = &conntrackFilter{}
	filterV6.addProtocol(132)
	filterV6.addPort(conntrackOrigSrcPort, 1000)

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 1 || v6Match != 1 {
		t.Fatalf("Error, there should be only 1 match, v4:%d, v6:%d", v4Match, v6Match)
	}

	// DstPort filter
	filterV4 = &conntrackFilter{}
	filterV4.addProtocol(6)
	filterV4.addPort(conntrackOrigDstPort, 6000)

	filterV6 = &conntrackFilter{}
	filterV6.addProtocol(132)
	filterV6.addPort(conntrackOrigDstPort, 2000)

	v4Match, v6Match = applyFilter(flowList, filterV4, filterV6)
	if v4Match != 1 || v6Match != 1 {
		t.Fatalf("Error, there should be only 1 match, v4:%d, v6:%d", v4Match, v6Match)
	}
}
