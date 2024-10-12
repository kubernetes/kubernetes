//go:build linux
// +build linux

/*
Copyright 2024 The Kubernetes Authors.

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
	"testing"

	"github.com/stretchr/testify/require"
	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"

	netutils "k8s.io/utils/net"
)

func applyFilter(flowList []netlink.ConntrackFlow, ipv4Filter *conntrackFilter, ipv6Filter *conntrackFilter) (ipv4Match, ipv6Match int) {
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
	flow1.Forward.SrcIP = netutils.ParseIPSloppy("10.0.0.1")
	flow1.Forward.DstIP = netutils.ParseIPSloppy("20.0.0.1")
	flow1.Forward.SrcPort = 1000
	flow1.Forward.DstPort = 2000
	flow1.Forward.Protocol = 17
	flow1.Reverse.SrcIP = netutils.ParseIPSloppy("20.0.0.1")
	flow1.Reverse.DstIP = netutils.ParseIPSloppy("192.168.1.1")
	flow1.Reverse.SrcPort = 2000
	flow1.Reverse.DstPort = 1000
	flow1.Reverse.Protocol = 17

	flow2 := netlink.ConntrackFlow{}
	flow2.FamilyType = unix.AF_INET
	flow2.Forward.SrcIP = netutils.ParseIPSloppy("10.0.0.2")
	flow2.Forward.DstIP = netutils.ParseIPSloppy("20.0.0.2")
	flow2.Forward.SrcPort = 5000
	flow2.Forward.DstPort = 6000
	flow2.Forward.Protocol = 6
	flow2.Reverse.SrcIP = netutils.ParseIPSloppy("20.0.0.2")
	flow2.Reverse.DstIP = netutils.ParseIPSloppy("192.168.1.1")
	flow2.Reverse.SrcPort = 6000
	flow2.Reverse.DstPort = 5000
	flow2.Reverse.Protocol = 6

	flow3 := netlink.ConntrackFlow{}
	flow3.FamilyType = unix.AF_INET6
	flow3.Forward.SrcIP = netutils.ParseIPSloppy("eeee:eeee:eeee:eeee:eeee:eeee:eeee:eeee")
	flow3.Forward.DstIP = netutils.ParseIPSloppy("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd")
	flow3.Forward.SrcPort = 1000
	flow3.Forward.DstPort = 2000
	flow3.Forward.Protocol = 132
	flow3.Reverse.SrcIP = netutils.ParseIPSloppy("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd")
	flow3.Reverse.DstIP = netutils.ParseIPSloppy("eeee:eeee:eeee:eeee:eeee:eeee:eeee:eeee")
	flow3.Reverse.SrcPort = 2000
	flow3.Reverse.DstPort = 1000
	flow3.Reverse.Protocol = 132
	flowList = append(flowList, flow1, flow2, flow3)

	testCases := []struct {
		name              string
		filterV4          *conntrackFilter
		filterV6          *conntrackFilter
		expectedV4Matches int
		expectedV6Matches int
	}{
		{
			name:              "Empty filter",
			filterV4:          &conntrackFilter{},
			filterV6:          &conntrackFilter{},
			expectedV4Matches: 0,
			expectedV6Matches: 0,
		},
		{
			name:              "Protocol filter",
			filterV4:          &conntrackFilter{protocol: 6},
			filterV6:          &conntrackFilter{protocol: 17},
			expectedV4Matches: 1,
			expectedV6Matches: 1,
		},
		{
			name:              "Original Source IP filter",
			filterV4:          &conntrackFilter{original: &connectionTuple{srcIP: netutils.ParseIPSloppy("10.0.0.1")}},
			filterV6:          &conntrackFilter{original: &connectionTuple{srcIP: netutils.ParseIPSloppy("eeee:eeee:eeee:eeee:eeee:eeee:eeee:eeee")}},
			expectedV4Matches: 1,
			expectedV6Matches: 1,
		},
		{
			name:              "Original Destination IP filter",
			filterV4:          &conntrackFilter{original: &connectionTuple{dstIP: netutils.ParseIPSloppy("20.0.0.1")}},
			filterV6:          &conntrackFilter{original: &connectionTuple{dstIP: netutils.ParseIPSloppy("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd")}},
			expectedV4Matches: 1,
			expectedV6Matches: 1,
		},
		{
			name:              "Original Source Port Filter",
			filterV4:          &conntrackFilter{protocol: 6, original: &connectionTuple{srcPort: 5000}},
			filterV6:          &conntrackFilter{protocol: 132, original: &connectionTuple{srcPort: 1000}},
			expectedV4Matches: 1,
			expectedV6Matches: 1,
		},
		{
			name:              "Original Destination Port Filter",
			filterV4:          &conntrackFilter{protocol: 6, original: &connectionTuple{dstPort: 6000}},
			filterV6:          &conntrackFilter{protocol: 132, original: &connectionTuple{dstPort: 2000}},
			expectedV4Matches: 1,
			expectedV6Matches: 1,
		},
		{
			name:              "Reply Source IP filter",
			filterV4:          &conntrackFilter{reply: &connectionTuple{srcIP: netutils.ParseIPSloppy("20.0.0.1")}},
			filterV6:          &conntrackFilter{reply: &connectionTuple{srcIP: netutils.ParseIPSloppy("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd")}},
			expectedV4Matches: 1,
			expectedV6Matches: 1,
		},
		{
			name:              "Reply Destination IP filter",
			filterV4:          &conntrackFilter{reply: &connectionTuple{dstIP: netutils.ParseIPSloppy("192.168.1.1")}},
			filterV6:          &conntrackFilter{reply: &connectionTuple{dstIP: netutils.ParseIPSloppy("dddd:dddd:dddd:dddd:dddd:dddd:dddd:dddd")}},
			expectedV4Matches: 2,
			expectedV6Matches: 0,
		},
		{
			name:              "Reply Source Port filter",
			filterV4:          &conntrackFilter{protocol: 17, reply: &connectionTuple{srcPort: 2000}},
			filterV6:          &conntrackFilter{protocol: 132, reply: &connectionTuple{srcPort: 2000}},
			expectedV4Matches: 1,
			expectedV6Matches: 1,
		},
		{
			name:              "Reply Destination Port filter",
			filterV4:          &conntrackFilter{protocol: 6, reply: &connectionTuple{dstPort: 5000}},
			filterV6:          &conntrackFilter{protocol: 132, reply: &connectionTuple{dstPort: 1000}},
			expectedV4Matches: 1,
			expectedV6Matches: 1,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			v4Matches, v6Matches := applyFilter(flowList, tc.filterV4, tc.filterV6)
			require.Equal(t, tc.expectedV4Matches, v4Matches)
			require.Equal(t, tc.expectedV6Matches, v6Matches)
		})
	}
}
