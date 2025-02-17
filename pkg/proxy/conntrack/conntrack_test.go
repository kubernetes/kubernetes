//go:build linux
// +build linux

/*
Copyright 2015 The Kubernetes Authors.

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

func TestConntracker_ClearEntries(t *testing.T) {
	testCases := []struct {
		name     string
		ipFamily uint8
		filters  []netlink.CustomConntrackFilter
		flows    []*netlink.ConntrackFlow
	}{
		{
			name:     "single IPv6 filter",
			ipFamily: unix.AF_INET6,
			filters: []netlink.CustomConntrackFilter{
				&conntrackFilter{
					protocol: 17,
					original: &connectionTuple{dstPort: 8000},
					reply:    &connectionTuple{srcIP: netutils.ParseIPSloppy("2001:db8:1::2")},
				},
			},
			flows: []*netlink.ConntrackFlow{
				{
					FamilyType: unix.AF_INET6,
					Forward: netlink.IPTuple{
						DstIP:    netutils.ParseIPSloppy("2001:db8:10::20"),
						DstPort:  8000,
						Protocol: unix.IPPROTO_UDP,
					},
					Reverse: netlink.IPTuple{
						Protocol: unix.IPPROTO_UDP,
						SrcIP:    netutils.ParseIPSloppy("2001:db8:1::2"),
						SrcPort:  54321,
					},
				},
			},
		},
		{
			name:     "multiple IPv4 filters",
			ipFamily: unix.AF_INET,
			filters: []netlink.CustomConntrackFilter{
				&conntrackFilter{
					protocol: 6,
					original: &connectionTuple{dstPort: 3000},
				},
				&conntrackFilter{
					protocol: 17,
					original: &connectionTuple{dstPort: 5000},
					reply:    &connectionTuple{srcIP: netutils.ParseIPSloppy("10.244.0.3")},
				},
				&conntrackFilter{
					protocol: 132,
					original: &connectionTuple{dstIP: netutils.ParseIPSloppy("10.96.0.10")},
					reply:    &connectionTuple{srcIP: netutils.ParseIPSloppy("10.244.0.3")},
				},
			},
			flows: []*netlink.ConntrackFlow{
				{
					FamilyType: unix.AF_INET,
					Forward: netlink.IPTuple{
						DstPort:  3000,
						DstIP:    netutils.ParseIPSloppy("10.96.0.1"),
						Protocol: unix.IPPROTO_TCP,
					},
					Reverse: netlink.IPTuple{
						Protocol: unix.IPPROTO_TCP,
						SrcIP:    netutils.ParseIPSloppy("10.96.0.10"),
						SrcPort:  54321,
					},
				},
				{
					FamilyType: unix.AF_INET,
					Forward: netlink.IPTuple{
						DstPort:  5000,
						DstIP:    netutils.ParseIPSloppy("10.96.0.1"),
						Protocol: unix.IPPROTO_UDP,
					},
					Reverse: netlink.IPTuple{
						Protocol: unix.IPPROTO_UDP,
						SrcIP:    netutils.ParseIPSloppy("10.244.0.3"),
						SrcPort:  54321,
					},
				},
				{
					FamilyType: unix.AF_INET,
					Forward: netlink.IPTuple{
						DstIP:    netutils.ParseIPSloppy("10.96.0.10"),
						DstPort:  5000,
						Protocol: unix.IPPROTO_SCTP,
					},
					Reverse: netlink.IPTuple{
						Protocol: unix.IPPROTO_SCTP,
						SrcIP:    netutils.ParseIPSloppy("10.244.0.3"),
						SrcPort:  54321,
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			handler := &fakeHandler{}
			handler.entries = tc.flows
			handler.deleteErrors = []error{unix.EINTR, nil}
			ct := newConntracker(handler)
			_, err := ct.ClearEntries(tc.ipFamily, tc.filters...)
			require.NoError(t, err)
			require.Equal(t, netlink.ConntrackTableType(netlink.ConntrackTable), handler.tableType)
			require.Equal(t, netlink.InetFamily(tc.ipFamily), handler.ipFamily)
			require.Equal(t, len(tc.filters), len(handler.filters))
			for i := 0; i < len(tc.filters); i++ {
				require.Equal(t, tc.filters[i], handler.filters[i])
			}
			require.Empty(t, len(handler.entries))
		})
	}
}
