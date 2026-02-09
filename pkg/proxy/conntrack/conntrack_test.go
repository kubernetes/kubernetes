//go:build linux

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

	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"

	netutils "k8s.io/utils/net"
)

func TestConntracker_DeleteEntries(t *testing.T) {
	// Define some common flows
	flow1 := &netlink.ConntrackFlow{
		Forward: netlink.IPTuple{
			SrcIP:    netutils.ParseIPSloppy("1.1.1.1"),
			DstIP:    netutils.ParseIPSloppy("2.2.2.2"),
			SrcPort:  1000,
			DstPort:  2000,
			Protocol: 6,
		},
		Reverse: netlink.IPTuple{
			SrcIP:    netutils.ParseIPSloppy("2.2.2.2"),
			DstIP:    netutils.ParseIPSloppy("1.1.1.1"),
			SrcPort:  2000,
			DstPort:  1000,
			Protocol: 6,
		},
	}
	flow2 := &netlink.ConntrackFlow{
		Forward: netlink.IPTuple{
			SrcIP:    netutils.ParseIPSloppy("3.3.3.3"),
			DstIP:    netutils.ParseIPSloppy("4.4.4.4"),
			SrcPort:  3000,
			DstPort:  4000,
			Protocol: 17,
		},
		Reverse: netlink.IPTuple{
			SrcIP:    netutils.ParseIPSloppy("4.4.4.4"),
			DstIP:    netutils.ParseIPSloppy("3.3.3.3"),
			SrcPort:  4000,
			DstPort:  3000,
			Protocol: 17,
		},
	}
	flow3 := &netlink.ConntrackFlow{
		Forward: netlink.IPTuple{
			SrcIP:    netutils.ParseIPSloppy("5.5.5.5"),
			DstIP:    netutils.ParseIPSloppy("6.6.6.6"),
			SrcPort:  5000,
			DstPort:  6000,
			Protocol: 132,
		},
		Reverse: netlink.IPTuple{
			SrcIP:    netutils.ParseIPSloppy("6.6.6.6"),
			DstIP:    netutils.ParseIPSloppy("5.5.5.5"),
			SrcPort:  6000,
			DstPort:  5000,
			Protocol: 132,
		},
	}
	// Edge case: Partial flow (missing some fields)
	flowEmpty := &netlink.ConntrackFlow{
		Forward: netlink.IPTuple{
			Protocol: 6,
		},
	}

	testCases := []struct {
		name            string
		initialEntries  []*netlink.ConntrackFlow
		deleteFlows     []*netlink.ConntrackFlow
		expectedDeleted int
		expectedLeft    int
	}{
		{
			name:            "Delete single flow",
			initialEntries:  []*netlink.ConntrackFlow{flow1, flow2, flow3},
			deleteFlows:     []*netlink.ConntrackFlow{flow1},
			expectedDeleted: 1,
			expectedLeft:    2,
		},
		{
			name:            "Delete multiple flows",
			initialEntries:  []*netlink.ConntrackFlow{flow1, flow2, flow3},
			deleteFlows:     []*netlink.ConntrackFlow{flow1, flow3},
			expectedDeleted: 2,
			expectedLeft:    1,
		},
		{
			name:            "Delete non-existent flow",
			initialEntries:  []*netlink.ConntrackFlow{flow1, flow2},
			deleteFlows:     []*netlink.ConntrackFlow{flow3},
			expectedDeleted: 0,
			expectedLeft:    2,
		},
		{
			name:            "Delete flow with empty fields (should match nothing)",
			initialEntries:  []*netlink.ConntrackFlow{flow1, flow2, flow3},
			deleteFlows:     []*netlink.ConntrackFlow{flowEmpty},
			expectedDeleted: 0,
			expectedLeft:    3,
		},
		{
			name:            "Delete all flows",
			initialEntries:  []*netlink.ConntrackFlow{flow1, flow2, flow3},
			deleteFlows:     []*netlink.ConntrackFlow{flow1, flow2, flow3},
			expectedDeleted: 3,
			expectedLeft:    0,
		},
		{
			name:            "Delete with empty list",
			initialEntries:  []*netlink.ConntrackFlow{flow1, flow2},
			deleteFlows:     []*netlink.ConntrackFlow{},
			expectedDeleted: 0,
			expectedLeft:    2,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			// avoid mutations of entries between tests
			entries := make([]*netlink.ConntrackFlow, len(tc.initialEntries))
			copy(entries, tc.initialEntries)

			handler := &fakeHandler{
				entries: entries,
			}
			ct := newConntracker(handler)

			n, err := ct.DeleteEntries(unix.AF_INET, tc.deleteFlows)
			if err != nil {
				t.Fatalf("DeleteEntries() error = %v, want nil", err)
			}

			if n != tc.expectedDeleted {
				t.Errorf("DeleteEntries() = %d, want %d", n, tc.expectedDeleted)
			}

			left, err := ct.ListEntries(unix.AF_INET)
			if err != nil {
				t.Fatalf("ListEntries() error = %v", err)
			}
			if len(left) != tc.expectedLeft {
				t.Errorf("ListEntries() left = %d, want %d", len(left), tc.expectedLeft)
			}
		})
	}
}
