//go:build linux

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
	"sync"

	"github.com/vishvananda/netlink"
)

// NewFake creates a new FakeInterface
func NewFake() Interface {
	return newConntracker(
		&fakeHandler{
			entries: make([]*netlink.ConntrackFlow, 0),
		},
	)
}

var _ netlinkHandler = (*fakeHandler)(nil)

type fakeHandler struct {
	mu              sync.Mutex
	entries         []*netlink.ConntrackFlow
	netlinkRequests int // try to get the estimated number of netlink request
}

func (f *fakeHandler) ConntrackTableList(_ netlink.ConntrackTableType, _ netlink.InetFamily) ([]*netlink.ConntrackFlow, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.netlinkRequests++
	return f.entries, nil
}

func (f *fakeHandler) ConntrackDeleteFilters(tableType netlink.ConntrackTableType, family netlink.InetFamily, netlinkFilters ...netlink.CustomConntrackFilter) (uint, error) {
	f.mu.Lock()
	defer f.mu.Unlock()

	// 1 netlink request to dump the table
	// https://github.com/vishvananda/netlink/blob/0af32151e72b990c271ef6268e8aadb7e015f2bd/conntrack_linux.go#L163
	f.netlinkRequests++
	var dataplaneFlows []*netlink.ConntrackFlow
	before := len(f.entries)

	for _, flow := range f.entries {
		var matched bool
		for _, filter := range netlinkFilters {
			matched = filter.MatchConntrackFlow(flow)
			if matched {
				// 1 netlink request to delete the flow
				// https://github.com/vishvananda/netlink/blob/0af32151e72b990c271ef6268e8aadb7e015f2bd/conntrack_linux.go#L182
				f.netlinkRequests++
			}
		}
		// no filter matched, keep the flow
		if !matched {
			dataplaneFlows = append(dataplaneFlows, flow)
		}
	}
	f.entries = dataplaneFlows
	return uint(before - len(f.entries)), nil
}
