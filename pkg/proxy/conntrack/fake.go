//go:build linux
// +build linux

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
	"errors"
	"math/rand/v2"

	"github.com/vishvananda/netlink"
	"golang.org/x/sys/unix"
)

type fakeHandler struct {
	tableType netlink.ConntrackTableType
	ipFamily  netlink.InetFamily
	filters   []*conntrackFilter

	entries         []*netlink.ConntrackFlow
	tableListErrors []error
	deleteErrors    []error

	// callsConntrackTableList is a counter that gets incremented each time ConntrackTableList is called
	callsConntrackTableList int
	// callsConntrackDeleteFilters is a counter that gets incremented each time ConntrackDeleteFilters is called
	callsConntrackDeleteFilters int
}

// ConntrackTableList is part of netlinkHandler interface.
func (fake *fakeHandler) ConntrackTableList(_ netlink.ConntrackTableType, _ netlink.InetFamily) ([]*netlink.ConntrackFlow, error) {
	// special case for consumers outside of this package (fake-proxiers) where cleanup is called
	// on fakeHandler without any error mocking.
	if len(fake.tableListErrors) == 0 {
		return fake.entries, nil
	}

	calls := fake.callsConntrackTableList
	fake.callsConntrackTableList++

	// we panic when ConntrackTableList() is called more times than the length of configured
	// tableListErrors. This indicates improper configuration of fakeHandler for the test.
	if calls > len(fake.tableListErrors)-1 {
		panic("exceeded the allowed number of ConntrackTableList calls")
	}

	var err = fake.tableListErrors[calls]
	var flows []*netlink.ConntrackFlow

	// return all flow entries if no error, a random subset if interrupted (EINTR), or no entries
	// for any other error condition.
	if errors.Is(err, unix.EINTR) {
		flows = fake.entries[:rand.IntN(len(fake.entries))]
	} else if err == nil {
		flows = fake.entries
	}
	return flows, err
}

// ConntrackDeleteFilters is part of netlinkHandler interface.
func (fake *fakeHandler) ConntrackDeleteFilters(tableType netlink.ConntrackTableType, family netlink.InetFamily, netlinkFilters ...netlink.CustomConntrackFilter) (uint, error) {
	fake.tableType = tableType
	fake.ipFamily = family
	fake.filters = make([]*conntrackFilter, 0, len(netlinkFilters))
	for _, netlinkFilter := range netlinkFilters {
		fake.filters = append(fake.filters, netlinkFilter.(*conntrackFilter))
	}

	// flows is the total flows to be considered for deletion
	var flows []*netlink.ConntrackFlow
	// remainingFlows is the total flows remaining after simulating the deletion
	var remainingFlows []*netlink.ConntrackFlow

	var err error

	calls := fake.callsConntrackDeleteFilters
	fake.callsConntrackDeleteFilters++

	// we ignore the cases when deleteErrors is not configured at all, this will happen when called
	// from outside the current package (fake-proxiers), we don't simulate any error in this case.
	if len(fake.deleteErrors) > 0 {
		// we panic when ConntrackDeleteFilters() is called more times than the length of configured
		// deleteErrors. This indicates improper configuration of fakeHandler for the test.
		if calls > len(fake.deleteErrors)-1 {
			panic("exceeded the allowed number of ConntrackDeleteFilters calls")
		} else {
			err = fake.deleteErrors[calls]
		}
	}

	// the actual implementation netlink.ConntrackDeleteFilters does a dump call internally
	// before deleting the flow entries, which can return a partial list of entries. We simulate
	// the same behaviour here in our fake implementation.
	if errors.Is(err, unix.EINTR) {
		randInt := rand.IntN(len(fake.entries))
		// simulate deletion on partial list in case of interrupt
		flows = fake.entries[:randInt]
		remainingFlows = fake.entries[randInt:]
	} else if err != nil {
		remainingFlows = fake.entries
	} else {
		// simulate deletion on complete list of flows
		flows = fake.entries
	}

	before := len(flows)
	for _, flow := range flows {
		var matched bool
		for _, filter := range fake.filters {
			matched = filter.MatchConntrackFlow(flow)
			if matched {
				break
			}
		}
		if !matched {
			remainingFlows = append(remainingFlows, flow)
		}
	}
	fake.entries = remainingFlows
	return uint(before - len(fake.entries)), err
}

var _ netlinkHandler = (*fakeHandler)(nil)

// NewFake creates a new Interface
func NewFake() Interface {
	return &conntracker{handler: &fakeHandler{}}
}
