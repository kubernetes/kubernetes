/*
Copyright 2020 The Kubernetes Authors.

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

package testing

import (
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/util/conntrack"
)

// NetworkFlow defines a network flow in the conntrack table
type NetworkFlow struct {
	OrigSrcIP   string
	OrigSrcPort int
	OrigDstIP   string
	OrigDstPort int
	NatSrcIP    string
	NatSrcPort  int
	NatDstIP    string
	NatDstPort  int
	Proto       v1.Protocol
}

// FakeConntrack is no-op implementation of conntrack Clearer.
// it stores the conntrack entries in a table internally
// and stores the results of the operations performed on the
// internal table
type FakeConntrack struct {
	ConntrackTable []NetworkFlow
	DeletedFlows   int
}

var _ = conntrack.Clearer(&FakeConntrack{})

// NewFakeClearer returns a no-op conntrack.Clearer
func NewFakeClearer() *FakeConntrack {
	return &FakeConntrack{}
}

// Exists returns true if conntrack binary is installed.
func (f *FakeConntrack) Exists() bool {
	return true
}

// ClearEntriesForIP delete conntrack entries by the destination IP and protocol
func (f *FakeConntrack) ClearEntriesForIP(ip string, protocol v1.Protocol) error {
	filter := NetworkFlow{
		Proto:     protocol,
		OrigDstIP: ip,
	}

	f.DeletedFlows = f.filter(filter)
	if f.DeletedFlows == 0 {
		return fmt.Errorf(conntrack.NoConnectionToDelete)
	}
	return nil
}

// ClearEntriesForPort delete conntrack entries by the destination Port and protocol
func (f *FakeConntrack) ClearEntriesForPort(port int, isIPv6 bool, protocol v1.Protocol) error {
	filter := NetworkFlow{
		Proto:       protocol,
		OrigDstPort: port,
	}

	f.DeletedFlows = f.filter(filter)
	if f.DeletedFlows == 0 {
		return fmt.Errorf(conntrack.NoConnectionToDelete)
	}
	return nil
}

// ClearEntriesForNAT delete conntrack entries by the NAT source and destination IP and protocol
func (f *FakeConntrack) ClearEntriesForNAT(origin, dest string, protocol v1.Protocol) error {
	filter := NetworkFlow{
		Proto:     protocol,
		NatDstIP:  dest,
		OrigDstIP: origin,
	}

	f.DeletedFlows = f.filter(filter)
	if f.DeletedFlows == 0 {
		return fmt.Errorf(conntrack.NoConnectionToDelete)
	}
	return nil
}

// ClearEntriesForPortNAT delete conntrack entries by the NAT destination IP and Port and protocol
func (f *FakeConntrack) ClearEntriesForPortNAT(dest string, port int, protocol v1.Protocol) error {
	filter := NetworkFlow{
		Proto:       protocol,
		NatDstIP:    dest,
		OrigDstPort: port,
	}

	f.DeletedFlows = f.filter(filter)
	if f.DeletedFlows == 0 {
		return fmt.Errorf(conntrack.NoConnectionToDelete)
	}
	return nil
}

// filter returns the number of flows that should be filtered
// by the passed filter
func (f *FakeConntrack) filter(flowFilter NetworkFlow) int {
	count := 0
	for _, flow := range f.ConntrackTable {
		if matchConntrackFlow(flow, flowFilter) {
			count++
		}
	}
	return count
}

// matchConntrackFlow return true if the flow match the filter fields
func matchConntrackFlow(flow, flowFilter NetworkFlow) bool {
	if flowFilter.Proto != flow.Proto {
		// different Layer 4 protocol always not match
		return false
	}

	if flowFilter.OrigSrcIP != "" && flowFilter.OrigSrcIP != flow.OrigSrcIP {
		return false
	}

	if flowFilter.OrigDstIP != "" && flowFilter.OrigDstIP != flow.OrigDstIP {
		return false
	}

	if flowFilter.NatSrcIP != "" && flowFilter.NatSrcIP != flow.NatSrcIP {
		return false
	}

	if flowFilter.NatDstIP != "" && flowFilter.NatDstIP != flow.NatDstIP {
		return false
	}

	if flowFilter.OrigSrcPort > 0 && flowFilter.OrigSrcPort != flow.OrigSrcPort {
		return false
	}

	if flowFilter.OrigDstPort > 0 && flowFilter.OrigDstPort != flow.OrigDstPort {
		return false
	}

	if flowFilter.NatSrcPort > 0 && flowFilter.NatSrcPort != flow.NatSrcPort {
		return false
	}

	if flowFilter.NatDstPort > 0 && flowFilter.NatDstPort != flow.NatDstPort {
		return false
	}

	return true
}
