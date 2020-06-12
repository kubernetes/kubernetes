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

// FakeConntrack is no-op implementation of conntrack Clearer.
type FakeConntrack struct {
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

var _ = conntrack.Clearer(&FakeConntrack{})

// NewFakeClearer returns a no-op conntrack.Clearer
func NewFakeClearer() *FakeConntrack {
	return &FakeConntrack{}
}

// Exists returns true if conntrack binary is installed.
func (f FakeConntrack) Exists() bool {
	return true
}

// ClearEntriesForIP delete conntrack entries by the destination IP and protocol
func (f FakeConntrack) ClearEntriesForIP(ip string, protocol v1.Protocol) error {
	if protocol == f.Proto && ip == f.OrigDstIP {
		return nil
	}
	return fmt.Errorf(conntrack.NoConnectionToDelete)
}

// ClearEntriesForPort delete conntrack entries by the destination Port and protocol
func (f FakeConntrack) ClearEntriesForPort(port int, isIPv6 bool, protocol v1.Protocol) error {
	if protocol == f.Proto && port == f.OrigDstPort {
		return nil
	}
	return fmt.Errorf(conntrack.NoConnectionToDelete)
}

// ClearEntriesForNAT delete conntrack entries by the NAT source and destination IP and protocol
func (f FakeConntrack) ClearEntriesForNAT(origin, dest string, protocol v1.Protocol) error {
	if protocol == f.Proto &&
		origin == f.OrigDstIP &&
		dest == f.NatDstIP {
		return nil
	}
	return fmt.Errorf(conntrack.NoConnectionToDelete)
}

// ClearEntriesForPortNAT delete conntrack entries by the NAT destination IP and Port and protocol
func (f FakeConntrack) ClearEntriesForPortNAT(dest string, port int, protocol v1.Protocol) error {
	if protocol == f.Proto &&
		port == f.OrigDstPort &&
		dest == f.NatDstIP {
		return nil
	}
	return fmt.Errorf(conntrack.NoConnectionToDelete)
}
