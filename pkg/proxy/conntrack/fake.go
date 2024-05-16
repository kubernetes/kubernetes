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
	"fmt"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

// FakeInterface implements Interface by just recording entries that have been cleared.
type FakeInterface struct {
	ClearedIPs      sets.Set[string]
	ClearedPorts    sets.Set[int]
	ClearedNATs     map[string]string // origin -> dest
	ClearedPortNATs map[int]string    // port -> dest
}

var _ Interface = &FakeInterface{}

// NewFake creates a new FakeInterface
func NewFake() *FakeInterface {
	fake := &FakeInterface{}
	fake.Reset()
	return fake
}

// Reset clears fake's sets/maps
func (fake *FakeInterface) Reset() {
	fake.ClearedIPs = sets.New[string]()
	fake.ClearedPorts = sets.New[int]()
	fake.ClearedNATs = make(map[string]string)
	fake.ClearedPortNATs = make(map[int]string)
}

// ClearEntriesForIP is part of Interface
func (fake *FakeInterface) ClearEntriesForIP(ip string, protocol v1.Protocol) error {
	if protocol != v1.ProtocolUDP {
		return fmt.Errorf("FakeInterface currently only supports UDP")
	}

	fake.ClearedIPs.Insert(ip)
	return nil
}

// ClearEntriesForPort is part of Interface
func (fake *FakeInterface) ClearEntriesForPort(port int, isIPv6 bool, protocol v1.Protocol) error {
	if protocol != v1.ProtocolUDP {
		return fmt.Errorf("FakeInterface currently only supports UDP")
	}

	fake.ClearedPorts.Insert(port)
	return nil
}

// ClearEntriesForNAT is part of Interface
func (fake *FakeInterface) ClearEntriesForNAT(origin, dest string, protocol v1.Protocol) error {
	if protocol != v1.ProtocolUDP {
		return fmt.Errorf("FakeInterface currently only supports UDP")
	}
	if previous, exists := fake.ClearedNATs[origin]; exists && previous != dest {
		return fmt.Errorf("ClearEntriesForNAT called with same origin (%s), different destination (%s / %s)", origin, previous, dest)
	}

	fake.ClearedNATs[origin] = dest
	return nil
}

// ClearEntriesForPortNAT is part of Interface
func (fake *FakeInterface) ClearEntriesForPortNAT(dest string, port int, protocol v1.Protocol) error {
	if protocol != v1.ProtocolUDP {
		return fmt.Errorf("FakeInterface currently only supports UDP")
	}
	if previous, exists := fake.ClearedPortNATs[port]; exists && previous != dest {
		return fmt.Errorf("ClearEntriesForPortNAT called with same port (%d), different destination (%s / %s)", port, previous, dest)
	}

	fake.ClearedPortNATs[port] = dest
	return nil
}
