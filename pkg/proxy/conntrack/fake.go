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

	"github.com/vishvananda/netlink"

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

// ClearEntries is part of Interface
func (fake *FakeInterface) ClearEntries(_ uint8, filters ...netlink.CustomConntrackFilter) error {
	for _, anyFilter := range filters {
		filter := anyFilter.(*conntrackFilter)
		if filter.protocol != protocolMap[v1.ProtocolUDP] {
			return fmt.Errorf("FakeInterface currently only supports UDP")
		}

		// record IP and Port entries
		if filter.original != nil && filter.reply == nil {
			if filter.original.dstIP != nil {
				fake.ClearedIPs.Insert(filter.original.dstIP.String())
			}
			if filter.original.dstPort != 0 {
				fake.ClearedPorts.Insert(int(filter.original.dstPort))
			}
		}

		// record NAT and NATPort entries
		if filter.original != nil && filter.reply != nil {
			if filter.original.dstIP != nil && filter.reply.srcIP != nil {
				origin := filter.original.dstIP.String()
				dest := filter.reply.srcIP.String()
				if previous, exists := fake.ClearedNATs[origin]; exists && previous != dest {
					return fmt.Errorf("filter for NAT passed with same origin (%s), different destination (%s / %s)", origin, previous, dest)
				}
				fake.ClearedNATs[filter.original.dstIP.String()] = filter.reply.srcIP.String()
			}

			if filter.original.dstPort != 0 && filter.reply.srcIP != nil {
				dest := filter.reply.srcIP.String()
				port := int(filter.original.dstPort)
				if previous, exists := fake.ClearedPortNATs[port]; exists && previous != dest {
					return fmt.Errorf("filter for PortNAT passed with same port (%d), different destination (%s / %s)", port, previous, dest)
				}
				fake.ClearedPortNATs[port] = dest
			}
		}
	}
	return nil
}
