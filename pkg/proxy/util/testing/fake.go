/*
Copyright 2018 The Kubernetes Authors.

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

import "net"

// FakeNetwork implements the NetworkInterfacer interface for test purpose.
type FakeNetwork struct {
	NetworkInterfaces []net.Interface
	// The key of map Addrs is the network interface name
	Address map[string][]net.Addr
}

// NewFakeNetwork initializes a FakeNetwork.
func NewFakeNetwork() *FakeNetwork {
	return &FakeNetwork{
		NetworkInterfaces: make([]net.Interface, 0),
		Address:           make(map[string][]net.Addr),
	}
}

// AddInterfaceAddr create an interface and its associated addresses for FakeNetwork implementation.
func (f *FakeNetwork) AddInterfaceAddr(intf *net.Interface, addrs []net.Addr) {
	f.NetworkInterfaces = append(f.NetworkInterfaces, *intf)
	f.Address[intf.Name] = addrs
}

// InterfaceAddrs is part of NetworkInterfacer interface.
func (f *FakeNetwork) InterfaceAddrs() ([]net.Addr, error) {
	addrs := make([]net.Addr, 0)
	for _, value := range f.Address {
		addrs = append(addrs, value...)
	}
	return addrs, nil
}
