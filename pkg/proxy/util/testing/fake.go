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

// Addrs is part of NetworkInterfacer interface.
func (f *FakeNetwork) Addrs(intf *net.Interface) ([]net.Addr, error) {
	return f.Address[intf.Name], nil
}

// Interfaces is part of NetworkInterfacer interface.
func (f *FakeNetwork) Interfaces() ([]net.Interface, error) {
	return f.NetworkInterfaces, nil
}

// AddrStruct implements the net.Addr for test purpose.
type AddrStruct struct{ Val string }

// Network is part of net.Addr interface.
func (a AddrStruct) Network() string {
	return a.Val
}

// String is part of net.Addr interface.
func (a AddrStruct) String() string {
	return a.Val
}

var _ net.Addr = &AddrStruct{}
