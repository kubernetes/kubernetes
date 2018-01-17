/*
Copyright 2017 The Kubernetes Authors.

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

package util

import (
	"net"
)

// NetworkInterface defines an interface for several net library functions. Production
// code will forward to net library functions, and unit tests will override the methods
// for testing purposes.
type NetworkInterface interface {
	Addrs(intf *net.Interface) ([]net.Addr, error)
	Interfaces() ([]net.Interface, error)
}

// RealNetwork implements the NetworkInterface interface for production code, just
// wrapping the underlying net library function calls.
type RealNetwork struct{}

// Addrs wraps net.Interface.Addrs()
func (_ RealNetwork) Addrs(intf *net.Interface) ([]net.Addr, error) {
	return intf.Addrs()
}

// Interfaces wraps net.Interfaces()
func (_ RealNetwork) Interfaces() ([]net.Interface, error) {
	return net.Interfaces()
}

// RealNetwork implements the NetworkInterface interface for production code, just
// wrapping the underlying net library function calls.
type FakeNetwork struct {
	NetworkInterfaces []net.Interface
	// The key of map Addrs is interface name
	Address map[string][]net.Addr
}

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

// Addrs is part of FakeNetwork interface.
func (f *FakeNetwork) Addrs(intf *net.Interface) ([]net.Addr, error) {
	return f.Address[intf.Name], nil
}

// Interfaces is part of FakeNetwork interface.
func (f *FakeNetwork) Interfaces() ([]net.Interface, error) {
	return f.NetworkInterfaces, nil
}

type AddrStruct struct{ Val string }

func (a AddrStruct) Network() string {
	return a.Val
}
func (a AddrStruct) String() string {
	return a.Val
}
