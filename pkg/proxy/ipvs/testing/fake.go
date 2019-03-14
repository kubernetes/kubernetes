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

package testing

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
)

// FakeNetlinkHandle mock implementation of proxy NetlinkHandle
type FakeNetlinkHandle struct {
	// localAddresses is a network interface name to all of its IP addresses map, e.g.
	// eth0 -> [1.2.3.4, 10.20.30.40]
	localAddresses map[string][]string
}

// NewFakeNetlinkHandle will create a new FakeNetlinkHandle
func NewFakeNetlinkHandle() *FakeNetlinkHandle {
	fake := &FakeNetlinkHandle{
		localAddresses: make(map[string][]string),
	}
	return fake
}

// EnsureAddressBind is a mock implementation
func (h *FakeNetlinkHandle) EnsureAddressBind(address, devName string) (exist bool, err error) {
	if len(devName) == 0 {
		return false, fmt.Errorf("Device name can't be empty")
	}
	if _, ok := h.localAddresses[devName]; !ok {
		return false, fmt.Errorf("Error bind address: %s to a non-exist interface: %s", address, devName)
	}
	for _, addr := range h.localAddresses[devName] {
		if addr == address {
			// return true if the address is already bound to device
			return true, nil
		}
	}
	h.localAddresses[devName] = append(h.localAddresses[devName], address)
	return false, nil
}

// UnbindAddress is a mock implementation
func (h *FakeNetlinkHandle) UnbindAddress(address, devName string) error {
	if len(devName) == 0 {
		return fmt.Errorf("Device name can't be empty")
	}
	if _, ok := h.localAddresses[devName]; !ok {
		return fmt.Errorf("Error unbind address: %s from a non-exist interface: %s", address, devName)
	}
	for i, addr := range h.localAddresses[devName] {
		if addr == address {
			// delete address from slice h.localAddresses[devName]
			h.localAddresses[devName] = append(h.localAddresses[devName][:i], h.localAddresses[devName][i+1:]...)
			return nil
		}
	}
	// return error message if address is not found in slice h.localAddresses[devName]
	return fmt.Errorf("Address: %s is not found in interface: %s", address, devName)
}

// EnsureDummyDevice is a mock implementation
func (h *FakeNetlinkHandle) EnsureDummyDevice(devName string) (bool, error) {
	if len(devName) == 0 {
		return false, fmt.Errorf("Device name can't be empty")
	}
	if _, ok := h.localAddresses[devName]; !ok {
		// create dummy interface if devName is not found in localAddress map
		h.localAddresses[devName] = make([]string, 0)
		return false, nil
	}
	// return true if devName is already created in localAddress map
	return true, nil
}

// DeleteDummyDevice is a mock implementation
func (h *FakeNetlinkHandle) DeleteDummyDevice(devName string) error {
	if len(devName) == 0 {
		return fmt.Errorf("Device name can't be empty")
	}
	if _, ok := h.localAddresses[devName]; !ok {
		return fmt.Errorf("Error deleting a non-exist interface: %s", devName)
	}
	delete(h.localAddresses, devName)
	return nil
}

// ListBindAddress is a mock implementation
func (h *FakeNetlinkHandle) ListBindAddress(devName string) ([]string, error) {
	if len(devName) == 0 {
		return nil, fmt.Errorf("Device name can't be empty")
	}
	if _, ok := h.localAddresses[devName]; !ok {
		return nil, fmt.Errorf("Error list addresses from a non-exist interface: %s", devName)
	}
	return h.localAddresses[devName], nil
}

// GetLocalAddresses is a mock implementation
func (h *FakeNetlinkHandle) GetLocalAddresses(dev, filterDev string) (sets.String, error) {
	res := sets.NewString()
	if len(dev) != 0 {
		// list all addresses from a given network interface.
		for _, addr := range h.localAddresses[dev] {
			res.Insert(addr)
		}
		return res, nil
	}
	// If filterDev is not given, will list all addresses from all available network interface.
	for linkName := range h.localAddresses {
		if linkName == filterDev {
			continue
		}
		// list all addresses from a given network interface.
		for _, addr := range h.localAddresses[linkName] {
			res.Insert(addr)
		}
	}
	return res, nil
}

// SetLocalAddresses set IP addresses to the given interface device.  It's not part of interface.
func (h *FakeNetlinkHandle) SetLocalAddresses(dev string, ips ...string) error {
	if h.localAddresses == nil {
		h.localAddresses = make(map[string][]string)
	}
	if len(dev) == 0 {
		return fmt.Errorf("device name can't be empty")
	}
	h.localAddresses[dev] = make([]string, 0)
	h.localAddresses[dev] = append(h.localAddresses[dev], ips...)
	return nil
}
