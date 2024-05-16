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

package ipvs

import (
	"k8s.io/apimachinery/pkg/util/sets"
)

// NetLinkHandle for revoke netlink interface
type NetLinkHandle interface {
	// EnsureAddressBind checks if address is bound to the interface and, if not, binds it.  If the address is already bound, return true.
	EnsureAddressBind(address, devName string) (exist bool, err error)
	// UnbindAddress unbind address from the interface
	UnbindAddress(address, devName string) error
	// EnsureDummyDevice checks if dummy device is exist and, if not, create one.  If the dummy device is already exist, return true.
	EnsureDummyDevice(devName string) (exist bool, err error)
	// DeleteDummyDevice deletes the given dummy device by name.
	DeleteDummyDevice(devName string) error
	// ListBindAddress will list all IP addresses which are bound in a given interface
	ListBindAddress(devName string) ([]string, error)
	// GetAllLocalAddresses return all local addresses on the node.
	// Only the addresses of the current family are returned.
	// IPv6 link-local and loopback addresses are excluded.
	GetAllLocalAddresses() (sets.Set[string], error)
	// GetLocalAddresses return all local addresses for an interface.
	// Only the addresses of the current family are returned.
	// IPv6 link-local and loopback addresses are excluded.
	GetLocalAddresses(dev string) (sets.Set[string], error)
	// GetAllLocalAddressesExcept return all local addresses on the node, except from the passed dev.
	// This is not the same as to take the diff between GetAllLocalAddresses and GetLocalAddresses
	// since an address can be assigned to many interfaces. This problem raised
	// https://github.com/kubernetes/kubernetes/issues/114815
	GetAllLocalAddressesExcept(dev string) (sets.Set[string], error)
}
