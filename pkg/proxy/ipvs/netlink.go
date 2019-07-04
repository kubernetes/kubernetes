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
	// GetLocalAddresses returns all unique local type IP addresses based on specified device and filter device
	// If device is not specified, it will list all unique local type addresses except filter device addresses
	GetLocalAddresses(dev, filterDev string) (sets.String, error)
}
