/*
   Copyright © 2021 The CDI Authors

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

package cdi

import (
	"sync"

	oci "github.com/opencontainers/runtime-spec/specs-go"
)

//
// Registry keeps a cache of all CDI Specs installed or generated on
// the host. Registry is the primary interface clients should use to
// interact with CDI.
//
// The most commonly used Registry functions are for refreshing the
// registry and injecting CDI devices into an OCI Spec.
//
type Registry interface {
	RegistryResolver
	RegistryRefresher
	DeviceDB() RegistryDeviceDB
	SpecDB() RegistrySpecDB
}

// RegistryRefresher is the registry interface for refreshing the
// cache of CDI Specs and devices.
//
// Refresh rescans all CDI Spec directories and updates the
// state of the cache to reflect any changes. It returns any
// errors encountered during the refresh.
//
// GetErrors returns all errors encountered for any of the scanned
// Spec files during the last cache refresh.
//
// GetSpecDirectories returns the set up CDI Spec directories
// currently in use. The directories are returned in the scan
// order of Refresh().
type RegistryRefresher interface {
	Refresh() error
	GetErrors() map[string][]error
	GetSpecDirectories() []string
}

// RegistryResolver is the registry interface for injecting CDI
// devices into an OCI Spec.
//
// InjectDevices takes an OCI Spec and injects into it a set of
// CDI devices given by qualified name. It returns the names of
// any unresolved devices and an error if injection fails.
type RegistryResolver interface {
	InjectDevices(spec *oci.Spec, device ...string) (unresolved []string, err error)
}

// RegistryDeviceDB is the registry interface for querying devices.
//
// GetDevice returns the CDI device for the given qualified name. If
// the device is not GetDevice returns nil.
//
// ListDevices returns a slice with the names of qualified device
// known. The returned slice is sorted.
type RegistryDeviceDB interface {
	GetDevice(device string) *Device
	ListDevices() []string
}

// RegistrySpecDB is the registry interface for querying CDI Specs.
//
// ListVendors returns a slice with all vendors known. The returned
// slice is sorted.
//
// ListClasses returns a slice with all classes known. The returned
// slice is sorted.
//
// GetVendorSpecs returns a slice of all Specs for the vendor.
//
// GetSpecErrors returns any errors for the Spec encountered during
// the last cache refresh.
type RegistrySpecDB interface {
	ListVendors() []string
	ListClasses() []string
	GetVendorSpecs(vendor string) []*Spec
	GetSpecErrors(*Spec) []error
}

type registry struct {
	*Cache
}

var _ Registry = &registry{}

var (
	reg      *registry
	initOnce sync.Once
)

// GetRegistry returns the CDI registry. If any options are given, those
// are applied to the registry.
func GetRegistry(options ...Option) Registry {
	var new bool
	initOnce.Do(func() {
		reg, _ = getRegistry(options...)
		new = true
	})
	if !new && len(options) > 0 {
		reg.Configure(options...)
		reg.Refresh()
	}
	return reg
}

// DeviceDB returns the registry interface for querying devices.
func (r *registry) DeviceDB() RegistryDeviceDB {
	return r
}

// SpecDB returns the registry interface for querying Specs.
func (r *registry) SpecDB() RegistrySpecDB {
	return r
}

func getRegistry(options ...Option) (*registry, error) {
	c, err := NewCache(options...)
	return &registry{c}, err
}
