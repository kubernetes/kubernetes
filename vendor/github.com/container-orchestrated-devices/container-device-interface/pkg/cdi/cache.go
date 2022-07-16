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
	"path/filepath"
	"sort"
	"strings"
	"sync"

	"github.com/hashicorp/go-multierror"
	oci "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
)

// Option is an option to change some aspect of default CDI behavior.
type Option func(*Cache) error

// Cache stores CDI Specs loaded from Spec directories.
type Cache struct {
	sync.Mutex
	specDirs []string
	specs    map[string][]*Spec
	devices  map[string]*Device
	errors   map[string][]error
}

// NewCache creates a new CDI Cache. The cache is populated from a set
// of CDI Spec directories. These can be specified using a WithSpecDirs
// option. The default set of directories is exposed in DefaultSpecDirs.
func NewCache(options ...Option) (*Cache, error) {
	c := &Cache{}

	if err := c.Configure(options...); err != nil {
		return nil, err
	}
	if len(c.specDirs) == 0 {
		c.Configure(WithSpecDirs(DefaultSpecDirs...))
	}

	return c, c.Refresh()
}

// Configure applies options to the cache. Updates the cache if options have
// changed.
func (c *Cache) Configure(options ...Option) error {
	if len(options) == 0 {
		return nil
	}

	c.Lock()
	defer c.Unlock()

	for _, o := range options {
		if err := o(c); err != nil {
			return errors.Wrapf(err, "failed to apply cache options")
		}
	}

	return nil
}

// Refresh rescans the CDI Spec directories and refreshes the Cache.
func (c *Cache) Refresh() error {
	var (
		specs      = map[string][]*Spec{}
		devices    = map[string]*Device{}
		conflicts  = map[string]struct{}{}
		specErrors = map[string][]error{}
		result     []error
	)

	// collect errors per spec file path and once globally
	collectError := func(err error, paths ...string) {
		result = append(result, err)
		for _, path := range paths {
			specErrors[path] = append(specErrors[path], err)
		}
	}
	// resolve conflicts based on device Spec priority (order of precedence)
	resolveConflict := func(name string, dev *Device, old *Device) bool {
		devSpec, oldSpec := dev.GetSpec(), old.GetSpec()
		devPrio, oldPrio := devSpec.GetPriority(), oldSpec.GetPriority()
		switch {
		case devPrio > oldPrio:
			return false
		case devPrio == oldPrio:
			devPath, oldPath := devSpec.GetPath(), oldSpec.GetPath()
			collectError(errors.Errorf("conflicting device %q (specs %q, %q)",
				name, devPath, oldPath), devPath, oldPath)
			conflicts[name] = struct{}{}
		}
		return true
	}

	_ = scanSpecDirs(c.specDirs, func(path string, priority int, spec *Spec, err error) error {
		path = filepath.Clean(path)
		if err != nil {
			collectError(errors.Wrapf(err, "failed to load CDI Spec"), path)
			return nil
		}

		vendor := spec.GetVendor()
		specs[vendor] = append(specs[vendor], spec)

		for _, dev := range spec.devices {
			qualified := dev.GetQualifiedName()
			other, ok := devices[qualified]
			if ok {
				if resolveConflict(qualified, dev, other) {
					continue
				}
			}
			devices[qualified] = dev
		}

		return nil
	})

	for conflict := range conflicts {
		delete(devices, conflict)
	}

	c.Lock()
	defer c.Unlock()

	c.specs = specs
	c.devices = devices
	c.errors = specErrors

	if len(result) > 0 {
		return multierror.Append(nil, result...)
	}

	return nil
}

// InjectDevices injects the given qualified devices to an OCI Spec. It
// returns any unresolvable devices and an error if injection fails for
// any of the devices.
func (c *Cache) InjectDevices(ociSpec *oci.Spec, devices ...string) ([]string, error) {
	var unresolved []string

	if ociSpec == nil {
		return devices, errors.Errorf("can't inject devices, nil OCI Spec")
	}

	c.Lock()
	defer c.Unlock()

	edits := &ContainerEdits{}
	specs := map[*Spec]struct{}{}

	for _, device := range devices {
		d := c.devices[device]
		if d == nil {
			unresolved = append(unresolved, device)
			continue
		}
		if _, ok := specs[d.GetSpec()]; !ok {
			specs[d.GetSpec()] = struct{}{}
			edits.Append(d.GetSpec().edits())
		}
		edits.Append(d.edits())
	}

	if unresolved != nil {
		return unresolved, errors.Errorf("unresolvable CDI devices %s",
			strings.Join(devices, ", "))
	}

	if err := edits.Apply(ociSpec); err != nil {
		return nil, errors.Wrap(err, "failed to inject devices")
	}

	return nil, nil
}

// GetDevice returns the cached device for the given qualified name.
func (c *Cache) GetDevice(device string) *Device {
	c.Lock()
	defer c.Unlock()

	return c.devices[device]
}

// ListDevices lists all cached devices by qualified name.
func (c *Cache) ListDevices() []string {
	var devices []string

	c.Lock()
	defer c.Unlock()

	for name := range c.devices {
		devices = append(devices, name)
	}
	sort.Strings(devices)

	return devices
}

// ListVendors lists all vendors known to the cache.
func (c *Cache) ListVendors() []string {
	var vendors []string

	c.Lock()
	defer c.Unlock()

	for vendor := range c.specs {
		vendors = append(vendors, vendor)
	}
	sort.Strings(vendors)

	return vendors
}

// ListClasses lists all device classes known to the cache.
func (c *Cache) ListClasses() []string {
	var (
		cmap    = map[string]struct{}{}
		classes []string
	)

	c.Lock()
	defer c.Unlock()

	for _, specs := range c.specs {
		for _, spec := range specs {
			cmap[spec.GetClass()] = struct{}{}
		}
	}
	for class := range cmap {
		classes = append(classes, class)
	}
	sort.Strings(classes)

	return classes
}

// GetVendorSpecs returns all specs for the given vendor.
func (c *Cache) GetVendorSpecs(vendor string) []*Spec {
	c.Lock()
	defer c.Unlock()

	return c.specs[vendor]
}

// GetSpecErrors returns all errors encountered for the spec during the
// last cache refresh.
func (c *Cache) GetSpecErrors(spec *Spec) []error {
	return c.errors[spec.GetPath()]
}

// GetErrors returns all errors encountered during the last
// cache refresh.
func (c *Cache) GetErrors() map[string][]error {
	return c.errors
}

// GetSpecDirectories returns the CDI Spec directories currently in use.
func (c *Cache) GetSpecDirectories() []string {
	dirs := make([]string, len(c.specDirs))
	copy(dirs, c.specDirs)
	return dirs
}
