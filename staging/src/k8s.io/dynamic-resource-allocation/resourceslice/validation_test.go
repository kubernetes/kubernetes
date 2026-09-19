/*
Copyright 2025 The Kubernetes Authors.

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

package resourceslice

import (
	"testing"

	resourceapi "k8s.io/api/resource/v1"
	"k8s.io/apimachinery/pkg/api/resource"
)

func TestValidateDriverResources(t *testing.T) {
	const (
		driverName = "driver.example.com"
		otherName  = "other.example.com"
		poolName   = "pool"
		deviceName = "device"
	)

	device := func(modify func(*resourceapi.Device)) resourceapi.Device {
		d := resourceapi.Device{Name: deviceName}
		modify(&d)
		return d
	}
	resources := func(d resourceapi.Device) *DriverResources {
		return &DriverResources{
			Pools: map[string]Pool{
				poolName: {
					Slices: []Slice{
						{Devices: []resourceapi.Device{d}},
					},
				},
			},
		}
	}

	for name, tc := range map[string]struct {
		device  resourceapi.Device
		wantErr bool
	}{
		"unqualified-attribute": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"foo": {StringValue: new("bar")},
				}
			}),
		},
		"foreign-domain-attribute": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					resourceapi.QualifiedName(otherName + "/foo"): {StringValue: new("bar")},
				}
			}),
		},
		"driver-qualified-attribute": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					resourceapi.QualifiedName(driverName + "/foo"): {StringValue: new("bar")},
				}
			}),
			wantErr: true,
		},
		"unqualified-capacity": {
			device: device(func(d *resourceapi.Device) {
				d.Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					"foo": {Value: resource.MustParse("1")},
				}
			}),
		},
		"foreign-domain-capacity": {
			device: device(func(d *resourceapi.Device) {
				d.Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					resourceapi.QualifiedName(otherName + "/foo"): {Value: resource.MustParse("1")},
				}
			}),
		},
		"driver-qualified-capacity": {
			device: device(func(d *resourceapi.Device) {
				d.Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					resourceapi.QualifiedName(driverName + "/foo"): {Value: resource.MustParse("1")},
				}
			}),
			wantErr: true,
		},
		"driver-qualified-attribute-and-unqualified-capacity": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					resourceapi.QualifiedName(driverName + "/foo"): {StringValue: new("bar")},
				}
				d.Capacity = map[resourceapi.QualifiedName]resourceapi.DeviceCapacity{
					"foo": {Value: resource.MustParse("1")},
				}
			}),
			wantErr: true,
		},
		// A name that merely has a "/" prefixed with the empty string, i.e. an
		// empty domain, is not the same as being qualified with the driver name
		// and must not be rejected here.
		"empty-domain-attribute": {
			device: device(func(d *resourceapi.Device) {
				d.Attributes = map[resourceapi.QualifiedName]resourceapi.DeviceAttribute{
					"/foo": {StringValue: new("bar")},
				}
			}),
		},
	} {
		t.Run(name, func(t *testing.T) {
			err := validateDriverResources(driverName, resources(tc.device))
			if tc.wantErr && err == nil {
				t.Fatal("expected an error, got none")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("expected no error, got: %v", err)
			}
		})
	}
}
