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

package deviceplugin

import (
	"fmt"
	"strings"

	"k8s.io/api/core/v1"

	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/device-plugin/v1alpha1"
)

// NewDevice returns a pluginapi.Device
func NewDevice(name, kind, vendor string) *pluginapi.Device {
	return &pluginapi.Device{
		Name:   name,
		Kind:   kind,
		Vendor: vendor,
	}
}

func copyDevices(devs []*pluginapi.Device) []*pluginapi.Device {
	var clones []*pluginapi.Device
	for _, d := range devs {
		clones = append(clones, d)
	}

	return clones
}

// HasDevice returns then index of the device and if the device
// is present in the list
func HasDevice(d *pluginapi.Device, devs []*pluginapi.Device) (int, bool) {
	name := DeviceKey(d)

	for i, d := range devs {
		if DeviceKey(d) != name {
			continue
		}

		return i, true
	}

	return -1, false
}

func deleteDev(d *pluginapi.Device, devs []*pluginapi.Device) []*pluginapi.Device {
	i, ok := HasDevice(d, devs)
	if !ok {
		return devs
	}

	devs[i], devs[len(devs)-1] = devs[len(devs)-1], devs[i]

	return devs[:len(devs)-1]
}

func deleteDevAt(i int, devs []*pluginapi.Device) []*pluginapi.Device {
	devs[i], devs[len(devs)-1] = devs[len(devs)-1], devs[i]
	return devs[:len(devs)-1]
}

// IsDevsValid returns an error if any of the Device fields are invalid
func IsDevsValid(devs []*pluginapi.Device, vendor string) error {
	for _, d := range devs {
		if err := IsDevValid(d, vendor); err != nil {
			return err
		}
	}

	return nil
}

// IsDevValid returns an error if any of the field has an error
func IsDevValid(d *pluginapi.Device, vendor string) error {
	d.Vendor = strings.TrimSpace(d.Vendor)
	d.Kind = strings.TrimSpace(d.Kind)
	d.Name = strings.TrimSpace(d.Name)

	if d.Vendor != vendor {
		return fmt.Errorf(pluginapi.ErrVendorMismatch+" %v, %v", d, vendor)
	}

	if d.Kind == "" || d.Name == "" {
		return fmt.Errorf(pluginapi.ErrEmptyDevice+" %+vs", d)
	}

	if d.Kind != "nvidia-gpu" && strings.ContainsAny(d.Kind, pluginapi.InvalidChars) {
		return fmt.Errorf(pluginapi.ErrInvalidDeviceKind + " " + d.Kind)
	}
	return nil
}

// IsVendorValid returns an error if the vendor is invalid,
// Expecting vendor to be trimed
func IsVendorValid(vendor string) error {
	if vendor == "" {
		return fmt.Errorf(pluginapi.ErrEmptyVendor)
	}

	if strings.ContainsAny(vendor, pluginapi.InvalidChars) {
		return fmt.Errorf(pluginapi.ErrInvalidVendor + " " + vendor)
	}

	return nil
}

// DeviceName returns true if the ResourceName points to a device plugin name
// returns the trimed device plugin name
func DeviceName(k v1.ResourceName) (bool, string) {
	key := string(k)
	if k != v1.ResourceNvidiaGPU && !strings.HasPrefix(key, v1.ResourceOpaqueIntPrefix) {
		return false, ""
	}
	var name string
	if k == v1.ResourceNvidiaGPU {
		name = "nvidia-gpu"
	} else {
		name = strings.TrimPrefix(key, v1.ResourceOpaqueIntPrefix)
	}

	return true, name
}

// VendorDeviceKey returns a Device plugin key
func VendorDeviceKey(d *pluginapi.Device) string {
	return d.Vendor + "-" + DeviceKey(d)
}

// DeviceKey returns a Device plugin key
func DeviceKey(d *pluginapi.Device) string {
	return d.Kind + "-" + d.Name
}
