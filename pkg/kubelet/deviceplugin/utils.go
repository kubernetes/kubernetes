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

package deviceplugin

import (
	"fmt"

	"k8s.io/api/core/v1"
	v1helper "k8s.io/kubernetes/pkg/api/v1/helper"
	pluginapi "k8s.io/kubernetes/pkg/kubelet/apis/deviceplugin/v1alpha1"
)

func cloneDevice(d *pluginapi.Device) *pluginapi.Device {
	return &pluginapi.Device{
		ID:     d.ID,
		Health: d.Health,
	}

}

func copyDevices(devs map[string]*pluginapi.Device) []*pluginapi.Device {
	var clones []*pluginapi.Device

	for _, d := range devs {
		clones = append(clones, cloneDevice(d))
	}

	return clones
}

// IsResourceNameValid returns an error if the resource is invalid or is not an
// extended resource name.
func IsResourceNameValid(resourceName string) error {
	if resourceName == "" {
		return fmt.Errorf(errEmptyResourceName)
	}
	if !IsDeviceName(v1.ResourceName(resourceName)) {
		return fmt.Errorf(errInvalidResourceName)
	}
	return nil
}

// IsDeviceName returns whether the ResourceName points to an extended resource
// name exported by a device plugin.
func IsDeviceName(k v1.ResourceName) bool {
	return v1helper.IsExtendedResourceName(k)
}
