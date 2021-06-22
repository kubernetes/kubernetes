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

package device

import pluginapi "k8s.io/kubelet/pkg/apis/deviceplugin/v1beta1"

// DeviceInstances is a mapping device name -> plugin device data
type DeviceInstances map[string]pluginapi.Device

// ResourceDeviceInstances is a mapping resource name -> DeviceInstances
type ResourceDeviceInstances map[string]DeviceInstances

func NewResourceDeviceInstances() ResourceDeviceInstances {
	return make(ResourceDeviceInstances)
}

func (rdev ResourceDeviceInstances) Clone() ResourceDeviceInstances {
	clone := NewResourceDeviceInstances()
	for resourceName, resourceDevs := range rdev {
		clone[resourceName] = make(map[string]pluginapi.Device)
		for devID, dev := range resourceDevs {
			clone[resourceName][devID] = dev
		}
	}
	return clone
}
