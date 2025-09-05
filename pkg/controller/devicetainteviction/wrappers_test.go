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

package devicetainteviction

import (
	"fmt"

	resourceapi "k8s.io/api/resource/v1"
	draapi "k8s.io/dynamic-resource-allocation/api"
)

var u = draapi.MakeUniqueString

type ResourceSliceWrapper struct {
	draapi.ResourceSlice
}

func makeResourceSlice(nodeName, driverName string) *ResourceSliceWrapper {
	wrapper := new(ResourceSliceWrapper)
	wrapper.Name = nodeName + "-" + driverName
	wrapper.Spec.NodeName = &nodeName
	wrapper.Spec.Pool.Name = u(nodeName)
	wrapper.Spec.Pool.ResourceSliceCount = 1
	wrapper.Spec.Driver = u(driverName)
	return wrapper
}

func (wrapper *ResourceSliceWrapper) obj() *draapi.ResourceSlice {
	return &wrapper.ResourceSlice
}

// device extends the devices field of the inner object.
// The device must have a name and may have arbitrary additional fields.
func (wrapper *ResourceSliceWrapper) device(name string, otherFields ...any) *ResourceSliceWrapper {
	device := draapi.Device{Name: u(name)}
	for _, field := range otherFields {
		switch typedField := field.(type) {
		case map[draapi.QualifiedName]draapi.DeviceAttribute:
			device.Attributes = typedField
		case map[draapi.QualifiedName]draapi.DeviceCapacity:
			device.Capacity = typedField
		case []resourceapi.DeviceTaint:
			device.Taints = append(device.Taints, typedField...)
		case resourceapi.DeviceTaint:
			device.Taints = append(device.Taints, typedField)
		default:
			panic(fmt.Sprintf("expected a type which matches a field in BasicDevice, got %T", field))
		}
	}
	wrapper.Spec.Devices = append(wrapper.Spec.Devices, device)
	return wrapper
}

func mustConvertResourceSlice(in *draapi.ResourceSlice) *resourceapi.ResourceSlice {
	var out resourceapi.ResourceSlice
	if err := draapi.Convert_api_ResourceSlice_To_v1_ResourceSlice(in, &out, nil); err != nil {
		panic(err)
	}
	return &out
}
