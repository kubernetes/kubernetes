/*
Copyright 2024 The Kubernetes Authors.

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

package api

import (
	"errors"
	"fmt"
	"unique"

	corev1 "k8s.io/api/core/v1"
	v1beta1 "k8s.io/api/resource/v1beta1"
	conversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
)

var (
	localSchemeBuilder runtime.SchemeBuilder
	AddToScheme        = localSchemeBuilder.AddToScheme
)

func Convert_api_UniqueString_To_string(in *UniqueString, out *string, s conversion.Scope) error {
	if *in == NullUniqueString {
		*out = ""
		return nil
	}
	*out = in.String()
	return nil
}

func Convert_string_To_api_UniqueString(in *string, out *UniqueString, s conversion.Scope) error {
	if *in == "" {
		*out = NullUniqueString
		return nil
	}
	*out = UniqueString(unique.Make(*in))
	return nil
}

func Convert_v1beta1_ResourceSlice_To_api_ResourceSlice(in *v1beta1.ResourceSlice, out *ResourceSlice, s conversion.Scope) error {
	_, err := sliceContext(s)
	if err != nil {
		return err
	}
	return autoConvert_v1beta1_ResourceSlice_To_api_ResourceSlice(in, out, s)
}

func Convert_v1beta1_ResourceSliceSpec_To_api_ResourceSliceSpec(in *v1beta1.ResourceSliceSpec, out *ResourceSliceSpec, s conversion.Scope) error {
	sliceContext, err := sliceContext(s)
	if err != nil {
		return err
	}

	inCopy := in.DeepCopy()
	inCopy.Devices = make([]v1beta1.Device, 0)

	for _, device := range in.Devices {
		// Always keep basic devices.
		if device.Basic != nil {
			inCopy.Devices = append(inCopy.Devices, device)
			continue
		}
		// Drop composite devices if the partitionable devices feature is not enabled.
		if device.Composite != nil && !sliceContext.PartitionableDevicesEnabled {
			continue
		}

		var noMatch bool
		switch {
		case device.Composite.NodeName != "":
			if device.Composite.NodeName != sliceContext.Node.Name {
				noMatch = true
			}
		case device.Composite.AllNodes:
			// Do nothing
		case device.Composite.NodeSelector != nil:
			selector, err := nodeaffinity.NewNodeSelector(device.Composite.NodeSelector)
			if err != nil {
				return fmt.Errorf("node selector in device %s: %w", device.Name, err)
			}
			if !selector.Match(sliceContext.Node) {
				noMatch = true
			}
		}
		if !noMatch {
			inCopy.Devices = append(inCopy.Devices, device)
		}
	}
	return autoConvert_v1beta1_ResourceSliceSpec_To_api_ResourceSliceSpec(inCopy, out, s)
}

func Convert_v1beta1_DeviceCapacityConsumption_To_api_DeviceCapacityConsumption(in *v1beta1.DeviceCapacityConsumption, out *DeviceCapacityConsumption, s conversion.Scope) error {
	return autoConvert_v1beta1_DeviceCapacityConsumption_To_api_DeviceCapacityConsumption(in, out, s)
}

type SliceScope struct {
	SliceContext SliceContext
}

func (s SliceScope) Convert(src, dest interface{}) error {
	return errors.New("conversion.Scope.Convert not implemented")
}

func (s SliceScope) Meta() *conversion.Meta {
	return &conversion.Meta{Context: s.SliceContext}
}

type SliceContext struct {
	Slice                       *v1beta1.ResourceSlice
	Node                        *corev1.Node
	PartitionableDevicesEnabled bool
}

func sliceContext(s conversion.Scope) (SliceContext, error) {
	if s == nil {
		return SliceContext{}, fmt.Errorf("scope must be provided with context of type SliceContext")
	}
	sliceContext, ok := s.Meta().Context.(SliceContext)
	if !ok {
		return SliceContext{}, fmt.Errorf("context in scope must be of type SliceContext")
	}
	return sliceContext, nil
}

func Convert_v1beta1_CapacityPool_To_api_CapacityPool(in *v1beta1.CapacityPool, out *CapacityPool, s conversion.Scope) error {
	sliceContext, err := sliceContext(s)
	if err != nil {
		return err
	}
	slice := sliceContext.Slice

	capacityPoolCapacities := make([]map[v1beta1.QualifiedName]v1beta1.DeviceCapacity, 0)
	for _, entry := range in.Includes {
		var mixin v1beta1.CapacityPoolMixin
		for _, m := range slice.Spec.Mixins.CapacityPool {
			if entry.Name == m.Name {
				mixin = m
			}
		}
		capacityPoolCapacities = append(capacityPoolCapacities, mixin.Capacity)
	}
	capacityPoolCapacities = append(capacityPoolCapacities, in.Capacity)

	if err := autoConvert_v1beta1_CapacityPool_To_api_CapacityPool(in, out, s); err != nil {
		return err
	}
	out.Capacity = flattenCapacity(capacityPoolCapacities...)
	return nil
}

func Convert_v1beta1_Device_To_api_Device(in *v1beta1.Device, out *Device, s conversion.Scope) error {
	out.Name = MakeUniqueString(in.Name)

	if in.Basic != nil {
		out.Attributes = flattenAttributes(in.Basic.Attributes)
		out.Capacity = flattenCapacity(in.Basic.Capacity)
	}
	if in.Composite != nil {
		if in.Composite.NodeName != "" {
			out.NodeName = MakeUniqueString(in.Composite.NodeName)
		}
		out.NodeSelector = in.Composite.NodeSelector
		out.AllNodes = in.Composite.AllNodes

		sliceContext, err := sliceContext(s)
		if err != nil {
			return err
		}
		slice := sliceContext.Slice

		deviceAttributes := make([]map[v1beta1.QualifiedName]v1beta1.DeviceAttribute, 0)
		deviceCapacities := make([]map[v1beta1.QualifiedName]v1beta1.DeviceCapacity, 0)
		for _, entry := range in.Composite.Includes {
			var mixin v1beta1.DeviceMixin
			for _, m := range slice.Spec.Mixins.Device {
				if entry.Name == m.Name {
					mixin = m
				}
			}
			if mixin.Composite == nil {
				continue
			}
			deviceAttributes = append(deviceAttributes, mixin.Composite.Attributes)
			deviceCapacities = append(deviceCapacities, mixin.Composite.Capacity)
		}
		deviceAttributes = append(deviceAttributes, in.Composite.Attributes)
		deviceCapacities = append(deviceCapacities, in.Composite.Capacity)
		out.Attributes = flattenAttributes(deviceAttributes...)
		out.Capacity = flattenCapacity(deviceCapacities...)

		outConsumesCapacity := make([]DeviceCapacityConsumption, 0)
		for _, consumesCapacity := range in.Composite.ConsumesCapacity {
			deviceConsumesCapacities := make([]map[v1beta1.QualifiedName]v1beta1.DeviceCapacity, 0)
			for _, entry := range consumesCapacity.Includes {
				var mixin v1beta1.DeviceCapacityConsumptionMixin
				for _, dcc := range slice.Spec.Mixins.DeviceCapacityConsumption {
					if entry.Name == dcc.Name {
						mixin = dcc
					}
				}
				deviceConsumesCapacities = append(deviceConsumesCapacities, mixin.Capacity)
			}
			deviceConsumesCapacities = append(deviceConsumesCapacities, consumesCapacity.Capacity)
			outConsumesCapacity = append(outConsumesCapacity, DeviceCapacityConsumption{
				CapacityPool: MakeUniqueString(consumesCapacity.CapacityPool),
				Capacity:     flattenCapacity(deviceConsumesCapacities...),
			})
		}
		if len(outConsumesCapacity) > 0 {
			out.ConsumesCapacity = outConsumesCapacity
		}
	}

	return nil
}

func Convert_api_Device_To_v1beta1_Device(in *Device, out *v1beta1.Device, s conversion.Scope) error {
	return fmt.Errorf("api_Device_To_v1beta1_Device not implemented")
}

func flattenAttributes(attributes ...map[v1beta1.QualifiedName]v1beta1.DeviceAttribute) map[QualifiedName]DeviceAttribute {
	var hasNonEmptyAttributes bool
	for _, attrs := range attributes {
		if len(attrs) > 0 {
			hasNonEmptyAttributes = true
			break
		}
	}
	if !hasNonEmptyAttributes {
		return nil
	}

	flattenedAttributes := make(map[QualifiedName]DeviceAttribute)
	for _, attrs := range attributes {
		for name, attr := range attrs {
			var outAttr DeviceAttribute
			if err := Convert_v1beta1_DeviceAttribute_To_api_DeviceAttribute(&attr, &outAttr, nil); err != nil {
				continue
			}
			flattenedAttributes[QualifiedName(name)] = outAttr
		}
	}
	return flattenedAttributes
}

func flattenCapacity(capacities ...map[v1beta1.QualifiedName]v1beta1.DeviceCapacity) map[QualifiedName]DeviceCapacity {
	var hasNonEmptyCapacity bool
	for _, capacity := range capacities {
		if len(capacity) > 0 {
			hasNonEmptyCapacity = true
			break
		}
	}
	if !hasNonEmptyCapacity {
		return nil
	}

	flattenedCapacity := make(map[QualifiedName]DeviceCapacity)
	for _, capacity := range capacities {
		for name, cap := range capacity {
			var outCap DeviceCapacity
			if err := Convert_v1beta1_DeviceCapacity_To_api_DeviceCapacity(&cap, &outCap, nil); err != nil {
				continue
			}
			flattenedCapacity[QualifiedName(name)] = outCap
		}
	}
	return flattenedCapacity
}

func Convert_api_Attributes_To_v1beta1_Attributes(in map[QualifiedName]DeviceAttribute, out map[v1beta1.QualifiedName]v1beta1.DeviceAttribute) error {
	for name, attr := range in {
		var outDeviceAttribute v1beta1.DeviceAttribute
		if err := Convert_api_DeviceAttribute_To_v1beta1_DeviceAttribute(&attr, &outDeviceAttribute, nil); err != nil {
			return err
		}
		out[v1beta1.QualifiedName(name)] = outDeviceAttribute
	}
	return nil
}

func Convert_api_Capacity_To_v1beta1_Capacity(in map[QualifiedName]DeviceCapacity, out map[v1beta1.QualifiedName]v1beta1.DeviceCapacity) error {
	for name, cap := range in {
		var outDeviceCapacity v1beta1.DeviceCapacity
		if err := Convert_api_DeviceCapacity_To_v1beta1_DeviceCapacity(&cap, &outDeviceCapacity, nil); err != nil {
			return err
		}
		out[v1beta1.QualifiedName(name)] = outDeviceCapacity
	}
	return nil
}
