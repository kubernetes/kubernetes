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
	"unique"

	v1beta1 "k8s.io/api/resource/v1beta1"
	conversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
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
	if s == nil {
		s = sliceScope{
			slice: in,
		}
	}
	return autoConvert_v1beta1_ResourceSlice_To_api_ResourceSlice(in, out, s)
}

func Convert_v1beta1_ResourceSliceSpec_To_api_ResourceSliceSpec(in *v1beta1.ResourceSliceSpec, out *ResourceSliceSpec, s conversion.Scope) error {
	return nil
}

func Convert_v1beta1_DeviceCapacityConsumption_To_api_DeviceCapacityConsumption(in *v1beta1.DeviceCapacityConsumption, out *DeviceCapacityConsumption, s conversion.Scope) error {
	return nil
}

type sliceScope struct {
	slice *v1beta1.ResourceSlice
}

func (s sliceScope) Convert(src, dest interface{}) error {
	return errors.New("conversion.Scope.Convert not implemented")
}

func (s sliceScope) Meta() *conversion.Meta {
	return &conversion.Meta{Context: s.slice}
}

func Convert_v1beta1_CapacityPool_To_api_CapacityPool(in *v1beta1.CapacityPool, out *CapacityPool, s conversion.Scope) error {
	return autoConvert_v1beta1_CapacityPool_To_api_CapacityPool(in, out, s)
}

func Convert_v1beta1_CompositeDevice_To_api_CompositeDevice(in *v1beta1.CompositeDevice, out *CompositeDevice, s conversion.Scope) error {

	err := autoConvert_v1beta1_CompositeDevice_To_api_CompositeDevice(in, out, s)
	if err != nil {
		return err
	}

	slice, ok := s.Meta().Context.(*v1beta1.ResourceSlice)
	if !ok {
		return nil
	}

	outDeviceAttributes := make(map[QualifiedName]DeviceAttribute)
	outDeviceCapacity := make(map[QualifiedName]DeviceCapacity)
	for _, entry := range in.Includes {
		var mixin v1beta1.DeviceMixin
		for _, m := range slice.Spec.Mixins.Device {
			if entry.Name == m.Name {
				mixin = m
			}
		}
		for name, inAttr := range mixin.Composite.Attributes {
			var outAttr DeviceAttribute
			if err := Convert_v1beta1_DeviceAttribute_To_api_DeviceAttribute(&inAttr, &outAttr, s); err != nil {
				continue
			}
			outDeviceAttributes[QualifiedName(name)] = outAttr
		}
		for name, inCap := range mixin.Composite.Capacity {
			var outCap DeviceCapacity
			if err := Convert_v1beta1_DeviceCapacity_To_api_DeviceCapacity(&inCap, &outCap, s); err != nil {
				continue
			}
			outDeviceCapacity[QualifiedName(name)] = outCap
		}
	}
	for name, attr := range out.Attributes {
		outDeviceAttributes[name] = attr
	}
	for name, cap := range out.Capacity {
		outDeviceCapacity[name] = cap
	}
	out.Attributes = outDeviceAttributes
	out.Capacity = outDeviceCapacity

	for i, consumesCapacity := range in.ConsumesCapacity {
		outConsumesCapacity := make(map[QualifiedName]DeviceCapacity)
		for _, entry := range consumesCapacity.Includes {
			var mixin v1beta1.DeviceCapacityConsumptionMixin
			for _, dcc := range slice.Spec.Mixins.DeviceCapacityConsumption {
				if entry.Name == dcc.Name {
					mixin = dcc
				}
			}
			for name, inCap := range mixin.Capacity {
				var outCap DeviceCapacity
				if err := Convert_v1beta1_DeviceCapacity_To_api_DeviceCapacity(&inCap, &outCap, s); err != nil {
					continue
				}
				outConsumesCapacity[QualifiedName(name)] = outCap
			}
		}
		for name, cap := range out.ConsumesCapacity[i].Capacity {
			outConsumesCapacity[name] = cap
		}
		out.ConsumesCapacity[i].Capacity = outConsumesCapacity
	}

	return nil
}
