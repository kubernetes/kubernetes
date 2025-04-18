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
	"slices"
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

func Convert_v1beta1_ResourceSliceSpec_To_api_ResourceSliceSpec(in *v1beta1.ResourceSliceSpec, out *ResourceSliceSpec, s conversion.Scope) error {
	if err := autoConvert_v1beta1_ResourceSliceSpec_To_api_ResourceSliceSpec(in, out, s); err != nil {
		return err
	}
	// The autoconversion drops the Mixins field. This is ok since we flatten the mixins during
	// conversion, so the base mixins are no longer needed.
	return nil
}

func Convert_api_ResourceSliceSpec_To_v1beta1_ResourceSliceSpec(in *ResourceSliceSpec, out *v1beta1.ResourceSliceSpec, s conversion.Scope) error {
	return errors.New("conversion to v1beta1.ResourceSliceSpec not supported")
}

func Convert_v1beta1_BasicDevice_To_api_BasicDevice(in *v1beta1.BasicDevice, out *BasicDevice, s conversion.Scope) error {
	if err := autoConvert_v1beta1_BasicDevice_To_api_BasicDevice(in, out, s); err != nil {
		return err
	}

	sliceContext, ok := sliceContext(s)
	if !ok {
		return nil
	}
	slice := sliceContext.Slice
	if slice.Spec.Mixins == nil {
		return nil
	}

	// Create a new set of maps, since we need to layer attributes and capacity
	// from the actual device over anything that comes from the mixins.
	outAttributes := make(map[QualifiedName]DeviceAttribute)
	outCapacity := make(map[QualifiedName]DeviceCapacity)
	for _, ref := range in.Includes {
		i := slices.IndexFunc(slice.Spec.Mixins.Device, func(mixin v1beta1.DeviceMixin) bool {
			return mixin.Name == ref.Name
		})
		if i >= 0 {
			mixin := slice.Spec.Mixins.Device[i]
			for name, inAttr := range mixin.Attributes {
				var outAttr DeviceAttribute
				if err := Convert_v1beta1_DeviceAttribute_To_api_DeviceAttribute(&inAttr, &outAttr, s); err != nil {
					continue
				}
				outAttributes[QualifiedName(name)] = outAttr
			}
			for name, inCap := range mixin.Capacity {
				var outCap DeviceCapacity
				if err := Convert_v1beta1_DeviceCapacity_To_api_DeviceCapacity(&inCap, &outCap, s); err != nil {
					continue
				}
				outCapacity[QualifiedName(name)] = outCap
			}
		}
	}
	// Layer the attributes and capacity from the autoconverted device over what
	// came from the mixins.
	for name, outAttr := range out.Attributes {
		outAttributes[name] = outAttr
	}
	for name, outCap := range out.Capacity {
		outCapacity[name] = outCap
	}
	// Do we need a nil/empty check here?
	if len(outAttributes) > 0 {
		out.Attributes = outAttributes
	}
	if len(outCapacity) > 0 {
		out.Capacity = outCapacity
	}
	return nil
}

func Convert_api_Device_To_v1beta1_Device(in *Device, out *v1beta1.Device, s conversion.Scope) error {
	return errors.New("conversion to v1beta1.Device not supported")
}

func Convert_v1beta1_CounterSet_To_api_CounterSet(in *v1beta1.CounterSet, out *CounterSet, s conversion.Scope) error {
	if err := autoConvert_v1beta1_CounterSet_To_api_CounterSet(in, out, s); err != nil {
		return err
	}

	sliceContext, ok := sliceContext(s)
	if !ok {
		return nil
	}
	slice := sliceContext.Slice
	if slice.Spec.Mixins == nil {
		return nil
	}

	outCounters := make(map[string]Counter)
	for _, ref := range in.Includes {
		i := slices.IndexFunc(slice.Spec.Mixins.CounterSet, func(mixin v1beta1.CounterSetMixin) bool {
			return mixin.Name == ref.Name
		})
		if i >= 0 {
			mixin := slice.Spec.Mixins.CounterSet[i]
			for name, inCounter := range mixin.Counters {
				var outCounter Counter
				if err := Convert_v1beta1_Counter_To_api_Counter(&inCounter, &outCounter, s); err != nil {
					continue
				}
				outCounters[name] = outCounter
			}
		}
	}
	for name, outCounter := range out.Counters {
		outCounters[name] = outCounter
	}
	if len(outCounters) > 0 {
		out.Counters = outCounters
	}
	return nil
}

func Convert_api_CounterSet_To_v1beta1_CounterSet(in *CounterSet, out *v1beta1.CounterSet, s conversion.Scope) error {
	return errors.New("conversion to v1beta1.CounterSet not supported")
}

func Convert_v1beta1_DeviceCounterConsumption_To_api_DeviceCounterConsumption(in *v1beta1.DeviceCounterConsumption, out *DeviceCounterConsumption, s conversion.Scope) error {
	if err := autoConvert_v1beta1_DeviceCounterConsumption_To_api_DeviceCounterConsumption(in, out, s); err != nil {
		return err
	}

	sliceContext, ok := sliceContext(s)
	if !ok {
		return nil
	}
	slice := sliceContext.Slice
	if slice.Spec.Mixins == nil {
		return nil
	}

	outCounters := make(map[string]Counter)
	for _, ref := range in.Includes {
		i := slices.IndexFunc(slice.Spec.Mixins.DeviceCounterConsumption, func(mixin v1beta1.DeviceCounterConsumptionMixin) bool {
			return mixin.Name == ref.Name
		})
		if i >= 0 {
			mixin := slice.Spec.Mixins.DeviceCounterConsumption[i]
			for name, inCounter := range mixin.Counters {
				var outCounter Counter
				if err := Convert_v1beta1_Counter_To_api_Counter(&inCounter, &outCounter, s); err != nil {
					continue
				}
				outCounters[name] = outCounter
			}
		}
	}
	for name, outCounter := range out.Counters {
		outCounters[name] = outCounter
	}
	if len(outCounters) > 0 {
		out.Counters = outCounters
	}

	return nil
}

func Convert_api_DeviceCounterConsumption_To_v1beta1_DeviceCounterConsumption(in *DeviceCounterConsumption, out *v1beta1.DeviceCounterConsumption, s conversion.Scope) error {
	return errors.New("conversion to v1beta1.DeviceCounterConsumption not supported")
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
	Slice *v1beta1.ResourceSlice
}

func sliceContext(s conversion.Scope) (SliceContext, bool) {
	if s == nil {
		return SliceContext{}, false
	}
	sliceContext, ok := s.Meta().Context.(SliceContext)
	if !ok {
		return SliceContext{}, false
	}
	return sliceContext, true
}
