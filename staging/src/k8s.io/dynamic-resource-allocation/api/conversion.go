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
	"unique"

	v1 "k8s.io/api/resource/v1"
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

func Convert_api_ResourceSliceMixins_To_v1_ResourceSliceMixins(in *ResourceSliceMixins, out *v1.ResourceSliceMixins, s conversion.Scope) error {
	if err := autoConvert_api_ResourceSliceMixins_To_v1_ResourceSliceMixins(in, out, s); err != nil {
		return err
	}
	if len(in.Device) > 0 {
		out.Device = make([]v1.DeviceMixin, 0, len(in.Device))
		for name, mixin := range in.Device {
			item := v1.DeviceMixin{}
			if err := Convert_api_DeviceMixin_To_v1_DeviceMixin(&mixin, &item, s); err != nil {
				return err
			}
			item.Name = name
			out.Device = append(out.Device, item)
		}
	}

	if len(in.CounterSet) > 0 {
		out.CounterSet = make([]v1.CounterSetMixin, 0, len(in.CounterSet))
		for name, mixin := range in.CounterSet {
			item := v1.CounterSetMixin{}
			if err := Convert_api_CounterSetMixin_To_v1_CounterSetMixin(&mixin, &item, s); err != nil {
				return err
			}
			item.Name = name
			out.CounterSet = append(out.CounterSet, item)
		}
	}

	if len(in.DeviceCounterConsumption) > 0 {
		out.DeviceCounterConsumption = make([]v1.DeviceCounterConsumptionMixin, 0, len(in.DeviceCounterConsumption))
		for name, mixin := range in.DeviceCounterConsumption {
			item := v1.DeviceCounterConsumptionMixin{}
			if err := Convert_api_DeviceCounterConsumptionMixin_To_v1_DeviceCounterConsumptionMixin(&mixin, &item, s); err != nil {
				return err
			}
			item.Name = name
			out.DeviceCounterConsumption = append(out.DeviceCounterConsumption, item)
		}
	}
	return nil
}

func Convert_v1_ResourceSliceMixins_To_api_ResourceSliceMixins(in *v1.ResourceSliceMixins, out *ResourceSliceMixins, s conversion.Scope) error {
	if err := autoConvert_v1_ResourceSliceMixins_To_api_ResourceSliceMixins(in, out, s); err != nil {
		return err
	}
	if len(in.Device) > 0 {
		out.Device = make(map[string]DeviceMixin, len(in.Device))
		for _, mixin := range in.Device {
			item := DeviceMixin{}
			if err := Convert_v1_DeviceMixin_To_api_DeviceMixin(&mixin, &item, s); err != nil {
				return err
			}
			out.Device[mixin.Name] = item
		}
	}

	if len(in.CounterSet) > 0 {
		out.CounterSet = make(map[string]CounterSetMixin, len(in.CounterSet))
		for _, mixin := range in.CounterSet {
			item := CounterSetMixin{}
			if err := Convert_v1_CounterSetMixin_To_api_CounterSetMixin(&mixin, &item, s); err != nil {
				return err
			}
			out.CounterSet[mixin.Name] = item
		}
	}

	if len(in.DeviceCounterConsumption) > 0 {
		out.DeviceCounterConsumption = make(map[string]DeviceCounterConsumptionMixin, len(in.DeviceCounterConsumption))
		for _, mixin := range in.DeviceCounterConsumption {
			item := DeviceCounterConsumptionMixin{}
			if err := Convert_v1_DeviceCounterConsumptionMixin_To_api_DeviceCounterConsumptionMixin(&mixin, &item, s); err != nil {
				return err
			}
			out.DeviceCounterConsumption[mixin.Name] = item
		}
	}
	return nil
}

func Convert_v1_DeviceCounterConsumptionMixin_To_api_DeviceCounterConsumptionMixin(in *v1.DeviceCounterConsumptionMixin, out *DeviceCounterConsumptionMixin, s conversion.Scope) error {
	if err := autoConvert_v1_DeviceCounterConsumptionMixin_To_api_DeviceCounterConsumptionMixin(in, out, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1_CounterSetMixin_To_api_CounterSetMixin(in *v1.CounterSetMixin, out *CounterSetMixin, s conversion.Scope) error {
	if err := autoConvert_v1_CounterSetMixin_To_api_CounterSetMixin(in, out, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1_DeviceMixin_To_api_DeviceMixin(in *v1.DeviceMixin, out *DeviceMixin, s conversion.Scope) error {
	if err := autoConvert_v1_DeviceMixin_To_api_DeviceMixin(in, out, s); err != nil {
		return err
	}
	return nil
}
