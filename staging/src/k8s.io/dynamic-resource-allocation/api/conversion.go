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

func Convert_api_BasicDevice_To_v1beta1_BasicDevice(in *BasicDevice, out *v1beta1.BasicDevice, s conversion.Scope) error {
	if err := autoConvert_api_BasicDevice_To_v1beta1_BasicDevice(in, out, s); err != nil {
		return err
	}
	if len(in.MixinRefs) > 0 {
		out.Includes = make([]string, len(in.MixinRefs))
		for i := range in.MixinRefs {
			out.Includes[i] = in.MixinRefs[i].String()
		}
	}
	return nil
}

func Convert_v1beta1_BasicDevice_To_api_BasicDevice(in *v1beta1.BasicDevice, out *BasicDevice, s conversion.Scope) error {
	if err := autoConvert_v1beta1_BasicDevice_To_api_BasicDevice(in, out, s); err != nil {
		return err
	}
	if len(in.Includes) > 0 {
		out.MixinRefs = make([]UniqueString, len(in.Includes))
		for i := range in.Includes {
			out.MixinRefs[i] = MakeUniqueString(in.Includes[i])
		}
	}
	return nil
}

func Convert_api_CounterSet_To_v1beta1_CounterSet(in *CounterSet, out *v1beta1.CounterSet, s conversion.Scope) error {
	if err := autoConvert_api_CounterSet_To_v1beta1_CounterSet(in, out, s); err != nil {
		return err
	}
	if len(in.MixinRefs) > 0 {
		out.Includes = make([]string, len(in.MixinRefs))
		for i := range in.MixinRefs {
			out.Includes[i] = in.MixinRefs[i].String()
		}
	}
	return nil
}

func Convert_v1beta1_CounterSet_To_api_CounterSet(in *v1beta1.CounterSet, out *CounterSet, s conversion.Scope) error {
	if err := autoConvert_v1beta1_CounterSet_To_api_CounterSet(in, out, s); err != nil {
		return err
	}
	if len(in.Includes) > 0 {
		out.MixinRefs = make([]UniqueString, len(in.Includes))
		for i := range in.Includes {
			out.MixinRefs[i] = MakeUniqueString(in.Includes[i])
		}
	}
	return nil
}

func Convert_api_DeviceCounterConsumption_To_v1beta1_DeviceCounterConsumption(in *DeviceCounterConsumption, out *v1beta1.DeviceCounterConsumption, s conversion.Scope) error {
	if err := autoConvert_api_DeviceCounterConsumption_To_v1beta1_DeviceCounterConsumption(in, out, s); err != nil {
		return err
	}
	if len(in.MixinRefs) > 0 {
		out.Includes = make([]string, len(in.MixinRefs))
		for i := range in.MixinRefs {
			out.Includes[i] = in.MixinRefs[i].String()
		}
	}
	return nil
}

func Convert_v1beta1_DeviceCounterConsumption_To_api_DeviceCounterConsumption(in *v1beta1.DeviceCounterConsumption, out *DeviceCounterConsumption, s conversion.Scope) error {
	if err := autoConvert_v1beta1_DeviceCounterConsumption_To_api_DeviceCounterConsumption(in, out, s); err != nil {
		return err
	}
	if len(in.Includes) > 0 {
		out.MixinRefs = make([]UniqueString, len(in.Includes))
		for i := range in.Includes {
			out.MixinRefs[i] = MakeUniqueString(in.Includes[i])
		}
	}
	return nil
}

func Convert_api_ResourceSliceMixins_To_v1beta1_ResourceSliceMixins(in *ResourceSliceMixins, out *v1beta1.ResourceSliceMixins, s conversion.Scope) error {
	if err := autoConvert_api_ResourceSliceMixins_To_v1beta1_ResourceSliceMixins(in, out, s); err != nil {
		return err
	}
	if len(in.Device) > 0 {
		out.Device = make([]v1beta1.DeviceMixin, 0, len(in.Device))
		for name, mixin := range in.Device {
			item := v1beta1.DeviceMixin{}
			if err := Convert_api_DeviceMixin_To_v1beta1_DeviceMixin(&mixin, &item, s); err != nil {
				return err
			}
			item.Name = name.String()
			out.Device = append(out.Device, item)
		}
	}

	if len(in.CounterSet) > 0 {
		out.CounterSet = make([]v1beta1.CounterSetMixin, 0, len(in.CounterSet))
		for name, mixin := range in.CounterSet {
			item := v1beta1.CounterSetMixin{}
			if err := Convert_api_CounterSetMixin_To_v1beta1_CounterSetMixin(&mixin, &item, s); err != nil {
				return err
			}
			item.Name = name.String()
			out.CounterSet = append(out.CounterSet, item)
		}
	}

	if len(in.DeviceCounterConsumption) > 0 {
		out.DeviceCounterConsumption = make([]v1beta1.DeviceCounterConsumptionMixin, 0, len(in.DeviceCounterConsumption))
		for name, mixin := range in.DeviceCounterConsumption {
			item := v1beta1.DeviceCounterConsumptionMixin{}
			if err := Convert_api_DeviceCounterConsumptionMixin_To_v1beta1_DeviceCounterConsumptionMixin(&mixin, &item, s); err != nil {
				return err
			}
			item.Name = name.String()
			out.DeviceCounterConsumption = append(out.DeviceCounterConsumption, item)
		}
	}
	return nil
}

func Convert_v1beta1_ResourceSliceMixins_To_api_ResourceSliceMixins(in *v1beta1.ResourceSliceMixins, out *ResourceSliceMixins, s conversion.Scope) error {
	if err := autoConvert_v1beta1_ResourceSliceMixins_To_api_ResourceSliceMixins(in, out, s); err != nil {
		return err
	}
	if len(in.Device) > 0 {
		out.Device = make(map[UniqueString]DeviceMixin, len(in.Device))
		for _, mixin := range in.Device {
			item := DeviceMixin{}
			if err := Convert_v1beta1_DeviceMixin_To_api_DeviceMixin(&mixin, &item, s); err != nil {
				return err
			}
			name := MakeUniqueString(mixin.Name)
			out.Device[name] = item
		}
	}

	if len(in.CounterSet) > 0 {
		out.CounterSet = make(map[UniqueString]CounterSetMixin, len(in.CounterSet))
		for _, mixin := range in.CounterSet {
			item := CounterSetMixin{}
			if err := Convert_v1beta1_CounterSetMixin_To_api_CounterSetMixin(&mixin, &item, s); err != nil {
				return err
			}
			name := MakeUniqueString(mixin.Name)
			out.CounterSet[name] = item
		}
	}

	if len(in.DeviceCounterConsumption) > 0 {
		out.DeviceCounterConsumption = make(map[UniqueString]DeviceCounterConsumptionMixin, len(in.DeviceCounterConsumption))
		for _, mixin := range in.DeviceCounterConsumption {
			item := DeviceCounterConsumptionMixin{}
			if err := Convert_v1beta1_DeviceCounterConsumptionMixin_To_api_DeviceCounterConsumptionMixin(&mixin, &item, s); err != nil {
				return err
			}
			name := MakeUniqueString(mixin.Name)
			out.DeviceCounterConsumption[name] = item
		}
	}
	return nil
}

func Convert_v1beta1_DeviceCounterConsumptionMixin_To_api_DeviceCounterConsumptionMixin(in *v1beta1.DeviceCounterConsumptionMixin, out *DeviceCounterConsumptionMixin, s conversion.Scope) error {
	if err := autoConvert_v1beta1_DeviceCounterConsumptionMixin_To_api_DeviceCounterConsumptionMixin(in, out, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1beta1_CounterSetMixin_To_api_CounterSetMixin(in *v1beta1.CounterSetMixin, out *CounterSetMixin, s conversion.Scope) error {
	if err := autoConvert_v1beta1_CounterSetMixin_To_api_CounterSetMixin(in, out, s); err != nil {
		return err
	}
	return nil
}

func Convert_v1beta1_DeviceMixin_To_api_DeviceMixin(in *v1beta1.DeviceMixin, out *DeviceMixin, s conversion.Scope) error {
	if err := autoConvert_v1beta1_DeviceMixin_To_api_DeviceMixin(in, out, s); err != nil {
		return err
	}
	return nil
}
