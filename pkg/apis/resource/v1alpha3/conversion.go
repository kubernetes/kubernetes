/*
Copyright 2022 The Kubernetes Authors.

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

package v1alpha3

import (
	"fmt"
	unsafe "unsafe"

	corev1 "k8s.io/api/core/v1"
	resourcev1alpha3 "k8s.io/api/resource/v1alpha3"
	"k8s.io/apimachinery/pkg/api/resource"
	conversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	core "k8s.io/kubernetes/pkg/apis/core"
	resourceapi "k8s.io/kubernetes/pkg/apis/resource"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	if err := scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("ResourceSlice"),
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name", resourceapi.ResourceSliceSelectorNodeName, resourceapi.ResourceSliceSelectorDriver:
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported for %s: %s", SchemeGroupVersion.WithKind("ResourceSlice"), label)
			}
		}); err != nil {
		return err
	}

	return nil
}

func Convert_resource_DeviceCapacity_To_resource_Quantity(in *resourceapi.DeviceCapacity, out *resource.Quantity, s conversion.Scope) error {
	*out = in.Value
	return nil
}

func Convert_resource_Quantity_To_resource_DeviceCapacity(in *resource.Quantity, out *resourceapi.DeviceCapacity, s conversion.Scope) error {
	out.Value = *in
	return nil
}

func Convert_v1alpha3_DeviceRequest_To_resource_DeviceRequest(in *resourcev1alpha3.DeviceRequest, out *resourceapi.DeviceRequest, s conversion.Scope) error {
	if err := autoConvert_v1alpha3_DeviceRequest_To_resource_DeviceRequest(in, out, s); err != nil {
		return err
	}
	// If any fields on the main request is set, we create a ExactDeviceRequest
	// and set the Exactly field. It might be invalid but that will be caught in validation.
	if hasAnyMainRequestFieldsSet(in) {
		var exactDeviceRequest resourceapi.ExactDeviceRequest
		exactDeviceRequest.DeviceClassName = in.DeviceClassName
		if in.Selectors != nil {
			selectors := make([]resourceapi.DeviceSelector, 0, len(in.Selectors))
			for i := range in.Selectors {
				var selector resourceapi.DeviceSelector
				err := Convert_v1alpha3_DeviceSelector_To_resource_DeviceSelector(&in.Selectors[i], &selector, s)
				if err != nil {
					return err
				}
				selectors = append(selectors, selector)
			}
			exactDeviceRequest.Selectors = selectors
		}
		exactDeviceRequest.AllocationMode = resourceapi.DeviceAllocationMode(in.AllocationMode)
		exactDeviceRequest.Count = in.Count
		exactDeviceRequest.AdminAccess = in.AdminAccess
		var tolerations []resourceapi.DeviceToleration
		for _, e := range in.Tolerations {
			var toleration resourceapi.DeviceToleration
			if err := Convert_v1alpha3_DeviceToleration_To_resource_DeviceToleration(&e, &toleration, s); err != nil {
				return err
			}
			tolerations = append(tolerations, toleration)
		}
		exactDeviceRequest.Tolerations = tolerations
		out.Exactly = &exactDeviceRequest
	}
	return nil
}

func hasAnyMainRequestFieldsSet(deviceRequest *resourcev1alpha3.DeviceRequest) bool {
	return deviceRequest.DeviceClassName != "" ||
		deviceRequest.Selectors != nil ||
		deviceRequest.AllocationMode != "" ||
		deviceRequest.Count != 0 ||
		deviceRequest.AdminAccess != nil ||
		deviceRequest.Tolerations != nil
}

func Convert_resource_DeviceRequest_To_v1alpha3_DeviceRequest(in *resourceapi.DeviceRequest, out *resourcev1alpha3.DeviceRequest, s conversion.Scope) error {
	if err := autoConvert_resource_DeviceRequest_To_v1alpha3_DeviceRequest(in, out, s); err != nil {
		return err
	}
	if in.Exactly != nil {
		out.DeviceClassName = in.Exactly.DeviceClassName
		if in.Exactly.Selectors != nil {
			selectors := make([]resourcev1alpha3.DeviceSelector, 0, len(in.Exactly.Selectors))
			for i := range in.Exactly.Selectors {
				var selector resourcev1alpha3.DeviceSelector
				err := Convert_resource_DeviceSelector_To_v1alpha3_DeviceSelector(&in.Exactly.Selectors[i], &selector, s)
				if err != nil {
					return err
				}
				selectors = append(selectors, selector)
			}
			out.Selectors = selectors
		}
		out.AllocationMode = resourcev1alpha3.DeviceAllocationMode(in.Exactly.AllocationMode)
		out.Count = in.Exactly.Count
		out.AdminAccess = in.Exactly.AdminAccess
		var tolerations []resourcev1alpha3.DeviceToleration
		for _, e := range in.Exactly.Tolerations {
			var toleration resourcev1alpha3.DeviceToleration
			if err := Convert_resource_DeviceToleration_To_v1alpha3_DeviceToleration(&e, &toleration, s); err != nil {
				return err
			}
			tolerations = append(tolerations, toleration)
		}
		out.Tolerations = tolerations
	}
	return nil
}

func Convert_v1alpha3_ResourceSliceSpec_To_resource_ResourceSliceSpec(in *resourcev1alpha3.ResourceSliceSpec, out *resourceapi.ResourceSliceSpec, s conversion.Scope) error {
	if err := autoConvert_v1alpha3_ResourceSliceSpec_To_resource_ResourceSliceSpec(in, out, s); err != nil {
		return err
	}
	if in.NodeName == "" {
		out.NodeName = nil
	} else {
		out.NodeName = &in.NodeName
	}
	if !in.AllNodes {
		out.AllNodes = nil
	} else {
		out.AllNodes = &in.AllNodes
	}
	return nil
}

func Convert_resource_ResourceSliceSpec_To_v1alpha3_ResourceSliceSpec(in *resourceapi.ResourceSliceSpec, out *resourcev1alpha3.ResourceSliceSpec, s conversion.Scope) error {
	if err := autoConvert_resource_ResourceSliceSpec_To_v1alpha3_ResourceSliceSpec(in, out, s); err != nil {
		return err
	}
	if in.NodeName == nil {
		out.NodeName = ""
	} else {
		out.NodeName = *in.NodeName
	}
	if in.AllNodes == nil {
		out.AllNodes = false
	} else {
		out.AllNodes = *in.AllNodes
	}
	return nil
}

func Convert_v1alpha3_Device_To_resource_Device(in *resourcev1alpha3.Device, out *resourceapi.Device, s conversion.Scope) error {
	if err := autoConvert_v1alpha3_Device_To_resource_Device(in, out, s); err != nil {
		return err
	}
	if in.Basic != nil {
		basic := in.Basic
		if len(basic.Attributes) > 0 {
			attributes := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
			if err := convert_v1alpha3_Attributes_To_resource_Attributes(basic.Attributes, attributes, s); err != nil {
				return err
			}
			out.Attributes = attributes
		}

		if len(basic.Capacity) > 0 {
			capacity := make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity)
			if err := convert_v1alpha3_Capacity_To_resource_Capacity(basic.Capacity, capacity, s); err != nil {
				return err
			}
			out.Capacity = capacity
		}
		var consumesCounters []resourceapi.DeviceCounterConsumption
		for _, e := range basic.ConsumesCounters {
			var deviceCounterConsumption resourceapi.DeviceCounterConsumption
			if err := Convert_v1alpha3_DeviceCounterConsumption_To_resource_DeviceCounterConsumption(&e, &deviceCounterConsumption, s); err != nil {
				return err
			}
			consumesCounters = append(consumesCounters, deviceCounterConsumption)
		}
		out.ConsumesCounters = consumesCounters
		out.NodeName = basic.NodeName
		out.NodeSelector = (*core.NodeSelector)(unsafe.Pointer(basic.NodeSelector))
		out.AllNodes = basic.AllNodes
		var taints []resourceapi.DeviceTaint
		for _, e := range basic.Taints {
			var taint resourceapi.DeviceTaint
			if err := Convert_v1alpha3_DeviceTaint_To_resource_DeviceTaint(&e, &taint, s); err != nil {
				return err
			}
			taints = append(taints, taint)
		}
		out.Taints = taints
	}
	return nil
}

func Convert_resource_Device_To_v1alpha3_Device(in *resourceapi.Device, out *resourcev1alpha3.Device, s conversion.Scope) error {
	if err := autoConvert_resource_Device_To_v1alpha3_Device(in, out, s); err != nil {
		return err
	}
	out.Basic = &resourcev1alpha3.BasicDevice{}
	if len(in.Attributes) > 0 {
		attributes := make(map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute)
		if err := convert_resource_Attributes_To_v1alpha3_Attributes(in.Attributes, attributes, s); err != nil {
			return err
		}
		out.Basic.Attributes = attributes
	}

	if len(in.Capacity) > 0 {
		capacity := make(map[resourcev1alpha3.QualifiedName]resource.Quantity)
		if err := convert_resource_Capacity_To_v1alpha3_Capacity(in.Capacity, capacity, s); err != nil {
			return err
		}
		out.Basic.Capacity = capacity
	}
	var consumesCounters []resourcev1alpha3.DeviceCounterConsumption
	for _, e := range in.ConsumesCounters {
		var deviceCounterConsumption resourcev1alpha3.DeviceCounterConsumption
		if err := Convert_resource_DeviceCounterConsumption_To_v1alpha3_DeviceCounterConsumption(&e, &deviceCounterConsumption, s); err != nil {
			return err
		}
		consumesCounters = append(consumesCounters, deviceCounterConsumption)
	}
	out.Basic.ConsumesCounters = consumesCounters
	out.Basic.NodeName = in.NodeName
	out.Basic.NodeSelector = (*corev1.NodeSelector)(unsafe.Pointer(in.NodeSelector))
	out.Basic.AllNodes = in.AllNodes
	var taints []resourcev1alpha3.DeviceTaint
	for _, e := range in.Taints {
		var taint resourcev1alpha3.DeviceTaint
		if err := Convert_resource_DeviceTaint_To_v1alpha3_DeviceTaint(&e, &taint, s); err != nil {
			return err
		}
		taints = append(taints, taint)
	}
	out.Basic.Taints = taints
	return nil
}

func convert_resource_Attributes_To_v1alpha3_Attributes(in map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, out map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute, s conversion.Scope) error {
	for k, v := range in {
		var a resourcev1alpha3.DeviceAttribute
		if err := Convert_resource_DeviceAttribute_To_v1alpha3_DeviceAttribute(&v, &a, s); err != nil {
			return err
		}
		out[resourcev1alpha3.QualifiedName(k)] = a
	}
	return nil
}

func convert_resource_Capacity_To_v1alpha3_Capacity(in map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, out map[resourcev1alpha3.QualifiedName]resource.Quantity, s conversion.Scope) error {
	for k, v := range in {
		var c resource.Quantity
		if err := Convert_resource_DeviceCapacity_To_resource_Quantity(&v, &c, s); err != nil {
			return err
		}
		out[resourcev1alpha3.QualifiedName(k)] = c
	}
	return nil
}

func convert_v1alpha3_Attributes_To_resource_Attributes(in map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute, out map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, s conversion.Scope) error {
	for k, v := range in {
		var a resourceapi.DeviceAttribute
		if err := Convert_v1alpha3_DeviceAttribute_To_resource_DeviceAttribute(&v, &a, s); err != nil {
			return err
		}
		out[resourceapi.QualifiedName(k)] = a
	}
	return nil
}

func convert_v1alpha3_Capacity_To_resource_Capacity(in map[resourcev1alpha3.QualifiedName]resource.Quantity, out map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, s conversion.Scope) error {
	for k, v := range in {
		var c resourceapi.DeviceCapacity
		if err := Convert_resource_Quantity_To_resource_DeviceCapacity(&v, &c, s); err != nil {
			return err
		}
		out[resourceapi.QualifiedName(k)] = c
	}
	return nil
}
