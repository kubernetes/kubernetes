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

package v1beta1

import (
	unsafe "unsafe"

	corev1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	conversion "k8s.io/apimachinery/pkg/conversion"
)

func Convert_v1beta1_DeviceRequest_To_v1_DeviceRequest(in *resourcev1beta1.DeviceRequest, out *resourceapi.DeviceRequest, s conversion.Scope) error {
	if err := autoConvert_v1beta1_DeviceRequest_To_v1_DeviceRequest(in, out, s); err != nil {
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
				err := Convert_v1beta1_DeviceSelector_To_v1_DeviceSelector(&in.Selectors[i], &selector, s)
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
			if err := Convert_v1beta1_DeviceToleration_To_v1_DeviceToleration(&e, &toleration, s); err != nil {
				return err
			}
			tolerations = append(tolerations, toleration)
		}
		exactDeviceRequest.Tolerations = tolerations
		out.Exactly = &exactDeviceRequest
	}
	return nil
}

func hasAnyMainRequestFieldsSet(deviceRequest *resourcev1beta1.DeviceRequest) bool {
	return deviceRequest.DeviceClassName != "" ||
		deviceRequest.Selectors != nil ||
		deviceRequest.AllocationMode != "" ||
		deviceRequest.Count != 0 ||
		deviceRequest.AdminAccess != nil ||
		deviceRequest.Tolerations != nil
}

func Convert_v1_DeviceRequest_To_v1beta1_DeviceRequest(in *resourceapi.DeviceRequest, out *resourcev1beta1.DeviceRequest, s conversion.Scope) error {
	if err := autoConvert_v1_DeviceRequest_To_v1beta1_DeviceRequest(in, out, s); err != nil {
		return err
	}
	if in.Exactly != nil {
		out.DeviceClassName = in.Exactly.DeviceClassName
		if in.Exactly.Selectors != nil {
			selectors := make([]resourcev1beta1.DeviceSelector, 0, len(in.Exactly.Selectors))
			for i := range in.Exactly.Selectors {
				var selector resourcev1beta1.DeviceSelector
				err := Convert_v1_DeviceSelector_To_v1beta1_DeviceSelector(&in.Exactly.Selectors[i], &selector, s)
				if err != nil {
					return err
				}
				selectors = append(selectors, selector)
			}
			out.Selectors = selectors
		}
		out.AllocationMode = resourcev1beta1.DeviceAllocationMode(in.Exactly.AllocationMode)
		out.Count = in.Exactly.Count
		out.AdminAccess = in.Exactly.AdminAccess
		var tolerations []resourcev1beta1.DeviceToleration
		for _, e := range in.Exactly.Tolerations {
			var toleration resourcev1beta1.DeviceToleration
			if err := Convert_v1_DeviceToleration_To_v1beta1_DeviceToleration(&e, &toleration, s); err != nil {
				return err
			}
			tolerations = append(tolerations, toleration)
		}
		out.Tolerations = tolerations
	}
	return nil
}

func Convert_v1beta1_ResourceSliceSpec_To_v1_ResourceSliceSpec(in *resourcev1beta1.ResourceSliceSpec, out *resourceapi.ResourceSliceSpec, s conversion.Scope) error {
	if err := autoConvert_v1beta1_ResourceSliceSpec_To_v1_ResourceSliceSpec(in, out, s); err != nil {
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

func Convert_v1_ResourceSliceSpec_To_v1beta1_ResourceSliceSpec(in *resourceapi.ResourceSliceSpec, out *resourcev1beta1.ResourceSliceSpec, s conversion.Scope) error {
	if err := autoConvert_v1_ResourceSliceSpec_To_v1beta1_ResourceSliceSpec(in, out, s); err != nil {
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

func Convert_v1beta1_Device_To_v1_Device(in *resourcev1beta1.Device, out *resourceapi.Device, s conversion.Scope) error {
	if err := autoConvert_v1beta1_Device_To_v1_Device(in, out, s); err != nil {
		return err
	}
	if in.Basic != nil {
		basic := in.Basic
		if len(basic.Attributes) > 0 {
			attributes := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
			if err := convert_v1beta1_Attributes_To_v1_Attributes(basic.Attributes, attributes, s); err != nil {
				return err
			}
			out.Attributes = attributes
		}
		if len(basic.Capacity) > 0 {
			capacity := make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity)
			if err := convert_v1beta1_Capacity_To_v1_Capacity(basic.Capacity, capacity, s); err != nil {
				return err
			}
			out.Capacity = capacity
		}
		var consumesCounters []resourceapi.DeviceCounterConsumption
		for _, e := range basic.ConsumesCounters {
			var deviceCounterConsumption resourceapi.DeviceCounterConsumption
			if err := Convert_v1beta1_DeviceCounterConsumption_To_v1_DeviceCounterConsumption(&e, &deviceCounterConsumption, s); err != nil {
				return err
			}
			consumesCounters = append(consumesCounters, deviceCounterConsumption)
		}
		out.ConsumesCounters = consumesCounters
		out.NodeName = basic.NodeName
		out.NodeSelector = basic.NodeSelector
		out.AllNodes = basic.AllNodes
		var taints []resourceapi.DeviceTaint
		for _, e := range basic.Taints {
			var taint resourceapi.DeviceTaint
			if err := Convert_v1beta1_DeviceTaint_To_v1_DeviceTaint(&e, &taint, s); err != nil {
				return err
			}
			taints = append(taints, taint)
		}
		out.Taints = taints
	}
	return nil
}

func Convert_v1_Device_To_v1beta1_Device(in *resourceapi.Device, out *resourcev1beta1.Device, s conversion.Scope) error {
	if err := autoConvert_v1_Device_To_v1beta1_Device(in, out, s); err != nil {
		return err
	}
	out.Basic = &resourcev1beta1.BasicDevice{}
	if len(in.Attributes) > 0 {
		attributes := make(map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceAttribute)
		if err := convert_v1_Attributes_To_v1beta1_Attributes(in.Attributes, attributes, s); err != nil {
			return err
		}
		out.Basic.Attributes = attributes
	}

	if len(in.Capacity) > 0 {
		capacity := make(map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceCapacity)
		if err := convert_v1_Capacity_To_v1beta1_Capacity(in.Capacity, capacity, s); err != nil {
			return err
		}
		out.Basic.Capacity = capacity
	}
	var consumesCounters []resourcev1beta1.DeviceCounterConsumption
	for _, e := range in.ConsumesCounters {
		var deviceCounterConsumption resourcev1beta1.DeviceCounterConsumption
		if err := Convert_v1_DeviceCounterConsumption_To_v1beta1_DeviceCounterConsumption(&e, &deviceCounterConsumption, s); err != nil {
			return err
		}
		consumesCounters = append(consumesCounters, deviceCounterConsumption)
	}
	out.Basic.ConsumesCounters = consumesCounters
	out.Basic.NodeName = in.NodeName
	out.Basic.NodeSelector = (*corev1.NodeSelector)(unsafe.Pointer(in.NodeSelector))
	out.Basic.AllNodes = in.AllNodes
	var taints []resourcev1beta1.DeviceTaint
	for _, e := range in.Taints {
		var taint resourcev1beta1.DeviceTaint
		if err := Convert_v1_DeviceTaint_To_v1beta1_DeviceTaint(&e, &taint, s); err != nil {
			return err
		}
		taints = append(taints, taint)
	}
	out.Basic.Taints = taints
	return nil
}

func convert_v1_Attributes_To_v1beta1_Attributes(in map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, out map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceAttribute, s conversion.Scope) error {
	for k, v := range in {
		var a resourcev1beta1.DeviceAttribute
		if err := Convert_v1_DeviceAttribute_To_v1beta1_DeviceAttribute(&v, &a, s); err != nil {
			return err
		}
		out[resourcev1beta1.QualifiedName(k)] = a
	}
	return nil
}

func convert_v1_Capacity_To_v1beta1_Capacity(in map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, out map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceCapacity, s conversion.Scope) error {
	for k, v := range in {
		var c resourcev1beta1.DeviceCapacity
		if err := Convert_v1_DeviceCapacity_To_v1beta1_DeviceCapacity(&v, &c, s); err != nil {
			return err
		}
		out[resourcev1beta1.QualifiedName(k)] = c
	}
	return nil
}

func convert_v1beta1_Attributes_To_v1_Attributes(in map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceAttribute, out map[resourceapi.QualifiedName]resourceapi.DeviceAttribute, s conversion.Scope) error {
	for k, v := range in {
		var a resourceapi.DeviceAttribute
		if err := Convert_v1beta1_DeviceAttribute_To_v1_DeviceAttribute(&v, &a, s); err != nil {
			return err
		}
		out[resourceapi.QualifiedName(k)] = a
	}
	return nil
}

func convert_v1beta1_Capacity_To_v1_Capacity(in map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceCapacity, out map[resourceapi.QualifiedName]resourceapi.DeviceCapacity, s conversion.Scope) error {
	for k, v := range in {
		var c resourceapi.DeviceCapacity
		if err := Convert_v1beta1_DeviceCapacity_To_v1_DeviceCapacity(&v, &c, s); err != nil {
			return err
		}
		out[resourceapi.QualifiedName(k)] = c
	}
	return nil
}
