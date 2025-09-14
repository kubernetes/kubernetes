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

package v1beta1

import (
	"fmt"
	unsafe "unsafe"

	corev1 "k8s.io/api/core/v1"
	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	conversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
	core "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/resource"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	if err := scheme.AddFieldLabelConversionFunc(SchemeGroupVersion.WithKind("ResourceSlice"),
		func(label, value string) (string, string, error) {
			switch label {
			case "metadata.name", resourcev1beta1.ResourceSliceSelectorNodeName, resourcev1beta1.ResourceSliceSelectorDriver:
				return label, value, nil
			default:
				return "", "", fmt.Errorf("field label not supported for %s: %s", SchemeGroupVersion.WithKind("ResourceSlice"), label)
			}
		}); err != nil {
		return err
	}

	return nil
}

func Convert_v1beta1_DeviceRequest_To_resource_DeviceRequest(in *resourcev1beta1.DeviceRequest, out *resource.DeviceRequest, s conversion.Scope) error {
	if err := autoConvert_v1beta1_DeviceRequest_To_resource_DeviceRequest(in, out, s); err != nil {
		return err
	}
	// If any fields on the main request is set, we create a ExactDeviceRequest
	// and set the Exactly field. It might be invalid but that will be caught in validation.
	if hasAnyMainRequestFieldsSet(in) {
		var exactDeviceRequest resource.ExactDeviceRequest
		exactDeviceRequest.DeviceClassName = in.DeviceClassName
		if in.Selectors != nil {
			selectors := make([]resource.DeviceSelector, 0, len(in.Selectors))
			for i := range in.Selectors {
				var selector resource.DeviceSelector
				err := Convert_v1beta1_DeviceSelector_To_resource_DeviceSelector(&in.Selectors[i], &selector, s)
				if err != nil {
					return err
				}
				selectors = append(selectors, selector)
			}
			exactDeviceRequest.Selectors = selectors
		}
		exactDeviceRequest.AllocationMode = resource.DeviceAllocationMode(in.AllocationMode)
		exactDeviceRequest.Count = in.Count
		exactDeviceRequest.AdminAccess = in.AdminAccess
		var tolerations []resource.DeviceToleration
		for _, e := range in.Tolerations {
			var toleration resource.DeviceToleration
			if err := Convert_v1beta1_DeviceToleration_To_resource_DeviceToleration(&e, &toleration, s); err != nil {
				return err
			}
			tolerations = append(tolerations, toleration)
		}
		exactDeviceRequest.Tolerations = tolerations
		if in.Capacity != nil {
			var capacity resource.CapacityRequirements
			if err := Convert_v1beta1_CapacityRequirements_To_resource_CapacityRequirements(in.Capacity, &capacity, s); err != nil {
				return err
			}
			exactDeviceRequest.Capacity = &capacity
		}

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
		deviceRequest.Tolerations != nil ||
		deviceRequest.Capacity != nil
}

func Convert_resource_DeviceRequest_To_v1beta1_DeviceRequest(in *resource.DeviceRequest, out *resourcev1beta1.DeviceRequest, s conversion.Scope) error {
	if err := autoConvert_resource_DeviceRequest_To_v1beta1_DeviceRequest(in, out, s); err != nil {
		return err
	}
	if in.Exactly != nil {
		out.DeviceClassName = in.Exactly.DeviceClassName
		if in.Exactly.Selectors != nil {
			selectors := make([]resourcev1beta1.DeviceSelector, 0, len(in.Exactly.Selectors))
			for i := range in.Exactly.Selectors {
				var selector resourcev1beta1.DeviceSelector
				err := Convert_resource_DeviceSelector_To_v1beta1_DeviceSelector(&in.Exactly.Selectors[i], &selector, s)
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
			if err := Convert_resource_DeviceToleration_To_v1beta1_DeviceToleration(&e, &toleration, s); err != nil {
				return err
			}
			tolerations = append(tolerations, toleration)
		}
		out.Tolerations = tolerations
		if in.Exactly.Capacity != nil {
			var capacity resourcev1beta1.CapacityRequirements
			if err := Convert_resource_CapacityRequirements_To_v1beta1_CapacityRequirements(in.Exactly.Capacity, &capacity, s); err != nil {
				return err
			}
			out.Capacity = &capacity
		}
	}
	return nil
}

func Convert_v1beta1_ResourceSliceSpec_To_resource_ResourceSliceSpec(in *resourcev1beta1.ResourceSliceSpec, out *resource.ResourceSliceSpec, s conversion.Scope) error {
	if err := autoConvert_v1beta1_ResourceSliceSpec_To_resource_ResourceSliceSpec(in, out, s); err != nil {
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

func Convert_resource_ResourceSliceSpec_To_v1beta1_ResourceSliceSpec(in *resource.ResourceSliceSpec, out *resourcev1beta1.ResourceSliceSpec, s conversion.Scope) error {
	if err := autoConvert_resource_ResourceSliceSpec_To_v1beta1_ResourceSliceSpec(in, out, s); err != nil {
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

func Convert_v1beta1_Device_To_resource_Device(in *resourcev1beta1.Device, out *resource.Device, s conversion.Scope) error {
	if err := autoConvert_v1beta1_Device_To_resource_Device(in, out, s); err != nil {
		return err
	}
	if in.Basic != nil {
		basic := in.Basic
		if len(basic.Attributes) > 0 {
			attributes := make(map[resource.QualifiedName]resource.DeviceAttribute)
			if err := convert_v1beta1_Attributes_To_resource_Attributes(basic.Attributes, attributes, s); err != nil {
				return err
			}
			out.Attributes = attributes
		}
		if len(basic.Capacity) > 0 {
			capacity := make(map[resource.QualifiedName]resource.DeviceCapacity)
			if err := convert_v1beta1_Capacity_To_resource_Capacity(basic.Capacity, capacity, s); err != nil {
				return err
			}
			out.Capacity = capacity
		}
		var consumesCounters []resource.DeviceCounterConsumption
		for _, e := range basic.ConsumesCounters {
			var deviceCounterConsumption resource.DeviceCounterConsumption
			if err := Convert_v1beta1_DeviceCounterConsumption_To_resource_DeviceCounterConsumption(&e, &deviceCounterConsumption, s); err != nil {
				return err
			}
			consumesCounters = append(consumesCounters, deviceCounterConsumption)
		}
		out.ConsumesCounters = consumesCounters
		out.NodeName = basic.NodeName
		out.NodeSelector = (*core.NodeSelector)(unsafe.Pointer(basic.NodeSelector))
		out.AllNodes = basic.AllNodes
		var taints []resource.DeviceTaint
		for _, e := range basic.Taints {
			var taint resource.DeviceTaint
			if err := Convert_v1beta1_DeviceTaint_To_resource_DeviceTaint(&e, &taint, s); err != nil {
				return err
			}
			taints = append(taints, taint)
		}
		out.Taints = taints
		out.BindsToNode = basic.BindsToNode
		out.BindingConditions = basic.BindingConditions
		out.BindingFailureConditions = basic.BindingFailureConditions
		out.AllowMultipleAllocations = in.Basic.AllowMultipleAllocations
	}
	return nil
}

func Convert_resource_Device_To_v1beta1_Device(in *resource.Device, out *resourcev1beta1.Device, s conversion.Scope) error {
	if err := autoConvert_resource_Device_To_v1beta1_Device(in, out, s); err != nil {
		return err
	}
	out.Basic = &resourcev1beta1.BasicDevice{}
	if len(in.Attributes) > 0 {
		attributes := make(map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceAttribute)
		if err := convert_resource_Attributes_To_v1beta1_Attributes(in.Attributes, attributes, s); err != nil {
			return err
		}
		out.Basic.Attributes = attributes
	}

	if len(in.Capacity) > 0 {
		capacity := make(map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceCapacity)
		if err := convert_resource_Capacity_To_v1beta1_Capacity(in.Capacity, capacity, s); err != nil {
			return err
		}
		out.Basic.Capacity = capacity
	}
	var consumesCounters []resourcev1beta1.DeviceCounterConsumption
	for _, e := range in.ConsumesCounters {
		var deviceCounterConsumption resourcev1beta1.DeviceCounterConsumption
		if err := Convert_resource_DeviceCounterConsumption_To_v1beta1_DeviceCounterConsumption(&e, &deviceCounterConsumption, s); err != nil {
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
		if err := Convert_resource_DeviceTaint_To_v1beta1_DeviceTaint(&e, &taint, s); err != nil {
			return err
		}
		taints = append(taints, taint)
	}
	out.Basic.Taints = taints
	out.Basic.BindsToNode = in.BindsToNode
	out.Basic.BindingConditions = in.BindingConditions
	out.Basic.BindingFailureConditions = in.BindingFailureConditions
	out.Basic.AllowMultipleAllocations = in.AllowMultipleAllocations
	return nil
}

func convert_resource_Attributes_To_v1beta1_Attributes(in map[resource.QualifiedName]resource.DeviceAttribute, out map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceAttribute, s conversion.Scope) error {
	for k, v := range in {
		var a resourcev1beta1.DeviceAttribute
		if err := Convert_resource_DeviceAttribute_To_v1beta1_DeviceAttribute(&v, &a, s); err != nil {
			return err
		}
		out[resourcev1beta1.QualifiedName(k)] = a
	}
	return nil
}

func convert_resource_Capacity_To_v1beta1_Capacity(in map[resource.QualifiedName]resource.DeviceCapacity, out map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceCapacity, s conversion.Scope) error {
	for k, v := range in {
		var c resourcev1beta1.DeviceCapacity
		if err := Convert_resource_DeviceCapacity_To_v1beta1_DeviceCapacity(&v, &c, s); err != nil {
			return err
		}
		out[resourcev1beta1.QualifiedName(k)] = c
	}
	return nil
}

func convert_v1beta1_Attributes_To_resource_Attributes(in map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceAttribute, out map[resource.QualifiedName]resource.DeviceAttribute, s conversion.Scope) error {
	for k, v := range in {
		var a resource.DeviceAttribute
		if err := Convert_v1beta1_DeviceAttribute_To_resource_DeviceAttribute(&v, &a, s); err != nil {
			return err
		}
		out[resource.QualifiedName(k)] = a
	}
	return nil
}

func convert_v1beta1_Capacity_To_resource_Capacity(in map[resourcev1beta1.QualifiedName]resourcev1beta1.DeviceCapacity, out map[resource.QualifiedName]resource.DeviceCapacity, s conversion.Scope) error {
	for k, v := range in {
		var c resource.DeviceCapacity
		if err := Convert_v1beta1_DeviceCapacity_To_resource_DeviceCapacity(&v, &c, s); err != nil {
			return err
		}
		out[resource.QualifiedName(k)] = c
	}
	return nil
}
