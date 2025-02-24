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
	"slices"
	"strings"
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
	out.Name = in.Name
	for i := range in.FirstAvailable {
		var deviceSubRequest resourceapi.DeviceSubRequest
		err := Convert_v1alpha3_DeviceSubRequest_To_resource_DeviceSubRequest(&in.FirstAvailable[i], &deviceSubRequest, s)
		if err != nil {
			return err
		}
		out.FirstAvailable = append(out.FirstAvailable, deviceSubRequest)
	}

	// If any fields on the main request is set, we create a SpecificDeviceRequest
	// and set the Exactly field. It might be invalid but that will be caught in validation.
	if hasAnyMainRequestFieldsSet(in) {
		var specificDeviceRequest resourceapi.SpecificDeviceRequest
		specificDeviceRequest.DeviceClassName = in.DeviceClassName
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
			specificDeviceRequest.Selectors = selectors
		}
		specificDeviceRequest.AllocationMode = resourceapi.DeviceAllocationMode(in.AllocationMode)
		specificDeviceRequest.Count = in.Count
		specificDeviceRequest.AdminAccess = in.AdminAccess
		out.Exactly = &specificDeviceRequest
	}
	return nil
}

func hasAnyMainRequestFieldsSet(deviceRequest *resourcev1alpha3.DeviceRequest) bool {
	return deviceRequest.DeviceClassName != "" ||
		deviceRequest.Selectors != nil ||
		deviceRequest.AllocationMode != "" ||
		deviceRequest.Count != 0 ||
		deviceRequest.AdminAccess != nil
}

func Convert_resource_DeviceRequest_To_v1alpha3_DeviceRequest(in *resourceapi.DeviceRequest, out *resourcev1alpha3.DeviceRequest, s conversion.Scope) error {
	out.Name = in.Name
	for i := range in.FirstAvailable {
		var deviceSubRequest resourcev1alpha3.DeviceSubRequest
		err := Convert_resource_DeviceSubRequest_To_v1alpha3_DeviceSubRequest(&in.FirstAvailable[i], &deviceSubRequest, s)
		if err != nil {
			return err
		}
		out.FirstAvailable = append(out.FirstAvailable, deviceSubRequest)
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
	}
	return nil
}

const (
	basicDeviceNamesAnnotation = "resource.k8s.io/basic-device-names"
)

func Convert_resource_ResourceSlice_To_v1alpha3_ResourceSlice(in *resourceapi.ResourceSlice, out *resourcev1alpha3.ResourceSlice, s conversion.Scope) error {
	names, found := in.Annotations[basicDeviceNamesAnnotation]
	if found {
		splits := strings.Split(names, ",")
		s.Meta().Context = splits
	}
	err := autoConvert_resource_ResourceSlice_To_v1alpha3_ResourceSlice(in, out, s)
	if err != nil {
		return err
	}
	annos := out.Annotations
	delete(annos, basicDeviceNamesAnnotation)
	if len(annos) > 0 {
		out.Annotations = annos
	} else {
		out.Annotations = nil
	}
	return nil
}

func Convert_v1alpha3_ResourceSlice_To_resource_ResourceSlice(in *resourcev1alpha3.ResourceSlice, out *resourceapi.ResourceSlice, s conversion.Scope) error {
	s.Meta().Context = []string{}
	err := autoConvert_v1alpha3_ResourceSlice_To_resource_ResourceSlice(in, out, s)
	if err != nil {
		return err
	}
	names := s.Meta().Context.([]string)
	if len(names) > 0 {
		if out.Annotations == nil {
			out.Annotations = make(map[string]string)
		}
		out.Annotations[basicDeviceNamesAnnotation] = strings.Join(names, ",")
	}
	return nil
}

func Convert_resource_Device_To_v1alpha3_Device(in *resourceapi.Device, out *resourcev1alpha3.Device, s conversion.Scope) error {
	out.Name = in.Name
	basicDeviceNames, ok := s.Meta().Context.([]string)
	if !ok {
		basicDeviceNames = []string{}
	}
	if slices.Contains(basicDeviceNames, in.Name) {
		basic := &resourcev1alpha3.BasicDevice{}
		if len(in.Attributes) > 0 {
			attributes := make(map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute)
			if err := convert_resource_Attributes_To_v1alpha3_Attributes(in.Attributes, attributes, s); err != nil {
				return err
			}
			basic.Attributes = attributes
		}

		if len(in.Capacity) > 0 {
			capacity := make(map[resourcev1alpha3.QualifiedName]resource.Quantity)
			if err := convert_resource_Capacity_To_v1alpha3_Capacity(in.Capacity, capacity, s); err != nil {
				return err
			}
			basic.Capacity = capacity
		}
		out.Basic = basic
		return nil
	}

	composite := &resourcev1alpha3.CompositeDevice{}

	var includes []resourcev1alpha3.DeviceMixinRef
	for _, e := range in.Includes {
		var deviceMixinRef resourcev1alpha3.DeviceMixinRef
		if err := Convert_resource_DeviceMixinRef_To_v1alpha3_DeviceMixinRef(&e, &deviceMixinRef, s); err != nil {
			return err
		}
		includes = append(includes, deviceMixinRef)
	}
	composite.Includes = includes

	if len(in.Attributes) > 0 {
		attributes := make(map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute)
		if err := convert_resource_Attributes_To_v1alpha3_Attributes(in.Attributes, attributes, s); err != nil {
			return err
		}
		composite.Attributes = attributes
	}

	if len(in.Attributes) > 0 {
		capacity := make(map[resourcev1alpha3.QualifiedName]resource.Quantity)
		if err := convert_resource_Capacity_To_v1alpha3_Capacity(in.Capacity, capacity, s); err != nil {
			return err
		}
		composite.Capacity = capacity
	}

	var consumesCapacity []resourcev1alpha3.DeviceCapacityConsumption
	for _, e := range in.ConsumesCapacity {
		var deviceCapacityConsumption resourcev1alpha3.DeviceCapacityConsumption
		if err := Convert_resource_DeviceCapacityConsumption_To_v1alpha3_DeviceCapacityConsumption(&e, &deviceCapacityConsumption, s); err != nil {
			return err
		}
		consumesCapacity = append(consumesCapacity, deviceCapacityConsumption)
	}
	composite.ConsumesCapacity = consumesCapacity

	composite.NodeName = in.NodeName
	composite.NodeSelector = (*corev1.NodeSelector)(unsafe.Pointer(in.NodeSelector))
	composite.AllNodes = in.AllNodes
	out.Composite = composite

	return nil
}

func Convert_v1alpha3_Device_To_resource_Device(in *resourcev1alpha3.Device, out *resourceapi.Device, s conversion.Scope) error {
	out.Name = in.Name
	if in.Basic != nil {
		basicDeviceNames, ok := s.Meta().Context.([]string)
		if !ok {
			return fmt.Errorf("context has unexpected type")
		}
		basicDeviceNames = append(basicDeviceNames, in.Name)
		s.Meta().Context = basicDeviceNames
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
		return nil
	}
	if in.Composite != nil {
		composite := in.Composite

		var includes []resourceapi.DeviceMixinRef
		for _, e := range composite.Includes {
			var deviceMixinRef resourceapi.DeviceMixinRef
			if err := Convert_v1alpha3_DeviceMixinRef_To_resource_DeviceMixinRef(&e, &deviceMixinRef, s); err != nil {
				return err
			}
			includes = append(includes, deviceMixinRef)
		}
		out.Includes = includes

		if len(composite.Attributes) > 0 {
			attributes := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
			if err := convert_v1alpha3_Attributes_To_resource_Attributes(composite.Attributes, attributes, s); err != nil {
				return err
			}
			out.Attributes = attributes
		}

		if len(composite.Capacity) > 0 {
			capacity := make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity)
			if err := convert_v1alpha3_Capacity_To_resource_Capacity(composite.Capacity, capacity, s); err != nil {
				return err
			}
			out.Capacity = capacity
		}

		var consumesCapacity []resourceapi.DeviceCapacityConsumption
		for _, e := range composite.ConsumesCapacity {
			var deviceCapacityConsumption resourceapi.DeviceCapacityConsumption
			if err := Convert_v1alpha3_DeviceCapacityConsumption_To_resource_DeviceCapacityConsumption(&e, &deviceCapacityConsumption, s); err != nil {
				return err
			}
			consumesCapacity = append(consumesCapacity, deviceCapacityConsumption)
		}
		out.ConsumesCapacity = consumesCapacity

		out.NodeName = composite.NodeName
		out.NodeSelector = (*core.NodeSelector)(unsafe.Pointer(composite.NodeSelector))
		out.AllNodes = composite.AllNodes
	}
	return nil
}

func Convert_resource_DeviceMixin_To_v1alpha3_DeviceMixin(in *resourceapi.DeviceMixin, out *resourcev1alpha3.DeviceMixin, s conversion.Scope) error {
	out.Name = in.Name

	var compositeDeviceMixin resourcev1alpha3.CompositeDeviceMixin
	if len(in.Attributes) > 0 {
		attributes := make(map[resourcev1alpha3.QualifiedName]resourcev1alpha3.DeviceAttribute)
		if err := convert_resource_Attributes_To_v1alpha3_Attributes(in.Attributes, attributes, s); err != nil {
			return err
		}
		compositeDeviceMixin.Attributes = attributes
	}

	if len(in.Capacity) > 0 {
		capacity := make(map[resourcev1alpha3.QualifiedName]resource.Quantity)
		if err := convert_resource_Capacity_To_v1alpha3_Capacity(in.Capacity, capacity, s); err != nil {
			return err
		}
		compositeDeviceMixin.Capacity = capacity
	}
	out.Composite = &compositeDeviceMixin
	return nil
}

func Convert_v1alpha3_DeviceMixin_To_resource_DeviceMixin(in *resourcev1alpha3.DeviceMixin, out *resourceapi.DeviceMixin, s conversion.Scope) error {
	out.Name = in.Name

	if in.Composite != nil {
		composite := in.Composite
		if len(composite.Attributes) > 0 {
			attributes := make(map[resourceapi.QualifiedName]resourceapi.DeviceAttribute)
			if err := convert_v1alpha3_Attributes_To_resource_Attributes(composite.Attributes, attributes, s); err != nil {
				return err
			}
			out.Attributes = attributes
		}

		if len(composite.Capacity) > 0 {
			capacity := make(map[resourceapi.QualifiedName]resourceapi.DeviceCapacity)
			if err := convert_v1alpha3_Capacity_To_resource_Capacity(composite.Capacity, capacity, s); err != nil {
				return err
			}
			out.Capacity = capacity
		}
	}
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
