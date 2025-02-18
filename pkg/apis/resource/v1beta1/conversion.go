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

	resourcev1beta1 "k8s.io/api/resource/v1beta1"
	conversion "k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/runtime"
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
	out.Name = in.Name
	for i := range in.FirstAvailable {
		var deviceSubRequest resource.DeviceSubRequest
		err := Convert_v1beta1_DeviceSubRequest_To_resource_DeviceSubRequest(&in.FirstAvailable[i], &deviceSubRequest, s)
		if err != nil {
			return err
		}
		out.FirstAvailable = append(out.FirstAvailable, deviceSubRequest)
	}

	// If any fields on the main request is set, we create a SpecificDeviceRequest
	// and set the Exactly field. It might be invalid but that will be caught in validation.
	if hasAnyMainRequestFieldsSet(in) {
		var specificDeviceRequest resource.SpecificDeviceRequest
		specificDeviceRequest.DeviceClassName = in.DeviceClassName
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
			specificDeviceRequest.Selectors = selectors
		}
		specificDeviceRequest.AllocationMode = resource.DeviceAllocationMode(in.AllocationMode)
		specificDeviceRequest.Count = in.Count
		specificDeviceRequest.AdminAccess = in.AdminAccess
		out.Exactly = &specificDeviceRequest
	}
	return nil
}

func hasAnyMainRequestFieldsSet(deviceRequest *resourcev1beta1.DeviceRequest) bool {
	return deviceRequest.DeviceClassName != "" ||
		deviceRequest.Selectors != nil ||
		deviceRequest.AllocationMode != "" ||
		deviceRequest.Count != 0 ||
		deviceRequest.AdminAccess != nil
}

func Convert_resource_DeviceRequest_To_v1beta1_DeviceRequest(in *resource.DeviceRequest, out *resourcev1beta1.DeviceRequest, s conversion.Scope) error {
	out.Name = in.Name
	for i := range in.FirstAvailable {
		var deviceSubRequest resourcev1beta1.DeviceSubRequest
		err := Convert_resource_DeviceSubRequest_To_v1beta1_DeviceSubRequest(&in.FirstAvailable[i], &deviceSubRequest, s)
		if err != nil {
			return err
		}
		out.FirstAvailable = append(out.FirstAvailable, deviceSubRequest)
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
	}
	return nil
}
