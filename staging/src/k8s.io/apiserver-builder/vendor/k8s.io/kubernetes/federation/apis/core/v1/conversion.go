/*
Copyright 2016 The Kubernetes Authors.

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

package v1

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api/v1"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add non-generated conversion functions
	err := scheme.AddConversionFuncs(
		v1.Convert_v1_DeleteOptions_To_api_DeleteOptions,
		v1.Convert_api_DeleteOptions_To_v1_DeleteOptions,
		v1.Convert_v1_List_To_api_List,
		v1.Convert_api_List_To_v1_List,
		v1.Convert_v1_ListOptions_To_api_ListOptions,
		v1.Convert_api_ListOptions_To_v1_ListOptions,
		v1.Convert_v1_ObjectFieldSelector_To_api_ObjectFieldSelector,
		v1.Convert_api_ObjectFieldSelector_To_v1_ObjectFieldSelector,
		v1.Convert_v1_ObjectMeta_To_api_ObjectMeta,
		v1.Convert_api_ObjectMeta_To_v1_ObjectMeta,
		v1.Convert_v1_ObjectReference_To_api_ObjectReference,
		v1.Convert_api_ObjectReference_To_v1_ObjectReference,
		v1.Convert_v1_Secret_To_api_Secret,
		v1.Convert_api_Secret_To_v1_Secret,
		v1.Convert_v1_SecretList_To_api_SecretList,
		v1.Convert_api_SecretList_To_v1_SecretList,
		v1.Convert_v1_Service_To_api_Service,
		v1.Convert_api_Service_To_v1_Service,
		v1.Convert_v1_ServiceList_To_api_ServiceList,
		v1.Convert_api_ServiceList_To_v1_ServiceList,
		v1.Convert_v1_ServicePort_To_api_ServicePort,
		v1.Convert_api_ServicePort_To_v1_ServicePort,
		v1.Convert_v1_ServiceProxyOptions_To_api_ServiceProxyOptions,
		v1.Convert_api_ServiceProxyOptions_To_v1_ServiceProxyOptions,
		v1.Convert_v1_ServiceSpec_To_api_ServiceSpec,
		v1.Convert_api_ServiceSpec_To_v1_ServiceSpec,
		v1.Convert_v1_ServiceStatus_To_api_ServiceStatus,
		v1.Convert_api_ServiceStatus_To_v1_ServiceStatus,
	)
	if err != nil {
		return err
	}

	// Add field label conversions for kinds having selectable nothing but ObjectMeta fields.
	for _, kind := range []string{
		"Service",
	} {
		err = scheme.AddFieldLabelConversionFunc("v1", kind,
			func(label, value string) (string, string, error) {
				switch label {
				case "metadata.namespace",
					"metadata.name":
					return label, value, nil
				default:
					return "", "", fmt.Errorf("field label %q not supported for %q", label, kind)
				}
			})
		if err != nil {
			return err
		}
	}
	if err := v1.AddFieldLabelConversionsForEvent(scheme); err != nil {
		return nil
	}
	if err := v1.AddFieldLabelConversionsForNamespace(scheme); err != nil {
		return nil
	}
	if err := v1.AddFieldLabelConversionsForSecret(scheme); err != nil {
		return nil
	}
	return nil
}
