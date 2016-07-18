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

package core

import (
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
)

func addDefaultingFuncs(scheme *runtime.Scheme) {
	scheme.AddDefaultingFuncs(
		func(obj *api.ListOptions) {
			if obj.LabelSelector == nil {
				obj.LabelSelector = labels.Everything()
			}
			if obj.FieldSelector == nil {
				obj.FieldSelector = fields.Everything()
			}
		},
	)
}

func addConversionFuncs(scheme *runtime.Scheme) {
	// Add non-generated conversion functions
	err := scheme.AddConversionFuncs(
		api.Convert_unversioned_TypeMeta_To_unversioned_TypeMeta,
		api.Convert_unversioned_ListMeta_To_unversioned_ListMeta,
		api.Convert_intstr_IntOrString_To_intstr_IntOrString,
		api.Convert_unversioned_Time_To_unversioned_Time,
		api.Convert_Slice_string_To_unversioned_Time,
		api.Convert_string_To_labels_Selector,
		api.Convert_string_To_fields_Selector,
		api.Convert_Pointer_bool_To_bool,
		api.Convert_bool_To_Pointer_bool,
		api.Convert_Pointer_string_To_string,
		api.Convert_string_To_Pointer_string,
		api.Convert_labels_Selector_To_string,
		api.Convert_fields_Selector_To_string,
		api.Convert_resource_Quantity_To_resource_Quantity,
	)
	if err != nil {
		// If one of the conversion functions is malformed, detect it immediately.
		panic(err)
	}
}
