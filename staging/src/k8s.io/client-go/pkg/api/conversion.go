/*
Copyright 2014 The Kubernetes Authors.

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
	"fmt"

	"k8s.io/client-go/pkg/api/resource"
	metav1 "k8s.io/client-go/pkg/apis/meta/v1"
	"k8s.io/client-go/pkg/conversion"
	"k8s.io/client-go/pkg/fields"
	"k8s.io/client-go/pkg/labels"
	"k8s.io/client-go/pkg/runtime"
	"k8s.io/client-go/pkg/util/intstr"
	utillabels "k8s.io/client-go/pkg/util/labels"
	"k8s.io/client-go/pkg/util/validation/field"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	return scheme.AddConversionFuncs(
		Convert_v1_TypeMeta_To_v1_TypeMeta,

		Convert_unversioned_ListMeta_To_unversioned_ListMeta,

		Convert_intstr_IntOrString_To_intstr_IntOrString,

		Convert_unversioned_Time_To_unversioned_Time,

		Convert_Slice_string_To_unversioned_Time,

		Convert_resource_Quantity_To_resource_Quantity,

		Convert_string_To_labels_Selector,
		Convert_labels_Selector_To_string,

		Convert_string_To_fields_Selector,
		Convert_fields_Selector_To_string,

		Convert_Pointer_bool_To_bool,
		Convert_bool_To_Pointer_bool,

		Convert_Pointer_string_To_string,
		Convert_string_To_Pointer_string,

		Convert_Pointer_int64_To_int,
		Convert_int_To_Pointer_int64,

		Convert_Pointer_int32_To_int32,
		Convert_int32_To_Pointer_int32,

		Convert_Pointer_float64_To_float64,
		Convert_float64_To_Pointer_float64,

		Convert_map_to_unversioned_LabelSelector,
		Convert_unversioned_LabelSelector_to_map,
	)
}

func Convert_Pointer_float64_To_float64(in **float64, out *float64, s conversion.Scope) error {
	if *in == nil {
		*out = 0
		return nil
	}
	*out = float64(**in)
	return nil
}

func Convert_float64_To_Pointer_float64(in *float64, out **float64, s conversion.Scope) error {
	temp := float64(*in)
	*out = &temp
	return nil
}

func Convert_Pointer_int32_To_int32(in **int32, out *int32, s conversion.Scope) error {
	if *in == nil {
		*out = 0
		return nil
	}
	*out = int32(**in)
	return nil
}

func Convert_int32_To_Pointer_int32(in *int32, out **int32, s conversion.Scope) error {
	temp := int32(*in)
	*out = &temp
	return nil
}

func Convert_Pointer_int64_To_int(in **int64, out *int, s conversion.Scope) error {
	if *in == nil {
		*out = 0
		return nil
	}
	*out = int(**in)
	return nil
}

func Convert_int_To_Pointer_int64(in *int, out **int64, s conversion.Scope) error {
	temp := int64(*in)
	*out = &temp
	return nil
}

func Convert_Pointer_string_To_string(in **string, out *string, s conversion.Scope) error {
	if *in == nil {
		*out = ""
		return nil
	}
	*out = **in
	return nil
}

func Convert_string_To_Pointer_string(in *string, out **string, s conversion.Scope) error {
	if in == nil {
		stringVar := ""
		*out = &stringVar
		return nil
	}
	*out = in
	return nil
}

func Convert_Pointer_bool_To_bool(in **bool, out *bool, s conversion.Scope) error {
	if *in == nil {
		*out = false
		return nil
	}
	*out = **in
	return nil
}

func Convert_bool_To_Pointer_bool(in *bool, out **bool, s conversion.Scope) error {
	if in == nil {
		boolVar := false
		*out = &boolVar
		return nil
	}
	*out = in
	return nil
}

// +k8s:conversion-fn=drop
func Convert_v1_TypeMeta_To_v1_TypeMeta(in, out *metav1.TypeMeta, s conversion.Scope) error {
	// These values are explicitly not copied
	//out.APIVersion = in.APIVersion
	//out.Kind = in.Kind
	return nil
}

// +k8s:conversion-fn=copy-only
func Convert_unversioned_ListMeta_To_unversioned_ListMeta(in, out *metav1.ListMeta, s conversion.Scope) error {
	*out = *in
	return nil
}

// +k8s:conversion-fn=copy-only
func Convert_intstr_IntOrString_To_intstr_IntOrString(in, out *intstr.IntOrString, s conversion.Scope) error {
	*out = *in
	return nil
}

// +k8s:conversion-fn=copy-only
func Convert_unversioned_Time_To_unversioned_Time(in *metav1.Time, out *metav1.Time, s conversion.Scope) error {
	// Cannot deep copy these, because time.Time has unexported fields.
	*out = *in
	return nil
}

// Convert_Slice_string_To_unversioned_Time allows converting a URL query parameter value
func Convert_Slice_string_To_unversioned_Time(input *[]string, out *metav1.Time, s conversion.Scope) error {
	str := ""
	if len(*input) > 0 {
		str = (*input)[0]
	}
	return out.UnmarshalQueryParameter(str)
}

func Convert_string_To_labels_Selector(in *string, out *labels.Selector, s conversion.Scope) error {
	selector, err := labels.Parse(*in)
	if err != nil {
		return err
	}
	*out = selector
	return nil
}

func Convert_string_To_fields_Selector(in *string, out *fields.Selector, s conversion.Scope) error {
	selector, err := fields.ParseSelector(*in)
	if err != nil {
		return err
	}
	*out = selector
	return nil
}

func Convert_labels_Selector_To_string(in *labels.Selector, out *string, s conversion.Scope) error {
	if *in == nil {
		return nil
	}
	*out = (*in).String()
	return nil
}

func Convert_fields_Selector_To_string(in *fields.Selector, out *string, s conversion.Scope) error {
	if *in == nil {
		return nil
	}
	*out = (*in).String()
	return nil
}

// +k8s:conversion-fn=copy-only
func Convert_resource_Quantity_To_resource_Quantity(in *resource.Quantity, out *resource.Quantity, s conversion.Scope) error {
	*out = *in
	return nil
}

func Convert_map_to_unversioned_LabelSelector(in *map[string]string, out *metav1.LabelSelector, s conversion.Scope) error {
	if in == nil {
		return nil
	}
	out = new(metav1.LabelSelector)
	for labelKey, labelValue := range *in {
		utillabels.AddLabelToSelector(out, labelKey, labelValue)
	}
	return nil
}

func Convert_unversioned_LabelSelector_to_map(in *metav1.LabelSelector, out *map[string]string, s conversion.Scope) error {
	var err error
	*out, err = metav1.LabelSelectorAsMap(in)
	if err != nil {
		err = field.Invalid(field.NewPath("labelSelector"), *in, fmt.Sprintf("cannot convert to old selector: %v", err))
	}
	return err
}
