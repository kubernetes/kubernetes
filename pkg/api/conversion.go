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
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func init() {
	Scheme.AddDefaultingFuncs(
		func(obj *ListOptions) {
			if obj.LabelSelector == nil {
				obj.LabelSelector = labels.Everything()
			}
			if obj.FieldSelector == nil {
				obj.FieldSelector = fields.Everything()
			}
		},
	)
	Scheme.AddConversionFuncs(
		Convert_unversioned_TypeMeta_To_unversioned_TypeMeta,
		Convert_unversioned_ListMeta_To_unversioned_ListMeta,
		Convert_intstr_IntOrString_To_intstr_IntOrString,
		Convert_unversioned_Time_To_unversioned_Time,
		Convert_Slice_string_To_unversioned_Time,
		Convert_string_To_labels_Selector,
		Convert_string_To_fields_Selector,
		Convert_Pointer_bool_To_bool,
		Convert_bool_To_Pointer_bool,
		Convert_Pointer_string_To_string,
		Convert_string_To_Pointer_string,
		Convert_labels_Selector_To_string,
		Convert_fields_Selector_To_string,
		Convert_resource_Quantity_To_resource_Quantity,
	)
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

func Convert_unversioned_TypeMeta_To_unversioned_TypeMeta(in, out *unversioned.TypeMeta, s conversion.Scope) error {
	// These values are explicitly not copied
	//out.APIVersion = in.APIVersion
	//out.Kind = in.Kind
	return nil
}

func Convert_unversioned_ListMeta_To_unversioned_ListMeta(in, out *unversioned.ListMeta, s conversion.Scope) error {
	out.ResourceVersion = in.ResourceVersion
	out.SelfLink = in.SelfLink
	return nil
}

func Convert_intstr_IntOrString_To_intstr_IntOrString(in, out *intstr.IntOrString, s conversion.Scope) error {
	out.Type = in.Type
	out.IntVal = in.IntVal
	out.StrVal = in.StrVal
	return nil
}

func Convert_unversioned_Time_To_unversioned_Time(in *unversioned.Time, out *unversioned.Time, s conversion.Scope) error {
	// Cannot deep copy these, because time.Time has unexported fields.
	*out = *in
	return nil
}

// Convert_Slice_string_To_unversioned_Time allows converting a URL query parameter value
func Convert_Slice_string_To_unversioned_Time(input *[]string, out *unversioned.Time, s conversion.Scope) error {
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
func Convert_resource_Quantity_To_resource_Quantity(in *resource.Quantity, out *resource.Quantity, s conversion.Scope) error {
	*out = *in
	return nil
}
