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

package v1

import (
	"fmt"
	"net/url"
	"strconv"
	"strings"

	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
)

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

func Convert_Pointer_int64_To_int64(in **int64, out *int64, s conversion.Scope) error {
	if *in == nil {
		*out = 0
		return nil
	}
	*out = int64(**in)
	return nil
}

func Convert_int64_To_Pointer_int64(in *int64, out **int64, s conversion.Scope) error {
	temp := int64(*in)
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
func Convert_v1_TypeMeta_To_v1_TypeMeta(in, out *TypeMeta, s conversion.Scope) error {
	// These values are explicitly not copied
	//out.APIVersion = in.APIVersion
	//out.Kind = in.Kind
	return nil
}

// +k8s:conversion-fn=copy-only
func Convert_v1_ListMeta_To_v1_ListMeta(in, out *ListMeta, s conversion.Scope) error {
	*out = *in
	return nil
}

// +k8s:conversion-fn=copy-only
func Convert_v1_DeleteOptions_To_v1_DeleteOptions(in, out *DeleteOptions, s conversion.Scope) error {
	*out = *in
	return nil
}

// +k8s:conversion-fn=copy-only
func Convert_intstr_IntOrString_To_intstr_IntOrString(in, out *intstr.IntOrString, s conversion.Scope) error {
	*out = *in
	return nil
}

func Convert_Pointer_intstr_IntOrString_To_intstr_IntOrString(in **intstr.IntOrString, out *intstr.IntOrString, s conversion.Scope) error {
	if *in == nil {
		*out = intstr.IntOrString{} // zero value
		return nil
	}
	*out = **in // copy
	return nil
}

func Convert_intstr_IntOrString_To_Pointer_intstr_IntOrString(in *intstr.IntOrString, out **intstr.IntOrString, s conversion.Scope) error {
	temp := *in // copy
	*out = &temp
	return nil
}

// +k8s:conversion-fn=copy-only
func Convert_v1_Time_To_v1_Time(in *Time, out *Time, s conversion.Scope) error {
	// Cannot deep copy these, because time.Time has unexported fields.
	*out = *in
	return nil
}

// +k8s:conversion-fn=copy-only
func Convert_v1_MicroTime_To_v1_MicroTime(in *MicroTime, out *MicroTime, s conversion.Scope) error {
	// Cannot deep copy these, because time.Time has unexported fields.
	*out = *in
	return nil
}

func Convert_Pointer_v1_Duration_To_v1_Duration(in **Duration, out *Duration, s conversion.Scope) error {
	if *in == nil {
		*out = Duration{} // zero duration
		return nil
	}
	*out = **in // copy
	return nil
}

func Convert_v1_Duration_To_Pointer_v1_Duration(in *Duration, out **Duration, s conversion.Scope) error {
	temp := *in //copy
	*out = &temp
	return nil
}

// Convert_Slice_string_To_v1_Time allows converting a URL query parameter value
func Convert_Slice_string_To_v1_Time(in *[]string, out *Time, s conversion.Scope) error {
	str := ""
	if len(*in) > 0 {
		str = (*in)[0]
	}
	return out.UnmarshalQueryParameter(str)
}

func Convert_Slice_string_To_Pointer_v1_Time(in *[]string, out **Time, s conversion.Scope) error {
	if in == nil {
		return nil
	}
	str := ""
	if len(*in) > 0 {
		str = (*in)[0]
	}
	temp := Time{}
	if err := temp.UnmarshalQueryParameter(str); err != nil {
		return err
	}
	*out = &temp
	return nil
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

func Convert_Map_string_To_string_To_v1_LabelSelector(in *map[string]string, out *LabelSelector, s conversion.Scope) error {
	if in == nil {
		return nil
	}
	for labelKey, labelValue := range *in {
		AddLabelToSelector(out, labelKey, labelValue)
	}
	return nil
}

func Convert_v1_LabelSelector_To_Map_string_To_string(in *LabelSelector, out *map[string]string, s conversion.Scope) error {
	var err error
	*out, err = LabelSelectorAsMap(in)
	return err
}

// Convert_Slice_string_To_Slice_int32 converts multiple query parameters or
// a single query parameter with a comma delimited value to multiple int32.
// This is used for port forwarding which needs the ports as int32.
func Convert_Slice_string_To_Slice_int32(in *[]string, out *[]int32, s conversion.Scope) error {
	for _, s := range *in {
		for _, v := range strings.Split(s, ",") {
			x, err := strconv.ParseUint(v, 10, 16)
			if err != nil {
				return fmt.Errorf("cannot convert to []int32: %v", err)
			}
			*out = append(*out, int32(x))
		}
	}
	return nil
}

// Convert_Slice_string_To_Pointer_v1_DeletionPropagation allows converting a URL query parameter propagationPolicy
func Convert_Slice_string_To_Pointer_v1_DeletionPropagation(in *[]string, out **DeletionPropagation, s conversion.Scope) error {
	var str string
	if len(*in) > 0 {
		str = (*in)[0]
	} else {
		str = ""
	}
	temp := DeletionPropagation(str)
	*out = &temp
	return nil
}

// Convert_Slice_string_To_v1_IncludeObjectPolicy allows converting a URL query parameter value
func Convert_Slice_string_To_v1_IncludeObjectPolicy(in *[]string, out *IncludeObjectPolicy, s conversion.Scope) error {
	if len(*in) > 0 {
		*out = IncludeObjectPolicy((*in)[0])
	}
	return nil
}

// Convert_url_Values_To_v1_DeleteOptions allows converting a URL to DeleteOptions.
func Convert_url_Values_To_v1_DeleteOptions(in *url.Values, out *DeleteOptions, s conversion.Scope) error {
	if err := autoConvert_url_Values_To_v1_DeleteOptions(in, out, s); err != nil {
		return err
	}

	uid := types.UID("")
	if values, ok := (*in)["uid"]; ok && len(values) > 0 {
		uid = types.UID(values[0])
	}

	resourceVersion := ""
	if values, ok := (*in)["resourceVersion"]; ok && len(values) > 0 {
		resourceVersion = values[0]
	}

	if len(uid) > 0 || len(resourceVersion) > 0 {
		if out.Preconditions == nil {
			out.Preconditions = &Preconditions{}
		}
		if len(uid) > 0 {
			out.Preconditions.UID = &uid
		}
		if len(resourceVersion) > 0 {
			out.Preconditions.ResourceVersion = &resourceVersion
		}
	}
	return nil
}
