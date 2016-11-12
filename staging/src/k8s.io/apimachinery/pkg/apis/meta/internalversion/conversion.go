/*
Copyright 2017 The Kubernetes Authors.

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

package internalversion

import (
	"fmt"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/util/validation/field"
)

func Convert_internalversion_ListOptions_To_v1_ListOptions(in *ListOptions, out *metav1.ListOptions, s conversion.Scope) error {
	if err := metav1.Convert_fields_Selector_To_string(&in.FieldSelector, &out.FieldSelector, s); err != nil {
		return err
	}
	if err := metav1.Convert_labels_Selector_To_string(&in.LabelSelector, &out.LabelSelector, s); err != nil {
		return err
	}
	out.IncludeUninitialized = in.IncludeUninitialized
	out.ResourceVersion = in.ResourceVersion
	out.TimeoutSeconds = in.TimeoutSeconds
	out.Watch = in.Watch
	return nil
}

func Convert_v1_ListOptions_To_internalversion_ListOptions(in *metav1.ListOptions, out *ListOptions, s conversion.Scope) error {
	if err := metav1.Convert_string_To_fields_Selector(&in.FieldSelector, &out.FieldSelector, s); err != nil {
		return err
	}
	if err := metav1.Convert_string_To_labels_Selector(&in.LabelSelector, &out.LabelSelector, s); err != nil {
		return err
	}
	out.IncludeUninitialized = in.IncludeUninitialized
	out.ResourceVersion = in.ResourceVersion
	out.TimeoutSeconds = in.TimeoutSeconds
	out.Watch = in.Watch
	return nil
}

func Convert_map_to_v1_LabelSelector(in *map[string]string, out *metav1.LabelSelector, s conversion.Scope) error {
	if in == nil {
		return nil
	}
	out = new(metav1.LabelSelector)
	for labelKey, labelValue := range *in {
		metav1.AddLabelToSelector(out, labelKey, labelValue)
	}
	return nil
}

func Convert_v1_LabelSelector_to_map(in *metav1.LabelSelector, out *map[string]string, s conversion.Scope) error {
	var err error
	*out, err = metav1.LabelSelectorAsMap(in)
	if err != nil {
		err = field.Invalid(field.NewPath("labelSelector"), *in, fmt.Sprintf("cannot convert to old selector: %v", err))
	}
	return err
}
