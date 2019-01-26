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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
)

func Convert_internalversion_ListOptions_To_v1_ListOptions(in *ListOptions, out *metav1.ListOptions, s conversion.Scope) error {
	if err := metav1.Convert_fields_Selector_To_string(&in.FieldSelector, &out.FieldSelector, s); err != nil {
		return err
	}
	if err := metav1.Convert_labels_Selector_To_string(&in.LabelSelector, &out.LabelSelector, s); err != nil {
		return err
	}
	out.ResourceVersion = in.ResourceVersion
	out.TimeoutSeconds = in.TimeoutSeconds
	out.Watch = in.Watch
	out.Limit = in.Limit
	out.Continue = in.Continue
	return nil
}

func Convert_v1_ListOptions_To_internalversion_ListOptions(in *metav1.ListOptions, out *ListOptions, s conversion.Scope) error {
	if err := metav1.Convert_string_To_fields_Selector(&in.FieldSelector, &out.FieldSelector, s); err != nil {
		return err
	}
	if err := metav1.Convert_string_To_labels_Selector(&in.LabelSelector, &out.LabelSelector, s); err != nil {
		return err
	}
	out.ResourceVersion = in.ResourceVersion
	out.TimeoutSeconds = in.TimeoutSeconds
	out.Watch = in.Watch
	out.Limit = in.Limit
	out.Continue = in.Continue
	return nil
}
