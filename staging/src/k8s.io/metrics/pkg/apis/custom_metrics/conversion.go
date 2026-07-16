/*
Copyright 2020 The Kubernetes Authors.

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

package custom_metrics

import (
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/conversion"
)

func Convert_v1_ObjectReference_To_custom_metrics_ObjectReference(in *v1.ObjectReference, out *ObjectReference, s conversion.Scope) error {
	out.APIVersion = in.APIVersion

	out.Kind = in.Kind
	out.Namespace = in.Namespace
	out.Name = in.Name
	out.UID = in.UID
	out.ResourceVersion = in.ResourceVersion
	out.FieldPath = in.FieldPath
	return nil
}

func Convert_custom_metrics_ObjectReference_To_v1_ObjectReference(in *ObjectReference, out *v1.ObjectReference, s conversion.Scope) error {
	out.APIVersion = in.APIVersion

	out.Kind = in.Kind
	out.Namespace = in.Namespace
	out.Name = in.Name
	out.UID = in.UID
	out.ResourceVersion = in.ResourceVersion
	out.FieldPath = in.FieldPath
	return nil
}
