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

package v1alpha1

import (
	"strings"

	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apiserver/pkg/apis/audit"
)

func Convert_audit_ObjectReference_To_v1alpha1_ObjectReference(in *audit.ObjectReference, out *ObjectReference, s conversion.Scope) error {
	// Begin by copying all fields
	if err := autoConvert_audit_ObjectReference_To_v1alpha1_ObjectReference(in, out, s); err != nil {
		return err
	}
	// empty string means the core api group
	if in.APIGroup == "" {
		out.APIVersion = in.APIVersion
	} else {
		out.APIVersion = in.APIGroup + "/" + in.APIVersion
	}
	return nil
}

func Convert_v1alpha1_ObjectReference_To_audit_ObjectReference(in *ObjectReference, out *audit.ObjectReference, s conversion.Scope) error {
	// Begin by copying all fields
	if err := autoConvert_v1alpha1_ObjectReference_To_audit_ObjectReference(in, out, s); err != nil {
		return err
	}
	i := strings.LastIndex(in.APIVersion, "/")
	if i == -1 {
		// In fact it should always contain a "/"
		out.APIVersion = in.APIVersion
	} else {
		out.APIGroup = in.APIVersion[:i]
		out.APIVersion = in.APIVersion[i+1:]
	}
	return nil
}
