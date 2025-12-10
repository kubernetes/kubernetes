/*
Copyright 2021 The Kubernetes Authors.

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

package v1alpha2

import (
	v1 "k8s.io/code-generator/examples/crd/apis/gateway-api/v1"
)

// Copied from Gateway API as a minimal reproducer for #131533

type PolicyAncestorStatus struct {
	AncestorRef v1.ParentReference `json:"ancestorRef"`
}

type PolicyStatus struct {
	Ancestors []PolicyAncestorStatus `json:"ancestors"`
}

func (in *PolicyAncestorStatus) DeepCopyInto(out *PolicyAncestorStatus) {
	*out = *in
	in.AncestorRef.DeepCopyInto(&out.AncestorRef)
}

func (in *PolicyStatus) DeepCopyInto(out *PolicyStatus) {
	*out = *in
	if in.Ancestors != nil {
		in, out := &in.Ancestors, &out.Ancestors
		*out = make([]PolicyAncestorStatus, len(*in))
		for i := range *in {
			(*in)[i].DeepCopyInto(&(*out)[i])
		}
	}
}
