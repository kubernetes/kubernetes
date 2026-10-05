/*
Copyright The Kubernetes Authors.

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

package v1beta1

import (
	nodev1beta1 "k8s.io/api/node/v1beta1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/node"
)

// Convert_v1beta1_RuntimeClass_To_node_RuntimeClass clears fields that this
// unserved version cannot represent, including when the destination is reused.
func Convert_v1beta1_RuntimeClass_To_node_RuntimeClass(in *nodev1beta1.RuntimeClass, out *node.RuntimeClass, s conversion.Scope) error {
	if err := autoConvert_v1beta1_RuntimeClass_To_node_RuntimeClass(in, out, s); err != nil {
		return err
	}
	out.PodCheckpoint = nil
	return nil
}

// Convert_node_RuntimeClass_To_v1beta1_RuntimeClass omits the v1-only checkpoint
// policy because this unserved version predates it.
func Convert_node_RuntimeClass_To_v1beta1_RuntimeClass(in *node.RuntimeClass, out *nodev1beta1.RuntimeClass, s conversion.Scope) error {
	return autoConvert_node_RuntimeClass_To_v1beta1_RuntimeClass(in, out, s)
}
