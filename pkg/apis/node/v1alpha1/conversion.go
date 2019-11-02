/*
Copyright 2019 The Kubernetes Authors.

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
	v1alpha1 "k8s.io/api/node/v1alpha1"
	conversion "k8s.io/apimachinery/pkg/conversion"
	node "k8s.io/kubernetes/pkg/apis/node"
)

// Convert_v1alpha1_RuntimeClass_To_node_RuntimeClass must override the automatic
// conversion since we unnested the spec struct after v1alpha1
func Convert_v1alpha1_RuntimeClass_To_node_RuntimeClass(in *v1alpha1.RuntimeClass, out *node.RuntimeClass, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	out.Handler = in.Spec.RuntimeHandler

	if in.Spec.Overhead != nil {
		out.Overhead = &node.Overhead{}
		if err := Convert_v1alpha1_Overhead_To_node_Overhead(in.Spec.Overhead, out.Overhead, s); err != nil {
			return err
		}
	}
	if in.Spec.Scheduling != nil {
		out.Scheduling = &node.Scheduling{}
		if err := Convert_v1alpha1_Scheduling_To_node_Scheduling(in.Spec.Scheduling, out.Scheduling, s); err != nil {
			return err
		}
	}
	return nil
}

// Convert_node_RuntimeClass_To_v1alpha1_RuntimeClass must override the automatic
// conversion since we unnested the spec struct after v1alpha1
func Convert_node_RuntimeClass_To_v1alpha1_RuntimeClass(in *node.RuntimeClass, out *v1alpha1.RuntimeClass, s conversion.Scope) error {
	out.ObjectMeta = in.ObjectMeta
	out.Spec.RuntimeHandler = in.Handler

	if in.Overhead != nil {
		out.Spec.Overhead = &v1alpha1.Overhead{}
		if err := Convert_node_Overhead_To_v1alpha1_Overhead(in.Overhead, out.Spec.Overhead, s); err != nil {
			return err
		}
	}
	if in.Scheduling != nil {
		out.Spec.Scheduling = &v1alpha1.Scheduling{}
		if err := Convert_node_Scheduling_To_v1alpha1_Scheduling(in.Scheduling, out.Spec.Scheduling, s); err != nil {
			return err
		}
	}
	return nil
}
