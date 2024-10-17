/*
Copyright 2024 The Kubernetes Authors.

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
	coordinationv1 "k8s.io/api/coordination/v1"
	coordinationv1alpha1 "k8s.io/api/coordination/v1alpha1"
	conversion "k8s.io/apimachinery/pkg/conversion"
	coordination "k8s.io/kubernetes/pkg/apis/coordination"
)

func Convert_v1alpha1_LeaseCandidateSpec_To_coordination_LeaseCandidateSpec(in *coordinationv1alpha1.LeaseCandidateSpec, out *coordination.LeaseCandidateSpec, s conversion.Scope) error {
	if err := autoConvert_v1alpha1_LeaseCandidateSpec_To_coordination_LeaseCandidateSpec(in, out, s); err != nil {
		return err
	}

	if len(in.PreferredStrategies) > 0 {
		out.Strategy = (coordination.CoordinatedLeaseStrategy)(in.PreferredStrategies[0])
	}
	return nil
}

func Convert_coordination_LeaseCandidateSpec_To_v1alpha1_LeaseCandidateSpec(in *coordination.LeaseCandidateSpec, out *coordinationv1alpha1.LeaseCandidateSpec, s conversion.Scope) error {
	if err := autoConvert_coordination_LeaseCandidateSpec_To_v1alpha1_LeaseCandidateSpec(in, out, s); err != nil {
		return err
	}

	out.PreferredStrategies = []coordinationv1.CoordinatedLeaseStrategy{(coordinationv1.CoordinatedLeaseStrategy)(in.Strategy)}
	return nil
}
