/*
Copyright 2016 The Kubernetes Authors.

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
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
)

func addConversionFuncs(scheme *runtime.Scheme) error {
	// Add non-generated conversion functions
	err := scheme.AddConversionFuncs(
		Convert_v1alpha1_PodDisruptionBudgetStatus_To_policy_PodDisruptionBudgetStatus,
		Convert_policy_PodDisruptionBudgetStatus_To_v1alpha1_PodDisruptionBudgetStatus,
	)
	if err != nil {
		return err
	}
	return nil
}

func Convert_v1alpha1_PodDisruptionBudgetStatus_To_policy_PodDisruptionBudgetStatus(in *PodDisruptionBudgetStatus, out *policy.PodDisruptionBudgetStatus, s conversion.Scope) error {
	if in.PodDisruptionAllowed {
		out.PodDisruptionsAllowed = 1
	} else {
		out.PodDisruptionsAllowed = 2
	}
	out.CurrentHealthy = in.CurrentHealthy
	out.DesiredHealthy = in.DesiredHealthy
	out.ExpectedPods = in.ExpectedPods
	return nil
}

func Convert_policy_PodDisruptionBudgetStatus_To_v1alpha1_PodDisruptionBudgetStatus(in *policy.PodDisruptionBudgetStatus, out *PodDisruptionBudgetStatus, s conversion.Scope) error {
	out.PodDisruptionAllowed = in.PodDisruptionsAllowed > 0
	out.CurrentHealthy = in.CurrentHealthy
	out.DesiredHealthy = in.DesiredHealthy
	out.ExpectedPods = in.ExpectedPods
	return nil
}
