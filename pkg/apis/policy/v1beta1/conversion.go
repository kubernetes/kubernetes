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

package v1beta1

import (
	"k8s.io/api/policy/v1beta1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/policy"
)

func Convert_v1beta1_PodDisruptionBudget_To_policy_PodDisruptionBudget(in *v1beta1.PodDisruptionBudget, out *policy.PodDisruptionBudget, s conversion.Scope) error {
	if err := autoConvert_v1beta1_PodDisruptionBudget_To_policy_PodDisruptionBudget(in, out, s); err != nil {
		return err
	}

	switch {
	case apiequality.Semantic.DeepEqual(in.Spec.Selector, policy.V1beta1MatchNoneSelector):
		// If the v1beta1 version has a non-nil but empty selector, it should be
		// selecting no pods, even when used with the internal or v1 api. We
		// add a selector that is non-empty but will never match any pods.
		out.Spec.Selector = policy.NonV1beta1MatchNoneSelector.DeepCopy()
	case apiequality.Semantic.DeepEqual(in.Spec.Selector, policy.V1beta1MatchAllSelector):
		// If the v1beta1 version has our v1beta1-specific "match-all" selector,
		// swap that out for a simpler empty "match-all" selector for v1
		out.Spec.Selector = policy.NonV1beta1MatchAllSelector.DeepCopy()
	default:
		// otherwise, make sure the label intended to be used in a match-all or match-none selector
		// never gets combined with user-specified fields
		policy.StripPDBV1beta1Label(out.Spec.Selector)
	}
	return nil
}

func Convert_policy_PodDisruptionBudget_To_v1beta1_PodDisruptionBudget(in *policy.PodDisruptionBudget, out *v1beta1.PodDisruptionBudget, s conversion.Scope) error {
	if err := autoConvert_policy_PodDisruptionBudget_To_v1beta1_PodDisruptionBudget(in, out, s); err != nil {
		return err
	}

	switch {
	case apiequality.Semantic.DeepEqual(in.Spec.Selector, policy.NonV1beta1MatchNoneSelector):
		// If the internal version has our v1beta1-specific "match-none" selector,
		// swap that out for a simpler empty "match-none" selector for v1beta1
		out.Spec.Selector = policy.V1beta1MatchNoneSelector.DeepCopy()
	case apiequality.Semantic.DeepEqual(in.Spec.Selector, policy.NonV1beta1MatchAllSelector):
		// If the internal version has a non-nil but empty selector, we want it to
		// select all pods. We make sure this happens even with the v1beta1 api by
		// adding a non-empty selector that selects all pods.
		out.Spec.Selector = policy.V1beta1MatchAllSelector.DeepCopy()
	}
	return nil
}
