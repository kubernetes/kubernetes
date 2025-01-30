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

package v1

import (
	"k8s.io/api/policy/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/kubernetes/pkg/apis/policy"
)

func Convert_v1_PodDisruptionBudget_To_policy_PodDisruptionBudget(in *v1.PodDisruptionBudget, out *policy.PodDisruptionBudget, s conversion.Scope) error {
	if err := autoConvert_v1_PodDisruptionBudget_To_policy_PodDisruptionBudget(in, out, s); err != nil {
		return err
	}

	switch {
	case apiequality.Semantic.DeepEqual(in.Spec.Selector, policy.NonV1beta1MatchNoneSelector):
		// no-op, preserve
	case apiequality.Semantic.DeepEqual(in.Spec.Selector, policy.NonV1beta1MatchAllSelector):
		// no-op, preserve
	default:
		// otherwise, make sure the label intended to be used in a match-all or match-none selector
		// never gets combined with user-specified fields
		policy.StripPDBV1beta1Label(out.Spec.Selector)
	}
	return nil
}
