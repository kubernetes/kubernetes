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

package poddisruptionbudget

import (
	policy "k8s.io/api/policy/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// UpdateDisruptionAllowedCondition updates the DisruptionAllowed condition
// on a PodDisruptionBudget based on the value of the DisruptionsAllowed field.
func UpdateDisruptionAllowedCondition(pdb *policy.PodDisruptionBudget) {
	if pdb.Status.Conditions == nil {
		pdb.Status.Conditions = make([]metav1.Condition, 0)
	}
	if pdb.Status.DisruptionsAllowed > 0 {
		apimeta.SetStatusCondition(&pdb.Status.Conditions, metav1.Condition{
			Type:               policy.DisruptionAllowedCondition,
			Reason:             policy.SufficientPodsReason,
			Status:             metav1.ConditionTrue,
			ObservedGeneration: pdb.Status.ObservedGeneration,
		})
	} else {
		apimeta.SetStatusCondition(&pdb.Status.Conditions, metav1.Condition{
			Type:               policy.DisruptionAllowedCondition,
			Reason:             policy.InsufficientPodsReason,
			Status:             metav1.ConditionFalse,
			ObservedGeneration: pdb.Status.ObservedGeneration,
		})
	}
}

// ConditionsAreUpToDate checks whether the status and reason for the
// DisruptionAllowed condition are set to the correct values based on the
// DisruptionsAllowed field.
func ConditionsAreUpToDate(pdb *policy.PodDisruptionBudget) bool {
	cond := apimeta.FindStatusCondition(pdb.Status.Conditions, policy.DisruptionAllowedCondition)
	if cond == nil {
		return false
	}

	if pdb.Status.ObservedGeneration != pdb.Generation {
		return false
	}

	if pdb.Status.DisruptionsAllowed > 0 {
		return cond.Status == metav1.ConditionTrue && cond.Reason == policy.SufficientPodsReason
	}
	return cond.Status == metav1.ConditionFalse && cond.Reason == policy.InsufficientPodsReason
}
