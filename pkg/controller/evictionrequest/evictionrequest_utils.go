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

package evictionrequest

import (
	coordinationv1alpha1 "k8s.io/api/coordination/v1alpha1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1ac "k8s.io/client-go/applyconfigurations/meta/v1"
)

// setCondition creates or updates a condition in the apply configuration,
// preserving the LastTransitionTime when the status has not changed.
// Inspired by k8s.io/apimachinery/pkg/api/meta.SetStatusCondition.
func setCondition(
	existingConditions []metav1.Condition,
	conditionType coordinationv1alpha1.EvictionRequestConditionType,
	status metav1.ConditionStatus,
	reason, message string,
) *metav1ac.ConditionApplyConfiguration {
	now := metav1.Now()
	transitionTime := now

	existing := meta.FindStatusCondition(existingConditions, string(conditionType))
	if existing != nil && existing.Status == status {
		// Status unchanged, so we preserve the original transition time
		transitionTime = existing.LastTransitionTime
	}

	return metav1ac.Condition().
		WithType(string(conditionType)).
		WithStatus(status).
		WithReason(reason).
		WithMessage(message).
		WithLastTransitionTime(transitionTime)
}

// isTerminal returns true if the EvictionRequest has reached
// a terminal state (Canceled or Evicted condition is True).
func isTerminal(evictionRequest *coordinationv1alpha1.EvictionRequest) bool {
	return meta.IsStatusConditionTrue(evictionRequest.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionCanceled)) ||
		meta.IsStatusConditionTrue(evictionRequest.Status.Conditions, string(coordinationv1alpha1.EvictionRequestConditionEvicted))
}
