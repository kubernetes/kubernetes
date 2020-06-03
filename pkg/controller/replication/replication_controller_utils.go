/*
Copyright 2015 The Kubernetes Authors.

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

// If you make changes to this file, you should also make the corresponding change in ReplicaSet.

package replication

import (
	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// NewReplicationControllerCondition creates a new replication controller condition.
func NewReplicationControllerCondition(condType corev1.ReplicationControllerConditionType, status corev1.ConditionStatus, reason, msg string) corev1.ReplicationControllerCondition {
	return corev1.ReplicationControllerCondition{
		Type:               condType,
		Status:             status,
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            msg,
	}
}

// GetCondition returns a replication controller condition with the provided type if it exists.
func GetCondition(status corev1.ReplicationControllerStatus, condType corev1.ReplicationControllerConditionType) *corev1.ReplicationControllerCondition {
	for i := range status.Conditions {
		c := status.Conditions[i]
		if c.Type == condType {
			return &c
		}
	}
	return nil
}

// SetCondition adds/replaces the given condition in the replication controller status.
func SetCondition(status *corev1.ReplicationControllerStatus, condition corev1.ReplicationControllerCondition) {
	currentCond := GetCondition(*status, condition.Type)
	if currentCond != nil && currentCond.Status == condition.Status && currentCond.Reason == condition.Reason {
		return
	}
	status.Conditions = filterOutCondition(status.Conditions, condition.Type)
	status.Conditions = append(status.Conditions, condition)
}

// RemoveCondition removes the condition with the provided type from the replication controller status.
func RemoveCondition(status *corev1.ReplicationControllerStatus, condType corev1.ReplicationControllerConditionType) {
	status.Conditions = filterOutCondition(status.Conditions, condType)
}

// filterOutCondition returns a new slice of replication controller conditions without conditions with the provided type.
func filterOutCondition(conditions []corev1.ReplicationControllerCondition, condType corev1.ReplicationControllerConditionType) []corev1.ReplicationControllerCondition {
	var newConditions []corev1.ReplicationControllerCondition
	for _, c := range conditions {
		if c.Type == condType {
			continue
		}
		newConditions = append(newConditions, c)
	}
	return newConditions
}
