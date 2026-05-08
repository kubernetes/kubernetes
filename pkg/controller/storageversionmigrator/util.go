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

package storageversionmigrator

import (
	"k8s.io/apimachinery/pkg/api/meta"

	svmv1 "k8s.io/api/storagemigration/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func setStatusConditions(
	toBeUpdatedSVM *svmv1.StorageVersionMigration,
	conditionType svmv1.MigrationConditionType,
	reason, message string,
) *svmv1.StorageVersionMigration {
	// Cannot set the condition twice.
	if meta.IsStatusConditionTrue(toBeUpdatedSVM.Status.Conditions, string(conditionType)) ||
		meta.IsStatusConditionFalse(toBeUpdatedSVM.Status.Conditions, string(conditionType)) {
		return toBeUpdatedSVM
	}

	toBeUpdatedSVM.Status.Conditions = append(toBeUpdatedSVM.Status.Conditions, metav1.Condition{
		Type:               string(conditionType),
		Status:             metav1.ConditionTrue,
		LastTransitionTime: metav1.Now(),
		Reason:             reason,
		Message:            message,
	})

	return toBeUpdatedSVM
}
