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

	svmv1beta1 "k8s.io/api/storagemigration/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func setStatusConditions(
	toBeUpdatedSVM *svmv1beta1.StorageVersionMigration,
	conditionType svmv1beta1.MigrationConditionType,
	reason, message string,
) *svmv1beta1.StorageVersionMigration {
	// Cannot set the condition twice.
	if meta.IsStatusConditionTrue(toBeUpdatedSVM.Status.Conditions, string(conditionType)) ||
		meta.IsStatusConditionFalse(toBeUpdatedSVM.Status.Conditions, string(conditionType)) {
		return toBeUpdatedSVM
	}

	if conditionType == svmv1beta1.MigrationSucceeded || conditionType == svmv1beta1.MigrationFailed {
		// set running condition to false if we're finished
		runningCond := meta.FindStatusCondition(toBeUpdatedSVM.Status.Conditions, string(svmv1beta1.MigrationRunning))
		if runningCond != nil {
			runningCond.Status = metav1.ConditionFalse
		}
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
