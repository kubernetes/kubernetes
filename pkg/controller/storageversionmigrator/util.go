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
	"fmt"
	"strconv"

	"k8s.io/apimachinery/pkg/runtime/schema"

	corev1 "k8s.io/api/core/v1"
	svmv1alpha1 "k8s.io/api/storagemigration/v1alpha1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func convertResourceVersionToInt(rv string) (int64, error) {
	resourceVersion, err := strconv.ParseInt(rv, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("failed to parse resource version %q: %w", rv, err)
	}

	return resourceVersion, nil
}

func getGVRFromResource(svm *svmv1alpha1.StorageVersionMigration) schema.GroupVersionResource {
	return schema.GroupVersionResource{
		Group:    svm.Spec.Resource.Group,
		Version:  svm.Spec.Resource.Version,
		Resource: svm.Spec.Resource.Resource,
	}
}

// IsConditionTrue returns true if the StorageVersionMigration has the given condition
// It is exported for use in tests
func IsConditionTrue(svm *svmv1alpha1.StorageVersionMigration, conditionType svmv1alpha1.MigrationConditionType) bool {
	return indexOfCondition(svm, conditionType) != -1
}

func indexOfCondition(svm *svmv1alpha1.StorageVersionMigration, conditionType svmv1alpha1.MigrationConditionType) int {
	for i, c := range svm.Status.Conditions {
		if c.Type == conditionType && c.Status == corev1.ConditionTrue {
			return i
		}
	}
	return -1
}

func setStatusConditions(
	toBeUpdatedSVM *svmv1alpha1.StorageVersionMigration,
	conditionType svmv1alpha1.MigrationConditionType,
	reason string,
) *svmv1alpha1.StorageVersionMigration {
	if !IsConditionTrue(toBeUpdatedSVM, conditionType) {
		if conditionType == svmv1alpha1.MigrationSucceeded || conditionType == svmv1alpha1.MigrationFailed {
			runningConditionIdx := indexOfCondition(toBeUpdatedSVM, svmv1alpha1.MigrationRunning)
			if runningConditionIdx != -1 {
				toBeUpdatedSVM.Status.Conditions[runningConditionIdx].Status = corev1.ConditionFalse
			}
		}

		toBeUpdatedSVM.Status.Conditions = append(toBeUpdatedSVM.Status.Conditions, svmv1alpha1.MigrationCondition{
			Type:           conditionType,
			Status:         corev1.ConditionTrue,
			LastUpdateTime: metav1.Now(),
			Reason:         reason,
		})
	}

	return toBeUpdatedSVM
}
