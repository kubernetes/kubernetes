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

package validation

import (
	"testing"

	"k8s.io/kubernetes/pkg/apis/storagemigration"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// TestValidateStorageVersionMigration tests the ValidateStorageVersionMigration function
func TestValidateStorageVersionMigration(t *testing.T) {
	tests := []struct {
		name        string
		svm         *storagemigration.StorageVersionMigration
		errorString string
	}{
		{
			name: "when all fields are non-empty",
			svm: &storagemigration.StorageVersionMigration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-svm",
				},
				Spec: storagemigration.StorageVersionMigrationSpec{
					Resource: storagemigration.GroupVersionResource{
						Group:    "non-empty",
						Version:  "non-empty",
						Resource: "non-empty",
					},
				},
			},
			errorString: "",
		},
		{
			name: "when all fields are empty",
			svm: &storagemigration.StorageVersionMigration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-svm",
				},
				Spec: storagemigration.StorageVersionMigrationSpec{
					Resource: storagemigration.GroupVersionResource{
						Group:    "",
						Version:  "",
						Resource: "",
					},
				},
			},
			errorString: "[spec.resource.resource: Required value: resource is required, spec.resource.version: Required value: version is required]",
		},
		{
			name: "when resource is empty",
			svm: &storagemigration.StorageVersionMigration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-svm",
				},
				Spec: storagemigration.StorageVersionMigrationSpec{
					Resource: storagemigration.GroupVersionResource{
						Group:    "non-empty",
						Version:  "non-empty",
						Resource: "",
					},
				},
			},
			errorString: "spec.resource.resource: Required value: resource is required",
		},
		{
			name: "when version is empty",
			svm: &storagemigration.StorageVersionMigration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-svm",
				},
				Spec: storagemigration.StorageVersionMigrationSpec{
					Resource: storagemigration.GroupVersionResource{
						Group:    "non-empty",
						Version:  "",
						Resource: "non-empty",
					},
				},
			},
			errorString: "spec.resource.version: Required value: version is required",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errors := ValidateStorageVersionMigration(test.svm)

			errorString := ""
			if len(errors) == 0 {
				errorString = ""
			} else {
				errorString = errors.ToAggregate().Error()
			}

			if errorString != test.errorString {
				t.Errorf("Expected error string %s, got %s", test.errorString, errorString)
			}
		})
	}
}
