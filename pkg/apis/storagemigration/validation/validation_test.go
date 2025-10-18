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
	"fmt"
	"testing"

	"k8s.io/kubernetes/pkg/apis/storagemigration"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

const dns1035ErrMsg = "a DNS-1035 label must consist of lower case alphanumeric characters or '-', start with an alphabetic character, and end with an alphanumeric character (e.g. 'my-name',  or 'abc-123', regex used for validation is '[a-z]([-a-z0-9]*[a-z0-9])?')"
const dns1123ErrMsg = "a lowercase RFC 1123 subdomain must consist of lower case alphanumeric characters, '-' or '.', and must start and end with an alphanumeric character (e.g. 'example.com', regex used for validation is '[a-z0-9]([-a-z0-9]*[a-z0-9])?(\\.[a-z0-9]([-a-z0-9]*[a-z0-9])?)*')"

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
			errorString: fmt.Sprintf("[spec.resource.resource: Invalid value: \"\": %s, spec.resource.version: Invalid value: \"\": %s]", dns1123ErrMsg, dns1035ErrMsg),
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
			errorString: fmt.Sprintf("spec.resource.resource: Invalid value: \"\": %s", dns1123ErrMsg),
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
						Resource: ".",
					},
				},
			},
			errorString: fmt.Sprintf("spec.resource.resource: Invalid value: \".\": %s", dns1123ErrMsg),
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
			errorString: fmt.Sprintf("spec.resource.version: Invalid value: \"\": %s", dns1035ErrMsg),
		},
		{
			name: "when version is invalid",
			svm: &storagemigration.StorageVersionMigration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-svm",
				},
				Spec: storagemigration.StorageVersionMigrationSpec{
					Resource: storagemigration.GroupVersionResource{
						Group:    "non-empty",
						Version:  "1",
						Resource: "non-empty",
					},
				},
			},
			errorString: fmt.Sprintf("spec.resource.version: Invalid value: \"1\": %s", dns1035ErrMsg),
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
