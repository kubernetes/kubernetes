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
	"strings"
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
					Resource: metav1.GroupResource{
						Group:    "non-empty",
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
					Resource: metav1.GroupResource{
						Group:    "",
						Resource: "",
					},
				},
			},
			errorString: "spec.resource.resource: Required value: resource is required to be set",
		},
		{
			name: "when resource is empty",
			svm: &storagemigration.StorageVersionMigration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-svm",
				},
				Spec: storagemigration.StorageVersionMigrationSpec{
					Resource: metav1.GroupResource{
						Group:    "non-empty",
						Resource: "",
					},
				},
			},
			errorString: "spec.resource.resource: Required value: resource is required to be set",
		},
		{
			name: "when resource is invalid",
			svm: &storagemigration.StorageVersionMigration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-svm",
				},
				Spec: storagemigration.StorageVersionMigrationSpec{
					Resource: metav1.GroupResource{
						Group:    "non-empty",
						Resource: ".",
					},
				},
			},
			errorString: fmt.Sprintf("spec.resource.resource: Invalid value: \".\": %s", dns1035ErrMsg),
		},
		{
			name: "when group is invalid",
			svm: &storagemigration.StorageVersionMigration{
				ObjectMeta: metav1.ObjectMeta{
					Name: "test-svm",
				},
				Spec: storagemigration.StorageVersionMigrationSpec{
					Resource: metav1.GroupResource{
						Group:    "-",
						Resource: "non-empty",
					},
				},
			},
			errorString: fmt.Sprintf("spec.resource.group: Invalid value: \"-\": %s", dns1123ErrMsg),
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
func TestValidateStorageVersionMigrationUpdate(t *testing.T) {
	validSVM := &storagemigration.StorageVersionMigration{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test-svm",
			ResourceVersion: "1",
		},
		Spec: storagemigration.StorageVersionMigrationSpec{
			Resource: metav1.GroupResource{
				Group:    "example.com",
				Resource: "myresources",
			},
		},
	}

	tests := []struct {
		name           string
		newSVM         *storagemigration.StorageVersionMigration
		oldSVM         *storagemigration.StorageVersionMigration
		errorSubstring string
	}{
		{
			name:           "valid update (no change)",
			newSVM:         validSVM.DeepCopy(),
			oldSVM:         validSVM.DeepCopy(),
			errorSubstring: "",
		},
		{
			name: "valid update (metadata change)",
			newSVM: func() *storagemigration.StorageVersionMigration {
				svm := validSVM.DeepCopy()
				svm.ObjectMeta.Labels = map[string]string{"a": "b"}
				return svm
			}(),
			oldSVM:         validSVM.DeepCopy(),
			errorSubstring: "",
		},
		{
			name: "invalid update (spec changed)",
			newSVM: func() *storagemigration.StorageVersionMigration {
				svm := validSVM.DeepCopy()
				svm.Spec.Resource.Group = "new.example.com"
				return svm
			}(),
			oldSVM:         validSVM.DeepCopy(),
			errorSubstring: "spec: Invalid value",
		},
		{
			name: "invalid update (new object is invalid)",
			newSVM: func() *storagemigration.StorageVersionMigration {
				svm := validSVM.DeepCopy()
				svm.Spec.Resource.Resource = ""
				return svm
			}(),
			oldSVM:         validSVM.DeepCopy(),
			errorSubstring: "spec.resource.resource: Required value",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errors := ValidateStorageVersionMigrationUpdate(test.newSVM, test.oldSVM)

			if test.errorSubstring == "" {
				if len(errors) != 0 {
					t.Errorf("Expected no error, got %s", errors.ToAggregate().Error())
				}
			} else {
				if len(errors) == 0 {
					t.Errorf("Expected error containing %q, got no error", test.errorSubstring)
				} else if !strings.Contains(errors.ToAggregate().Error(), test.errorSubstring) {
					t.Errorf("Expected error containing %q, got %s", test.errorSubstring, errors.ToAggregate().Error())
				}
			}
		})
	}
}

var (
	// Add a time variable
	now = metav1.Now()

	// Add LastTransitionTime to all conditions
	runningCond   = metav1.Condition{Type: string(storagemigration.MigrationRunning), Status: metav1.ConditionTrue, Reason: "Running", LastTransitionTime: now}
	succeededCond = metav1.Condition{Type: string(storagemigration.MigrationSucceeded), Status: metav1.ConditionTrue, Reason: "Succeeded", LastTransitionTime: now}
	failedCond    = metav1.Condition{Type: string(storagemigration.MigrationFailed), Status: metav1.ConditionTrue, Reason: "Failed", LastTransitionTime: now}

	runningFalseCond   = metav1.Condition{Type: string(storagemigration.MigrationRunning), Status: metav1.ConditionFalse, Reason: "NotRunning", LastTransitionTime: now}
	succeededFalseCond = metav1.Condition{Type: string(storagemigration.MigrationSucceeded), Status: metav1.ConditionFalse, Reason: "NotSucceeded", LastTransitionTime: now}
	failedFalseCond    = metav1.Condition{Type: string(storagemigration.MigrationFailed), Status: metav1.ConditionFalse, Reason: "NotFailed", LastTransitionTime: now}
)

func newTestSVM(rv string, conditions ...metav1.Condition) *storagemigration.StorageVersionMigration {
	return &storagemigration.StorageVersionMigration{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "test-svm",
			ResourceVersion: "1",
		},
		Spec: storagemigration.StorageVersionMigrationSpec{
			Resource: metav1.GroupResource{
				Group:    "example.com",
				Resource: "myresources",
			},
		},
		Status: storagemigration.StorageVersionMigrationStatus{
			ResourceVersion: rv,
			Conditions:      conditions,
		},
	}
}

func TestValidateStorageVersionMigrationStatusUpdate(t *testing.T) {
	tests := []struct {
		name           string
		newSVM         *storagemigration.StorageVersionMigration
		oldSVM         *storagemigration.StorageVersionMigration
		errorSubstring string
	}{
		{
			name:           "valid: initial status set with RV",
			newSVM:         newTestSVM("123", runningCond),
			oldSVM:         newTestSVM(""),
			errorSubstring: "",
		},
		{
			name:           "valid: transition running to succeeded",
			newSVM:         newTestSVM("123", runningFalseCond, succeededCond),
			oldSVM:         newTestSVM("123", runningCond),
			errorSubstring: "",
		},
		{
			name:           "valid: transition running to failed",
			newSVM:         newTestSVM("123", runningFalseCond, failedCond),
			oldSVM:         newTestSVM("123", runningCond),
			errorSubstring: "",
		},
		{
			name:           "valid: succeeded stays succeeded",
			newSVM:         newTestSVM("123", succeededCond),
			oldSVM:         newTestSVM("123", succeededCond),
			errorSubstring: "",
		},
		{
			name:           "valid: failed stays failed",
			newSVM:         newTestSVM("123", failedCond),
			oldSVM:         newTestSVM("123", failedCond),
			errorSubstring: "",
		},
		{
			name:           "invalid: resource version change",
			newSVM:         newTestSVM("456", runningCond),
			oldSVM:         newTestSVM("123", runningCond),
			errorSubstring: "status.resourceVersion: Invalid value: \"456\": field is immutable",
		},
		{
			name:           "invalid: bad resource version format",
			newSVM:         newTestSVM("abc", runningCond),
			oldSVM:         newTestSVM(""),
			errorSubstring: "status.resourceVersion: Invalid value: \"abc\": resource version is not well formed: abc",
		},
		{
			name:           "invalid: both succeeded and failed are true",
			newSVM:         newTestSVM("123", succeededCond, failedCond),
			oldSVM:         newTestSVM("123", runningCond),
			errorSubstring: "Both success and failed conditions cannot be true at the same time",
		},
		{
			name:           "invalid: succeeded changed from true to false",
			newSVM:         newTestSVM("123", succeededFalseCond),
			oldSVM:         newTestSVM("123", succeededCond),
			errorSubstring: "Success condition cannot be set to false once it is true",
		},
		{
			name:           "invalid: failed changed from true to false",
			newSVM:         newTestSVM("123", failedFalseCond),
			oldSVM:         newTestSVM("123", failedCond),
			errorSubstring: "Failed condition cannot be set to false once it is true",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			errors := ValidateStorageVersionMigrationStatusUpdate(test.newSVM, test.oldSVM)

			if test.errorSubstring == "" {
				if len(errors) != 0 {
					t.Errorf("Expected no error, got %s", errors.ToAggregate().Error())
				}
			} else {
				if len(errors) == 0 {
					t.Errorf("Expected error containing %q, got no error", test.errorSubstring)
				} else if !strings.Contains(errors.ToAggregate().Error(), test.errorSubstring) {
					t.Errorf("Expected error containing %q, got %s", test.errorSubstring, errors.ToAggregate().Error())
				}
			}
		})
	}
}
