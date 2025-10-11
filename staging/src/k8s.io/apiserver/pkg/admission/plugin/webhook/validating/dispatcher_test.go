/*
Copyright 2025 The Kubernetes Authors.

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

package validating

import (
	"errors"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestAggregateWebhookErrors(t *testing.T) {
	tests := []struct {
		name                 string
		errors               []error
		expectedErrorMessage string
		expectedCode         int32
		expectedUnmerged     int // number of unmerged errors
	}{
		{
			name:                 "single error",
			errors:               []error{apierrors.NewBadRequest("bad request")},
			expectedErrorMessage: "bad request",
			expectedCode:         400,
			expectedUnmerged:     0,
		},
		{
			name: "no StatusErrors - returns first error",
			errors: []error{
				errors.New("first regular error"),
				errors.New("second regular error"),
			},
			expectedErrorMessage: "first regular error",
			expectedCode:         0,
			expectedUnmerged:     1,
		},
		{
			name: "mixed error types - StatusError and regular error",
			errors: []error{
				errors.New("regular error"),
				apierrors.NewBadRequest("bad request"),
			},
			expectedErrorMessage: "bad request",
			expectedCode:         400,
			expectedUnmerged:     1,
		},
		{
			name: "multiple mergeable errors are combined into one",
			errors: []error{
				apierrors.NewBadRequest("first bad request"),
				apierrors.NewBadRequest("second bad request"),
			},
			expectedErrorMessage: "first bad request; second bad request",
			expectedCode:         400,
			expectedUnmerged:     0,
		},
		{
			name: "errors with StatusDetails merging",
			errors: []error{
				&apierrors.StatusError{ErrStatus: metav1.Status{
					Code: 400, Reason: metav1.StatusReasonBadRequest, Message: "first error",
					Details: &metav1.StatusDetails{Name: "test", Causes: []metav1.StatusCause{{Type: "FieldValueInvalid", Field: "field1"}}},
				}},
				&apierrors.StatusError{ErrStatus: metav1.Status{
					Code: 400, Reason: metav1.StatusReasonBadRequest, Message: "second error",
					Details: &metav1.StatusDetails{Name: "test", Causes: []metav1.StatusCause{{Type: "FieldValueInvalid", Field: "field2"}}},
				}},
			},
			expectedErrorMessage: "first error; second error",
			expectedCode:         400,
			expectedUnmerged:     0,
		},
		{
			name: "errors with nil Details merging",
			errors: []error{
				&apierrors.StatusError{ErrStatus: metav1.Status{
					Code: 400, Reason: metav1.StatusReasonBadRequest, Message: "first error",
					Details: nil,
				}},
				&apierrors.StatusError{ErrStatus: metav1.Status{
					Code: 400, Reason: metav1.StatusReasonBadRequest, Message: "second error",
					Details: nil,
				}},
			},
			expectedErrorMessage: "first error; second error",
			expectedCode:         400,
			expectedUnmerged:     0,
		},
		{
			name: "non-mergeable errors with different codes",
			errors: []error{
				apierrors.NewBadRequest("bad request"),
				apierrors.NewForbidden(schema.GroupResource{Group: "apps", Resource: "deployments"}, "test", errors.New("forbidden")),
			},
			expectedErrorMessage: "bad request",
			expectedCode:         400,
			expectedUnmerged:     1,
		},
		{
			name: "multiple errors with priority - BadRequest should be selected over InternalError",
			errors: []error{
				apierrors.NewInternalError(errors.New("internal error")),
				apierrors.NewBadRequest("bad request"),
				apierrors.NewServiceUnavailable("service unavailable"),
			},
			expectedErrorMessage: "bad request",
			expectedCode:         400,
			expectedUnmerged:     2, // internal error and service unavailable are unmerged
		},
		{
			name: "priority order test - Invalid takes precedence over Forbidden and Unauthorized",
			errors: []error{
				apierrors.NewUnauthorized("unauthorized"),
				apierrors.NewForbidden(schema.GroupResource{Group: "apps", Resource: "deployments"}, "test", errors.New("forbidden")),
				apierrors.NewInvalid(schema.GroupKind{Group: "apps", Kind: "Deployment"}, "test", nil),
			},
			expectedErrorMessage: "Deployment.apps \"test\" is invalid",
			expectedCode:         422,
			expectedUnmerged:     2, // unauthorized and forbidden are unmerged
		},
		{
			name: "complete priority hierarchy test",
			errors: []error{
				apierrors.NewInternalError(errors.New("internal error")),
				apierrors.NewServiceUnavailable("service unavailable"),
				apierrors.NewUnauthorized("unauthorized"),
				apierrors.NewForbidden(schema.GroupResource{Group: "apps", Resource: "deployments"}, "test", errors.New("forbidden")),
				apierrors.NewInvalid(schema.GroupKind{Group: "apps", Kind: "Deployment"}, "test", nil),
				apierrors.NewBadRequest("bad request"),
			},
			expectedErrorMessage: "bad request",
			expectedCode:         400,
			expectedUnmerged:     5, // all other errors are unmerged
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result, unmergedErrs := aggregateWebhookErrors(tt.errors)
			if result == nil {
				t.Fatalf("Expected error but got nil")
			}

			if result.Error() != tt.expectedErrorMessage {
				t.Errorf("Expected error message %q, got %q", tt.expectedErrorMessage, result.Error())
			}

			if statusErr, ok := result.(*apierrors.StatusError); ok { //nolint:errorlint
				if statusErr.Status().Code != tt.expectedCode {
					t.Errorf("Expected status code %d, got %d", tt.expectedCode, statusErr.Status().Code)
				}
			} else if tt.expectedCode != 0 {
				t.Errorf("Expected StatusError but got %T", result)
			}

			if len(unmergedErrs) != tt.expectedUnmerged {
				t.Errorf("Expected %d unmerged errors, got %d", tt.expectedUnmerged, len(unmergedErrs))
			}
		})
	}
}

func TestCanMergeStatus(t *testing.T) {
	tests := []struct {
		name     string
		status1  *metav1.Status
		status2  *metav1.Status
		expected bool
	}{
		{
			name:     "nil statuses",
			status1:  nil,
			status2:  nil,
			expected: false,
		},
		{
			name:     "one nil status",
			status1:  &metav1.Status{Code: 400, Reason: metav1.StatusReasonBadRequest},
			status2:  nil,
			expected: false,
		},
		{
			name:     "different codes",
			status1:  &metav1.Status{Code: 400, Reason: metav1.StatusReasonBadRequest},
			status2:  &metav1.Status{Code: 403, Reason: metav1.StatusReasonBadRequest},
			expected: false,
		},
		{
			name:     "different reasons",
			status1:  &metav1.Status{Code: 400, Reason: metav1.StatusReasonBadRequest},
			status2:  &metav1.Status{Code: 400, Reason: metav1.StatusReasonInvalid},
			expected: false,
		},
		{
			name:     "same code and reason, no details",
			status1:  &metav1.Status{Code: 400, Reason: metav1.StatusReasonBadRequest},
			status2:  &metav1.Status{Code: 400, Reason: metav1.StatusReasonBadRequest},
			expected: true,
		},
		{
			name:     "same code and reason, same resource details",
			status1:  &metav1.Status{Code: 400, Reason: metav1.StatusReasonBadRequest, Details: &metav1.StatusDetails{Name: "test", Kind: "Pod"}},
			status2:  &metav1.Status{Code: 400, Reason: metav1.StatusReasonBadRequest, Details: &metav1.StatusDetails{Name: "test", Kind: "Pod"}},
			expected: true,
		},
		{
			name:     "same code and reason, different resource names",
			status1:  &metav1.Status{Code: 400, Reason: metav1.StatusReasonBadRequest, Details: &metav1.StatusDetails{Name: "test1", Kind: "Pod"}},
			status2:  &metav1.Status{Code: 400, Reason: metav1.StatusReasonBadRequest, Details: &metav1.StatusDetails{Name: "test2", Kind: "Pod"}},
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := canMergeStatus(tt.status1, tt.status2)
			if result != tt.expected {
				t.Errorf("Expected %v, got %v", tt.expected, result)
			}
		})
	}
}

func TestMergeStatus(t *testing.T) {
	tests := []struct {
		name                      string
		dest                      *metav1.Status
		source                    *metav1.Status
		expectedStatus            string
		expectedReason            metav1.StatusReason
		expectedCode              int32
		expectedRetryAfterSeconds int32
		expectedMessage           string
		expectedCauses            int
		shouldDestRemainNil       bool
		shouldMerge               bool // indicates if mergeStatus should actually merge (assumes canMergeStatus would return true)
	}{
		{
			name:                "nil dest",
			dest:                nil,
			source:              &metav1.Status{Message: "source message"},
			expectedMessage:     "",
			expectedCauses:      0,
			shouldDestRemainNil: true,
			shouldMerge:         false,
		},
		{
			name:            "nil source",
			dest:            &metav1.Status{Message: "dest message", Status: "Failure", Code: 400},
			source:          nil,
			expectedMessage: "dest message",
			expectedCauses:  0,
			expectedStatus:  "Failure",
			expectedCode:    400,
			shouldMerge:     false,
		},
		{
			name:            "compatible statuses - same code and reason",
			dest:            &metav1.Status{Message: "first error", Status: "Failure", Reason: "Invalid", Code: 422},
			source:          &metav1.Status{Message: "second error", Status: "Failure", Reason: "Invalid", Code: 422},
			expectedMessage: "first error; second error",
			expectedCauses:  0,
			expectedStatus:  "Failure",
			expectedReason:  "Invalid",
			expectedCode:    422,
			shouldMerge:     true,
		},
		{
			name:            "merge empty source message",
			dest:            &metav1.Status{Message: "first error", Status: "Failure", Reason: "Invalid", Code: 422},
			source:          &metav1.Status{Message: "", Status: "Failure", Reason: "Invalid", Code: 422},
			expectedMessage: "first error",
			expectedCauses:  0,
			expectedStatus:  "Failure",
			expectedReason:  "Invalid",
			expectedCode:    422,
			shouldMerge:     true,
		},
		{
			name:            "empty dest message with source message",
			dest:            &metav1.Status{Message: "", Status: "Failure", Reason: "Invalid", Code: 422},
			source:          &metav1.Status{Message: "source error", Status: "Failure", Reason: "Invalid", Code: 422},
			expectedMessage: "source error",
			expectedCauses:  0,
			expectedStatus:  "Failure",
			expectedReason:  "Invalid",
			expectedCode:    422,
			shouldMerge:     true,
		},
		{
			name: "merge compatible causes and details - same resource",
			dest: &metav1.Status{
				Message: "first error",
				Status:  "Failure",
				Reason:  "Invalid",
				Code:    422,
				Details: &metav1.StatusDetails{
					Name:   "test-pod",
					Kind:   "Pod",
					Group:  "",
					Causes: []metav1.StatusCause{{Type: "FieldValueInvalid", Field: "field1"}},
				},
			},
			source: &metav1.Status{
				Message: "second error",
				Status:  "Failure",
				Reason:  "Invalid",
				Code:    422,
				Details: &metav1.StatusDetails{
					Name:   "test-pod", // Same resource
					Kind:   "Pod",
					Group:  "",
					Causes: []metav1.StatusCause{{Type: "FieldValueInvalid", Field: "field2"}},
				},
			},
			expectedMessage: "first error; second error",
			expectedCauses:  2,
			expectedStatus:  "Failure",
			expectedReason:  "Invalid",
			expectedCode:    422,
			shouldMerge:     true,
		},
		{
			name: "merge details into nil dest details - both nil",
			dest: &metav1.Status{
				Message: "first error",
				Status:  "Failure",
				Reason:  "Invalid",
				Code:    422,
				Details: nil,
			},
			source: &metav1.Status{
				Message: "second error",
				Status:  "Failure",
				Reason:  "Invalid",
				Code:    422,
				Details: nil, // Both nil - should merge
			},
			expectedMessage: "first error; second error",
			expectedCauses:  0,
			expectedStatus:  "Failure",
			expectedReason:  "Invalid",
			expectedCode:    422,
			shouldMerge:     true,
		},
		{
			name: "merge retry after seconds - use higher value",
			dest: &metav1.Status{
				Message: "first error",
				Status:  "Failure",
				Reason:  "ServiceUnavailable",
				Code:    503,
				Details: &metav1.StatusDetails{
					Name:              "test-service",
					Kind:              "Service",
					RetryAfterSeconds: 10,
					Causes:            []metav1.StatusCause{{Type: "ServiceUnavailable", Field: "spec"}},
				},
			},
			source: &metav1.Status{
				Message: "second error",
				Status:  "Failure",
				Reason:  "ServiceUnavailable",
				Code:    503,
				Details: &metav1.StatusDetails{
					Name:              "test-service",
					Kind:              "Service",
					RetryAfterSeconds: 30,
					Causes:            []metav1.StatusCause{{Type: "ServiceUnavailable", Field: "metadata"}},
				},
			},
			expectedMessage:           "first error; second error",
			expectedCauses:            2,
			expectedRetryAfterSeconds: 30,
			expectedStatus:            "Failure",
			expectedReason:            "ServiceUnavailable",
			expectedCode:              503,
			shouldMerge:               true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var originalMessage string
			var originalCauses int
			if tt.dest != nil {
				originalMessage = tt.dest.Message
				originalCauses = getCausesCount(tt.dest)
			}
			mergeStatus(tt.dest, tt.source)

			if tt.shouldDestRemainNil {
				if tt.dest != nil {
					t.Errorf("Expected dest to remain nil, but got %+v", tt.dest)
				}
				return
			}

			if tt.dest == nil {
				t.Fatalf("Expected dest to be non-nil after merge")
			}

			if tt.dest.Message != tt.expectedMessage {
				t.Errorf("Expected message %q, got %q", tt.expectedMessage, tt.dest.Message)
			}

			actualCauses := getCausesCount(tt.dest)
			if actualCauses != tt.expectedCauses {
				t.Errorf("Expected %d causes, got %d", tt.expectedCauses, actualCauses)
			}

			checkField := func(expected, actual interface{}, name string) {
				if expected != "" && expected != actual {
					t.Errorf("Expected %s %v, got %v", name, expected, actual)
				}
			}
			checkField(tt.expectedStatus, tt.dest.Status, "Status")
			checkField(tt.expectedReason, tt.dest.Reason, "Reason")
			if tt.expectedCode != 0 && tt.dest.Code != tt.expectedCode {
				t.Errorf("Expected Code %d, got %d", tt.expectedCode, tt.dest.Code)
			}

			if tt.expectedRetryAfterSeconds != 0 {
				actualRetryAfterSeconds := int32(0)
				if tt.dest.Details != nil {
					actualRetryAfterSeconds = tt.dest.Details.RetryAfterSeconds
				}
				if actualRetryAfterSeconds != tt.expectedRetryAfterSeconds {
					t.Errorf("Expected RetryAfterSeconds %d, got %d", tt.expectedRetryAfterSeconds, actualRetryAfterSeconds)
				}
			}

			if !tt.shouldMerge && tt.dest != nil && tt.source != nil {
				// When shouldMerge is false, verify that no actual merging occurred
				if tt.dest.Message != originalMessage {
					t.Errorf("Expected no merge (shouldMerge=false), but message changed from %q to %q", originalMessage, tt.dest.Message)
				}
				currentCauses := getCausesCount(tt.dest)
				if currentCauses != originalCauses {
					t.Errorf("Expected no merge (shouldMerge=false), but causes changed from %d to %d", originalCauses, currentCauses)
				}
			}

		})
	}
}

func getCausesCount(status *metav1.Status) int {
	if status.Details != nil && status.Details.Causes != nil {
		return len(status.Details.Causes)
	}
	return 0
}
