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
		name          string
		errors        []error
		expectedError string
		expectedCode  int32
	}{
		{
			name:          "single error",
			errors:        []error{apierrors.NewBadRequest("bad request")},
			expectedError: "bad request",
			expectedCode:  400,
		},
		{
			name: "no StatusErrors - returns first error",
			errors: []error{
				errors.New("first regular error"),
				errors.New("second regular error"),
			},
			expectedError: "first regular error",
			expectedCode:  0,
		},
		{
			name: "mixed error types - StatusError and regular error",
			errors: []error{
				errors.New("regular error"),
				apierrors.NewBadRequest("bad request"),
			},
			expectedError: "bad request",
			expectedCode:  400,
		},
		{
			name: "multiple mergeable errors are combined into one",
			errors: []error{
				apierrors.NewBadRequest("first bad request"),
				apierrors.NewBadRequest("second bad request"),
			},
			expectedError: "first bad request; second bad request",
			expectedCode:  400,
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
			expectedError: "first error; second error",
			expectedCode:  400,
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
			expectedError: "first error; second error",
			expectedCode:  400,
		},
		{
			name: "non-mergeable errors with different codes",
			errors: []error{
				apierrors.NewBadRequest("bad request"),
				apierrors.NewForbidden(schema.GroupResource{Group: "apps", Resource: "deployments"}, "test", errors.New("forbidden")),
			},
			expectedError: "bad request",
			expectedCode:  400,
		},
		{
			name: "multiple errors with priority - BadRequest should be selected over InternalError",
			errors: []error{
				apierrors.NewInternalError(errors.New("internal error")),
				apierrors.NewBadRequest("bad request"),
				apierrors.NewServiceUnavailable("service unavailable"),
			},
			expectedError: "bad request",
			expectedCode:  400,
		},
		{
			name: "priority order test - Invalid takes precedence over Forbidden and Unauthorized",
			errors: []error{
				apierrors.NewUnauthorized("unauthorized"),
				apierrors.NewForbidden(schema.GroupResource{Group: "apps", Resource: "deployments"}, "test", errors.New("forbidden")),
				apierrors.NewInvalid(schema.GroupKind{Group: "apps", Kind: "Deployment"}, "test", nil),
			},
			expectedError: "Deployment.apps \"test\" is invalid",
			expectedCode:  422,
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
			expectedError: "bad request",
			expectedCode:  400,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := aggregateWebhookErrors(tt.errors)
			if result == nil {
				t.Fatalf("Expected error but got nil")
			}

			if result.Error() != tt.expectedError {
				t.Errorf("Expected error message %q, got %q", tt.expectedError, result.Error())
			}

			if statusErr, ok := result.(*apierrors.StatusError); ok {
				if statusErr.Status().Code != tt.expectedCode {
					t.Errorf("Expected status code %d, got %d", tt.expectedCode, statusErr.Status().Code)
				}
			} else if tt.expectedCode != 0 {
				t.Errorf("Expected StatusError but got %T", result)
			}
		})
	}
}
