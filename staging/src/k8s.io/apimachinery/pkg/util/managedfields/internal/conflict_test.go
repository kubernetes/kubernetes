/*
Copyright 2019 The Kubernetes Authors.

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

package internal_test

import (
	"net/http"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/managedfields/internal"
	"sigs.k8s.io/structured-merge-diff/v4/fieldpath"
	"sigs.k8s.io/structured-merge-diff/v4/merge"
)

// TestNewConflictError tests that NewConflictError creates the correct StatusError for a given smd Conflicts
func TestNewConflictError(t *testing.T) {
	testCases := []struct {
		conflict merge.Conflicts
		expected *errors.StatusError
	}{
		{
			conflict: merge.Conflicts{
				merge.Conflict{
					Manager: `{"manager":"foo","operation":"Update","apiVersion":"v1","time":"2001-02-03T04:05:06Z"}`,
					Path:    fieldpath.MakePathOrDie("spec", "replicas"),
				},
			},
			expected: &errors.StatusError{
				ErrStatus: metav1.Status{
					Status: metav1.StatusFailure,
					Code:   http.StatusConflict,
					Reason: metav1.StatusReasonConflict,
					Details: &metav1.StatusDetails{
						Causes: []metav1.StatusCause{
							{
								Type:    metav1.CauseTypeFieldManagerConflict,
								Message: `conflict with "foo" using v1 at 2001-02-03T04:05:06Z`,
								Field:   ".spec.replicas",
							},
						},
					},
					Message: `Apply failed with 1 conflict: conflict with "foo" using v1 at 2001-02-03T04:05:06Z: .spec.replicas`,
				},
			},
		},
		{
			conflict: merge.Conflicts{
				merge.Conflict{
					Manager: `{"manager":"foo","operation":"Update","apiVersion":"v1","time":"2001-02-03T04:05:06Z"}`,
					Path:    fieldpath.MakePathOrDie("spec", "replicas"),
				},
				merge.Conflict{
					Manager: `{"manager":"bar","operation":"Apply"}`,
					Path:    fieldpath.MakePathOrDie("metadata", "labels", "app"),
				},
			},
			expected: &errors.StatusError{
				ErrStatus: metav1.Status{
					Status: metav1.StatusFailure,
					Code:   http.StatusConflict,
					Reason: metav1.StatusReasonConflict,
					Details: &metav1.StatusDetails{
						Causes: []metav1.StatusCause{
							{
								Type:    metav1.CauseTypeFieldManagerConflict,
								Message: `conflict with "foo" using v1 at 2001-02-03T04:05:06Z`,
								Field:   ".spec.replicas",
							},
							{
								Type:    metav1.CauseTypeFieldManagerConflict,
								Message: `conflict with "bar"`,
								Field:   ".metadata.labels.app",
							},
						},
					},
					Message: `Apply failed with 2 conflicts: conflicts with "bar":
- .metadata.labels.app
conflicts with "foo" using v1 at 2001-02-03T04:05:06Z:
- .spec.replicas`,
				},
			},
		},
		{
			conflict: merge.Conflicts{
				merge.Conflict{
					Manager: `{"manager":"foo","operation":"Update","subresource":"scale","apiVersion":"v1","time":"2001-02-03T04:05:06Z"}`,
					Path:    fieldpath.MakePathOrDie("spec", "replicas"),
				},
			},
			expected: &errors.StatusError{
				ErrStatus: metav1.Status{
					Status: metav1.StatusFailure,
					Code:   http.StatusConflict,
					Reason: metav1.StatusReasonConflict,
					Details: &metav1.StatusDetails{
						Causes: []metav1.StatusCause{
							{
								Type:    metav1.CauseTypeFieldManagerConflict,
								Message: `conflict with "foo" with subresource "scale" using v1 at 2001-02-03T04:05:06Z`,
								Field:   ".spec.replicas",
							},
						},
					},
					Message: `Apply failed with 1 conflict: conflict with "foo" with subresource "scale" using v1 at 2001-02-03T04:05:06Z: .spec.replicas`,
				},
			},
		},
	}
	for _, tc := range testCases {
		actual := internal.NewConflictError(tc.conflict)
		if !reflect.DeepEqual(tc.expected, actual) {
			t.Errorf("Expected to get\n%+v\nbut got\n%+v", tc.expected.ErrStatus, actual.ErrStatus)
		}
	}
}
