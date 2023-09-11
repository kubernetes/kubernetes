/*
Copyright 2018 The Kubernetes Authors.

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

package conversion

import (
	"reflect"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/validation"

	v1 "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1"
	"k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1"
)

func TestRestoreObjectMeta(t *testing.T) {
	tests := []struct {
		name          string
		original      map[string]interface{}
		converted     map[string]interface{}
		expected      map[string]interface{}
		expectedError bool
	}{
		{"no converted metadata",
			map[string]interface{}{"metadata": map[string]interface{}{}, "spec": map[string]interface{}{}},
			map[string]interface{}{"spec": map[string]interface{}{}},
			map[string]interface{}{"spec": map[string]interface{}{}},
			true,
		},
		{"invalid converted metadata",
			map[string]interface{}{"metadata": map[string]interface{}{}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": []interface{}{"foo"}},
			map[string]interface{}{"metadata": []interface{}{"foo"}},
			true,
		},
		{"no original metadata",
			map[string]interface{}{"spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{}, "spec": map[string]interface{}{}},
			false,
		},
		{"invalid original metadata",
			map[string]interface{}{"metadata": []interface{}{"foo"}},
			map[string]interface{}{"metadata": map[string]interface{}{}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{}, "spec": map[string]interface{}{}},
			true,
		},
		{"changed label, annotations and non-label",
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      map[string]interface{}{"a": "A", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "2"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "abc",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			false,
		},
		{"added labels and annotations",
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo": "bar",
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "abc",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			false,
		},
		{"added labels and annotations, with nil before",
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      nil,
				"annotations": nil,
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "abc",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			false,
		},
		{"removed labels and annotations",
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo": "abc",
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo": "bar",
			}, "spec": map[string]interface{}{}},
			false,
		},
		{"nil'ed labels and annotations",
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "bar",
				"labels":      map[string]interface{}{"a": "AA", "b": "B"},
				"annotations": map[string]interface{}{"a": "1", "b": "22"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "abc",
				"labels":      nil,
				"annotations": nil,
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo": "bar",
			}, "spec": map[string]interface{}{}},
			false,
		},
		{"added labels and annotations",
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo": "bar",
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo":         "abc",
				"labels":      map[string]interface{}{"a": nil, "b": "B"},
				"annotations": map[string]interface{}{"a": nil, "b": "22"},
			}, "spec": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{
				"foo": "bar",
			}, "spec": map[string]interface{}{}},
			true,
		},
		{"invalid label key",
			map[string]interface{}{"metadata": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{"labels": map[string]interface{}{"some/non-qualified/label": "x"}}},
			map[string]interface{}{"metadata": map[string]interface{}{}},
			true,
		},
		{"invalid annotation key",
			map[string]interface{}{"metadata": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{"labels": map[string]interface{}{"some/non-qualified/label": "x"}}},
			map[string]interface{}{"metadata": map[string]interface{}{}},
			true,
		},
		{"invalid label value",
			map[string]interface{}{"metadata": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{"labels": map[string]interface{}{"foo": "üäö"}}},
			map[string]interface{}{"metadata": map[string]interface{}{}},
			true,
		},
		{"too big label value",
			map[string]interface{}{"metadata": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{"labels": map[string]interface{}{"foo": strings.Repeat("x", validation.LabelValueMaxLength+1)}}},
			map[string]interface{}{"metadata": map[string]interface{}{}},
			true,
		},
		{"too big annotation value",
			map[string]interface{}{"metadata": map[string]interface{}{}},
			map[string]interface{}{"metadata": map[string]interface{}{"annotations": map[string]interface{}{"foo": strings.Repeat("x", 256*(1<<10)+1)}}},
			map[string]interface{}{"metadata": map[string]interface{}{}},
			true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if err := restoreObjectMeta(&unstructured.Unstructured{Object: tt.original}, &unstructured.Unstructured{Object: tt.converted}); err == nil && tt.expectedError {
				t.Fatalf("expected error, but didn't get one")
			} else if err != nil && !tt.expectedError {
				t.Fatalf("unexpected error: %v", err)
			}

			if !reflect.DeepEqual(tt.converted, tt.expected) {
				t.Errorf("unexpected result: %s", cmp.Diff(tt.expected, tt.converted))
			}
		})
	}
}

func TestCreateConversionReviewObjects(t *testing.T) {
	objects := &unstructured.UnstructuredList{
		Items: []unstructured.Unstructured{
			{
				Object: map[string]interface{}{"apiVersion": "foo/v2", "Kind": "Widget"},
			},
		},
	}

	rawObjects := []runtime.RawExtension{
		{
			Object: &objects.Items[0],
		},
	}

	testcases := []struct {
		Name     string
		Versions []string

		ExpectRequest  runtime.Object
		ExpectResponse runtime.Object
		ExpectErr      string
	}{
		{
			Name:      "no supported versions",
			Versions:  []string{"vx"},
			ExpectErr: "no supported conversion review versions",
		},
		{
			Name:     "v1",
			Versions: []string{"v1", "v1beta1", "v2"},
			ExpectRequest: &v1.ConversionReview{
				Request:  &v1.ConversionRequest{UID: "uid", DesiredAPIVersion: "foo/v1", Objects: rawObjects},
				Response: &v1.ConversionResponse{},
			},
			ExpectResponse: &v1.ConversionReview{},
		},
		{
			Name:     "v1beta1",
			Versions: []string{"v1beta1", "v1", "v2"},
			ExpectRequest: &v1beta1.ConversionReview{
				Request:  &v1beta1.ConversionRequest{UID: "uid", DesiredAPIVersion: "foo/v1", Objects: rawObjects},
				Response: &v1beta1.ConversionResponse{},
			},
			ExpectResponse: &v1beta1.ConversionReview{},
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {
			request, response, err := createConversionReviewObjects(tc.Versions, objects, "foo/v1", "uid")

			if err == nil && len(tc.ExpectErr) > 0 {
				t.Errorf("expected error, got none")
			} else if err != nil && len(tc.ExpectErr) == 0 {
				t.Errorf("unexpected error %v", err)
			} else if err != nil && !strings.Contains(err.Error(), tc.ExpectErr) {
				t.Errorf("expected error containing %q, got %v", tc.ExpectErr, err)
			}

			if e, a := tc.ExpectRequest, request; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected diff: %s", cmp.Diff(e, a))
			}
			if e, a := tc.ExpectResponse, response; !reflect.DeepEqual(e, a) {
				t.Errorf("unexpected diff: %s", cmp.Diff(e, a))
			}
		})
	}
}

func TestGetConvertedObjectsFromResponse(t *testing.T) {
	v1Object := &unstructured.Unstructured{Object: map[string]interface{}{"apiVersion": "foo/v1", "kind": "Widget", "metadata": map[string]interface{}{"name": "myv1"}}}

	testcases := []struct {
		Name     string
		Response runtime.Object

		ExpectObjects []runtime.RawExtension
		ExpectErr     string
	}{
		{
			Name:      "nil response",
			Response:  nil,
			ExpectErr: "unrecognized response type",
		},
		{
			Name:      "unknown type",
			Response:  &unstructured.Unstructured{},
			ExpectErr: "unrecognized response type",
		},

		{
			Name: "minimal valid v1beta1",
			Response: &v1beta1.ConversionReview{
				// apiVersion/kind were not validated originally, preserve backward compatibility
				Response: &v1beta1.ConversionResponse{
					// uid was not validated originally, preserve backward compatibility
					Result: metav1.Status{Status: metav1.StatusSuccess},
				},
			},
			ExpectObjects: nil,
		},
		{
			Name: "valid v1beta1 with objects",
			Response: &v1beta1.ConversionReview{
				// apiVersion/kind were not validated originally, preserve backward compatibility
				Response: &v1beta1.ConversionResponse{
					// uid was not validated originally, preserve backward compatibility
					Result:           metav1.Status{Status: metav1.StatusSuccess},
					ConvertedObjects: []runtime.RawExtension{{Object: v1Object}},
				},
			},
			ExpectObjects: []runtime.RawExtension{{Object: v1Object}},
		},
		{
			Name: "error v1beta1, empty status",
			Response: &v1beta1.ConversionReview{
				Response: &v1beta1.ConversionResponse{
					Result: metav1.Status{Status: ""},
				},
			},
			ExpectErr: `response.result.status was '', not 'Success'`,
		},
		{
			Name: "error v1beta1, failure status",
			Response: &v1beta1.ConversionReview{
				Response: &v1beta1.ConversionResponse{
					Result: metav1.Status{Status: metav1.StatusFailure},
				},
			},
			ExpectErr: `response.result.status was 'Failure', not 'Success'`,
		},
		{
			Name: "error v1beta1, custom status",
			Response: &v1beta1.ConversionReview{
				Response: &v1beta1.ConversionResponse{
					Result: metav1.Status{Status: metav1.StatusFailure, Message: "some failure message"},
				},
			},
			ExpectErr: `some failure message`,
		},
		{
			Name:      "invalid v1beta1, no response",
			Response:  &v1beta1.ConversionReview{},
			ExpectErr: "no response provided",
		},

		{
			Name: "minimal valid v1",
			Response: &v1.ConversionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "apiextensions.k8s.io/v1", Kind: "ConversionReview"},
				Response: &v1.ConversionResponse{
					UID:    "uid",
					Result: metav1.Status{Status: metav1.StatusSuccess},
				},
			},
			ExpectObjects: nil,
		},
		{
			Name: "valid v1 with objects",
			Response: &v1.ConversionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "apiextensions.k8s.io/v1", Kind: "ConversionReview"},
				Response: &v1.ConversionResponse{
					UID:              "uid",
					Result:           metav1.Status{Status: metav1.StatusSuccess},
					ConvertedObjects: []runtime.RawExtension{{Object: v1Object}},
				},
			},
			ExpectObjects: []runtime.RawExtension{{Object: v1Object}},
		},
		{
			Name: "invalid v1, no uid",
			Response: &v1.ConversionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "apiextensions.k8s.io/v1", Kind: "ConversionReview"},
				Response: &v1.ConversionResponse{
					Result: metav1.Status{Status: metav1.StatusSuccess},
				},
			},
			ExpectErr: `expected response.uid="uid"`,
		},
		{
			Name: "invalid v1, no apiVersion",
			Response: &v1.ConversionReview{
				TypeMeta: metav1.TypeMeta{Kind: "ConversionReview"},
				Response: &v1.ConversionResponse{
					UID:    "uid",
					Result: metav1.Status{Status: metav1.StatusSuccess},
				},
			},
			ExpectErr: `expected webhook response of apiextensions.k8s.io/v1, Kind=ConversionReview`,
		},
		{
			Name: "invalid v1, no kind",
			Response: &v1.ConversionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "apiextensions.k8s.io/v1"},
				Response: &v1.ConversionResponse{
					UID:    "uid",
					Result: metav1.Status{Status: metav1.StatusSuccess},
				},
			},
			ExpectErr: `expected webhook response of apiextensions.k8s.io/v1, Kind=ConversionReview`,
		},
		{
			Name: "invalid v1, mismatched apiVersion",
			Response: &v1.ConversionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "apiextensions.k8s.io/v2", Kind: "ConversionReview"},
				Response: &v1.ConversionResponse{
					UID:    "uid",
					Result: metav1.Status{Status: metav1.StatusSuccess},
				},
			},
			ExpectErr: `expected webhook response of apiextensions.k8s.io/v1, Kind=ConversionReview`,
		},
		{
			Name: "invalid v1, mismatched kind",
			Response: &v1.ConversionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "apiextensions.k8s.io/v1", Kind: "ConversionReview2"},
				Response: &v1.ConversionResponse{
					UID:    "uid",
					Result: metav1.Status{Status: metav1.StatusSuccess},
				},
			},
			ExpectErr: `expected webhook response of apiextensions.k8s.io/v1, Kind=ConversionReview`,
		},
		{
			Name: "error v1, empty status",
			Response: &v1.ConversionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "apiextensions.k8s.io/v1", Kind: "ConversionReview"},
				Response: &v1.ConversionResponse{
					UID:    "uid",
					Result: metav1.Status{Status: ""},
				},
			},
			ExpectErr: `response.result.status was '', not 'Success'`,
		},
		{
			Name: "error v1, failure status",
			Response: &v1.ConversionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "apiextensions.k8s.io/v1", Kind: "ConversionReview"},
				Response: &v1.ConversionResponse{
					UID:    "uid",
					Result: metav1.Status{Status: metav1.StatusFailure},
				},
			},
			ExpectErr: `response.result.status was 'Failure', not 'Success'`,
		},
		{
			Name: "error v1, custom status",
			Response: &v1.ConversionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "apiextensions.k8s.io/v1", Kind: "ConversionReview"},
				Response: &v1.ConversionResponse{
					UID:    "uid",
					Result: metav1.Status{Status: metav1.StatusFailure, Message: "some failure message"},
				},
			},
			ExpectErr: `some failure message`,
		},
		{
			Name: "invalid v1, no response",
			Response: &v1.ConversionReview{
				TypeMeta: metav1.TypeMeta{APIVersion: "apiextensions.k8s.io/v1", Kind: "ConversionReview"},
			},
			ExpectErr: "no response provided",
		},
	}

	for _, tc := range testcases {
		t.Run(tc.Name, func(t *testing.T) {

			objects, err := getConvertedObjectsFromResponse("uid", tc.Response)

			if err == nil && len(tc.ExpectErr) > 0 {
				t.Errorf("expected error, got none")
			} else if err != nil && len(tc.ExpectErr) == 0 {
				t.Errorf("unexpected error %v", err)
			} else if err != nil && !strings.Contains(err.Error(), tc.ExpectErr) {
				t.Errorf("expected error containing %q, got %v", tc.ExpectErr, err)
			}

			if !reflect.DeepEqual(objects, tc.ExpectObjects) {
				t.Errorf("unexpected diff: %s", cmp.Diff(tc.ExpectObjects, objects))
			}

		})
	}
}
