/*
Copyright 2015 The Kubernetes Authors.

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

package secret

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	api "k8s.io/kubernetes/pkg/apis/core"

	// install all api groups for testing
	"fmt"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	_ "k8s.io/kubernetes/pkg/api/testapi"
)

func TestExportSecret(t *testing.T) {
	tests := []struct {
		objIn     runtime.Object
		objOut    runtime.Object
		exact     bool
		expectErr bool
	}{
		{
			objIn: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
				Data: map[string][]byte{
					"foo": []byte("bar"),
				},
			},
			objOut: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
				Data: map[string][]byte{
					"foo": []byte("bar"),
				},
			},
			exact: true,
		},
		{
			objIn: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
				},
				Type: api.SecretTypeServiceAccountToken,
			},
			expectErr: true,
		},
		{
			objIn: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
					Annotations: map[string]string{
						api.ServiceAccountUIDKey: "true",
					},
				},
			},
			expectErr: true,
		},
		{
			objIn:     &api.Pod{},
			expectErr: true,
		},
	}

	for _, test := range tests {
		err := Strategy.Export(genericapirequest.NewContext(), test.objIn, test.exact)
		if err != nil {
			if !test.expectErr {
				t.Errorf("unexpected error: %v", err)
			}
			continue
		}
		if test.expectErr {
			t.Error("unexpected non-error")
			continue
		}
		if !reflect.DeepEqual(test.objIn, test.objOut) {
			t.Errorf("expected:\n%v\nsaw:\n%v\n", test.objOut, test.objIn)
		}
	}
}

func TestGetAttrs(t *testing.T) {
	testcases := []struct {
		name           string
		objIn          runtime.Object
		expectedLabels labels.Set
		expectedFields fields.Set
		expectedBool   bool
		expectedErr    error
	}{
		{
			name:           "expected error",
			objIn:          &api.ResourceQuotaList{},
			expectedLabels: nil,
			expectedFields: nil,
			expectedBool:   false,
			expectedErr:    fmt.Errorf("not a secret"),
		},
		{
			name: "expected metadata.name=foo and metadata.namespace=bar",
			objIn: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "bar",
					Labels: map[string]string{
						"key": "value",
					},
				},
			},
			expectedLabels: labels.Set{"key": "value"},
			expectedFields: fields.Set{"metadata.name": "foo", "metadata.namespace": "bar", "type": ""},
			expectedBool:   false,
			expectedErr:    nil,
		},
		{
			name: "expected metadata.name=foo1 and metadata.namespace=bar1",
			objIn: &api.Secret{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo1",
					Namespace: "bar1",
					Labels: map[string]string{
						"key1": "value1",
					},
					Initializers: &metav1.Initializers{},
				},
			},
			expectedLabels: labels.Set{"key1": "value1"},
			expectedFields: fields.Set{"metadata.name": "foo1", "metadata.namespace": "bar1", "type": ""},
			expectedBool:   true,
			expectedErr:    nil,
		},
	}

	for _, testcase := range testcases {
		rLabels, rFields, rBool, rErr := GetAttrs(testcase.objIn)

		if !reflect.DeepEqual(testcase.expectedErr, rErr) {
			t.Errorf("unxpected error, expected: %v, actual: %v", testcase.expectedLabels, rErr)
		}
		if !reflect.DeepEqual(testcase.expectedLabels, rLabels) {
			t.Errorf("unxpected labels, expected: %v, actual: %v", testcase.expectedLabels, rLabels)
		}
		if !reflect.DeepEqual(testcase.expectedFields, rFields) {
			t.Errorf("unxpected fields, expected: %v, actual: %v", testcase.expectedLabels, rFields)
		}
		if testcase.expectedBool != rBool {
			t.Errorf("unxpected bool, expected: %v, actual: %v", testcase.expectedBool, rBool)
		}
	}
}

func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		"v1",
		"Secret",
		SelectableFields(&api.Secret{}),
		nil,
	)
}
