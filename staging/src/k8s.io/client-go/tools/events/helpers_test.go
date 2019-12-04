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

package events

import (
	"testing"

	"k8s.io/api/core/v1"
	"k8s.io/api/events/v1beta1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestGetFieldSelector(t *testing.T) {
	tests := []struct {
		desc               string
		groupVersion       schema.GroupVersion
		regardingName      string
		regardingNamespace string
		regardingKind      string
		regardingUID       string
		expected           fields.Set
		expectedErr        bool
	}{
		{
			desc:               "events.k8s.io/v1beta1 event with empty parameters",
			groupVersion:       schema.GroupVersion{Group: v1beta1.GroupName, Version: "v1beta1"},
			regardingName:      "",
			regardingNamespace: "",
			regardingKind:      "",
			regardingUID:       "",
			expected:           fields.Set{},
			expectedErr:        false,
		},
		{
			desc:               "events.k8s.io/v1beta1 event with non-empty parameters",
			groupVersion:       schema.GroupVersion{Group: v1beta1.GroupName, Version: "v1beta1"},
			regardingName:      "test-node",
			regardingNamespace: "default",
			regardingKind:      "Event",
			regardingUID:       "2c55cad7-ee4e-11e9-abe1-525400e7bc6b",
			expected: fields.Set{
				"regarding.name":      "test-node",
				"regarding.namespace": "default",
				"regarding.kind":      "Event",
				"regarding.uid":       "2c55cad7-ee4e-11e9-abe1-525400e7bc6b",
			},
			expectedErr: false,
		},
		{
			desc:               "v1 event with non-empty parameters",
			groupVersion:       schema.GroupVersion{Group: v1.GroupName, Version: "v1"},
			regardingName:      "",
			regardingNamespace: "",
			regardingKind:      "",
			regardingUID:       "",
			expected:           fields.Set{},
			expectedErr:        false,
		},
		{
			desc:               "v1 event with non-empty parameters",
			groupVersion:       schema.GroupVersion{Group: v1.GroupName, Version: "v1"},
			regardingName:      "test-node",
			regardingNamespace: "default",
			regardingKind:      "Event",
			regardingUID:       "2c55cad7-ee4e-11e9-abe1-525400e7bc6b",
			expected: fields.Set{
				"involvedObject.name":      "test-node",
				"involvedObject.namespace": "default",
				"involvedObject.kind":      "Event",
				"involvedObject.uid":       "2c55cad7-ee4e-11e9-abe1-525400e7bc6b",
			},
			expectedErr: false,
		},
		{
			desc:               "unknown group version",
			groupVersion:       schema.GroupVersion{Group: v1.GroupName, Version: "v1alpha1"},
			regardingName:      "test-node",
			regardingNamespace: "default",
			regardingKind:      "Event",
			regardingUID:       "2c55cad7-ee4e-11e9-abe1-525400e7bc6b",
			expected:           fields.Set{},
			expectedErr:        true,
		},
	}

	for _, test := range tests {
		result, err := GetFieldSelector(test.groupVersion, test.regardingName, test.regardingNamespace, test.regardingKind, test.regardingUID)
		if !test.expectedErr && err != nil {
			t.Errorf("Unable to get field selector with %v", err)
		}
		if test.expectedErr && err == nil {
			t.Errorf("Expect error but got nil")
		}
		if !result.Matches(test.expected) {
			t.Errorf("Test %s expected %v, but got %v", test.desc, test.expected.AsSelector(), result)
		}
	}
}
