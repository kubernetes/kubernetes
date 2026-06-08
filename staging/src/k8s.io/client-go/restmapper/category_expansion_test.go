/*
Copyright 2017 The Kubernetes Authors.

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

package restmapper

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

func TestCategoryExpansion(t *testing.T) {
	tests := []struct {
		name string
		arg  string

		expected   []schema.GroupResource
		expectedOk bool
	}{
		{
			name:     "no-replacement",
			arg:      "service",
			expected: nil,
		},
		{
			name: "all-replacement",
			arg:  "all",
			expected: []schema.GroupResource{
				{Resource: "one"},
				{Resource: "two"},
				{Resource: "three", Group: "alpha"},
				{Resource: "four", Group: "bravo"},
			},
			expectedOk: true,
		},
	}

	for _, test := range tests {
		simpleCategoryExpander := SimpleCategoryExpander{
			Expansions: map[string][]schema.GroupResource{
				"all": {
					{Group: "", Resource: "one"},
					{Group: "", Resource: "two"},
					{Group: "alpha", Resource: "three"},
					{Group: "bravo", Resource: "four"},
				},
			},
		}

		actual, actualOk := simpleCategoryExpander.Expand(test.arg)
		if e, a := test.expected, actual; !reflect.DeepEqual(e, a) {
			t.Errorf("%s:  expected %s, got %s", test.name, e, a)
		}
		if e, a := test.expectedOk, actualOk; e != a {
			t.Errorf("%s:  expected %v, got %v", test.name, e, a)
		}
	}
}

func TestDiscoveryCategoryExpander(t *testing.T) {
	tests := []struct {
		category       string
		serverResponse []*metav1.APIResourceList
		expected       []schema.GroupResource
	}{
		{
			category: "all",
			serverResponse: []*metav1.APIResourceList{
				{
					GroupVersion: "batch/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "jobs",
							ShortNames: []string{"jz"},
							Categories: []string{"all"},
						},
					},
				},
			},
			expected: []schema.GroupResource{
				{
					Group:    "batch",
					Resource: "jobs",
				},
			},
		},
		{
			category: "all",
			serverResponse: []*metav1.APIResourceList{
				{
					GroupVersion: "batch/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "jobs",
							ShortNames: []string{"jz"},
						},
					},
				},
			},
		},
		{
			category: "targaryens",
			serverResponse: []*metav1.APIResourceList{
				{
					GroupVersion: "batch/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "jobs",
							ShortNames: []string{"jz"},
							Categories: []string{"all"},
						},
					},
				},
			},
		},
	}

	dc := &fakeDiscoveryClient{}
	for _, test := range tests {
		dc.serverResourcesHandler = func() ([]*metav1.APIResourceList, error) {
			return test.serverResponse, nil
		}
		expander := NewDiscoveryCategoryExpander(dc)
		expanded, _ := expander.Expand(test.category)
		if !reflect.DeepEqual(expanded, test.expected) {
			t.Errorf("expected %v, got %v", test.expected, expanded)
		}
	}

}
