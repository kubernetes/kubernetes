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

package util

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api/testapi"
)

func TestKindForSuggestion(t *testing.T) {
	tests := []struct {
		in          schema.GroupVersionResource
		srvRes      []*metav1.APIResourceList
		suggestions []string
	}{
		{
			in:          schema.GroupVersionResource{Group: "storage.k8s.io", Version: "", Resource: "ss"},
			suggestions: []string{"sc.storage.k8s.io"},
			srvRes: []*metav1.APIResourceList{
				{
					GroupVersion: "storage.k8s.io/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "storageclasses",
							ShortNames: []string{"sc"},
						},
					},
				},
			},
		},
		{
			in:          schema.GroupVersionResource{Group: "", Version: "", Resource: "deploym"},
			suggestions: []string{"deploy"},
			srvRes:      []*metav1.APIResourceList{},
		},
		{
			in:          schema.GroupVersionResource{Group: "", Version: "", Resource: "ns1"},
			suggestions: []string{"ns"},
			srvRes:      []*metav1.APIResourceList{},
		},
		{
			in:          schema.GroupVersionResource{Group: "", Version: "", Resource: "deploymenst"},
			suggestions: []string{"deployments"},
			srvRes:      []*metav1.APIResourceList{},
		},
	}

	ds := &fakeDiscoveryClient{}
	expander := NewShortcutExpander(testapi.Default.RESTMapper(), ds)
	mapper := &suggestionRESTMapper{
		resourceDiscover: expander.resourceDiscover,
		delegate:         expander,
	}

	for i, test := range tests {
		ds.serverResourcesHandler = func() ([]*metav1.APIResourceList, error) {
			return test.srvRes, nil
		}
		_, err := mapper.KindFor(test.in)
		if err == nil {
			t.Errorf("%d: expected error; got nil", i)
		}

		switch err := err.(type) {
		case *NoResourceWithSuggestionError:
			if !reflect.DeepEqual(err.Suggestions, test.suggestions) {
				t.Errorf("%d: unexpected recommends %s", i, err.Suggestions)
			}
		default:
			t.Errorf("%d: unexpected error %s", i, err.Error())
		}
	}
}
