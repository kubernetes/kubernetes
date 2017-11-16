/*
Copyright 2016 The Kubernetes Authors.

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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api/testapi"
)

func TestReplaceAliases(t *testing.T) {
	tests := []struct {
		name     string
		arg      string
		expected schema.GroupVersionResource
		srvRes   []*metav1.APIResourceList
	}{
		{
			name:     "rc-resolves-to-replicationcontrollers",
			arg:      "rc",
			expected: schema.GroupVersionResource{Resource: "replicationcontrollers"},
			srvRes:   []*metav1.APIResourceList{},
		},
		{
			name:     "storageclasses-no-replacement",
			arg:      "storageclasses",
			expected: schema.GroupVersionResource{Resource: "storageclasses"},
			srvRes:   []*metav1.APIResourceList{},
		},
		{
			name:     "hpa-priority",
			arg:      "hpa",
			expected: schema.GroupVersionResource{Resource: "superhorizontalpodautoscalers"},
			srvRes: []*metav1.APIResourceList{
				{
					GroupVersion: "autoscaling/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "superhorizontalpodautoscalers",
							ShortNames: []string{"hpa"},
						},
					},
				},
				{
					GroupVersion: "autoscaling/v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "horizontalpodautoscalers",
							ShortNames: []string{"hpa"},
						},
					},
				},
			},
		},
	}

	ds := &fakeDiscoveryClient{}
	mapper, err := NewShortcutExpander(testapi.Default.RESTMapper(), ds)
	if err != nil {
		t.Fatalf("Unable to create shortcut expander, err %s", err.Error())
	}

	for _, test := range tests {
		ds.serverResourcesHandler = func() ([]*metav1.APIResourceList, error) {
			return test.srvRes, nil
		}
		actual := mapper.expandResourceShortcut(schema.GroupVersionResource{Resource: test.arg})
		if actual != test.expected {
			t.Errorf("%s: unexpected argument: expected %s, got %s", test.name, test.expected, actual)
		}
	}
}

func TestKindFor(t *testing.T) {
	tests := []struct {
		in       schema.GroupVersionResource
		expected schema.GroupVersionKind
		srvRes   []*metav1.APIResourceList
	}{
		{
			in:       schema.GroupVersionResource{Group: "storage.k8s.io", Version: "", Resource: "sc"},
			expected: schema.GroupVersionKind{Group: "storage.k8s.io", Version: "v1", Kind: "StorageClass"},
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
			in:       schema.GroupVersionResource{Group: "", Version: "", Resource: "sc"},
			expected: schema.GroupVersionKind{Group: "storage.k8s.io", Version: "v1", Kind: "StorageClass"},
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
	}

	ds := &fakeDiscoveryClient{}
	mapper, err := NewShortcutExpander(testapi.Default.RESTMapper(), ds)
	if err != nil {
		t.Fatalf("Unable to create shortcut expander, err %s", err.Error())
	}

	for i, test := range tests {
		ds.serverResourcesHandler = func() ([]*metav1.APIResourceList, error) {
			return test.srvRes, nil
		}
		ret, err := mapper.KindFor(test.in)
		if err != nil {
			t.Errorf("%d: unexpected error returned %s", i, err.Error())
		}
		if ret != test.expected {
			t.Errorf("%d: unexpected data returned %#v, expected %#v", i, ret, test.expected)
		}
	}
}
