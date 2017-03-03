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
	"reflect"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/kubernetes/pkg/api/testapi"
)

func TestReplaceAliases(t *testing.T) {
	tests := []struct {
		name     string
		arg      string
		expected string
		srvRes   []*metav1.APIResourceList
	}{
		{
			name:     "no-replacement",
			arg:      "service",
			expected: "service",
			srvRes:   []*metav1.APIResourceList{},
		},
		{
			name:     "all-replacement",
			arg:      "all",
			expected: "pods,replicationcontrollers,services,statefulsets,horizontalpodautoscalers,jobs,deployments,replicasets",
			srvRes:   []*metav1.APIResourceList{},
		},
		{
			name:     "alias-in-comma-separated-arg",
			arg:      "all,secrets",
			expected: "pods,replicationcontrollers,services,statefulsets,horizontalpodautoscalers,jobs,deployments,replicasets,secrets",
			srvRes:   []*metav1.APIResourceList{},
		},
		{
			name:     "rc-resolves-to-replicationcontrollers",
			arg:      "rc",
			expected: "replicationcontrollers",
			srvRes:   []*metav1.APIResourceList{},
		},
		{
			name:     "storageclasses-no-replacement",
			arg:      "storageclasses",
			expected: "storageclasses",
			srvRes:   []*metav1.APIResourceList{},
		},
		{
			name:     "hpa-priority",
			arg:      "hpa",
			expected: "superhorizontalpodautoscalers",
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
		{
			name:     "all: uses server user resources if none of the wanted resources is available on server",
			arg:      "all",
			expected: "knowntype",
			srvRes: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "knowntype",
							Namespaced: true,
						},
					},
				},
			},
		},
		{
			name:     "all: server user resources are namespaced",
			arg:      "all",
			expected: "knowntype",
			srvRes: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{
							Name:       "knowntype",
							Namespaced: true,
						},
					},
				},
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{
							Name: "unknown",
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
		resources := []string{}
		ds.serverResourcesHandler = func() ([]*metav1.APIResourceList, error) {
			return test.srvRes, nil
		}
		for _, arg := range strings.Split(test.arg, ",") {
			curr, _ := mapper.AliasesForResource(arg)
			resources = append(resources, curr...)
		}
		if strings.Join(resources, ",") != test.expected {
			t.Errorf("%s: unexpected argument: expected %s, got %s", test.name, test.expected, resources)
		}
	}
}

func TestFilterKnownUserResources(t *testing.T) {
	wantedUserResources := []schema.GroupResource{
		{Group: "", Resource: "pods"},
		{Group: "", Resource: "unknown"},
	}

	availableUserResources := []schema.GroupResource{
		{Group: "", Resource: "pods"},
	}

	serverUserResources := []*metav1.APIResourceList{
		{
			GroupVersion: "v1",
			APIResources: []metav1.APIResource{
				{
					Name:       "pods",
					Namespaced: true,
				},
			},
		},
	}

	ds := &fakeDiscoveryClient{}
	ds.serverResourcesHandler = func() ([]*metav1.APIResourceList, error) {
		return serverUserResources, nil
	}
	mapper, err := NewShortcutExpander(testapi.Default.RESTMapper(), ds)
	if err != nil {
		t.Fatalf("Unable to create shortcut expander, err %s", err.Error())
	}
	filtered := mapper.filterKnownUserResources(wantedUserResources)
	if !reflect.DeepEqual(availableUserResources, filtered) {
		t.Errorf("expected %s, got %s", availableUserResources, filtered)
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
			expected: schema.GroupVersionKind{Group: "storage.k8s.io", Version: "v1beta1", Kind: "StorageClass"},
			srvRes: []*metav1.APIResourceList{
				{
					GroupVersion: "storage.k8s.io/v1beta1",
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
			expected: schema.GroupVersionKind{Group: "storage.k8s.io", Version: "v1beta1", Kind: "StorageClass"},
			srvRes: []*metav1.APIResourceList{
				{
					GroupVersion: "storage.k8s.io/v1beta1",
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
