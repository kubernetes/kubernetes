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

package dynamic

import (
	"testing"

	metav1 "k8s.io/client-go/pkg/apis/meta/v1"
	"k8s.io/client-go/pkg/runtime/schema"
)

func TestDiscoveryRESTMapper(t *testing.T) {
	resources := []*metav1.APIResourceList{
		{
			GroupVersion: "test/beta1",
			APIResources: []metav1.APIResource{
				{
					Name:       "test_kinds",
					Namespaced: true,
					Kind:       "test_kind",
				},
			},
		},
	}

	gvk := schema.GroupVersionKind{
		Group:   "test",
		Version: "beta1",
		Kind:    "test_kind",
	}

	mapper, err := NewDiscoveryRESTMapper(resources, VersionInterfaces)
	if err != nil {
		t.Fatalf("unexpected error creating mapper: %s", err)
	}

	for _, res := range []schema.GroupVersionResource{
		{
			Group:    "test",
			Version:  "beta1",
			Resource: "test_kinds",
		},
		{
			Version:  "beta1",
			Resource: "test_kinds",
		},
		{
			Group:    "test",
			Resource: "test_kinds",
		},
		{
			Resource: "test_kinds",
		},
	} {
		got, err := mapper.KindFor(res)
		if err != nil {
			t.Errorf("KindFor(%#v) unexpected error: %s", res, err)
			continue
		}

		if got != gvk {
			t.Errorf("KindFor(%#v) = %#v; want %#v", res, got, gvk)
		}
	}
}
