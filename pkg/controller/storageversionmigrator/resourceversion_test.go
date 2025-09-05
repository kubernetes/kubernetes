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

package storageversionmigrator

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	fakediscovery "k8s.io/client-go/discovery/fake"
	kubetesting "k8s.io/client-go/testing"
)

func TestIsResourceMigratable(t *testing.T) {
	tcs := []struct {
		name      string
		resources []*metav1.APIResourceList
		resource  schema.GroupVersionResource
		want      bool
		wantErr   string
	}{
		{
			name: "migratable resource",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
						{Name: "events", Namespaced: true, Kind: "Event", Verbs: []string{"get", "list", "watch", "create", "delete"}},
						{Name: "configmaps", Namespaced: true, Kind: "Event", Verbs: []string{"get", "watch", "create", "delete", "update", "patch", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
			want:     true,
		},
		{
			name: "non-updatable resource",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
						{Name: "events", Namespaced: true, Kind: "Event", Verbs: []string{"get", "list", "watch", "create", "delete"}},
						{Name: "configmaps", Namespaced: true, Kind: "Event", Verbs: []string{"get", "watch", "create", "delete", "update", "patch", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "events"},
			want:     false,
		},
		{
			name: "non-patchable resource",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
						{Name: "events", Namespaced: true, Kind: "Event", Verbs: []string{"get", "list", "watch", "create", "delete"}},
						{Name: "configmaps", Namespaced: true, Kind: "Configmap", Verbs: []string{"get", "watch", "create", "delete", "update", "patch", "delete"}},
						{Name: "secrets", Namespaced: true, Kind: "Secret", Verbs: []string{"get", "watch", "create", "delete", "update", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "secrets"},
			want:     false,
		},
		{
			name: "non-listable resource",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
						{Name: "events", Namespaced: true, Kind: "Event", Verbs: []string{"get", "list", "watch", "create", "delete"}},
						{Name: "configmaps", Namespaced: true, Kind: "Event", Verbs: []string{"get", "watch", "create", "delete", "update", "patch", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"},
			want:     false,
		},
		{
			name: "unknown resource",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "pods", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
						{Name: "events", Namespaced: true, Kind: "Event", Verbs: []string{"get", "list", "watch", "create", "delete"}},
						{Name: "configmaps", Namespaced: true, Kind: "Configmap", Verbs: []string{"get", "watch", "create", "delete", "update", "patch", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "", Version: "v1", Resource: "foo"},
			wantErr:  "resource \"/v1, Resource=foo\" not found in discovery",
		},
		{
			name: "multiple versions",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "foo", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
					},
				},
				{
					GroupVersion: "v1alpha1",
					APIResources: []metav1.APIResource{
						{Name: "foo", Namespaced: true, Kind: "Pod", Verbs: []string{"get", "watch", "create", "update", "patch", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "", Version: "v1alpha", Resource: "foo"},
			want:     false,
		},
		{
			name: "multiple versions and groups",
			resources: []*metav1.APIResourceList{
				{
					GroupVersion: "v1",
					APIResources: []metav1.APIResource{
						{Name: "foo", Namespaced: true, Kind: "Foo", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
					},
				},
				{
					GroupVersion: "v1alpha1",
					APIResources: []metav1.APIResource{
						{Name: "foo", Namespaced: true, Kind: "Foo", Verbs: []string{"get", "list", "watch", "create", "update", "patch", "delete"}},
					},
				},
				{
					GroupVersion: "bar/v1alpha1",
					APIResources: []metav1.APIResource{
						{Name: "foo", Namespaced: true, Kind: "Foo", Group: "bar", Verbs: []string{"get", "watch", "create", "update", "patch", "delete"}},
					},
				},
			},
			resource: schema.GroupVersionResource{Group: "bar", Version: "v1alpha1", Resource: "foo"},
			want:     false,
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {}))
			defer server.Close()
			discoveryClient := fakediscovery.FakeDiscovery{Fake: &kubetesting.Fake{}}
			discoveryClient.Resources = tc.resources
			rvController := ResourceVersionController{
				discoveryClient: &discoveryClient,
			}

			isMigratable, err := rvController.isResourceMigratable(tc.resource)
			if err != nil {
				if !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("Unexpected error: %v, want: %v", err, tc.wantErr)
				}
				return
			}
			if isMigratable != tc.want {
				t.Errorf("Expected %v, got %v", tc.want, isMigratable)
			}
		})
	}

}
