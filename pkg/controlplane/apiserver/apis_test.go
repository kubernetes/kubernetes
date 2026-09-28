/*
Copyright The Kubernetes Authors.

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

package apiserver

import (
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/registry/rest"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/kubernetes/pkg/api/legacyscheme"

	_ "k8s.io/kubernetes/pkg/apis/certificates/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

type noStorage struct{}

func (noStorage) New() runtime.Object { return nil }
func (noStorage) Destroy()            {}

func TestGroupResourcesIn(t *testing.T) {
	storage := func(resources ...string) map[string]rest.Storage {
		ret := map[string]rest.Storage{}
		for _, resource := range resources {
			ret[resource] = noStorage{}
		}
		return ret
	}

	tests := []struct {
		name       string
		group      string
		storageMap map[string]map[string]rest.Storage
		want       sets.Set[schema.GroupResource]
	}{
		{
			name:       "empty storage map",
			group:      "one",
			storageMap: map[string]map[string]rest.Storage{},
			want:       sets.New[schema.GroupResource](),
		},
		{
			name:       "resources of one version",
			group:      "one",
			storageMap: map[string]map[string]rest.Storage{"v1": storage("first", "second")},
			want: sets.New(
				schema.GroupResource{Group: "one", Resource: "first"},
				schema.GroupResource{Group: "one", Resource: "second"},
			),
		},
		{
			name:  "resource served by several versions counts once",
			group: "one",
			storageMap: map[string]map[string]rest.Storage{
				"v1":      storage("first"),
				"v1beta1": storage("first", "second"),
			},
			want: sets.New(
				schema.GroupResource{Group: "one", Resource: "first"},
				schema.GroupResource{Group: "one", Resource: "second"},
			),
		},
		{
			name:       "subresources count towards their parent",
			group:      "one",
			storageMap: map[string]map[string]rest.Storage{"v1": storage("first", "first/status", "first/scale")},
			want:       sets.New(schema.GroupResource{Group: "one", Resource: "first"}),
		},
		{
			name:       "subresource without its parent still names the parent",
			group:      "one",
			storageMap: map[string]map[string]rest.Storage{"v1": storage("first/status")},
			want:       sets.New(schema.GroupResource{Group: "one", Resource: "first"}),
		},
		{
			name:       "legacy group",
			group:      "",
			storageMap: map[string]map[string]rest.Storage{"v1": storage("pods", "pods/status")},
			want:       sets.New(schema.GroupResource{Resource: "pods"}),
		},
		{
			name:       "version with no resources contributes nothing",
			group:      "one",
			storageMap: map[string]map[string]rest.Storage{"v1": storage(), "v1beta1": storage("first")},
			want:       sets.New(schema.GroupResource{Group: "one", Resource: "first"}),
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if got := groupResourcesIn(tc.group, tc.storageMap); !got.Equal(tc.want) {
				t.Errorf("groupResourcesIn(%q) = %v, want %v", tc.group, got.UnsortedList(), tc.want.UnsortedList())
			}
		})
	}
}

func TestRegisteredResourcesFor(t *testing.T) {
	tests := []struct {
		name      string
		group     string
		want      []schema.GroupVersionResource
		miss      []schema.GroupVersionResource
		wantEmpty bool
	}{
		{
			name:  "gated resource in every version that carries it",
			group: "certificates.k8s.io",
			want: []schema.GroupVersionResource{
				{Group: "certificates.k8s.io", Version: "v1", Resource: "clustertrustbundles"},
				{Group: "certificates.k8s.io", Version: "v1beta1", Resource: "clustertrustbundles"},
				{Group: "certificates.k8s.io", Version: "v1alpha1", Resource: "clustertrustbundles"},
				{Group: "certificates.k8s.io", Version: "v1", Resource: "podcertificaterequests"},
				{Group: "certificates.k8s.io", Version: "v1", Resource: "certificatesigningrequests"},
			},
			miss: []schema.GroupVersionResource{
				// only kinds are registered; subresources and kind names are not resources
				{Group: "certificates.k8s.io", Version: "v1", Resource: "clustertrustbundles/status"},
				{Group: "certificates.k8s.io", Version: "v1", Resource: "ClusterTrustBundle"},
				// podcertificaterequests was never served in v1alpha1
				{Group: "certificates.k8s.io", Version: "v1alpha1", Resource: "podcertificaterequests"},
			},
		},
		{
			name:  "resources served only by pre-release versions",
			group: "scheduling.k8s.io",
			want: []schema.GroupVersionResource{
				{Group: "scheduling.k8s.io", Version: "v1beta1", Resource: "workloads"},
				{Group: "scheduling.k8s.io", Version: "v1beta1", Resource: "podgroups"},
				{Group: "scheduling.k8s.io", Version: "v1alpha3", Resource: "compositepodgroups"},
				{Group: "scheduling.k8s.io", Version: "v1", Resource: "priorityclasses"},
			},
			miss: []schema.GroupVersionResource{
				{Group: "scheduling.k8s.io", Version: "v1", Resource: "workloads"},
				{Group: "scheduling.k8s.io", Version: "v1", Resource: "compositepodgroups"},
			},
		},
		{
			name:  "legacy group",
			group: "",
			want: []schema.GroupVersionResource{
				{Version: "v1", Resource: "pods"},
				{Version: "v1", Resource: "configmaps"},
			},
			miss: []schema.GroupVersionResource{
				{Version: "v1", Resource: "deployments"},
			},
		},
		{
			name:      "unregistered group has no resources",
			group:     "nonexistent.k8s.io",
			wantEmpty: true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			apiGroupInfo := genericapiserver.NewDefaultAPIGroupInfo(tc.group, legacyscheme.Scheme, legacyscheme.ParameterCodec, legacyscheme.Codecs)
			got := registeredResourcesFor(&apiGroupInfo)

			if tc.wantEmpty && got.Len() != 0 {
				t.Errorf("registeredResourcesFor(%q) = %v, want none", tc.group, got.UnsortedList())
			}
			for _, gvr := range tc.want {
				if !got.Has(gvr) {
					t.Errorf("registeredResourcesFor(%q) is missing %s", tc.group, gvr)
				}
			}
			for _, gvr := range tc.miss {
				if got.Has(gvr) {
					t.Errorf("registeredResourcesFor(%q) unexpectedly contains %s", tc.group, gvr)
				}
			}
		})
	}
}
