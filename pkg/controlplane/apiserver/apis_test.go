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

	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapiserver "k8s.io/apiserver/pkg/server"
	"k8s.io/kubernetes/pkg/api/legacyscheme"

	_ "k8s.io/kubernetes/pkg/apis/certificates/install"
	_ "k8s.io/kubernetes/pkg/apis/core/install"
	_ "k8s.io/kubernetes/pkg/apis/scheduling/install"
)

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
