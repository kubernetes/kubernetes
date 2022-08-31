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

package endpoints

import (
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestIsVowel(t *testing.T) {
	tests := []struct {
		name string
		arg  rune
		want bool
	}{
		{
			name: "yes",
			arg:  'E',
			want: true,
		},
		{
			name: "no",
			arg:  'n',
			want: false,
		},
	}
	for _, tt := range tests {
		if got := isVowel(tt.arg); got != tt.want {
			t.Errorf("%q. IsVowel() = %v, want %v", tt.name, got, tt.want)
		}
	}
}

func TestGetArticleForNoun(t *testing.T) {
	tests := []struct {
		noun    string
		padding string
		want    string
	}{
		{
			noun:    "Frog",
			padding: " ",
			want:    " a ",
		},
		{
			noun:    "frogs",
			padding: " ",
			want:    " ",
		},
		{
			noun:    "apple",
			padding: "",
			want:    "an",
		},
		{
			noun:    "Apples",
			padding: " ",
			want:    " ",
		},
		{
			noun:    "Ingress",
			padding: " ",
			want:    " an ",
		},
		{
			noun:    "Class",
			padding: " ",
			want:    " a ",
		},
		{
			noun:    "S",
			padding: " ",
			want:    " a ",
		},
		{
			noun:    "O",
			padding: " ",
			want:    " an ",
		},
	}
	for _, tt := range tests {
		if got := GetArticleForNoun(tt.noun, tt.padding); got != tt.want {
			t.Errorf("%q. GetArticleForNoun() = %v, want %v", tt.noun, got, tt.want)
		}
	}
}

func TestConvertResourceInfoToDiscovery(t *testing.T) {
	tests := []struct {
		name                     string
		resources                []*apiResourceInfo
		wantAPIResource          []metav1.APIResource
		wantAPIResourceDiscovery []metav1.APIResourceDiscovery
	}{
		{
			name: "Basic Test",
			resources: []*apiResourceInfo{
				{
					APIResource: metav1.APIResource{
						Name:       "pods",
						Namespaced: true,
						Kind:       "Pod",
						ShortNames: []string{"po"},
						Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					},
				},
			},
			wantAPIResourceDiscovery: []metav1.APIResourceDiscovery{
				{
					Resource: "pods",
					Scope:    metav1.ScopeNamespace,
					ReturnType: metav1.APIDiscoveryKind{
						Kind: "Pod",
					},
					ShortNames: []string{"po"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
		},
		{
			name: "Basic Group Version Test",
			resources: []*apiResourceInfo{
				{
					APIResource: metav1.APIResource{
						Name:       "cronjobs",
						Namespaced: true,
						Group: "batch",
						Version: "v1",
						Kind:       "CronJob",
						ShortNames: []string{"cj"},
						Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					},
				},
			},
			wantAPIResourceDiscovery: []metav1.APIResourceDiscovery{
				{
					Resource: "cronjobs",
					Scope:    metav1.ScopeNamespace,
					ReturnType: metav1.APIDiscoveryKind{
						Group:   "batch",
						Version: "v1",
						Kind:    "CronJob",
					},
					ShortNames: []string{"cj"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
		},
		{
			name: "Test with subresource",
			resources: []*apiResourceInfo{
				{
					APIResource: metav1.APIResource{
						Name:       "cronjobs",
						Namespaced: true,
						Kind:       "CronJob",
						Group:      "batch",
						Version:    "v1",
						ShortNames: []string{"cj"},
						Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					},
				},
				{
					Subresource: "status",
					APIResource: metav1.APIResource{
						Name:       "cronjobs",
						Namespaced: true,
						Kind:       "CronJob",
						Group:      "batch",
						Version:    "v1",
						ShortNames: []string{"cj"},
						Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					},
				},
			},
			wantAPIResourceDiscovery: []metav1.APIResourceDiscovery{
				{
					Resource: "cronjobs",
					Scope:    metav1.ScopeNamespace,
					ReturnType: metav1.APIDiscoveryKind{
						Group:   "batch",
						Version: "v1",
						Kind:    "CronJob",
					},
					ShortNames: []string{"cj"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					Subresources: []metav1.APISubresourceDiscovery{{
						Subresource: "status",
						ReturnType: &metav1.APIDiscoveryKind{
							Group:   "batch",
							Version: "v1",
							Kind:    "CronJob",
						},
						Verbs: []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					}},
				},
			},
		},
		{
			name: "Cluster Scope Test",
			resources: []*apiResourceInfo{
				{
					APIResource: metav1.APIResource{
						Name:       "nodes",
						Namespaced: false,
						Kind:       "Node",
						ShortNames: []string{"no"},
						Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					},
				},
			},
			wantAPIResourceDiscovery: []metav1.APIResourceDiscovery{
				{
					Resource: "nodes",
					Scope:    metav1.ScopeCluster,
					ReturnType: metav1.APIDiscoveryKind{
						Kind: "Node",
					},
					ShortNames: []string{"no"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
		},

	}

	for _, tt := range tests {
		_, discoveryAPIResources := ConvertResourceInfoToDiscovery(tt.resources)
		if !reflect.DeepEqual(discoveryAPIResources, tt.wantAPIResourceDiscovery) {
			t.Errorf("Error: Test %s, Expected %v, got %v\n", tt.name, tt.wantAPIResourceDiscovery, discoveryAPIResources)
		}
	}
}
