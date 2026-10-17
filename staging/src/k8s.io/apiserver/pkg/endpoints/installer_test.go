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
	"net/http"
	"testing"

	restful "github.com/emicklei/go-restful/v3"
	"github.com/stretchr/testify/require"
	apidiscoveryv2 "k8s.io/api/apidiscovery/v2"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

func TestAddObjectParamsCaching(t *testing.T) {
	ws1 := new(restful.WebService)
	route1 := ws1.GET("/items").To(func(*restful.Request, *restful.Response) {})
	require.NoError(t, AddObjectParams(ws1, route1, &metav1.ListOptions{}, "watch", "allowWatchBookmarks"))
	ws1.Route(route1)

	ws2 := new(restful.WebService)
	route2 := ws2.GET("/items").To(func(*restful.Request, *restful.Response) {})
	require.NoError(t, AddObjectParams(ws2, route2, &metav1.ListOptions{}))
	ws2.Route(route2)

	ws3 := new(restful.WebService)
	route3 := ws3.GET("/items").To(func(*restful.Request, *restful.Response) {})
	require.NoError(t, AddObjectParams(ws3, route3, &metav1.ListOptions{}))
	ws3.Route(route3)

	params1 := ws1.Routes()[0].ParameterDocs
	params2 := ws2.Routes()[0].ParameterDocs
	params3 := ws3.Routes()[0].ParameterDocs

	require.NotEmpty(t, params2)
	require.Len(t, params1, len(params2)-2)
	require.Len(t, params3, len(params2))

	for _, p := range params1 {
		require.NotEqual(t, "watch", p.Data().Name)
		require.NotEqual(t, "allowWatchBookmarks", p.Data().Name)
	}

	hasWatch := false
	for i := range params2 {
		require.Same(t, params2[i], params3[i], "expected cached *restful.Parameter pointer reuse for %s", params2[i].Data().Name)
		if params2[i].Data().Name == "watch" {
			hasWatch = true
		}
	}
	require.True(t, hasWatch, "expected unexcluded ListOptions route to include watch parameter")
}

func TestPrettyParameterUnchanged(t *testing.T) {
	before := prettyParameter.Data()

	ws := new(restful.WebService)
	route := ws.GET("/items").
		To(func(req *restful.Request, resp *restful.Response) {
			resp.WriteHeader(http.StatusOK)
		}).
		Param(prettyParameter)
	ws.Route(route)

	require.Same(t, prettyParameter, ws.Routes()[0].ParameterDocs[0])
	require.Equal(t, before, prettyParameter.Data())
}

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

func TestConvertAPIResourceToDiscovery(t *testing.T) {
	tests := []struct {
		name                     string
		resources                []metav1.APIResource
		wantAPIResourceDiscovery []apidiscoveryv2.APIResourceDiscovery
		wantErr                  bool
	}{
		{
			name: "Basic Test",
			resources: []metav1.APIResource{
				{

					Name:       "pods",
					Namespaced: true,
					Kind:       "Pod",
					ShortNames: []string{"po"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
			wantAPIResourceDiscovery: []apidiscoveryv2.APIResourceDiscovery{
				{
					Resource: "pods",
					Scope:    apidiscoveryv2.ScopeNamespace,
					ResponseKind: &metav1.GroupVersionKind{
						Kind: "Pod",
					},
					ShortNames: []string{"po"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
		},
		{
			name: "Basic Group Version Test",
			resources: []metav1.APIResource{
				{
					Name:       "cronjobs",
					Namespaced: true,
					Group:      "batch",
					Version:    "v1",
					Kind:       "CronJob",
					ShortNames: []string{"cj"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
			wantAPIResourceDiscovery: []apidiscoveryv2.APIResourceDiscovery{
				{
					Resource: "cronjobs",
					Scope:    apidiscoveryv2.ScopeNamespace,
					ResponseKind: &metav1.GroupVersionKind{
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
			resources: []metav1.APIResource{
				{
					Name:       "cronjobs",
					Namespaced: true,
					Kind:       "CronJob",
					Group:      "batch",
					Version:    "v1",
					ShortNames: []string{"cj"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
				{
					Name:       "cronjobs/status",
					Namespaced: true,
					Kind:       "CronJob",
					Group:      "batch",
					Version:    "v1",
					ShortNames: []string{"cj"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
			wantAPIResourceDiscovery: []apidiscoveryv2.APIResourceDiscovery{
				{
					Resource: "cronjobs",
					Scope:    apidiscoveryv2.ScopeNamespace,
					ResponseKind: &metav1.GroupVersionKind{
						Group:   "batch",
						Version: "v1",
						Kind:    "CronJob",
					},
					ShortNames: []string{"cj"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					Subresources: []apidiscoveryv2.APISubresourceDiscovery{{
						Subresource: "status",
						ResponseKind: &metav1.GroupVersionKind{
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
			name: "Test multiple resources and subresources",
			resources: []metav1.APIResource{
				{
					Name:       "cronjobs",
					Namespaced: true,
					Kind:       "CronJob",
					Group:      "batch",
					Version:    "v1",
					ShortNames: []string{"cj"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
				{
					Name:       "cronjobs/status",
					Namespaced: true,
					Kind:       "CronJob",
					Group:      "batch",
					Version:    "v1",
					ShortNames: []string{"cj"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
				{
					Name:       "deployments",
					Namespaced: true,
					Kind:       "Deployment",
					Group:      "apps",
					Version:    "v1",
					ShortNames: []string{"deploy"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
				{
					Name:       "deployments/status",
					Namespaced: true,
					Kind:       "Deployment",
					Group:      "apps",
					Version:    "v1",
					ShortNames: []string{"deploy"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
			wantAPIResourceDiscovery: []apidiscoveryv2.APIResourceDiscovery{
				{
					Resource: "cronjobs",
					Scope:    apidiscoveryv2.ScopeNamespace,
					ResponseKind: &metav1.GroupVersionKind{
						Group:   "batch",
						Version: "v1",
						Kind:    "CronJob",
					},
					ShortNames: []string{"cj"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					Subresources: []apidiscoveryv2.APISubresourceDiscovery{{
						Subresource: "status",
						ResponseKind: &metav1.GroupVersionKind{
							Group:   "batch",
							Version: "v1",
							Kind:    "CronJob",
						},
						Verbs: []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					}},
				}, {
					Resource: "deployments",
					Scope:    apidiscoveryv2.ScopeNamespace,
					ResponseKind: &metav1.GroupVersionKind{
						Group:   "apps",
						Version: "v1",
						Kind:    "Deployment",
					},
					ShortNames: []string{"deploy"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					Subresources: []apidiscoveryv2.APISubresourceDiscovery{{
						Subresource: "status",
						ResponseKind: &metav1.GroupVersionKind{
							Group:   "apps",
							Version: "v1",
							Kind:    "Deployment",
						},
						Verbs: []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					}},
				},
			},
		}, {
			name: "Test with subresource with no parent",
			resources: []metav1.APIResource{
				{
					Name:       "cronjobs/status",
					Namespaced: true,
					Kind:       "CronJob",
					Group:      "batch",
					Version:    "v1",
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
			wantAPIResourceDiscovery: []apidiscoveryv2.APIResourceDiscovery{
				{
					Resource: "cronjobs",
					Scope:    apidiscoveryv2.ScopeNamespace,
					// populated to avoid nil panics
					ResponseKind: &metav1.GroupVersionKind{},
					Subresources: []apidiscoveryv2.APISubresourceDiscovery{{
						Subresource: "status",
						ResponseKind: &metav1.GroupVersionKind{
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
			name: "Test with subresource with missing kind",
			resources: []metav1.APIResource{
				{
					Name:       "cronjobs/status",
					Namespaced: true,
					Group:      "batch",
					Version:    "v1",
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
			wantAPIResourceDiscovery: []apidiscoveryv2.APIResourceDiscovery{
				{
					Resource: "cronjobs",
					Scope:    apidiscoveryv2.ScopeNamespace,
					// populated to avoid nil panics
					ResponseKind: &metav1.GroupVersionKind{},
					Subresources: []apidiscoveryv2.APISubresourceDiscovery{{
						Subresource: "status",
						// populated to avoid nil panics
						ResponseKind: &metav1.GroupVersionKind{},
						Verbs:        []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
					}},
				},
			},
		},
		{
			name: "Test with mismatch parent and subresource scope",
			resources: []metav1.APIResource{
				{
					Name:       "cronjobs",
					Namespaced: true,
					Kind:       "CronJob",
					Group:      "batch",
					Version:    "v1",
					ShortNames: []string{"cj"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
				{
					Name:       "cronjobs/status",
					Namespaced: false,
					Kind:       "CronJob",
					Group:      "batch",
					Version:    "v1",
					ShortNames: []string{"cj"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
			wantAPIResourceDiscovery: []apidiscoveryv2.APIResourceDiscovery{},
			wantErr:                  true,
		},
		{
			name: "Cluster Scope Test",
			resources: []metav1.APIResource{
				{
					Name:       "nodes",
					Namespaced: false,
					Kind:       "Node",
					ShortNames: []string{"no"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
			wantAPIResourceDiscovery: []apidiscoveryv2.APIResourceDiscovery{
				{
					Resource: "nodes",
					Scope:    apidiscoveryv2.ScopeCluster,
					ResponseKind: &metav1.GroupVersionKind{
						Kind: "Node",
					},
					ShortNames: []string{"no"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
		},
		{
			name: "Namespace Scope Test",
			resources: []metav1.APIResource{
				{
					Name:       "nodes",
					Namespaced: true,
					Kind:       "Node",
					ShortNames: []string{"no"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
			wantAPIResourceDiscovery: []apidiscoveryv2.APIResourceDiscovery{
				{
					Resource: "nodes",
					Scope:    apidiscoveryv2.ScopeNamespace,
					ResponseKind: &metav1.GroupVersionKind{
						Kind: "Node",
					},
					ShortNames: []string{"no"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
		},
		{
			name: "Singular Resource Name",
			resources: []metav1.APIResource{
				{
					Name:         "nodes",
					SingularName: "node",
					Kind:         "Node",
					ShortNames:   []string{"no"},
					Verbs:        []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
			wantAPIResourceDiscovery: []apidiscoveryv2.APIResourceDiscovery{
				{
					Resource:         "nodes",
					SingularResource: "node",
					Scope:            apidiscoveryv2.ScopeCluster,
					ResponseKind: &metav1.GroupVersionKind{
						Kind: "Node",
					},
					ShortNames: []string{"no"},
					Verbs:      []string{"create", "delete", "deletecollection", "get", "list", "patch", "update", "watch"},
				},
			},
		},
	}

	for _, tt := range tests {
		discoveryAPIResources, err := ConvertGroupVersionIntoToDiscovery(tt.resources)
		if err != nil {
			if tt.wantErr == false {
				t.Error(err)
			}
		} else {
			require.Equal(t, tt.wantAPIResourceDiscovery, discoveryAPIResources)
		}
	}
}
