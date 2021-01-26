/*
Copyright 2021 The Kubernetes Authors.

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

package resourcequota

import (
	"testing"

	"github.com/google/go-cmp/cmp"

	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropDisabledFields(t *testing.T) {
	testCases := []struct {
		name        string
		enabled     bool
		oldResSpec  *api.ResourceQuotaSpec
		newResSpec  *api.ResourceQuotaSpec
		wantResSpec *api.ResourceQuotaSpec
	}{
		{
			name:        "nil scopes",
			newResSpec:  &api.ResourceQuotaSpec{},
			wantResSpec: &api.ResourceQuotaSpec{},
		},
		{
			name: "empty scopes",
			newResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{},
				},
			},
			wantResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{},
				},
			},
		},
		{
			name: "cross-namespace scope removed when old spec is nil",
			newResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopeCrossNamespacePodAffinity,
					api.ResourceQuotaScopePriorityClass,
				},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{
						{
							ScopeName: api.ResourceQuotaScopeCrossNamespacePodAffinity,
						},
						{
							ScopeName: api.ResourceQuotaScopePriorityClass,
						},
					},
				},
			},
			wantResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopePriorityClass,
				},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{
						{
							ScopeName: api.ResourceQuotaScopePriorityClass,
						},
					},
				},
			},
		},
		{
			name: "cross-namespace scope removed when old spec doesn't include the scope",
			newResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopeCrossNamespacePodAffinity,
					api.ResourceQuotaScopePriorityClass,
				},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{
						{
							ScopeName: api.ResourceQuotaScopeCrossNamespacePodAffinity,
						},
						{
							ScopeName: api.ResourceQuotaScopePriorityClass,
						},
					},
				},
			},
			oldResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopePriorityClass,
				},
				ScopeSelector: &api.ScopeSelector{},
			},
			wantResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopePriorityClass,
				},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{
						{
							ScopeName: api.ResourceQuotaScopePriorityClass,
						},
					},
				},
			},
		},
		{
			name: "cross-namespace scope not removed when old spec includes the scope",
			newResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopeCrossNamespacePodAffinity,
					api.ResourceQuotaScopePriorityClass,
				},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{
						{
							ScopeName: api.ResourceQuotaScopeCrossNamespacePodAffinity,
						},
						{
							ScopeName: api.ResourceQuotaScopePriorityClass,
						},
					},
				},
			},
			oldResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopeCrossNamespacePodAffinity,
				},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{
						{
							ScopeName: api.ResourceQuotaScopeCrossNamespacePodAffinity,
						},
					},
				},
			},
			wantResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopeCrossNamespacePodAffinity,
					api.ResourceQuotaScopePriorityClass,
				},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{
						{
							ScopeName: api.ResourceQuotaScopeCrossNamespacePodAffinity,
						},
						{
							ScopeName: api.ResourceQuotaScopePriorityClass,
						},
					},
				},
			},
		},
		{
			name:    "cross-namespace scope not when old spec is nil and feature is enabled",
			enabled: true,
			newResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopeCrossNamespacePodAffinity,
					api.ResourceQuotaScopePriorityClass,
				},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{
						{
							ScopeName: api.ResourceQuotaScopeCrossNamespacePodAffinity,
						},
						{
							ScopeName: api.ResourceQuotaScopePriorityClass,
						},
					},
				},
			},
			wantResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopeCrossNamespacePodAffinity,
					api.ResourceQuotaScopePriorityClass,
				},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{
						{
							ScopeName: api.ResourceQuotaScopeCrossNamespacePodAffinity,
						},
						{
							ScopeName: api.ResourceQuotaScopePriorityClass,
						},
					},
				},
			},
		},
		{
			name:    "cross-namespace scope not removed when old spec doesn't include the scope and feature is enabled",
			enabled: true,
			newResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopeCrossNamespacePodAffinity,
					api.ResourceQuotaScopePriorityClass,
				},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{
						{
							ScopeName: api.ResourceQuotaScopeCrossNamespacePodAffinity,
						},
						{
							ScopeName: api.ResourceQuotaScopePriorityClass,
						},
					},
				},
			},
			oldResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopePriorityClass,
				},
				ScopeSelector: &api.ScopeSelector{},
			},
			wantResSpec: &api.ResourceQuotaSpec{
				Scopes: []api.ResourceQuotaScope{
					api.ResourceQuotaScopeCrossNamespacePodAffinity,
					api.ResourceQuotaScopePriorityClass,
				},
				ScopeSelector: &api.ScopeSelector{
					MatchExpressions: []api.ScopedResourceSelectorRequirement{
						{
							ScopeName: api.ResourceQuotaScopeCrossNamespacePodAffinity,
						},
						{
							ScopeName: api.ResourceQuotaScopePriorityClass,
						},
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodAffinityNamespaceSelector, tc.enabled)()
			DropDisabledFields(tc.newResSpec, tc.oldResSpec)
			if diff := cmp.Diff(tc.wantResSpec, tc.newResSpec); diff != "" {
				t.Errorf("%v: unexpected diff (-want, +got):\n%s", tc.name, diff)
			}
		})
	}
}
