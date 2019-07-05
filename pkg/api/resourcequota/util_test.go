/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"testing"

	"k8s.io/apimachinery/pkg/util/diff"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
)

func TestDropDisabledFields(t *testing.T) {
	rqWithScopeSelector := func() *api.ResourceQuota {
		return &api.ResourceQuota{Spec: api.ResourceQuotaSpec{Scopes: []api.ResourceQuotaScope{"scope-1"}, ScopeSelector: &api.ScopeSelector{
			MatchExpressions: []api.ScopedResourceSelectorRequirement{
				{
					ScopeName: api.ResourceQuotaScopePriorityClass,
					Operator:  api.ScopeSelectorOpIn,
					Values:    []string{"scope-1"},
				},
			},
		}}}
	}
	rqWithoutScopeSelector := func() *api.ResourceQuota {
		return &api.ResourceQuota{Spec: api.ResourceQuotaSpec{Scopes: []api.ResourceQuotaScope{"scope-1"}, ScopeSelector: nil}}
	}

	rqInfo := []struct {
		description      string
		hasScopeSelector bool
		resourceQuota    func() *api.ResourceQuota
	}{
		{
			description:      "ResourceQuota without Scopes Selector",
			hasScopeSelector: false,
			resourceQuota:    rqWithoutScopeSelector,
		},
		{
			description:      "ResourceQuota with Scope Selector",
			hasScopeSelector: true,
			resourceQuota:    rqWithScopeSelector,
		},
		{
			description:      "is nil",
			hasScopeSelector: false,
			resourceQuota:    func() *api.ResourceQuota { return nil },
		},
	}

	for _, enabled := range []bool{true, false} {
		for _, oldRQInfo := range rqInfo {
			for _, newRQInfo := range rqInfo {
				oldRQHasSelector, oldrq := oldRQInfo.hasScopeSelector, oldRQInfo.resourceQuota()
				newRQHasSelector, newrq := newRQInfo.hasScopeSelector, newRQInfo.resourceQuota()
				if newrq == nil {
					continue
				}

				t.Run(fmt.Sprintf("feature enabled=%v, old ResourceQuota %v, new ResourceQuota %v", enabled, oldRQInfo.description, newRQInfo.description), func(t *testing.T) {
					defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.ResourceQuotaScopeSelectors, enabled)()

					var oldRQSpec *api.ResourceQuotaSpec
					if oldrq != nil {
						oldRQSpec = &oldrq.Spec
					}
					DropDisabledFields(&newrq.Spec, oldRQSpec)

					// old ResourceQuota should never be changed
					if !reflect.DeepEqual(oldrq, oldRQInfo.resourceQuota()) {
						t.Errorf("old ResourceQuota changed: %v", diff.ObjectReflectDiff(oldrq, oldRQInfo.resourceQuota()))
					}

					switch {
					case enabled || oldRQHasSelector:
						// new ResourceQuota should not be changed if the feature is enabled, or if the old ResourceQuota had ScopeSelector
						if !reflect.DeepEqual(newrq, newRQInfo.resourceQuota()) {
							t.Errorf("new ResourceQuota changed: %v", diff.ObjectReflectDiff(newrq, newRQInfo.resourceQuota()))
						}
					case newRQHasSelector:
						// new ResourceQuota should be changed
						if reflect.DeepEqual(newrq, newRQInfo.resourceQuota()) {
							t.Errorf("new ResourceQuota was not changed")
						}
						// new ResourceQuota should not have ScopeSelector
						if !reflect.DeepEqual(newrq, rqWithoutScopeSelector()) {
							t.Errorf("new ResourceQuota had ScopeSelector: %v", diff.ObjectReflectDiff(newrq, rqWithoutScopeSelector()))
						}
					default:
						// new ResourceQuota should not need to be changed
						if !reflect.DeepEqual(newrq, newRQInfo.resourceQuota()) {
							t.Errorf("new ResourceQuota changed: %v", diff.ObjectReflectDiff(newrq, newRQInfo.resourceQuota()))
						}
					}
				})
			}
		}
	}
}
