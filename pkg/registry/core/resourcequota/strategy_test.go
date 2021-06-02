/*
Copyright 2014 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/features"
)

func TestResourceQuotaStrategy(t *testing.T) {
	if !Strategy.NamespaceScoped() {
		t.Errorf("ResourceQuota should be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("ResourceQuota should not allow create on update")
	}
	resourceQuota := &api.ResourceQuota{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Status: api.ResourceQuotaStatus{
			Used: api.ResourceList{
				api.ResourceCPU:                    resource.MustParse("1"),
				api.ResourceMemory:                 resource.MustParse("1Gi"),
				api.ResourcePods:                   resource.MustParse("1"),
				api.ResourceServices:               resource.MustParse("1"),
				api.ResourceReplicationControllers: resource.MustParse("1"),
				api.ResourceQuotas:                 resource.MustParse("1"),
			},
			Hard: api.ResourceList{
				api.ResourceCPU:                    resource.MustParse("100"),
				api.ResourceMemory:                 resource.MustParse("4Gi"),
				api.ResourcePods:                   resource.MustParse("10"),
				api.ResourceServices:               resource.MustParse("10"),
				api.ResourceReplicationControllers: resource.MustParse("10"),
				api.ResourceQuotas:                 resource.MustParse("1"),
			},
		},
	}
	Strategy.PrepareForCreate(genericapirequest.NewContext(), resourceQuota)
	if resourceQuota.Status.Used != nil {
		t.Errorf("ResourceQuota does not allow setting status on create")
	}
}

func TestGetValidationOptionsFromResourceQuota(t *testing.T) {
	crossNamespaceAffinity := api.ResourceQuota{
		Spec: api.ResourceQuotaSpec{
			Scopes: []api.ResourceQuotaScope{api.ResourceQuotaScopeCrossNamespacePodAffinity},
		},
	}

	for name, tc := range map[string]struct {
		old                             *api.ResourceQuota
		namespaceSelectorFeatureEnabled bool
		wantOpts                        validation.ResourceQuotaValidationOptions
	}{
		"create-feature-enabled": {
			namespaceSelectorFeatureEnabled: true,
			wantOpts: validation.ResourceQuotaValidationOptions{
				AllowPodAffinityNamespaceSelector: true,
			},
		},
		"create-feature-disabled": {
			namespaceSelectorFeatureEnabled: false,
			wantOpts: validation.ResourceQuotaValidationOptions{
				AllowPodAffinityNamespaceSelector: false,
			},
		},
		"update-old-doesn't-include-scope-feature-enabled": {
			old:                             &api.ResourceQuota{},
			namespaceSelectorFeatureEnabled: true,
			wantOpts: validation.ResourceQuotaValidationOptions{
				AllowPodAffinityNamespaceSelector: true,
			},
		},
		"update-old-doesn't-include-scope-feature-disabled": {
			old:                             &api.ResourceQuota{},
			namespaceSelectorFeatureEnabled: false,
			wantOpts: validation.ResourceQuotaValidationOptions{
				AllowPodAffinityNamespaceSelector: false,
			},
		},
		"update-old-includes-scope-feature-disabled": {
			old:                             &crossNamespaceAffinity,
			namespaceSelectorFeatureEnabled: false,
			wantOpts: validation.ResourceQuotaValidationOptions{
				AllowPodAffinityNamespaceSelector: true,
			},
		},
		"update-old-includes-scope-feature-enabled": {
			old:                             &crossNamespaceAffinity,
			namespaceSelectorFeatureEnabled: true,
			wantOpts: validation.ResourceQuotaValidationOptions{
				AllowPodAffinityNamespaceSelector: true,
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodAffinityNamespaceSelector, tc.namespaceSelectorFeatureEnabled)()
			gotOpts := getValidationOptionsFromResourceQuota(nil, tc.old)
			if diff := cmp.Diff(tc.wantOpts, gotOpts); diff != "" {
				t.Errorf("unexpected opts (-want, +got):\n%s", diff)
			}
		})
	}
}
