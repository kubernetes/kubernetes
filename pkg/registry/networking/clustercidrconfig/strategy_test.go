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

package clustercidrconfig

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/networking"
)

func newClusterCIDRConfig() networking.ClusterCIDRConfig {
	return networking.ClusterCIDRConfig{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: networking.ClusterCIDRConfigSpec{
			IPv4: &networking.CIDRConfig{
				CIDR:            "10.1.0.0/16",
				PerNodeMaskSize: int32(24),
			},
			IPv6: &networking.CIDRConfig{
				CIDR:            "fd00:1:1::/64",
				PerNodeMaskSize: int32(120),
			},
			NodeSelector: &api.NodeSelector{
				NodeSelectorTerms: []api.NodeSelectorTerm{
					{
						MatchExpressions: []api.NodeSelectorRequirement{
							{
								Key:      "foo",
								Operator: api.NodeSelectorOpIn,
								Values:   []string{"bar"},
							},
						},
					},
				},
			},
		},
	}
}

func TestClusterCIDRConfigStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	apiRequest := genericapirequest.RequestInfo{APIGroup: "networking.k8s.io",
		APIVersion: "v1alpha1",
		Resource:   "clustercidrconfigs",
	}
	ctx = genericapirequest.WithRequestInfo(ctx, &apiRequest)
	if Strategy.NamespaceScoped() {
		t.Errorf("ClusterCIDRConfigs must be cluster scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("ClusterCIDRConfigs should not allow create on update")
	}

	ccc := newClusterCIDRConfig()
	Strategy.PrepareForCreate(ctx, &ccc)

	errs := Strategy.Validate(ctx, &ccc)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	invalidCCC := newClusterCIDRConfig()
	invalidCCC.ResourceVersion = "4"
	invalidCCC.Spec = networking.ClusterCIDRConfigSpec{}
	Strategy.PrepareForUpdate(ctx, &invalidCCC, &ccc)
	errs = Strategy.ValidateUpdate(ctx, &invalidCCC, &ccc)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	if invalidCCC.ResourceVersion != "4" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}
