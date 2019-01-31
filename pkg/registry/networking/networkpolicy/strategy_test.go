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

package networkpolicy

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/networking"
	"testing"
)

const (
	namespace = "test-namespace"
)

func TestNetworkPolicyStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("NetworkPolicy must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("NetworkPolicy should not allow create on update")
	}

	if !Strategy.AllowUnconditionalUpdate() {
		t.Errorf("Networking should allow unconditional update")
	}

	oldNetworkPolicy := &networking.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "valid-policy",
			Namespace: namespace,
		},
		Spec: networking.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{},
			Ingress: []networking.NetworkPolicyIngressRule{
				{
					From:  []networking.NetworkPolicyPeer{},
					Ports: []networking.NetworkPolicyPort{},
				},
			},
		},
	}

	Strategy.PrepareForCreate(ctx, oldNetworkPolicy)

	errs := Strategy.Validate(ctx, oldNetworkPolicy)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	newNetworkPolicy := &networking.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "valid-policy-2",
			Namespace:       namespace,
			ResourceVersion: "4",
		},
		Spec: networking.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{},
			Egress: []networking.NetworkPolicyEgressRule{
				{
					Ports: []networking.NetworkPolicyPort{},
					To:    []networking.NetworkPolicyPeer{},
				},
			},
		},
	}

	errs = Strategy.Validate(ctx, newNetworkPolicy)
	if len(errs) != 0 {
		t.Errorf("unexpected error validating %v", errs)
	}

	Strategy.PrepareForUpdate(ctx, newNetworkPolicy, oldNetworkPolicy)

	errs = Strategy.ValidateUpdate(ctx, newNetworkPolicy, oldNetworkPolicy)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}
