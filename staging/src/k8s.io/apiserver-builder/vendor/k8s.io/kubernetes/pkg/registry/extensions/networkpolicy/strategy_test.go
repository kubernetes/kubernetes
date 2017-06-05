/*
Copyright 2016 The Kubernetes Authors.

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
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/extensions"
)

func TestNetworkPolicyStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("NetworkPolicy must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("NetworkPolicy should not allow create on update")
	}

	validMatchLabels := map[string]string{"a": "b"}
	np := &extensions.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "abc", Namespace: metav1.NamespaceDefault},
		Spec: extensions.NetworkPolicySpec{
			PodSelector: metav1.LabelSelector{MatchLabels: validMatchLabels},
			Ingress:     []extensions.NetworkPolicyIngressRule{},
		},
	}

	Strategy.PrepareForCreate(ctx, np)
	errs := Strategy.Validate(ctx, np)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}

	invalidNp := &extensions.NetworkPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: "bar", ResourceVersion: "4"},
	}
	Strategy.PrepareForUpdate(ctx, invalidNp, np)
	errs = Strategy.ValidateUpdate(ctx, invalidNp, np)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	if invalidNp.ResourceVersion != "4" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}
