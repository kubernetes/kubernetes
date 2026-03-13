/*
Copyright 2020 The Kubernetes Authors.

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

package ingressclass

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/networking"
)

func TestIngressClassStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Errorf("IngressClass must not be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("IngressClass should not allow create on update")
	}

	ingressClass := networking.IngressClass{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "foo",
			ResourceVersion: "1",
		},
		Spec: networking.IngressClassSpec{
			Controller: "example.com/controller",
		},
	}

	Strategy.PrepareForCreate(ctx, &ingressClass)
	if ingressClass.Generation != 1 {
		t.Error("IngressClass generation should be 1")
	}
	errs := Strategy.Validate(ctx, &ingressClass)
	if len(errs) != 0 {
		t.Errorf("Unexpected error from validation for IngressClass: %v", errs)
	}

	newIngressClass := ingressClass.DeepCopy()
	Strategy.PrepareForUpdate(ctx, newIngressClass, &ingressClass)
	errs = Strategy.ValidateUpdate(ctx, newIngressClass, &ingressClass)
	if len(errs) != 0 {
		t.Errorf("Unexpected error from update validation for IngressClass: %v", errs)
	}

	ingressClass.Name = "invalid/name"

	errs = Strategy.Validate(ctx, &ingressClass)
	if len(errs) == 0 {
		t.Errorf("Expected error from validation for IngressClass, got none")
	}
	errs = Strategy.ValidateUpdate(ctx, &ingressClass, &ingressClass)
	if len(errs) == 0 {
		t.Errorf("Expected error from update validation for IngressClass, got none")
	}
}
