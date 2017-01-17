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

package ingress

import (
	"testing"

	genericapirequest "k8s.io/apiserver/pkg/request"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/util/intstr"
)

func newIngress() extensions.Ingress {
	defaultBackend := extensions.IngressBackend{
		ServiceName: "default-backend",
		ServicePort: intstr.FromInt(80),
	}
	return extensions.Ingress{
		ObjectMeta: api.ObjectMeta{
			Name:      "foo",
			Namespace: api.NamespaceDefault,
		},
		Spec: extensions.IngressSpec{
			Backend: &extensions.IngressBackend{
				ServiceName: "default-backend",
				ServicePort: intstr.FromInt(80),
			},
			Rules: []extensions.IngressRule{
				{
					Host: "foo.bar.com",
					IngressRuleValue: extensions.IngressRuleValue{
						HTTP: &extensions.HTTPIngressRuleValue{
							Paths: []extensions.HTTPIngressPath{
								{
									Path:    "/foo",
									Backend: defaultBackend,
								},
							},
						},
					},
				},
			},
		},
		Status: extensions.IngressStatus{
			LoadBalancer: api.LoadBalancerStatus{
				Ingress: []api.LoadBalancerIngress{
					{IP: "127.0.0.1"},
				},
			},
		},
	}
}

func TestIngressStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !Strategy.NamespaceScoped() {
		t.Errorf("Ingress must be namespace scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("Ingress should not allow create on update")
	}

	ingress := newIngress()
	Strategy.PrepareForCreate(ctx, &ingress)
	if len(ingress.Status.LoadBalancer.Ingress) != 0 {
		t.Error("Ingress should not allow setting status on create")
	}
	errs := Strategy.Validate(ctx, &ingress)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	invalidIngress := newIngress()
	invalidIngress.ResourceVersion = "4"
	invalidIngress.Spec = extensions.IngressSpec{}
	Strategy.PrepareForUpdate(ctx, &invalidIngress, &ingress)
	errs = Strategy.ValidateUpdate(ctx, &invalidIngress, &ingress)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
	if invalidIngress.ResourceVersion != "4" {
		t.Errorf("Incoming resource version on update should not be mutated")
	}
}

func TestIngressStatusStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if !StatusStrategy.NamespaceScoped() {
		t.Errorf("Ingress must be namespace scoped")
	}
	if StatusStrategy.AllowCreateOnUpdate() {
		t.Errorf("Ingress should not allow create on update")
	}
	oldIngress := newIngress()
	newIngress := newIngress()
	oldIngress.ResourceVersion = "4"
	newIngress.ResourceVersion = "4"
	newIngress.Spec.Backend.ServiceName = "ignore"
	newIngress.Status = extensions.IngressStatus{
		LoadBalancer: api.LoadBalancerStatus{
			Ingress: []api.LoadBalancerIngress{
				{IP: "127.0.0.2"},
			},
		},
	}
	StatusStrategy.PrepareForUpdate(ctx, &newIngress, &oldIngress)
	if newIngress.Status.LoadBalancer.Ingress[0].IP != "127.0.0.2" {
		t.Errorf("Ingress status updates should allow change of status fields")
	}
	if newIngress.Spec.Backend.ServiceName != "default-backend" {
		t.Errorf("PrepareForUpdate should have preserved old spec")
	}
	errs := StatusStrategy.ValidateUpdate(ctx, &newIngress, &oldIngress)
	if len(errs) != 0 {
		t.Errorf("Unexpected error %v", errs)
	}
}

func TestSelectableFieldLabelConversions(t *testing.T) {
	apitesting.TestSelectableFieldLabelConversionsOfKind(t,
		testapi.Extensions.GroupVersion().String(),
		"Ingress",
		IngressToSelectableFields(&extensions.Ingress{}),
		nil,
	)
}
