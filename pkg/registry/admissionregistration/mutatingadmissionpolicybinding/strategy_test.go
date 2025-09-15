/*
Copyright 2022 The Kubernetes Authors.

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

package mutatingadmissionpolicybinding

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/registry/admissionregistration/resolver"

	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

func TestPolicyBindingStrategy(t *testing.T) {
	strategy := NewStrategy(nil, nil, replicaLimitsResolver)
	ctx := genericapirequest.NewDefaultContext()
	if strategy.NamespaceScoped() {
		t.Error("PolicyBinding strategy must be cluster scoped")
	}
	if strategy.AllowCreateOnUpdate() {
		t.Errorf("PolicyBinding should not allow create on update")
	}

	for _, configuration := range validPolicyBindings() {
		strategy.PrepareForCreate(ctx, configuration)
		errs := strategy.Validate(ctx, configuration)
		if len(errs) != 0 {
			t.Errorf("Unexpected error validating %v", errs)
		}
		invalidConfiguration := &admissionregistration.MutatingAdmissionPolicyBinding{
			ObjectMeta: metav1.ObjectMeta{Name: ""},
		}
		strategy.PrepareForUpdate(ctx, invalidConfiguration, configuration)
		errs = strategy.ValidateUpdate(ctx, invalidConfiguration, configuration)
		if len(errs) == 0 {
			t.Errorf("Expected a validation error")
		}
	}
}

var replicaLimitsResolver resolver.ResourceResolverFunc = func(gvk schema.GroupVersionKind) (schema.GroupVersionResource, error) {
	return schema.GroupVersionResource{
		Group:    "rules.example.com",
		Version:  "v1",
		Resource: "replicalimits",
	}, nil
}

func validPolicyBindings() []*admissionregistration.MutatingAdmissionPolicyBinding {
	denyAction := admissionregistration.DenyAction
	return []*admissionregistration.MutatingAdmissionPolicyBinding{
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "replicalimit-policy.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "replica-limit-test.example.com",
					ParameterNotFoundAction: &denyAction,
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo-clusterwide",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "replicalimit-policy.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Name:                    "replica-limit-test.example.com",
					Namespace:               "default",
					ParameterNotFoundAction: &denyAction,
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo-selector",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "replicalimit-policy.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"label": "value",
						},
					},
					ParameterNotFoundAction: &denyAction,
				},
			},
		},
		{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo-selector-clusterwide",
			},
			Spec: admissionregistration.MutatingAdmissionPolicyBindingSpec{
				PolicyName: "replicalimit-policy.example.com",
				ParamRef: &admissionregistration.ParamRef{
					Namespace: "mynamespace",
					Selector: &metav1.LabelSelector{
						MatchLabels: map[string]string{
							"label": "value",
						},
					},
					ParameterNotFoundAction: &denyAction,
				},
			},
		},
	}
}
