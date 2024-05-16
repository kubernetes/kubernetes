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

package validatingadmissionpolicy

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

func TestValidatingAdmissionPolicyStrategy(t *testing.T) {
	strategy := NewStrategy(nil, nil)
	ctx := genericapirequest.NewDefaultContext()
	if strategy.NamespaceScoped() {
		t.Error("ValidatingAdmissionPolicy strategy must be cluster scoped")
	}
	if strategy.AllowCreateOnUpdate() {
		t.Errorf("ValidatingAdmissionPolicy should not allow create on update")
	}

	configuration := validValidatingAdmissionPolicy()
	strategy.PrepareForCreate(ctx, configuration)
	errs := strategy.Validate(ctx, configuration)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	invalidConfiguration := &admissionregistration.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{Name: ""},
	}
	strategy.PrepareForUpdate(ctx, invalidConfiguration, configuration)
	errs = strategy.ValidateUpdate(ctx, invalidConfiguration, configuration)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}
func validValidatingAdmissionPolicy() *admissionregistration.ValidatingAdmissionPolicy {
	ignore := admissionregistration.Ignore
	return &admissionregistration.ValidatingAdmissionPolicy{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: admissionregistration.ValidatingAdmissionPolicySpec{
			ParamKind: &admissionregistration.ParamKind{
				Kind:       "ReplicaLimit",
				APIVersion: "rules.example.com/v1",
			},
			Validations: []admissionregistration.Validation{
				{
					Expression: "object.spec.replicas <= params.maxReplicas",
				},
			},
			MatchConstraints: &admissionregistration.MatchResources{
				MatchPolicy: func() *admissionregistration.MatchPolicyType {
					r := admissionregistration.MatchPolicyType("Exact")
					return &r
				}(),
				ObjectSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				NamespaceSelector: &metav1.LabelSelector{
					MatchLabels: map[string]string{"a": "b"},
				},
				ResourceRules: []admissionregistration.NamedRuleWithOperations{
					{
						RuleWithOperations: admissionregistration.RuleWithOperations{
							Operations: []admissionregistration.OperationType{"CREATE"},
							Rule: admissionregistration.Rule{
								APIGroups:   []string{"a"},
								APIVersions: []string{"a"},
								Resources:   []string{"a"},
							},
						},
					},
				},
			},
			FailurePolicy: &ignore,
		},
	}
}
