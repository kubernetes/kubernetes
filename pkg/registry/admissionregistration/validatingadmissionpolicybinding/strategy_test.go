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

package validatingadmissionpolicybinding

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

func TestPolicyBindingStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Error("PolicyBinding strategy must be cluster scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("PolicyBinding should not allow create on update")
	}

	configuration := validPolicyBinding()
	Strategy.PrepareForCreate(ctx, configuration)
	errs := Strategy.Validate(ctx, configuration)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	invalidConfiguration := &admissionregistration.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{Name: ""},
	}
	Strategy.PrepareForUpdate(ctx, invalidConfiguration, configuration)
	errs = Strategy.ValidateUpdate(ctx, invalidConfiguration, configuration)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}
func validPolicyBinding() *admissionregistration.ValidatingAdmissionPolicyBinding {
	return &admissionregistration.ValidatingAdmissionPolicyBinding{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Spec: admissionregistration.ValidatingAdmissionPolicyBindingSpec{
			PolicyName: "replicalimit-policy.example.com",
			ParamRef: &admissionregistration.ParamRef{
				Name: "replica-limit-test.example.com",
			},
		},
	}
}
