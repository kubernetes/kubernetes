/*
Copyright 2021 The Kubernetes Authors.

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

package validatingwebhookconfiguration

import (
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
)

func TestValidatingWebhookConfigurationStrategy(t *testing.T) {
	ctx := genericapirequest.NewDefaultContext()
	if Strategy.NamespaceScoped() {
		t.Error("ValidatingWebhookConfiguration strategy must be cluster scoped")
	}
	if Strategy.AllowCreateOnUpdate() {
		t.Errorf("ValidatingWebhookConfiguration should not allow create on update")
	}

	configuration := validValidatingWebhookConfiguration()
	Strategy.PrepareForCreate(ctx, configuration)
	errs := Strategy.Validate(ctx, configuration)
	if len(errs) != 0 {
		t.Errorf("Unexpected error validating %v", errs)
	}
	invalidConfiguration := &admissionregistration.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{Name: ""},
	}
	Strategy.PrepareForUpdate(ctx, invalidConfiguration, configuration)
	errs = Strategy.ValidateUpdate(ctx, invalidConfiguration, configuration)
	if len(errs) == 0 {
		t.Errorf("Expected a validation error")
	}
}
func validValidatingWebhookConfiguration() *admissionregistration.ValidatingWebhookConfiguration {
	ignore := admissionregistration.Ignore
	exact := admissionregistration.Exact
	thirty := int32(30)
	none := admissionregistration.SideEffectClassNone
	servicePath := "/"
	return &admissionregistration.ValidatingWebhookConfiguration{
		ObjectMeta: metav1.ObjectMeta{
			Name: "foo",
		},
		Webhooks: []admissionregistration.ValidatingWebhook{{
			Name: "foo.example.io",
			ClientConfig: admissionregistration.WebhookClientConfig{
				Service: &admissionregistration.ServiceReference{
					Name:      "foo",
					Namespace: "bar",
					Path:      &servicePath,
					Port:      443,
				},
			},
			FailurePolicy:           &ignore,
			MatchPolicy:             &exact,
			TimeoutSeconds:          &thirty,
			NamespaceSelector:       &metav1.LabelSelector{},
			ObjectSelector:          &metav1.LabelSelector{},
			SideEffects:             &none,
			AdmissionReviewVersions: []string{"v1beta1"},
		}},
	}
}
