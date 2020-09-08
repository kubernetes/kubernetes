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

package admission

import (
	"fmt"

	admissionregistrationv1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
	apis "k8s.io/kubernetes/pkg/apis/admissionregistration"
	apisv1 "k8s.io/kubernetes/pkg/apis/admissionregistration/v1"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/validation"
)

type webhookDefaulter struct {
}

// NewWebhookDefaulter returns a webhook defaulter that sets defaults for validating and
// mutating webhook configurations
func NewWebhookDefaulter() generic.WebhookDefaulter {
	return &webhookDefaulter{}
}

func (w *webhookDefaulter) SetDefaultsForValidatingWebhookConfiguration(v *admissionregistrationv1.ValidatingWebhookConfiguration) {
	apisv1.SetObjectDefaults_ValidatingWebhookConfiguration(v)
}

func (w *webhookDefaulter) SetDefaultsForMutatingWebhookConfiguration(m *admissionregistrationv1.MutatingWebhookConfiguration) {
	apisv1.SetObjectDefaults_MutatingWebhookConfiguration(m)
}

type webhookValidator struct {
}

// NewWebhookValidator returns a webhook validator that validates validating and
// mutating webhook configurations
func NewWebhookValidator() generic.WebhookValidator {
	return &webhookValidator{}
}

func (w *webhookValidator) ValidateValidatingWebhookConfiguration(v *admissionregistrationv1.ValidatingWebhookConfiguration) error {
	v1Webhook := &apis.ValidatingWebhookConfiguration{}
	err := apisv1.Convert_v1_ValidatingWebhookConfiguration_To_admissionregistration_ValidatingWebhookConfiguration(v, v1Webhook, nil)
	if err != nil {
		return fmt.Errorf("conversion failed for webhook: %s", v.Name)
	}
	return validation.ValidateValidatingWebhookConfiguration(v1Webhook, admissionregistrationv1.SchemeGroupVersion).ToAggregate()
}

func (w *webhookValidator) ValidateMutatingWebhookConfiguration(m *admissionregistrationv1.MutatingWebhookConfiguration) error {
	v1Webhook := &apis.MutatingWebhookConfiguration{}
	err := apisv1.Convert_v1_MutatingWebhookConfiguration_To_admissionregistration_MutatingWebhookConfiguration(m, v1Webhook, nil)
	if err != nil {
		return fmt.Errorf("conversion failed for webhook: %s", m.Name)
	}
	return validation.ValidateMutatingWebhookConfiguration(v1Webhook, admissionregistrationv1.SchemeGroupVersion).ToAggregate()
}
