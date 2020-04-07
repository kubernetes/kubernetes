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

	v1 "k8s.io/api/admissionregistration/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/generic"
	"k8s.io/kubernetes/pkg/apis/admissionregistration"
	apis "k8s.io/kubernetes/pkg/apis/admissionregistration"
	apisv1 "k8s.io/kubernetes/pkg/apis/admissionregistration/v1"
	"k8s.io/kubernetes/pkg/apis/admissionregistration/validation"
)

type webhookDefaultor struct {
}

func NewWebhookDefaultor() generic.WebhookDefaultor {
	return &webhookDefaultor{}
}

func (w *webhookDefaultor) SetDefaultForValidatingWebhookConfiguration(v *v1.ValidatingWebhookConfiguration) {
		apisv1.SetObjectDefaults_ValidatingWebhookConfiguration(v)
}

func (w *webhookDefaultor) SetDefaultForMutatingWebhookConfiguration(m *v1.MutatingWebhookConfiguration) {
	apisv1.SetObjectDefaults_MutatingWebhookConfiguration(m)
}

type webhookValidator struct {
}

func NewWebhookValidator() generic.WebhookValidator {
	return &webhookValidator{}
}

func (w *webhookValidator) ValidateValidatingWebhookConfiguration(v *v1.ValidatingWebhookConfiguration) error {
	gv := schema.GroupVersion{Group: admissionregistration.GroupName, Version: "v1"}
	v1Webhook := &apis.ValidatingWebhookConfiguration{}
	err := apisv1.Convert_v1_ValidatingWebhookConfiguration_To_admissionregistration_ValidatingWebhookConfiguration(v, v1Webhook, nil)
	if err != nil {
		return fmt.Errorf("conversion failed for webhook: %s", v.Name)
	}
	return validation.ValidateValidatingWebhookConfiguration(v1Webhook, gv).ToAggregate()
}

func (w *webhookValidator) ValidateMutatingWebhookConfiguration(m *v1.MutatingWebhookConfiguration) error {
	gv := schema.GroupVersion{Group: admissionregistration.GroupName, Version: "v1"}
	v1Webhook := &apis.MutatingWebhookConfiguration{}
	err := apisv1.Convert_v1_MutatingWebhookConfiguration_To_admissionregistration_MutatingWebhookConfiguration(m, v1Webhook, nil)
	if err != nil {
		return fmt.Errorf("conversion failed for webhook: %s", m.Name)
	}
	return validation.ValidateMutatingWebhookConfiguration(v1Webhook, gv).ToAggregate()
}
