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

package webhook

import (
	"k8s.io/api/admissionregistration/v1beta1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// WebhookAccessor provides a common interface to both mutating and validating webhook types.
type WebhookAccessor interface {
	// GetUID gets a string that uniquely identifies the webhook.
	GetUID() string

	// GetConfigurationName gets the name of the webhook configuration that owns this webhook.
	GetConfigurationName() string

	// GetName gets the webhook Name field. Note that the name is scoped to the webhook
	// configuration and does not provide a globally unique identity, if a unique identity is
	// needed, use GetUID.
	GetName() string
	// GetClientConfig gets the webhook ClientConfig field.
	GetClientConfig() v1beta1.WebhookClientConfig
	// GetRules gets the webhook Rules field.
	GetRules() []v1beta1.RuleWithOperations
	// GetFailurePolicy gets the webhook FailurePolicy field.
	GetFailurePolicy() *v1beta1.FailurePolicyType
	// GetMatchPolicy gets the webhook MatchPolicy field.
	GetMatchPolicy() *v1beta1.MatchPolicyType
	// GetNamespaceSelector gets the webhook NamespaceSelector field.
	GetNamespaceSelector() *metav1.LabelSelector
	// GetObjectSelector gets the webhook ObjectSelector field.
	GetObjectSelector() *metav1.LabelSelector
	// GetSideEffects gets the webhook SideEffects field.
	GetSideEffects() *v1beta1.SideEffectClass
	// GetTimeoutSeconds gets the webhook TimeoutSeconds field.
	GetTimeoutSeconds() *int32
	// GetAdmissionReviewVersions gets the webhook AdmissionReviewVersions field.
	GetAdmissionReviewVersions() []string

	// GetMutatingWebhook if the accessor contains a MutatingWebhook, returns it and true, else returns false.
	GetMutatingWebhook() (*v1beta1.MutatingWebhook, bool)
	// GetValidatingWebhook if the accessor contains a ValidatingWebhook, returns it and true, else returns false.
	GetValidatingWebhook() (*v1beta1.ValidatingWebhook, bool)
}

// NewMutatingWebhookAccessor creates an accessor for a MutatingWebhook.
func NewMutatingWebhookAccessor(uid, configurationName string, h *v1beta1.MutatingWebhook) WebhookAccessor {
	return mutatingWebhookAccessor{uid: uid, configurationName: configurationName, MutatingWebhook: h}
}

type mutatingWebhookAccessor struct {
	*v1beta1.MutatingWebhook
	uid               string
	configurationName string
}

func (m mutatingWebhookAccessor) GetUID() string {
	return m.uid
}

func (m mutatingWebhookAccessor) GetConfigurationName() string {
	return m.configurationName
}

func (m mutatingWebhookAccessor) GetName() string {
	return m.Name
}

func (m mutatingWebhookAccessor) GetClientConfig() v1beta1.WebhookClientConfig {
	return m.ClientConfig
}

func (m mutatingWebhookAccessor) GetRules() []v1beta1.RuleWithOperations {
	return m.Rules
}

func (m mutatingWebhookAccessor) GetFailurePolicy() *v1beta1.FailurePolicyType {
	return m.FailurePolicy
}

func (m mutatingWebhookAccessor) GetMatchPolicy() *v1beta1.MatchPolicyType {
	return m.MatchPolicy
}

func (m mutatingWebhookAccessor) GetNamespaceSelector() *metav1.LabelSelector {
	return m.NamespaceSelector
}

func (m mutatingWebhookAccessor) GetObjectSelector() *metav1.LabelSelector {
	return m.ObjectSelector
}

func (m mutatingWebhookAccessor) GetSideEffects() *v1beta1.SideEffectClass {
	return m.SideEffects
}

func (m mutatingWebhookAccessor) GetTimeoutSeconds() *int32 {
	return m.TimeoutSeconds
}

func (m mutatingWebhookAccessor) GetAdmissionReviewVersions() []string {
	return m.AdmissionReviewVersions
}

func (m mutatingWebhookAccessor) GetMutatingWebhook() (*v1beta1.MutatingWebhook, bool) {
	return m.MutatingWebhook, true
}

func (m mutatingWebhookAccessor) GetValidatingWebhook() (*v1beta1.ValidatingWebhook, bool) {
	return nil, false
}

// NewValidatingWebhookAccessor creates an accessor for a ValidatingWebhook.
func NewValidatingWebhookAccessor(uid, configurationName string, h *v1beta1.ValidatingWebhook) WebhookAccessor {
	return validatingWebhookAccessor{uid: uid, configurationName: configurationName, ValidatingWebhook: h}
}

type validatingWebhookAccessor struct {
	*v1beta1.ValidatingWebhook
	uid               string
	configurationName string
}

func (v validatingWebhookAccessor) GetUID() string {
	return v.uid
}

func (v validatingWebhookAccessor) GetConfigurationName() string {
	return v.configurationName
}

func (v validatingWebhookAccessor) GetName() string {
	return v.Name
}

func (v validatingWebhookAccessor) GetClientConfig() v1beta1.WebhookClientConfig {
	return v.ClientConfig
}

func (v validatingWebhookAccessor) GetRules() []v1beta1.RuleWithOperations {
	return v.Rules
}

func (v validatingWebhookAccessor) GetFailurePolicy() *v1beta1.FailurePolicyType {
	return v.FailurePolicy
}

func (v validatingWebhookAccessor) GetMatchPolicy() *v1beta1.MatchPolicyType {
	return v.MatchPolicy
}

func (v validatingWebhookAccessor) GetNamespaceSelector() *metav1.LabelSelector {
	return v.NamespaceSelector
}

func (v validatingWebhookAccessor) GetObjectSelector() *metav1.LabelSelector {
	return v.ObjectSelector
}

func (v validatingWebhookAccessor) GetSideEffects() *v1beta1.SideEffectClass {
	return v.SideEffects
}

func (v validatingWebhookAccessor) GetTimeoutSeconds() *int32 {
	return v.TimeoutSeconds
}

func (v validatingWebhookAccessor) GetAdmissionReviewVersions() []string {
	return v.AdmissionReviewVersions
}

func (v validatingWebhookAccessor) GetMutatingWebhook() (*v1beta1.MutatingWebhook, bool) {
	return nil, false
}

func (v validatingWebhookAccessor) GetValidatingWebhook() (*v1beta1.ValidatingWebhook, bool) {
	return v.ValidatingWebhook, true
}
