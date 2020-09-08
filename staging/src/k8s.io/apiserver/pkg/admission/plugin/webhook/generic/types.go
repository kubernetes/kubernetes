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

package generic

import (
	v1 "k8s.io/api/admissionregistration/v1"
)

// WebhookValidator contains methods to validate ValidatingWebhookConfiguration
// and MutatingWebhookConfiguration
type WebhookValidator interface {
	ValidateValidatingWebhookConfiguration(*v1.ValidatingWebhookConfiguration) error
	ValidateMutatingWebhookConfiguration(*v1.MutatingWebhookConfiguration) error
}

// WebhookDefaulter contains methods to set defaults on ValidatingWebhookConfiguration
// and MutatingWebhookConfiguration.
type WebhookDefaulter interface {
	SetDefaultsForValidatingWebhookConfiguration(*v1.ValidatingWebhookConfiguration)
	SetDefaultsForMutatingWebhookConfiguration(*v1.MutatingWebhookConfiguration)
}

// ManifestWebhookWrapper contains the method to wrap a source with a manifest
// based webhook source. It also contains an initialization method.
type ManifestWebhookWrapper interface {
	Initialize(webhookType WebhookType) error
	WrapHookSource(s Source) Source
}

// WebhookType depicts the type of webhook being referred to
type WebhookType string

var (
	// MutatingWebhook represent the WebhookType for a mutating webhook.
	MutatingWebhook WebhookType = "Mutating"
	// ValidatingWebhook represent the WebhookType for a validating webhook.
	ValidatingWebhook WebhookType = "Validating"
	knownWebhookTypes             = []WebhookType{MutatingWebhook, ValidatingWebhook}
)

func knownWebhookType(webhookType WebhookType) bool {
	for _, t := range knownWebhookTypes {
		if t == webhookType {
			return true
		}
	}
	return false
}
