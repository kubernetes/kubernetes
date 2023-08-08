/*
Copyright 2023 The Kubernetes Authors.

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

package admissionregistration

import (
	genericfeatures "k8s.io/apiserver/pkg/features"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
)

// DropDisabledMutatingWebhookConfigurationFields removes disabled fields from the mutatingWebhookConfiguration metadata and spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a mutatingWebhookConfiguration
func DropDisabledMutatingWebhookConfigurationFields(mutatingWebhookConfiguration, oldMutatingWebhookConfiguration *MutatingWebhookConfiguration) {
	if !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.AdmissionWebhookMatchConditions) && !matchConditionsInUseMutatingWebhook(oldMutatingWebhookConfiguration) {
		for i := range mutatingWebhookConfiguration.Webhooks {
			mutatingWebhookConfiguration.Webhooks[i].MatchConditions = nil
		}
	}
}

func matchConditionsInUseMutatingWebhook(mutatingWebhookConfiguration *MutatingWebhookConfiguration) bool {
	if mutatingWebhookConfiguration == nil {
		return false
	}
	for _, webhook := range mutatingWebhookConfiguration.Webhooks {
		if len(webhook.MatchConditions) != 0 {
			return true
		}
	}
	return false
}

// DropDisabledValidatingWebhookConfigurationFields removes disabled fields from the validatingWebhookConfiguration metadata and spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a validatingWebhookConfiguration
func DropDisabledValidatingWebhookConfigurationFields(validatingWebhookConfiguration, oldValidatingWebhookConfiguration *ValidatingWebhookConfiguration) {
	if !utilfeature.DefaultFeatureGate.Enabled(genericfeatures.AdmissionWebhookMatchConditions) && !matchConditionsInUseValidatingWebhook(oldValidatingWebhookConfiguration) {
		for i := range validatingWebhookConfiguration.Webhooks {
			validatingWebhookConfiguration.Webhooks[i].MatchConditions = nil
		}
	}
}

func matchConditionsInUseValidatingWebhook(validatingWebhookConfiguration *ValidatingWebhookConfiguration) bool {
	if validatingWebhookConfiguration == nil {
		return false
	}
	for _, webhook := range validatingWebhookConfiguration.Webhooks {
		if len(webhook.MatchConditions) != 0 {
			return true
		}
	}
	return false
}
