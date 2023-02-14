package admissionregistration

import (
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/features"
)

// DropDisabledMutatingWebhookConfigurationFields removes disabled fields from the mutatingWebhookConfiguration metadata and spec.
// This should be called from PrepareForCreate/PrepareForUpdate for all resources containing a mutatingWebhookConfiguration
func DropDisabledMutatingWebhookConfigurationFields(mutatingWebhookConfiguration, oldMutatingWebhookConfiguration *MutatingWebhookConfiguration) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.AdmissionWebhookMatchConditions) && !matchConditionsInUseMutatingWebhook(oldMutatingWebhookConfiguration) {
		for i := range mutatingWebhookConfiguration.Webhooks {
			mutatingWebhookConfiguration.Webhooks[i].MatchConditions = []MatchCondition{}
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
	if !utilfeature.DefaultFeatureGate.Enabled(features.AdmissionWebhookMatchConditions) && !matchConditionsInUseValidatingWebhook(oldValidatingWebhookConfiguration) {
		for i := range validatingWebhookConfiguration.Webhooks {
			validatingWebhookConfiguration.Webhooks[i].MatchConditions = []MatchCondition{}
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
