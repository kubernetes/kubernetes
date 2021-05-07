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

/*
webhookinterceptingwebhooks.go contains utility functions that are used for
enabling webhook intercepting.
*/

package generic

import (
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	"k8s.io/apiserver/pkg/admission/plugin/webhook/config/apis/webhookadmission"
)

// webhookInterceptingWebhooksPrecomputedConfig contains configuration that is
// precomputed at startup and provides data-structures for efficient lookups.
type webhookInterceptingWebhooksPrecomputedConfig struct {
	// maintainerUsersSet is a set of maintainer users
	maintainerUsersSet sets.String
	// maintainerUsersSet is a set of maintainer groups
	maintainerGroupsSet sets.String
	// idSet is a set of strings, each computed by `buildWebhookStringID` and
	// uniquely identifies a webhook intercepting webhook. This set allows
	// efficient search of webhook intercepting webhooks.
	idSet sets.String
}

// webhookStringID is used for precomputing and searching idSet.
func webhookStringID(whType webhookadmission.WebhookType, configurationName, name string) string {
	return fmt.Sprintf(`%s\0%s\0%s`, whType, configurationName, name)
}

// precomputeWebhookInterceptingWebhooksConfig precomputes the data-structures
// used for efficient lookup of webhook intercepting webhooks.
func precomputeWebhookInterceptingWebhooksConfig(webhookConfig *webhookadmission.WebhookAdmission) *webhookInterceptingWebhooksPrecomputedConfig {
	if webhookConfig == nil || webhookConfig.WebhookInterceptingWebhooks == nil {
		return nil
	}
	whStringIDs := []string{}
	for _, whID := range webhookConfig.WebhookInterceptingWebhooks.Identifiers {
		whStringIDs = append(whStringIDs, webhookStringID(whID.Type, whID.ConfigurationName, whID.Name))
	}
	return &webhookInterceptingWebhooksPrecomputedConfig{
		maintainerUsersSet:  sets.NewString(webhookConfig.WebhookInterceptingWebhooks.Maintainers.Users...),
		maintainerGroupsSet: sets.NewString(webhookConfig.WebhookInterceptingWebhooks.Maintainers.Groups...),
		idSet:               sets.NewString(whStringIDs...),
	}
}

// selectWebhookInterceptingWebhooks scans the registered webhooks and selects
// the ones that intercept webhooks. This is only called when the admission
// request is normally not intercepted by other webhooks (e.g. a webhook
// admission request). A nil is returned if none of the registered webhooks will
// intercept the request.
func (a *Webhook) selectWebhookInterceptingWebhooks(attr admission.Attributes) func(input []webhook.WebhookAccessor) []webhook.WebhookAccessor {
	// Exit early if the config is not loaded, or no any webhook intercepting
	// webhook is set.
	if a.webhookInterceptingWebhooksConfig == nil ||
		a.webhookInterceptingWebhooksConfig.idSet.Len() == 0 {
		return nil
	}
	// Exit early if the requestor is a maintainer.
	userInfo := attr.GetUserInfo()
	if a.webhookInterceptingWebhooksConfig.maintainerUsersSet.Has(userInfo.GetName()) ||
		a.webhookInterceptingWebhooksConfig.maintainerGroupsSet.HasAny(userInfo.GetGroups()...) {
		return nil
	}
	return func(input []webhook.WebhookAccessor) []webhook.WebhookAccessor {
		var hooks []webhook.WebhookAccessor
		for _, h := range input {
			whType := webhookadmission.Validating
			_, isMutating := h.GetMutatingWebhook()
			if isMutating {
				whType = webhookadmission.Mutating
			}
			whID := webhookStringID(whType, h.GetConfigurationName(), h.GetName())
			if a.webhookInterceptingWebhooksConfig.idSet.Has(whID) {
				hooks = append(hooks, h)
			}
		}
		return hooks
	}
}
