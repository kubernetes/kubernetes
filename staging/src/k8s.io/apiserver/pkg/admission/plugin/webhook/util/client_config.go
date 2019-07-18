/*
Copyright 2018 The Kubernetes Authors.

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

package util

import (
	"k8s.io/apiserver/pkg/admission/plugin/webhook"
	webhookutil "k8s.io/apiserver/pkg/util/webhook"
)

// HookClientConfigForWebhook construct a webhookutil.ClientConfig using a WebhookAccessor to access
// v1beta1.MutatingWebhook and v1beta1.ValidatingWebhook API objects.  webhookutil.ClientConfig is used
// to create a HookClient and the purpose of the config struct is to share that with other packages
// that need to create a HookClient.
func HookClientConfigForWebhook(w webhook.WebhookAccessor) webhookutil.ClientConfig {
	ret := webhookutil.ClientConfig{Name: w.GetName(), CABundle: w.GetClientConfig().CABundle}
	if w.GetClientConfig().URL != nil {
		ret.URL = *w.GetClientConfig().URL
	}
	if w.GetClientConfig().Service != nil {
		ret.Service = &webhookutil.ClientConfigService{
			Name:      w.GetClientConfig().Service.Name,
			Namespace: w.GetClientConfig().Service.Namespace,
		}
		if w.GetClientConfig().Service.Port != nil {
			ret.Service.Port = *w.GetClientConfig().Service.Port
		} else {
			ret.Service.Port = 443
		}
		if w.GetClientConfig().Service.Path != nil {
			ret.Service.Path = *w.GetClientConfig().Service.Path
		}
	}
	return ret
}

// HasAdmissionReviewVersion check whether a version is accepted by a given webhook.
func HasAdmissionReviewVersion(a string, w webhook.WebhookAccessor) bool {
	for _, b := range w.GetAdmissionReviewVersions() {
		if b == a {
			return true
		}
	}
	return false
}
