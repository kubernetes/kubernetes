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
	"k8s.io/api/admissionregistration/v1beta1"
	"k8s.io/apiserver/pkg/util/webhook"
)

// HookClientConfigForWebhook construct a webhook.ClientConfig using a v1beta1.Webhook API object.
// webhook.ClientConfig is used to create a HookClient and the purpose of the config struct is to
// share that with other packages that need to create a HookClient.
func HookClientConfigForWebhook(w *v1beta1.Webhook) webhook.ClientConfig {
	ret := webhook.ClientConfig{Name: w.Name, CABundle: w.ClientConfig.CABundle}
	if w.ClientConfig.URL != nil {
		ret.URL = *w.ClientConfig.URL
	}
	if w.ClientConfig.Service != nil {
		ret.Service = &webhook.ClientConfigService{
			Name:      w.ClientConfig.Service.Name,
			Namespace: w.ClientConfig.Service.Namespace,
		}
		if w.ClientConfig.Service.Port != nil {
			ret.Service.Port = *w.ClientConfig.Service.Port
		} else {
			ret.Service.Port = 443
		}
		if w.ClientConfig.Service.Path != nil {
			ret.Service.Path = *w.ClientConfig.Service.Path
		}
	}
	return ret
}

// HasAdmissionReviewVersion check whether a version is accepted by a given webhook.
func HasAdmissionReviewVersion(a string, w *v1beta1.Webhook) bool {
	for _, b := range w.AdmissionReviewVersions {
		if b == a {
			return true
		}
	}
	return false
}
