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
	"k8s.io/api/auditregistration/v1alpha1"
	"k8s.io/apiserver/pkg/util/webhook"
)

// HookClientConfigForSink constructs a webhook.ClientConfig using a v1alpha1.AuditSink API object.
// webhook.ClientConfig is used to create a HookClient and the purpose of the config struct is to
// share that with other packages that need to create a HookClient.
func HookClientConfigForSink(a *v1alpha1.AuditSink) webhook.ClientConfig {
	c := a.Spec.Webhook.ClientConfig
	ret := webhook.ClientConfig{Name: a.Name, CABundle: c.CABundle}
	if c.URL != nil {
		ret.URL = *c.URL
	}
	if c.Service != nil {
		ret.Service = &webhook.ClientConfigService{
			Name:      c.Service.Name,
			Namespace: c.Service.Namespace,
		}
		if c.Service.Port != nil {
			ret.Service.Port = *c.Service.Port
		} else {
			ret.Service.Port = 443
		}

		if c.Service.Path != nil {
			ret.Service.Path = *c.Service.Path
		}
	}
	return ret
}
