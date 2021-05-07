/*
Copyright 2017 The Kubernetes Authors.

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

package v1alpha1

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

type WebhookType string

const (
	Mutating   WebhookType = "Mutating"
	Validating WebhookType = "Validating"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// WebhookAdmission provides configuration for the webhook admission controller.
type WebhookAdmission struct {
	metav1.TypeMeta `json:",inline"`

	// WebhookInterceptingWebhooks specifies the webhooks that could intercept
	// other webhooks.
	WebhookInterceptingWebhooks *WebhookInterceptingWebhooks `json:"webhookInterceptingWebhooks,omitempty"`

	// KubeConfigFile is the path to the kubeconfig file.
	KubeConfigFile string `json:"kubeConfigFile"`
}

type WebhookInterceptingWebhooks struct {
	// Identifiers specify the webhook intercepting webhooks
	Identifiers []WebhookIdentifier `json:"identifiers"`
	// Maintainers consist of identities which maintain (i.e. could
	// add/delete/modify) the webhook intercepting webhooks (WIWs). Normally
	// webhook admission requests are intercepted by WIWs, and could potentially
	// be denied (for instance when the webhook is misconfigured). However,
	// webhook admission requests made by maintainers will *not* be intercepted
	// by WIWs, enabling them to maintain these webhooks. A request is
	// considered to be made by a maintainer if either its username is in Users,
	// or one of its groups is in Groups.
	Maintainers WebhookMaintainers `json:"maintainers"`
}

// WebhookIdentifier uniquely identifies a webhook across all
// registered webhooks. A webhook's name by itself is not enough to do so, as
// webhookconfigurations with different names could have webhooks with
// the same name.
type WebhookIdentifier struct {
	// ConfigurationName is the name of the webhook configuration (e.g.
	// foo-config)
	ConfigurationName string `json:"configurationName"`
	// Name specifies the name of the webhook (e.g. foo.bar.com).
	Name string `json:"name"`
	// Type specifies the type of the webhook. It is set internally to the type
	// of the webhook admission plugin that the configuration belongs.
	Type WebhookType `json:"-"`
}

// WebhookMaintainers is a list of users and groups.
type WebhookMaintainers struct {
	Users  []string `json:"users"`
	Groups []string `json:"groups"`
}
