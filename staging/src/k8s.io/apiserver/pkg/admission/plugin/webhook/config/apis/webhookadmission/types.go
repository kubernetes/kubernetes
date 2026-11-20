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

package webhookadmission

import metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// WebhookAdmission provides configuration for the webhook admission controller.
type WebhookAdmission struct {
	metav1.TypeMeta

	// KubeConfigFile is the path to the kubeconfig file.
	KubeConfigFile string

	// StaticManifestsDir is the path to a directory containing static webhook
	// configurations to be loaded at startup. Files with extensions .yaml,
	// .yml, and .json are read. Only admissionregistration.k8s.io/v1
	// ValidatingWebhookConfiguration and MutatingWebhookConfiguration
	// resources are supported.
	// Using this field requires the ManifestBasedAdmissionControlConfig
	// feature gate to be enabled.
	// +optional
	StaticManifestsDir string

	// StaticOrder determines where the webhooks loaded from StaticManifestsDir are
	// placed relative to the webhooks read from the API. Mutating webhooks are
	// dispatched sequentially, so this controls the order in which they observe and
	// mutate an object.
	//
	// Allowed values are:
	//   - "StaticFirst": manifest-based webhooks run before all API-based webhooks.
	//   - "StaticLast": manifest-based webhooks run after all API-based webhooks.
	//   - "ByName": manifest-based and API-based webhooks are interleaved and stably
	//     sorted together by the name of the webhook configuration that contains them,
	//     which is how API-based webhooks are already ordered relative to each other.
	//
	// Empty means StaticFirst, which preserves the behavior from before this field
	// existed.
	// StaticFirst also guarantees that manifest-based webhooks observe an object
	// before any webhook registered through the API does. StaticLast and ByName give
	// up that guarantee: a webhook registered by anyone with write access to
	// mutatingwebhookconfigurations can then mutate the object first.
	// May only be set when StaticManifestsDir is set.
	// This field is only honored by the MutatingAdmissionWebhook plugin. Setting it to
	// any value other than StaticFirst on the ValidatingAdmissionWebhook plugin is a
	// configuration error, because validating webhooks are dispatched in parallel
	// against the same object and cannot have ordering dependencies.
	// Using this field requires the ManifestBasedAdmissionControlConfig feature gate
	// to be enabled.
	// +optional
	StaticOrder StaticOrder
}

// StaticOrder determines where manifest-based webhooks are placed relative to
// webhooks read from the API.
type StaticOrder string

const (
	// StaticFirst runs manifest-based webhooks before all API-based webhooks.
	StaticFirst StaticOrder = "StaticFirst"
	// StaticLast runs manifest-based webhooks after all API-based webhooks.
	StaticLast StaticOrder = "StaticLast"
	// ByName interleaves manifest-based and API-based webhooks, stably sorting them
	// together by the name of the webhook configuration that contains them. This is
	// how API-based webhooks are already ordered relative to each other, so naming a
	// manifest-based configuration after the API-based configuration it replaces
	// usually preserves that configuration's position.
	// It does not preserve the position relative to a configuration whose name starts
	// with "<replaced-name>-", or starts with "<replaced-name>." and sorts before
	// "<replaced-name>.static.k8s.io". For example, "foo-bar" and "foo.bar.io"
	// both sort after "foo" but before "foo.static.k8s.io".
	ByName StaticOrder = "ByName"
)
