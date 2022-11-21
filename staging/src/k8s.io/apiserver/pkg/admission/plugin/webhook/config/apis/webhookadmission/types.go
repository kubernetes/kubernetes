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
	// +optional
	KubeConfigFile string

	// ExclusionRules are a list of rules that the webhooks while ignore
	// +optional
	ExclusionRules []ExclusionRule //NOTE: using ExclusionRule rather than Rule with name/namespace because don't want to have scope field
}

type ExclusionRule struct {
	// APIGroups is the API groups the resources belong to. '*' is all groups.
	// If '*' is present, the length of the slice must be one.
	// Required.
	// +listType=atomic
	APIGroups []string `json:"apiGroups,omitempty"`

	// APIVersions is the API versions the resources belong to. '*' is all versions.
	// If '*' is present, the length of the slice must be one.
	// Required.
	// +listType=atomic
	APIVersions []string `json:"apiVersions,omitempty"`

	// Kind to exclude.
	// Required.
	Kind string `json:"kind,omitempty"`

	// Name is a list of object names this rule applies to.
	// Required.
	Name string `json:"name,omitempty"`

	// Namespace is a the namespace this rule applies to.
	// Required.
	Namespace string `json:"namespace,omitempty"`
}
