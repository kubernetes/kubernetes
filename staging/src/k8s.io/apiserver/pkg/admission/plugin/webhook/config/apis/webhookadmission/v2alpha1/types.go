/*
Copyright 2019 The Kubernetes Authors.

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

package v2alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// WebhookAdmission provides configuration for the webhook admission controller.
type WebhookAdmission struct {
	metav1.TypeMeta `json:",inline"`

	// KubeConfigFile is the path to the kubeconfig file.
	KubeConfigFile string `json:"kubeConfigFile"`

	// ResourceFilters associate a key with a list of rules that match resources.
	// +optional
	Metrics WebhookAdmissionSpecMetrics `json:"metrics,omitempty"`
}

type WebhookAdmissionSpecMetrics struct {
	// SLODuration specifies options when observing the admission webhook SLO duration metrics.
	// +optional
	SLODuration WebhookAdmissionSpecMetricsSLODuration `json:"sloDuration,omitempty"`
}

type WebhookAdmissionSpecMetricsSLODuration struct {
	// IncludeResourceLabelsFor is a list of rules which when matched by a resource will add
	// resource specific labels to the observed metric.
	// +optional
	IncludeResourceLabelsFor []Rule `json:"includeResourceLabelsFor,omitempty"`
}

// Rule matches resources matching the specified lists of groups versions and resources.
type Rule struct {

	// Groups match API groups. "*" entry matches all.
	Groups []string `json:"groups,omitempty"`

	// Versions match API version. "*" entry matches all.
	Versions []string `json:"versions,omitempty"`

	// Resources match API resource names. "*" entry matches all.
	Resources []string `json:"resources,omitempty"`
}
