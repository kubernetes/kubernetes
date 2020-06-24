/*
Copyright 2020 The Kubernetes Authors.

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

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ServiceExport declares that the associated service should be exported to
// other clusters.
type ServiceExport struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`
	// +optional
	Status ServiceExportStatus `json:"status,omitempty"`
}

// ServiceExportStatus contains the current status of an export.
type ServiceExportStatus struct {
	// +optional
	// +patchStrategy=merge
	// +patchMergeKey=type
	// +listType=map
	// +listMapKey=type
	Conditions []ServiceExportCondition `json:"conditions,omitempty" patchStrategy:"merge" patchMergeKey:"type"`
}

// ServiceExportConditionType identifies a specific condition.
type ServiceExportConditionType string

const (
	// ServiceExportValid means that the service referenced by this
	// service export has been recognized as valid by an mcs-controller.
	// This will be false if the service is found to be unexportable
	// (ExternalName, not found).
	ServiceExportValid ServiceExportConditionType = "Valid"
	// ServiceExportConflict means that there is a conflict between two
	// exports for the same Service. When "True", the condition message
	// should contain enough information to diagnose the conflict:
	// field(s) under contention, which cluster won, and why.
	// Users should not expect detailed per-cluster information in the
	// conflict message.
	ServiceExportConflict ServiceExportConditionType = "Conflict"
)

// ServiceExportCondition contains details for the current condition of this
// service export.
//
// Once [#1624](https://github.com/kubernetes/enhancements/pull/1624) is
// merged, this will be replaced by metav1.Condition.
type ServiceExportCondition struct {
	Type ServiceExportConditionType `json:"type"`
	// Status is one of {"True", "False", "Unknown"}
	Status v1.ConditionStatus `json:"status"`
	// +optional
	LastTransitionTime *metav1.Time `json:"lastTransitionTime,omitempty"`
	// +optional
	Reason *string `json:"reason,omitempty"`
	// +optional
	Message *string `json:"message,omitempty"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ServiceImport describes a service imported from clusters in a supercluster.
type ServiceImport struct {
	metav1.TypeMeta `json:",inline"`
	// +optional
	metav1.ObjectMeta `json:"metadata,omitempty"`
	// +optional
	Spec ServiceImportSpec `json:"spec,omitempty"`
	// +optional
	Status ServiceImportStatus `json:"status,omitempty"`
}

// ServiceImportType designates the type of a ServiceImport
type ServiceImportType string

const (
	// SuperclusterIP are only accessible via the Supercluster IP.
	SuperclusterIP ServiceImportType = "SuperclusterIP"
	// Headless services allow backend pods to be addressed directly.
	Headless ServiceImportType = "Headless"
)

// ServiceImportSpec describes an imported service and the information necessary to consume it.
type ServiceImportSpec struct {
	// +listType=atomic
	Ports []ServicePort `json:"ports"`
	// +optional
	IP string `json:"ip,omitempty"`
	// +optional
	Type ServiceImportType `json:"type"`
	// +optional
	SessionAffinity v1.ServiceAffinity `json:"sessionAffinity"`
	// +optional
	SessionAffinityConfig *v1.SessionAffinityConfig `json:"sessionAffinityConfig"`
}

// ServicePort represents the port on which the service is exposed
type ServicePort struct {
	// The name of this port within the service. This must be a DNS_LABEL.
	// All ports within a ServiceSpec must have unique names. When considering
	// the endpoints for a Service, this must match the 'name' field in the
	// EndpointPort.
	// Optional if only one ServicePort is defined on this service.
	// +optional
	Name string `json:"name,omitempty"`

	// The IP protocol for this port. Supports "TCP", "UDP", and "SCTP".
	// Default is TCP.
	// +optional
	Protocol v1.Protocol `json:"protocol,omitempty"`

	// The application protocol for this port.
	// This field follows standard Kubernetes label syntax.
	// Un-prefixed names are reserved for IANA standard service names (as per
	// RFC-6335 and http://www.iana.org/assignments/service-names).
	// Non-standard protocols should use prefixed names such as
	// mycompany.com/my-custom-protocol.
	// Field can be enabled with ServiceAppProtocol feature gate.
	// +optional
	AppProtocol *string `json:"appProtocol,omitempty"`

	// The port that will be exposed by this service.
	Port int32 `json:"port"`
}

// ServiceImportStatus describes derived state of an imported service.
type ServiceImportStatus struct {
	// +optional
	// +patchStrategy=merge
	// +patchMergeKey=cluster
	// +listType=map
	// +listMapKey=cluster
	Clusters []ClusterStatus `json:"clusters"`
}

// ClusterStatus contains service configuration mapped to a specific source cluster
type ClusterStatus struct {
	Cluster string `json:"cluster"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ServiceExportList represents a list of endpoint slices
type ServiceExportList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// +optional
	metav1.ListMeta `json:"metadata,omitempty"`
	// List of endpoint slices
	// +listType=set
	Items []ServiceExport `json:"items"`
}

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ServiceImportList represents a list of endpoint slices
type ServiceImportList struct {
	metav1.TypeMeta `json:",inline"`
	// Standard list metadata.
	// +optional
	metav1.ListMeta `json:"metadata,omitempty"`
	// List of endpoint slices
	// +listType=set
	Items []ServiceImport `json:"items"`
}
