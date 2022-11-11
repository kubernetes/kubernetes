package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// HostAssignmentAdmissionConfig is the configuration for the the route host assignment plugin.
type HostAssignmentAdmissionConfig struct {
	metav1.TypeMeta `json:",inline"`

	// domain is used to generate a default host name for a route when the
	// route's host name is empty. The generated host name will follow this
	// pattern: "<route-name>.<route-namespace>.<domain>".
	Domain string `json:"domain"`
}
