package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// RestrictedEndpointsAdmissionConfig is the configuration for which CIDRs services can't manage
type RestrictedEndpointsAdmissionConfig struct {
	metav1.TypeMeta `json:",inline"`

	// RestrictedCIDRs indicates what CIDRs will be disallowed for services.
	RestrictedCIDRs []string `json:"restrictedCIDRs"`
}
