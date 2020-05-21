package v1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object

// ExternalIPRangerAdmissionConfig is the configuration for which CIDRs services can't manage
type ExternalIPRangerAdmissionConfig struct {
	metav1.TypeMeta `json:",inline"`

	// ExternalIPNetworkCIDRs controls what values are acceptable for the service external IP field. If empty, no externalIP
	// may be set. It may contain a list of CIDRs which are checked for access. If a CIDR is prefixed with !, IPs in that
	// CIDR will be rejected. Rejections will be applied first, then the IP checked against one of the allowed CIDRs. You
	// should ensure this range does not overlap with your nodes, pods, or service CIDRs for security reasons.
	ExternalIPNetworkCIDRs []string `json:"externalIPNetworkCIDRs"`
	// AllowIngressIP indicates that ingress IPs should be allowed
	AllowIngressIP bool `json:"allowIngressIP"`
}
