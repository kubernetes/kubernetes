package crypto

import (
	configv1 "github.com/openshift/api/config/v1"
)

// ShouldHonorClusterTLSProfile returns true if the component should honor the
// cluster-wide TLS security profile settings from apiserver.config.openshift.io/cluster.
//
// When this returns true (StrictAllComponents mode), components must honor the
// cluster-wide TLS profile unless they have a component-specific TLS configuration
// that overrides it.
//
// Unknown enum values are treated as StrictAllComponents for forward compatibility
// and to default to the more secure behavior.
func ShouldHonorClusterTLSProfile(tlsAdherence configv1.TLSAdherencePolicy) bool {
	switch tlsAdherence {
	case configv1.TLSAdherencePolicyNoOpinion, configv1.TLSAdherencePolicyLegacyAdheringComponentsOnly:
		return false
	default:
		return true
	}
}
