package hardcodedauthorizer

import (
	"context"

	"k8s.io/apiserver/pkg/authorization/authorizer"
)

type metricsAuthorizer struct{}

// GetUser() user.Info - checked
// GetVerb() string - checked
// IsReadOnly() bool - na
// GetNamespace() string - na
// GetResource() string - na
// GetSubresource() string - na
// GetName() string - na
// GetAPIGroup() string - na
// GetAPIVersion() string - na
// IsResourceRequest() bool - checked
// GetPath() string - checked
func (metricsAuthorizer) Authorize(ctx context.Context, a authorizer.Attributes) (authorized authorizer.Decision, reason string, err error) {
	if a.GetUser().GetName() != "system:serviceaccount:openshift-monitoring:prometheus-k8s" {
		return authorizer.DecisionNoOpinion, "", nil
	}
	if !a.IsResourceRequest() &&
		a.GetVerb() == "get" &&
		a.GetPath() == "/metrics" {
		return authorizer.DecisionAllow, "requesting metrics is allowed", nil
	}

	return authorizer.DecisionNoOpinion, "", nil
}

// NewHardCodedMetricsAuthorizer returns a hardcoded authorizer for checking metrics.
func NewHardCodedMetricsAuthorizer() *metricsAuthorizer {
	return new(metricsAuthorizer)
}
