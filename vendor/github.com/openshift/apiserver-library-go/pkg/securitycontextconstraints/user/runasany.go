package user

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"

	securityv1 "github.com/openshift/api/security/v1"
)

// runAsAny implements the interface RunAsUserSecurityContextConstraintsStrategy.
type runAsAny struct{}

var _ RunAsUserSecurityContextConstraintsStrategy = &runAsAny{}

// NewRunAsAny provides a strategy that will return nil.
func NewRunAsAny(options *securityv1.RunAsUserStrategyOptions) (RunAsUserSecurityContextConstraintsStrategy, error) {
	return &runAsAny{}, nil
}

// Generate creates the uid based on policy rules.
func (s *runAsAny) Generate(pod *api.Pod, container *api.Container) (*int64, error) {
	return nil, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *runAsAny) Validate(fldPath *field.Path, _ *api.Pod, _ *api.Container, runAsNonRoot *bool, runAsUser *int64) field.ErrorList {
	return field.ErrorList{}
}
