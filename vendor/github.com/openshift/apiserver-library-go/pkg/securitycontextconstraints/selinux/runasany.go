package selinux

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	coreapi "k8s.io/kubernetes/pkg/apis/core"

	securityv1 "github.com/openshift/api/security/v1"
)

// runAsAny implements the SELinuxSecurityContextConstraintsStrategy interface.
type runAsAny struct{}

var _ SELinuxSecurityContextConstraintsStrategy = &runAsAny{}

// NewRunAsAny provides a strategy that will return the configured se linux context or nil.
func NewRunAsAny(options *securityv1.SELinuxContextStrategyOptions) (SELinuxSecurityContextConstraintsStrategy, error) {
	return &runAsAny{}, nil
}

// Generate creates the SELinuxOptions based on constraint rules.
func (s *runAsAny) Generate(pod *coreapi.Pod, container *coreapi.Container) (*coreapi.SELinuxOptions, error) {
	return nil, nil
}

// Validate ensures that the specified values fall within the range of the strategy.
func (s *runAsAny) Validate(fldPath *field.Path, _ *coreapi.Pod, _ *coreapi.Container, options *coreapi.SELinuxOptions) field.ErrorList {
	return field.ErrorList{}
}
