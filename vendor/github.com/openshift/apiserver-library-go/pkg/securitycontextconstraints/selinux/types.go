package selinux

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
)

// SELinuxSecurityContextConstraintsStrategy defines the interface for all SELinux constraint strategies.
type SELinuxSecurityContextConstraintsStrategy interface {
	// Generate creates the SELinuxOptions based on constraint rules.
	Generate(pod *coreapi.Pod, container *coreapi.Container) (*coreapi.SELinuxOptions, error)
	// Validate ensures that the specified values fall within the range of the strategy.
	Validate(fldPath *field.Path, pod *coreapi.Pod, container *coreapi.Container, options *coreapi.SELinuxOptions) field.ErrorList
}
