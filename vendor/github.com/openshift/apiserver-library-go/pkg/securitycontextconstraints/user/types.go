package user

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// RunAsUserSecurityContextConstraintsStrategy defines the interface for all uid constraint strategies.
type RunAsUserSecurityContextConstraintsStrategy interface {
	// Generate creates the uid based on policy rules.
	Generate(pod *api.Pod, container *api.Container) (*int64, error)
	// Validate ensures that the specified values fall within the range of the strategy.
	Validate(fldPath *field.Path, pod *api.Pod, container *api.Container, runAsNonRoot *bool, runAsUser *int64) field.ErrorList
}
