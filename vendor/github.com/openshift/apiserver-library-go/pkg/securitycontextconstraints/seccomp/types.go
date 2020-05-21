package seccomp

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// SeccompStrategy defines the interface for all seccomp constraint strategies.
type SeccompStrategy interface {
	// Generate creates the profile based on policy rules.
	Generate(pod *api.Pod) (string, error)
	// ValidatePod ensures that the specified values on the pod fall within the range
	// of the strategy.
	ValidatePod(pod *api.Pod) field.ErrorList
	// ValidateContainer ensures that the specified values on the container fall within
	// the range of the strategy.
	ValidateContainer(pod *api.Pod, container *api.Container) field.ErrorList
}
