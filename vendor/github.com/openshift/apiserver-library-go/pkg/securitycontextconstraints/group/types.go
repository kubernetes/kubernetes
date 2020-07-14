package group

import (
	"k8s.io/apimachinery/pkg/util/validation/field"
	api "k8s.io/kubernetes/pkg/apis/core"
)

// GroupSecurityContextConstraintsStrategy defines the interface for all group constraint strategies.
type GroupSecurityContextConstraintsStrategy interface {
	// Generate creates the group based on policy rules.  The underlying implementation can
	// decide whether it will return a full range of values or a subset of values from the
	// configured ranges.
	Generate(pod *api.Pod) ([]int64, error)
	// Generate a single value to be applied.  The underlying implementation decides which
	// value to return if configured with multiple ranges.  This is used for FSGroup.
	GenerateSingle(pod *api.Pod) (*int64, error)
	// Validate ensures that the specified values fall within the range of the strategy.
	Validate(pod *api.Pod, groups []int64) field.ErrorList
}
