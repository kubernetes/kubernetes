// pod_filters.go
package podautoscaler

import (
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/client-go/kubernetes"
)

type FilterOptions struct {
	ScaleTargetRef *autoscalingv2.CrossVersionObjectReference
}

// NewPodFilter creates a new PodFilter with the given strategy and options
func NewPodFilter(strategyName string, opts FilterOptions) PodFilter {
	if strategyName == string(autoscalingv2.OwnerReferences) {
		return &OwnerReferencesFilter{
			filterOptions: opts,
		}
	}
	// default to LabelSelector
	return &LabelSelectorFilter{
		filterOptions: opts,
	}
}

// PodFilter defines an interface for filtering pods based on various strategies
type PodFilter interface {
	// Filter returns the subset of pods that should be considered for metrics calculation
	Filter(pods []*v1.Pod) ([]*v1.Pod, error)
	// Name returns the name of the filter strategy for logging purposes
	Name() string
}

// OwnerReferencesFilter filters pods by ownership chain
type OwnerReferencesFilter struct {
	filterOptions FilterOptions
	Client        kubernetes.Interface
	RESTMapper    apimeta.RESTMapper
}

func (f *OwnerReferencesFilter) Filter(pods []*v1.Pod) ([]*v1.Pod, error) {
	// TBD
	return nil, nil
}

func (f *OwnerReferencesFilter) Name() string {
	return "OwnerReferences"
}

// LabelSelectorFilter uses the default label selector strategy
type LabelSelectorFilter struct {
	filterOptions FilterOptions
}

// The default behavior - keep all pods
func (f *LabelSelectorFilter) Filter(pods []*v1.Pod) ([]*v1.Pod, error) {
	return pods, nil
}

func (f *LabelSelectorFilter) Name() string {
	return "LabelSelector"
}
