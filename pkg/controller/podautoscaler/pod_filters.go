// pod_filters.go
package podautoscaler

import (
	autoscalingv2 "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/kubernetes"
)

func NewPodFilter(strategyName string) PodFilter {
	if strategyName == string(autoscalingv2.LabelSelector) {
		return &LabelSelectorFilter{}
	}
	if strategyName == string(autoscalingv2.OwnerReferences) {
		return &OwnerReferencesFilter{}
	}
	//default filer
	return &LabelSelectorFilter{}
}

// PodFilter defines an interface for filtering pods based on various strategies
type PodFilter interface {
	// Filter returns a subset of pods that match the filtering criteria
	Filter(hpa *autoscalingv2.HorizontalPodAutoscaler, pods []*v1.Pod, selector labels.Selector) ([]*v1.Pod, error)

	// Name returns the name of the filter strategy for logging purposes
	Name() string
}

// OwnerReferencesFilter filters pods by ownership chain
type OwnerReferencesFilter struct {
	// Dependencies needed for filtering
	Client     kubernetes.Interface
	RESTMapper apimeta.RESTMapper
}

func (f *OwnerReferencesFilter) Filter(hpa *autoscalingv2.HorizontalPodAutoscaler, pods []*v1.Pod, selector labels.Selector) ([]*v1.Pod, error) {
	//TBD
	return nil, nil
}

func (f *OwnerReferencesFilter) Name() string {
	return "OwnerReferences"
}

// LabelSelectorFilter uses the default label selector strategy
type LabelSelectorFilter struct{}

// The default behavior is to return all pods that matched the label selector
func (f *LabelSelectorFilter) Filter(hpa *autoscalingv2.HorizontalPodAutoscaler, pods []*v1.Pod, selector labels.Selector) ([]*v1.Pod, error) {
	return pods, nil
}

func (f *LabelSelectorFilter) Name() string {
	return "LabelSelector"
}
