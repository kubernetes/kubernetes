package podautoscaler

import (
	"fmt"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"

	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/client-go/dynamic"
	appsv1client "k8s.io/client-go/kubernetes/typed/apps/v1"
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
	// Filter returns the subset of pods that should be considered for metrics calculation,
	// along with the pods that were filtered out
	Filter(pods []*v1.Pod) (filtered []*v1.Pod, unfiltered []*v1.Pod, err error)
	// Name returns the name of the filter strategy for logging purposes
	Name() string
	WithRESTMapper(mapper apimeta.RESTMapper) PodFilter
	WithCache(cache *ControllerCache) PodFilter
	WithDynamicClient(client *dynamic.DynamicClient) PodFilter
}

// OwnerReferencesFilter filters pods by ownership chain
type OwnerReferencesFilter struct {
	filterOptions FilterOptions
	Client        appsv1client.AppsV1Interface
	RESTMapper    apimeta.RESTMapper
	Cache         *ControllerCache
	dynamicClient *dynamic.DynamicClient
}

func (f *OwnerReferencesFilter) WithClient(client appsv1client.AppsV1Interface) PodFilter {
	f.Client = client
	return f
}

func (f *OwnerReferencesFilter) WithRESTMapper(mapper apimeta.RESTMapper) PodFilter {
	f.RESTMapper = mapper
	return f
}

func (f *OwnerReferencesFilter) Name() string {
	return string(autoscalingv2.OwnerReferences)
}

func (f *OwnerReferencesFilter) WithCache(cache *ControllerCache) PodFilter {
	f.Cache = cache
	return f
}

func (f *OwnerReferencesFilter) WithDynamicClient(client *dynamic.DynamicClient) PodFilter {
	f.dynamicClient = client
	return f
}

func (f *OwnerReferencesFilter) Filter(pods []*v1.Pod) ([]*v1.Pod, []*v1.Pod, error) {
	if f.Cache == nil {
		return nil, nil, fmt.Errorf("cache is required for OwnerReferencesFilter") // TODO: how to handle this?
	}

	if f.filterOptions.ScaleTargetRef == nil {
		return nil, nil, fmt.Errorf("ScaleTargetRef is required for OwnerReferencesFilter")
	}

	// If no pods to filter, return empty slice
	if len(pods) == 0 {
		return pods, nil, nil
	}

	filteredPods := make([]*v1.Pod, 0, len(pods))
	unfilteredPods := make([]*v1.Pod, 0, len(pods))

	namespace := pods[0].Namespace
	targetRef := f.filterOptions.ScaleTargetRef

	// Check ownership for each pod
	for _, pod := range pods {
		isOwned, err := f.isPodOwnedByTarget(pod, *targetRef, namespace)
		if err != nil {
			// On error, assume pod is not owned
			// TODO: is this how to handle this?
			fmt.Println("Error ", err.Error())
			unfilteredPods = append(unfilteredPods, pod)
			continue
		}

		if isOwned {
			fmt.Println("Pod is ownded by targetRef", pod.Name)
			filteredPods = append(filteredPods, pod)
		} else {
			fmt.Println("Pod is not ownded by targetRef", pod.Name)
			unfilteredPods = append(unfilteredPods, pod)
		}
	}

	return filteredPods, unfilteredPods, nil
}

// isPodOwnedByTarget checks if a pod is owned by the target reference by traversing the ownership chain
func (f *OwnerReferencesFilter) isPodOwnedByTarget(pod *v1.Pod, targetRef autoscalingv2.CrossVersionObjectReference, namespace string) (bool, error) {
	const maxOwnershipChainLength = 10 // TODO: should we make this configurable?

	current := &unstructured.Unstructured{}
	current.SetName(pod.Name)
	current.SetNamespace(pod.Namespace)
	current.SetUID(pod.UID)
	current.SetOwnerReferences(pod.OwnerReferences)

	// Check each owner reference to see if any directly matches our target
	var nextOwner *unstructured.Unstructured

	var err error

	for depth := 0; depth < maxOwnershipChainLength; depth++ {
		// Check if current object is our target
		if f.isTargetMatch(current, targetRef) {
			return true, nil
		}

		ownerRefs := current.GetOwnerReferences()
		if len(ownerRefs) == 0 {
			// No more owners to check
			return false, nil
		}

		for _, ownerRef := range ownerRefs {
			// Check if this owner directly matches our target
			if f.isOwnerRefMatch(ownerRef, targetRef) {
				return true, nil
			}

			// If not a direct match, we need to fetch the owner and continue traversal
			if nextOwner == nil {
				nextOwner, err = f.Cache.GetResource(namespace, ownerRef)
				if err != nil {
					fmt.Println(err.Error())
					continue // TODO: what should we do here?
				}
			}
		}

		if nextOwner == nil {
			// No valid owners found that we could fetch
			return false, nil
		}

		// Continue with the owner
		current = nextOwner
	}

	return false, fmt.Errorf("maximum ownership chain depth (%d) exceeded", maxOwnershipChainLength)
}

// isTargetMatch checks if the current object matches the target reference
func (f *OwnerReferencesFilter) isTargetMatch(obj *unstructured.Unstructured, targetRef autoscalingv2.CrossVersionObjectReference) bool {
	if obj.GetName() != targetRef.Name {
		return false
	}

	// Get the object's GVK
	gvk := obj.GroupVersionKind()

	// Parse target's API version
	targetGV, err := schema.ParseGroupVersion(targetRef.APIVersion)
	if err != nil {
		return false
	}

	// Compare Kind and API Version
	return gvk.Kind == targetRef.Kind && gvk.Group == targetGV.Group && gvk.Version == targetGV.Version
}

// isOwnerRefMatch checks if an owner reference matches the target reference
func (f *OwnerReferencesFilter) isOwnerRefMatch(ownerRef metav1.OwnerReference, targetRef autoscalingv2.CrossVersionObjectReference) bool {
	return ownerRef.Kind == targetRef.Kind && ownerRef.Name == targetRef.Name && ownerRef.APIVersion == targetRef.APIVersion
}

// LabelSelectorFilter uses the default label selector strategy
type LabelSelectorFilter struct {
	filterOptions FilterOptions
}

// The default behavior - keep all pods
func (f *LabelSelectorFilter) Filter(pods []*v1.Pod) ([]*v1.Pod, []*v1.Pod, error) {
	return pods, []*v1.Pod{}, nil
}

func (f *LabelSelectorFilter) Name() string {
	return string(autoscalingv2.LabelSelector)
}

func (f *LabelSelectorFilter) WithDynamicClient(client *dynamic.DynamicClient) PodFilter {
	// No-op for label selector
	return f
}

func (f *LabelSelectorFilter) WithRESTMapper(mapper apimeta.RESTMapper) PodFilter {
	// No-op for label selector
	return f
}

func (f *LabelSelectorFilter) WithCache(cache *ControllerCache) PodFilter {
	// No-op for label selector
	return f
}
