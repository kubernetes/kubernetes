package podautoscaler

import (
	"context"
	"fmt"

	autoscalingv2 "k8s.io/api/autoscaling/v2"
	v1 "k8s.io/api/core/v1"
	apimeta "k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
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
	WithClient(client appsv1client.AppsV1Interface) PodFilter
	WithRESTMapper(mapper apimeta.RESTMapper) PodFilter
}

// OwnerReferencesFilter filters pods by ownership chain
type OwnerReferencesFilter struct {
	filterOptions FilterOptions
	Client        appsv1client.AppsV1Interface
	RESTMapper    apimeta.RESTMapper
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

func (f *OwnerReferencesFilter) Filter(pods []*v1.Pod) ([]*v1.Pod, []*v1.Pod, error) {
	if f.Client == nil {
		return nil, nil, fmt.Errorf("apps/v1 client is required for OwnerReferencesFilter")
	}

	if f.RESTMapper == nil {
		return nil, nil, fmt.Errorf("RESTMapper is required for OwnerReferencesFilter")
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

	targetKind := f.filterOptions.ScaleTargetRef.Kind
	targetName := f.filterOptions.ScaleTargetRef.Name
	// targetAPIVersion := f.filterOptions.ScaleTargetRef.APIVersion
	namespace := pods[0].Namespace

	// Map to track pods owned by the target
	ownedPods := make(map[types.UID]bool)

	// Handle different resource types with specific ownership patterns
	// TODO(omerap12): combine functions
	switch targetKind {
	case "Deployment":
		if err := f.handleDeployment(namespace, targetName, pods, ownedPods); err != nil {
			return nil, nil, err
		}
	case "StatefulSet":
		if err := f.handleStatefulSet(namespace, targetName, pods, ownedPods); err != nil {
			return nil, nil, err
		}
	case "ReplicaSet":
		if err := f.handleReplicaSet(namespace, targetName, pods, ownedPods); err != nil {
			return nil, nil, err
		}
	case "DaemonSet":
		if err := f.handleDaemonSet(namespace, targetName, pods, ownedPods); err != nil {
			return nil, nil, err
		}
	default:
		f.handleGenericResource(targetKind, targetName, pods, ownedPods)
	}

	for _, pod := range pods {
		if ownedPods[pod.UID] {
			filteredPods = append(filteredPods, pod)
		} else {
			unfilteredPods = append(unfilteredPods, pod)
		}
	}

	return filteredPods, unfilteredPods, nil
}

// Handle Deployment -> ReplicaSet -> Pod ownership chain
func (f *OwnerReferencesFilter) handleDeployment(namespace, name string, pods []*v1.Pod, ownedPods map[types.UID]bool) error {
	deployment, err := f.Client.Deployments(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get Deployment %s/%s: %v", namespace, name, err)
	}

	// Get ReplicaSets owned by this Deployment
	rsList, err := f.Client.ReplicaSets(namespace).List(context.TODO(), metav1.ListOptions{
		LabelSelector: metav1.FormatLabelSelector(deployment.Spec.Selector),
	})
	if err != nil {
		return fmt.Errorf("failed to list ReplicaSets for Deployment %s/%s: %v", namespace, name, err)
	}

	// Create a map of ReplicaSet UIDs that are owned by our deployment
	deploymentRSs := make(map[types.UID]bool)
	for _, rs := range rsList.Items {
		for _, ownerRef := range rs.OwnerReferences {
			if ownerRef.Kind == "Deployment" && ownerRef.Name == name && ownerRef.UID == deployment.UID {
				deploymentRSs[rs.UID] = true
				break
			}
		}
	}

	// Find pods owned by these ReplicaSets
	for _, pod := range pods {
		for _, ownerRef := range pod.OwnerReferences {
			if ownerRef.Kind == "ReplicaSet" && deploymentRSs[ownerRef.UID] {
				ownedPods[pod.UID] = true
				break
			}
		}
	}

	return nil
}

// Handle ReplicaSet -> Pod direct ownership
func (f *OwnerReferencesFilter) handleReplicaSet(namespace, name string, pods []*v1.Pod, ownedPods map[types.UID]bool) error {
	replicaSet, err := f.Client.ReplicaSets(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get ReplicaSet %s/%s: %v", namespace, name, err)
	}

	for _, pod := range pods {
		for _, ownerRef := range pod.OwnerReferences {
			if ownerRef.Kind == "ReplicaSet" && ownerRef.Name == name && ownerRef.UID == replicaSet.UID {
				ownedPods[pod.UID] = true
				break
			}
		}
	}

	return nil
}

// Handle DaemonSet -> Pod direct ownership
func (f *OwnerReferencesFilter) handleDaemonSet(namespace, name string, pods []*v1.Pod, ownedPods map[types.UID]bool) error {
	daemonSet, err := f.Client.DaemonSets(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get DaemonSet %s/%s: %v", namespace, name, err)
	}

	for _, pod := range pods {
		for _, ownerRef := range pod.OwnerReferences {
			if ownerRef.Kind == "DaemonSet" && ownerRef.Name == name && ownerRef.UID == daemonSet.UID {
				ownedPods[pod.UID] = true
				break
			}
		}
	}

	return nil
}

// Handle StatefulSet -> Pod direct ownership
func (f *OwnerReferencesFilter) handleStatefulSet(namespace, name string, pods []*v1.Pod, ownedPods map[types.UID]bool) error {
	statefulSet, err := f.Client.StatefulSets(namespace).Get(context.TODO(), name, metav1.GetOptions{})
	if err != nil {
		return fmt.Errorf("failed to get StatefulSet %s/%s: %v", namespace, name, err)
	}

	for _, pod := range pods {
		for _, ownerRef := range pod.OwnerReferences {
			if ownerRef.Kind == "StatefulSet" && ownerRef.Name == name && ownerRef.UID == statefulSet.UID {
				ownedPods[pod.UID] = true
				break
			}
		}
	}

	return nil
}

func (f *OwnerReferencesFilter) handleGenericResource(kind, name string, pods []*v1.Pod, ownedPods map[types.UID]bool) {
	// Simply check owner references on the pods
	for _, pod := range pods {
		for _, ownerRef := range pod.OwnerReferences {
			if ownerRef.Kind == kind && ownerRef.Name == name {
				ownedPods[pod.UID] = true
				break
			}
		}
	}
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

func (f *LabelSelectorFilter) WithClient(client appsv1client.AppsV1Interface) PodFilter {
	// No-op for label selector
	return f
}

func (f *LabelSelectorFilter) WithRESTMapper(mapper apimeta.RESTMapper) PodFilter {
	// No-op for label selector
	return f
}
