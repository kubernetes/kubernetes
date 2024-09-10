/*
Copyright 2019 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package framework

import (
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/equality"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/kubernetes/pkg/api/v1/resource"
	"k8s.io/kubernetes/pkg/features"
)

const (
	// PodAdd is the event when a new pod is added to API server.
	PodAdd = "PodAdd"
	// ScheduleAttemptFailure is the event when a schedule attempt fails.
	ScheduleAttemptFailure = "ScheduleAttemptFailure"
	// BackoffComplete is the event when a pod finishes backoff.
	BackoffComplete = "BackoffComplete"
	// ForceActivate is the event when a pod is moved from unschedulablePods/backoffQ
	// to activeQ. Usually it's triggered by plugin implementations.
	ForceActivate = "ForceActivate"
	// PodUpdate is the event when a pod is updated
	PodUpdate = "PodUpdate"
)

var (
	// AssignedPodAdd is the event when an assigned pod is added.
	AssignedPodAdd = ClusterEvent{Resource: Pod, ActionType: Add, Label: "AssignedPodAdd"}
	// NodeAdd is the event when a new node is added to the cluster.
	NodeAdd = ClusterEvent{Resource: Node, ActionType: Add, Label: "NodeAdd"}
	// NodeDelete is the event when a node is deleted from the cluster.
	NodeDelete = ClusterEvent{Resource: Node, ActionType: Delete, Label: "NodeDelete"}
	// AssignedPodUpdate is the event when an assigned pod is updated.
	AssignedPodUpdate = ClusterEvent{Resource: Pod, ActionType: Update, Label: "AssignedPodUpdate"}
	// UnscheduledPodAdd is the event when an unscheduled pod is added.
	UnscheduledPodAdd = ClusterEvent{Resource: Pod, ActionType: Update, Label: "UnschedulablePodAdd"}
	// UnscheduledPodUpdate is the event when an unscheduled pod is updated.
	UnscheduledPodUpdate = ClusterEvent{Resource: Pod, ActionType: Update, Label: "UnschedulablePodUpdate"}
	// UnscheduledPodDelete is the event when an unscheduled pod is deleted.
	UnscheduledPodDelete = ClusterEvent{Resource: Pod, ActionType: Update, Label: "UnschedulablePodDelete"}
	// assignedPodOtherUpdate is the event when an assigned pod got updated in fields that are not covered by UpdatePodXXX.
	assignedPodOtherUpdate = ClusterEvent{Resource: Pod, ActionType: updatePodOther, Label: "AssignedPodUpdate"}
	// AssignedPodDelete is the event when an assigned pod is deleted.
	AssignedPodDelete = ClusterEvent{Resource: Pod, ActionType: Delete, Label: "AssignedPodDelete"}
	// PodRequestScaledDown is the event when a pod's resource request is scaled down.
	PodRequestScaledDown = ClusterEvent{Resource: Pod, ActionType: UpdatePodScaleDown, Label: "PodRequestScaledDown"}
	// PodLabelChange is the event when a pod's label is changed.
	PodLabelChange = ClusterEvent{Resource: Pod, ActionType: UpdatePodLabel, Label: "PodLabelChange"}
	// PodTolerationChange is the event when a pod's toleration is changed.
	PodTolerationChange = ClusterEvent{Resource: Pod, ActionType: UpdatePodTolerations, Label: "PodTolerationChange"}
	// PodSchedulingGateEliminatedChange is the event when a pod's scheduling gate is changed.
	PodSchedulingGateEliminatedChange = ClusterEvent{Resource: Pod, ActionType: UpdatePodSchedulingGatesEliminated, Label: "PodSchedulingGateChange"}
	// NodeSpecUnschedulableChange is the event when unschedulable node spec is changed.
	NodeSpecUnschedulableChange = ClusterEvent{Resource: Node, ActionType: UpdateNodeTaint, Label: "NodeSpecUnschedulableChange"}
	// NodeAllocatableChange is the event when node allocatable is changed.
	NodeAllocatableChange = ClusterEvent{Resource: Node, ActionType: UpdateNodeAllocatable, Label: "NodeAllocatableChange"}
	// NodeLabelChange is the event when node label is changed.
	NodeLabelChange = ClusterEvent{Resource: Node, ActionType: UpdateNodeLabel, Label: "NodeLabelChange"}
	// NodeAnnotationChange is the event when node annotation is changed.
	NodeAnnotationChange = ClusterEvent{Resource: Node, ActionType: UpdateNodeAnnotation, Label: "NodeAnnotationChange"}
	// NodeTaintChange is the event when node taint is changed.
	NodeTaintChange = ClusterEvent{Resource: Node, ActionType: UpdateNodeTaint, Label: "NodeTaintChange"}
	// NodeConditionChange is the event when node condition is changed.
	NodeConditionChange = ClusterEvent{Resource: Node, ActionType: UpdateNodeCondition, Label: "NodeConditionChange"}
	// PvAdd is the event when a persistent volume is added in the cluster.
	PvAdd = ClusterEvent{Resource: PersistentVolume, ActionType: Add, Label: "PvAdd"}
	// PvUpdate is the event when a persistent volume is updated in the cluster.
	PvUpdate = ClusterEvent{Resource: PersistentVolume, ActionType: Update, Label: "PvUpdate"}
	// PvcAdd is the event when a persistent volume claim is added in the cluster.
	PvcAdd = ClusterEvent{Resource: PersistentVolumeClaim, ActionType: Add, Label: "PvcAdd"}
	// PvcUpdate is the event when a persistent volume claim is updated in the cluster.
	PvcUpdate = ClusterEvent{Resource: PersistentVolumeClaim, ActionType: Update, Label: "PvcUpdate"}
	// StorageClassAdd is the event when a StorageClass is added in the cluster.
	StorageClassAdd = ClusterEvent{Resource: StorageClass, ActionType: Add, Label: "StorageClassAdd"}
	// StorageClassUpdate is the event when a StorageClass is updated in the cluster.
	StorageClassUpdate = ClusterEvent{Resource: StorageClass, ActionType: Update, Label: "StorageClassUpdate"}
	// CSINodeAdd is the event when a CSI node is added in the cluster.
	CSINodeAdd = ClusterEvent{Resource: CSINode, ActionType: Add, Label: "CSINodeAdd"}
	// CSINodeUpdate is the event when a CSI node is updated in the cluster.
	CSINodeUpdate = ClusterEvent{Resource: CSINode, ActionType: Update, Label: "CSINodeUpdate"}
	// CSIDriverAdd is the event when a CSI driver is added in the cluster.
	CSIDriverAdd = ClusterEvent{Resource: CSIDriver, ActionType: Add, Label: "CSIDriverAdd"}
	// CSIDriverUpdate is the event when a CSI driver is updated in the cluster.
	CSIDriverUpdate = ClusterEvent{Resource: CSIDriver, ActionType: Update, Label: "CSIDriverUpdate"}
	// CSIStorageCapacityAdd is the event when a CSI storage capacity is added in the cluster.
	CSIStorageCapacityAdd = ClusterEvent{Resource: CSIStorageCapacity, ActionType: Add, Label: "CSIStorageCapacityAdd"}
	// CSIStorageCapacityUpdate is the event when a CSI storage capacity is updated in the cluster.
	CSIStorageCapacityUpdate = ClusterEvent{Resource: CSIStorageCapacity, ActionType: Update, Label: "CSIStorageCapacityUpdate"}
	// PodSchedulingContextAdd is the event when a pod scheduling context is added.
	PodSchedulingContextAdd = ClusterEvent{Resource: PodSchedulingContext, ActionType: Add, Label: "PodSchedulingContextAdd"}
	// PodSchedulingContextUpdate is the event when a pod scheduling context is updated.
	PodSchedulingContextUpdate = ClusterEvent{Resource: PodSchedulingContext, ActionType: Update, Label: "PodSchedulingContextUpdate"}
	// ResourceClaimAdd is the event when a resource claim is added.
	ResourceClaimAdd = ClusterEvent{Resource: ResourceClaim, ActionType: Add, Label: "ResourceClaimAdd"}
	// ResourceClaimUpdate is the event when a resource claim is updated.
	ResourceClaimUpdate = ClusterEvent{Resource: ResourceClaim, ActionType: Update, Label: "ResourceClaimUpdate"}
	// ResourceSliceAdd is the event when a resource slice is added.
	ResourceSliceAdd = ClusterEvent{Resource: ResourceSlice, ActionType: Add, Label: "ResourceSliceAdd"}
	// ResourceSliceUpdate is the event when a resource slice is updated.
	ResourceSliceUpdate = ClusterEvent{Resource: ResourceSlice, ActionType: Update, Label: "ResourceSliceUpdate"}
	// DeviceClassAdd is the event when a device class is added.
	DeviceClassAdd = ClusterEvent{Resource: DeviceClass, ActionType: Add, Label: "DeviceClassAdd"}
	// DeviceClassUpdate is the event when a device class is updated.
	DeviceClassUpdate = ClusterEvent{Resource: DeviceClass, ActionType: Update, Label: "DeviceClassUpdate"}
	// WildCardEvent semantically matches all resources on all actions.
	WildCardEvent = ClusterEvent{Resource: WildCard, ActionType: All, Label: "WildCardEvent"}
	// UnschedulableTimeout is the event when a pod stays in unschedulable for longer than timeout.
	UnschedulableTimeout = ClusterEvent{Resource: WildCard, ActionType: All, Label: "UnschedulableTimeout"}
	// AllEvents contains all events defined above.
	AllEvents = []ClusterEvent{
		AssignedPodAdd,
		NodeAdd,
		NodeDelete,
		AssignedPodUpdate,
		UnscheduledPodAdd,
		UnscheduledPodUpdate,
		UnscheduledPodDelete,
		assignedPodOtherUpdate,
		AssignedPodDelete,
		PodRequestScaledDown,
		PodLabelChange,
		PodTolerationChange,
		PodSchedulingGateEliminatedChange,
		NodeSpecUnschedulableChange,
		NodeAllocatableChange,
		NodeLabelChange,
		NodeAnnotationChange,
		NodeTaintChange,
		NodeConditionChange,
		PvAdd,
		PvUpdate,
		PvcAdd,
		PvcUpdate,
		StorageClassAdd,
		StorageClassUpdate,
		CSINodeAdd,
		CSINodeUpdate,
		CSIDriverAdd,
		CSIDriverUpdate,
		CSIStorageCapacityAdd,
		CSIStorageCapacityUpdate,
		PodSchedulingContextAdd,
		PodSchedulingContextUpdate,
		ResourceClaimAdd,
		ResourceClaimUpdate,
		ResourceSliceAdd,
		ResourceSliceUpdate,
		DeviceClassAdd,
		DeviceClassUpdate,
		WildCardEvent,
		UnschedulableTimeout,
	}
)

// PodSchedulingPropertiesChange interprets the update of a pod and returns corresponding UpdatePodXYZ event(s).
// Once we have other pod update events, we should update here as well.
func PodSchedulingPropertiesChange(newPod *v1.Pod, oldPod *v1.Pod) (events []ClusterEvent) {
	podChangeExtracters := []podChangeExtractor{
		extractPodLabelsChange,
		extractPodScaleDown,
		extractPodSchedulingGateEliminatedChange,
		extractPodTolerationChange,
	}

	for _, fn := range podChangeExtracters {
		if event := fn(newPod, oldPod); event != nil {
			events = append(events, *event)
		}
	}

	if len(events) == 0 {
		// When no specific event is found, we use AssignedPodOtherUpdate,
		// which should only trigger plugins registering a general Pod/Update event.
		events = append(events, assignedPodOtherUpdate)
	}

	return
}

type podChangeExtractor func(newNode *v1.Pod, oldNode *v1.Pod) *ClusterEvent

// extractPodScaleDown interprets the update of a pod and returns PodRequestScaledDown event if any pod's resource request(s) is scaled down.
func extractPodScaleDown(newPod, oldPod *v1.Pod) *ClusterEvent {
	opt := resource.PodResourcesOptions{
		InPlacePodVerticalScalingEnabled: utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling),
	}
	newPodRequests := resource.PodRequests(newPod, opt)
	oldPodRequests := resource.PodRequests(oldPod, opt)

	for rName, oldReq := range oldPodRequests {
		newReq, ok := newPodRequests[rName]
		if !ok {
			// The resource request of rName is removed.
			return &PodRequestScaledDown
		}

		if oldReq.MilliValue() > newReq.MilliValue() {
			// The resource request of rName is scaled down.
			return &PodRequestScaledDown
		}
	}

	return nil
}

func extractPodLabelsChange(newPod *v1.Pod, oldPod *v1.Pod) *ClusterEvent {
	if isLabelChanged(newPod.GetLabels(), oldPod.GetLabels()) {
		return &PodLabelChange
	}
	return nil
}

func extractPodTolerationChange(newPod *v1.Pod, oldPod *v1.Pod) *ClusterEvent {
	if len(newPod.Spec.Tolerations) != len(oldPod.Spec.Tolerations) {
		// A Pod got a new toleration.
		// Due to API validation, the user can add, but cannot modify or remove tolerations.
		// So, it's enough to just check the length of tolerations to notice the update.
		// And, any updates in tolerations could make Pod schedulable.
		return &PodTolerationChange
	}

	return nil
}

func extractPodSchedulingGateEliminatedChange(newPod *v1.Pod, oldPod *v1.Pod) *ClusterEvent {
	if len(newPod.Spec.SchedulingGates) == 0 && len(oldPod.Spec.SchedulingGates) != 0 {
		// A scheduling gate on the pod is completely removed.
		return &PodSchedulingGateEliminatedChange
	}

	return nil
}

// NodeSchedulingPropertiesChange interprets the update of a node and returns corresponding UpdateNodeXYZ event(s).
func NodeSchedulingPropertiesChange(newNode *v1.Node, oldNode *v1.Node) (events []ClusterEvent) {
	nodeChangeExtracters := []nodeChangeExtractor{
		extractNodeSpecUnschedulableChange,
		extractNodeAllocatableChange,
		extractNodeLabelsChange,
		extractNodeTaintsChange,
		extractNodeConditionsChange,
		extractNodeAnnotationsChange,
	}

	for _, fn := range nodeChangeExtracters {
		if event := fn(newNode, oldNode); event != nil {
			events = append(events, *event)
		}
	}
	return
}

type nodeChangeExtractor func(newNode *v1.Node, oldNode *v1.Node) *ClusterEvent

func extractNodeAllocatableChange(newNode *v1.Node, oldNode *v1.Node) *ClusterEvent {
	if !equality.Semantic.DeepEqual(oldNode.Status.Allocatable, newNode.Status.Allocatable) {
		return &NodeAllocatableChange
	}
	return nil
}

func extractNodeLabelsChange(newNode *v1.Node, oldNode *v1.Node) *ClusterEvent {
	if isLabelChanged(newNode.GetLabels(), oldNode.GetLabels()) {
		return &NodeLabelChange
	}
	return nil
}

func isLabelChanged(newLabels map[string]string, oldLabels map[string]string) bool {
	return !equality.Semantic.DeepEqual(newLabels, oldLabels)
}

func extractNodeTaintsChange(newNode *v1.Node, oldNode *v1.Node) *ClusterEvent {
	if !equality.Semantic.DeepEqual(newNode.Spec.Taints, oldNode.Spec.Taints) {
		return &NodeTaintChange
	}
	return nil
}

func extractNodeConditionsChange(newNode *v1.Node, oldNode *v1.Node) *ClusterEvent {
	strip := func(conditions []v1.NodeCondition) map[v1.NodeConditionType]v1.ConditionStatus {
		conditionStatuses := make(map[v1.NodeConditionType]v1.ConditionStatus, len(conditions))
		for i := range conditions {
			conditionStatuses[conditions[i].Type] = conditions[i].Status
		}
		return conditionStatuses
	}
	if !equality.Semantic.DeepEqual(strip(oldNode.Status.Conditions), strip(newNode.Status.Conditions)) {
		return &NodeConditionChange
	}
	return nil
}

func extractNodeSpecUnschedulableChange(newNode *v1.Node, oldNode *v1.Node) *ClusterEvent {
	if newNode.Spec.Unschedulable != oldNode.Spec.Unschedulable && !newNode.Spec.Unschedulable {
		return &NodeSpecUnschedulableChange
	}
	return nil
}

func extractNodeAnnotationsChange(newNode *v1.Node, oldNode *v1.Node) *ClusterEvent {
	if !equality.Semantic.DeepEqual(oldNode.GetAnnotations(), newNode.GetAnnotations()) {
		return &NodeAnnotationChange
	}
	return nil
}
