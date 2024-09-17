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
	AssignedPodAdd = ClusterEvent{Resource: AssignedPod, ActionType: Add, Label: "AssignedPodAdd"}
	// NodeAdd is the event when a new node is added to the cluster.
	NodeAdd = ClusterEvent{Resource: Node, ActionType: Add, Label: "NodeAdd"}
	// NodeDelete is the event when a node is deleted from the cluster.
	NodeDelete = ClusterEvent{Resource: Node, ActionType: Delete, Label: "NodeDelete"}
	// AssignedPodUpdate is the event when an assigned pod is updated.
	AssignedPodUpdate = ClusterEvent{Resource: AssignedPod, ActionType: Update, Label: "AssignedPodUpdate"}
	// UnscheduledPodAdd is the event when an unscheduled pod is added.
	UnscheduledPodAdd = ClusterEvent{Resource: UnscheduledPod, ActionType: Add, Label: "UnscheduledPodAdd"}
	// UnscheduledPodUpdate is the event when an unscheduled pod is updated.
	UnscheduledPodUpdate = ClusterEvent{Resource: UnscheduledPod, ActionType: Update, Label: "UnscheduledPodUpdate"}
	// PodItselfUpdate is the event when an unscheduled pod itself is updated.
	PodItselfUpdate = ClusterEvent{Resource: PodItself, ActionType: Update, Label: "PodItselfUpdate"}
	// UnscheduledPodDelete is the event when an unscheduled pod is deleted.
	UnscheduledPodDelete = ClusterEvent{Resource: UnscheduledPod, ActionType: Delete, Label: "UnscheduledPodDelete"}
	// assignedPodOtherUpdate is the event when an assigned pod got updated in fields that are not covered by UpdatePodXXX.
	assignedPodOtherUpdate = ClusterEvent{Resource: AssignedPod, ActionType: updatePodOther, Label: "assignedPodOtherUpdate"}
	// podItselfOtherUpdate is the event when an unscheduled pod itself got updated in fields that are not covered by UpdatePodXXX.
	podItselfOtherUpdate = ClusterEvent{Resource: PodItself, ActionType: updatePodOther, Label: "podItselfOtherUpdate"}
	// unscheduledPodOtherUpdate is the event when an unscheduled pod got updated in fields that are not covered by UpdatePodXXX.
	unscheduledPodOtherUpdate = ClusterEvent{Resource: UnscheduledPod, ActionType: updatePodOther, Label: "unscheduledPodOtherUpdate"}
	// AssignedPodDelete is the event when an assigned pod is deleted.
	AssignedPodDelete = ClusterEvent{Resource: AssignedPod, ActionType: Delete, Label: "AssignedPodDelete"}
	// AssignedPodRequestScaledDown is the event when a pod's resource request is scaled down.
	AssignedPodRequestScaledDown = ClusterEvent{Resource: AssignedPod, ActionType: UpdatePodScaleDown, Label: "AssignedPodRequestScaledDown"}
	// PodItselfRequestScaledDown is the event when an unscheduled pod itself has its resource request scaled down.
	PodItselfRequestScaledDown = ClusterEvent{Resource: PodItself, ActionType: UpdatePodScaleDown, Label: "PodItselfRequestScaledDown"}
	// UnscheduledPodRequestScaledDown is the event when an unscheduled pod has its resource request scaled down.
	UnscheduledPodRequestScaledDown = ClusterEvent{Resource: UnscheduledPod, ActionType: UpdatePodScaleDown, Label: "UnscheduledPodRequestScaledDown"}
	// AssignedPodLabelChange is the event when a AssignedPod's label is changed.
	AssignedPodLabelChange = ClusterEvent{Resource: AssignedPod, ActionType: UpdatePodLabel, Label: "AssignedPodLabelChange"}
	// PodItselfLabelChange is the event when an unscheduled pod itself has its label changed.
	PodItselfLabelChange = ClusterEvent{Resource: PodItself, ActionType: UpdatePodLabel, Label: "PodItselfLabelChange"}
	// UnscheduledPodLabelChange is the event when an unscheduled pod has its label changed.
	UnscheduledPodLabelChange = ClusterEvent{Resource: UnscheduledPod, ActionType: UpdatePodLabel, Label: "UnscheduledPodLabelChange"}
	// AssignedPodTolerationChange is the event when a AssignedPod's toleration is changed.
	AssignedPodTolerationChange = ClusterEvent{Resource: AssignedPod, ActionType: UpdatePodTolerations, Label: "AssignedPodTolerationChange"}
	// PodItselfTolerationChange is the event when an unscheduled pod itself has its toleration updated.
	PodItselfTolerationChange = ClusterEvent{Resource: PodItself, ActionType: UpdatePodTolerations, Label: "PodItselfTolerationChange"}
	// UnscheduledPodTolerationChange is the event when an unscheduled pod has its toleration updated.
	UnscheduledPodTolerationChange = ClusterEvent{Resource: UnscheduledPod, ActionType: UpdatePodTolerations, Label: "UnscheduledPodTolerationChange"}
	// PodItselfSchedulingGateEliminatedChange is the event when an unscheduled pod itself has its scheduling gate changed.
	PodItselfSchedulingGateEliminatedChange = ClusterEvent{Resource: PodItself, ActionType: UpdatePodSchedulingGatesEliminated, Label: "PodItselfSchedulingGateEliminatedChange"}
	// UnscheduledPodSchedulingGateEliminatedChange is the event when an unscheduled pod has its scheduling gate changed.
	UnscheduledPodSchedulingGateEliminatedChange = ClusterEvent{Resource: UnscheduledPod, ActionType: UpdatePodSchedulingGatesEliminated, Label: "UnscheduledPodSchedulingGateEliminatedChange"}
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
		PodItselfUpdate,
		UnscheduledPodDelete,
		assignedPodOtherUpdate,
		podItselfOtherUpdate,
		unscheduledPodOtherUpdate,
		AssignedPodDelete,
		AssignedPodRequestScaledDown,
		PodItselfRequestScaledDown,
		UnscheduledPodRequestScaledDown,
		AssignedPodLabelChange,
		PodItselfLabelChange,
		UnscheduledPodLabelChange,
		AssignedPodTolerationChange,
		PodItselfTolerationChange,
		UnscheduledPodTolerationChange,
		PodItselfSchedulingGateEliminatedChange,
		UnscheduledPodSchedulingGateEliminatedChange,
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
		WildCardEvent,
		UnschedulableTimeout,
	}
)

// PodSchedulingPropertiesChange interprets the update of a pod and returns corresponding UpdatePodXYZ event(s).
// Once we have other pod update events, we should update here as well.
func PodSchedulingPropertiesChange(newPod *v1.Pod, oldPod *v1.Pod, isSelf bool) (events []ClusterEvent) {
	podChangeExtracters := []podChangeExtractor{
		extractPodLabelsChange,
		extractPodScaleDown,
		extractPodSchedulingGateEliminatedChange,
		extractPodTolerationChange,
	}

	for _, fn := range podChangeExtracters {
		if event := fn(newPod, oldPod, isSelf); event != nil {
			events = append(events, *event)
		}
	}

	if len(events) == 0 {
		// When no specific event is found, we use PodOtherUpdate,
		// which should only trigger plugins registering a general Pod/Update event.
		if len(newPod.Spec.NodeName) != 0 {
			events = append(events, assignedPodOtherUpdate)
			return
		}
		if isSelf {
			events = append(events, podItselfOtherUpdate)
			return
		}
		events = append(events, unscheduledPodOtherUpdate)
	}

	return
}

type podChangeExtractor func(newNode *v1.Pod, oldNode *v1.Pod, isSelf bool) *ClusterEvent

// extractPodScaleDown interprets the update of a pod and returns PodRequestScaledDown event if any pod's resource request(s) is scaled down.
func extractPodScaleDown(newPod, oldPod *v1.Pod, isSelf bool) *ClusterEvent {
	opt := resource.PodResourcesOptions{
		InPlacePodVerticalScalingEnabled: utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling),
	}
	newPodRequests := resource.PodRequests(newPod, opt)
	oldPodRequests := resource.PodRequests(oldPod, opt)

	for rName, oldReq := range oldPodRequests {
		newReq, ok := newPodRequests[rName]
		if !ok || oldReq.MilliValue() > newReq.MilliValue() {
			// The resource request of rName is removed or scaled down.
			if len(oldPod.Spec.NodeName) != 0 {
				return &AssignedPodRequestScaledDown
			}
			if isSelf {
				return &PodItselfRequestScaledDown
			}
			return &UnscheduledPodRequestScaledDown
		}
	}

	return nil
}

func extractPodLabelsChange(newPod *v1.Pod, oldPod *v1.Pod, isSelf bool) *ClusterEvent {
	if isLabelChanged(newPod.GetLabels(), oldPod.GetLabels()) {
		if len(oldPod.Spec.NodeName) != 0 {
			return &AssignedPodLabelChange
		}
		if isSelf {
			return &PodItselfLabelChange
		}
		return &UnscheduledPodLabelChange
	}
	return nil
}

func extractPodTolerationChange(newPod *v1.Pod, oldPod *v1.Pod, isSelf bool) *ClusterEvent {
	if len(newPod.Spec.Tolerations) != len(oldPod.Spec.Tolerations) {
		// A Pod got a new toleration.
		// Due to API validation, the user can add, but cannot modify or remove tolerations.
		// So, it's enough to just check the length of tolerations to notice the update.
		// And, any updates in tolerations could make Pod schedulable.
		if len(oldPod.Spec.NodeName) != 0 {
			return &AssignedPodTolerationChange
		}
		if isSelf {
			return &PodItselfTolerationChange
		}
		return &UnscheduledPodTolerationChange
	}

	return nil
}

func extractPodSchedulingGateEliminatedChange(newPod *v1.Pod, oldPod *v1.Pod, isSelf bool) *ClusterEvent {
	if len(newPod.Spec.SchedulingGates) == 0 && len(oldPod.Spec.SchedulingGates) != 0 {
		// A scheduling gate on the pod is completely removed.
		if isSelf {
			return &PodItselfSchedulingGateEliminatedChange
		}
		return &UnscheduledPodSchedulingGateEliminatedChange
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
