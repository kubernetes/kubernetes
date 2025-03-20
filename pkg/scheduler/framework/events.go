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
	"k8s.io/component-helpers/resource"
	"k8s.io/dynamic-resource-allocation/resourceclaim"
	"k8s.io/kubernetes/pkg/features"
)

// Special event labels.
const (
	// ScheduleAttemptFailure is the event when a schedule attempt fails.
	ScheduleAttemptFailure = "ScheduleAttemptFailure"
	// BackoffComplete is the event when a pod finishes backoff.
	BackoffComplete = "BackoffComplete"
	// PopFromBackoffQ is the event when a pod is popped from backoffQ when activeQ is empty.
	PopFromBackoffQ = "PopFromBackoffQ"
	// ForceActivate is the event when a pod is moved from unschedulablePods/backoffQ
	// to activeQ. Usually it's triggered by plugin implementations.
	ForceActivate = "ForceActivate"
	// UnschedulableTimeout is the event when a pod is moved from unschedulablePods
	// due to the timeout specified at pod-max-in-unschedulable-pods-duration.
	UnschedulableTimeout = "UnschedulableTimeout"
)

var (
	// EventAssignedPodAdd is the event when an assigned pod is added.
	EventAssignedPodAdd = ClusterEvent{Resource: assignedPod, ActionType: Add}
	// EventAssignedPodUpdate is the event when an assigned pod is updated.
	EventAssignedPodUpdate = ClusterEvent{Resource: assignedPod, ActionType: Update}
	// EventAssignedPodDelete is the event when an assigned pod is deleted.
	EventAssignedPodDelete = ClusterEvent{Resource: assignedPod, ActionType: Delete}
	// EventUnscheduledPodAdd is the event when an unscheduled pod is added.
	EventUnscheduledPodAdd = ClusterEvent{Resource: unschedulablePod, ActionType: Add}
	// EventUnscheduledPodUpdate is the event when an unscheduled pod is updated.
	EventUnscheduledPodUpdate = ClusterEvent{Resource: unschedulablePod, ActionType: Update}
	// EventUnscheduledPodDelete is the event when an unscheduled pod is deleted.
	EventUnscheduledPodDelete = ClusterEvent{Resource: unschedulablePod, ActionType: Delete}
	// EventUnschedulableTimeout is the event when a pod stays in unschedulable for longer than timeout.
	EventUnschedulableTimeout = ClusterEvent{Resource: WildCard, ActionType: All, label: UnschedulableTimeout}
	// EventForceActivate is the event when a pod is moved from unschedulablePods/backoffQ to activeQ.
	EventForceActivate = ClusterEvent{Resource: WildCard, ActionType: All, label: ForceActivate}
)

// PodSchedulingPropertiesChange interprets the update of a pod and returns corresponding UpdatePodXYZ event(s).
// Once we have other pod update events, we should update here as well.
func PodSchedulingPropertiesChange(newPod *v1.Pod, oldPod *v1.Pod) (events []ClusterEvent) {
	r := assignedPod
	if newPod.Spec.NodeName == "" {
		r = unschedulablePod
	}

	podChangeExtracters := []podChangeExtractor{
		extractPodLabelsChange,
		extractPodScaleDown,
		extractPodSchedulingGateEliminatedChange,
		extractPodTolerationChange,
	}
	if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
		podChangeExtracters = append(podChangeExtracters, extractPodGeneratedResourceClaimChange)
	}

	for _, fn := range podChangeExtracters {
		if event := fn(newPod, oldPod); event != none {
			events = append(events, ClusterEvent{Resource: r, ActionType: event})
		}
	}

	if len(events) == 0 {
		// When no specific event is found, we use AssignedPodOtherUpdate,
		// which should only trigger plugins registering a general Pod/Update event.
		events = append(events, ClusterEvent{Resource: r, ActionType: updatePodOther})
	}

	return
}

type podChangeExtractor func(newPod *v1.Pod, oldPod *v1.Pod) ActionType

// extractPodScaleDown interprets the update of a pod and returns PodRequestScaledDown event if any pod's resource request(s) is scaled down.
func extractPodScaleDown(newPod, oldPod *v1.Pod) ActionType {
	opt := resource.PodResourcesOptions{
		UseStatusResources: utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling),
	}
	newPodRequests := resource.PodRequests(newPod, opt)
	oldPodRequests := resource.PodRequests(oldPod, opt)

	for rName, oldReq := range oldPodRequests {
		newReq, ok := newPodRequests[rName]
		if !ok {
			// The resource request of rName is removed.
			return UpdatePodScaleDown
		}

		if oldReq.MilliValue() > newReq.MilliValue() {
			// The resource request of rName is scaled down.
			return UpdatePodScaleDown
		}
	}

	return none
}

func extractPodLabelsChange(newPod *v1.Pod, oldPod *v1.Pod) ActionType {
	if isLabelChanged(newPod.GetLabels(), oldPod.GetLabels()) {
		return UpdatePodLabel
	}
	return none
}

func extractPodTolerationChange(newPod *v1.Pod, oldPod *v1.Pod) ActionType {
	if len(newPod.Spec.Tolerations) != len(oldPod.Spec.Tolerations) {
		// A Pod got a new toleration.
		// Due to API validation, the user can add, but cannot modify or remove tolerations.
		// So, it's enough to just check the length of tolerations to notice the update.
		// And, any updates in tolerations could make Pod schedulable.
		return UpdatePodToleration
	}

	return none
}

func extractPodSchedulingGateEliminatedChange(newPod *v1.Pod, oldPod *v1.Pod) ActionType {
	if len(newPod.Spec.SchedulingGates) == 0 && len(oldPod.Spec.SchedulingGates) != 0 {
		// A scheduling gate on the pod is completely removed.
		return UpdatePodSchedulingGatesEliminated
	}

	return none
}

func extractPodGeneratedResourceClaimChange(newPod *v1.Pod, oldPod *v1.Pod) ActionType {
	if !resourceclaim.PodStatusEqual(newPod.Status.ResourceClaimStatuses, oldPod.Status.ResourceClaimStatuses) {
		return UpdatePodGeneratedResourceClaim
	}

	return none
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
		if event := fn(newNode, oldNode); event != none {
			events = append(events, ClusterEvent{Resource: Node, ActionType: event})
		}
	}
	return
}

type nodeChangeExtractor func(newNode *v1.Node, oldNode *v1.Node) ActionType

func extractNodeAllocatableChange(newNode *v1.Node, oldNode *v1.Node) ActionType {
	if !equality.Semantic.DeepEqual(oldNode.Status.Allocatable, newNode.Status.Allocatable) {
		return UpdateNodeAllocatable
	}
	return none
}

func extractNodeLabelsChange(newNode *v1.Node, oldNode *v1.Node) ActionType {
	if isLabelChanged(newNode.GetLabels(), oldNode.GetLabels()) {
		return UpdateNodeLabel
	}
	return none
}

func isLabelChanged(newLabels map[string]string, oldLabels map[string]string) bool {
	return !equality.Semantic.DeepEqual(newLabels, oldLabels)
}

func extractNodeTaintsChange(newNode *v1.Node, oldNode *v1.Node) ActionType {
	if !equality.Semantic.DeepEqual(newNode.Spec.Taints, oldNode.Spec.Taints) {
		return UpdateNodeTaint
	}
	return none
}

func extractNodeConditionsChange(newNode *v1.Node, oldNode *v1.Node) ActionType {
	strip := func(conditions []v1.NodeCondition) map[v1.NodeConditionType]v1.ConditionStatus {
		conditionStatuses := make(map[v1.NodeConditionType]v1.ConditionStatus, len(conditions))
		for i := range conditions {
			conditionStatuses[conditions[i].Type] = conditions[i].Status
		}
		return conditionStatuses
	}
	if !equality.Semantic.DeepEqual(strip(oldNode.Status.Conditions), strip(newNode.Status.Conditions)) {
		return UpdateNodeCondition
	}
	return none
}

func extractNodeSpecUnschedulableChange(newNode *v1.Node, oldNode *v1.Node) ActionType {
	if newNode.Spec.Unschedulable != oldNode.Spec.Unschedulable && !newNode.Spec.Unschedulable {
		// TODO: create UpdateNodeSpecUnschedulable ActionType
		return UpdateNodeTaint
	}
	return none
}

func extractNodeAnnotationsChange(newNode *v1.Node, oldNode *v1.Node) ActionType {
	if !equality.Semantic.DeepEqual(oldNode.GetAnnotations(), newNode.GetAnnotations()) {
		return UpdateNodeAnnotation
	}
	return none
}
