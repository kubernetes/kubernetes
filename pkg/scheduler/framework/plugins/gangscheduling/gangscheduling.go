/*
Copyright 2025 The Kubernetes Authors.

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

package gangscheduling

import (
	"context"
	"fmt"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	"k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	schedulinglisters "k8s.io/client-go/listers/scheduling/v1alpha3"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/helper"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/names"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

const (
	// Name is the name of the plugin used in the plugin registry and configurations.
	Name = names.GangScheduling
	// permitTimeoutDuration defines how long the gang pods should
	// wait at the permit stage for a quorum before being rejected.
	permitTimeoutDuration = 5 * time.Minute
)

// GangScheduling is a plugin that enforces "all-or-nothing" scheduling for pods
// belonging to a PodGroup with a Gang scheduling policy.
type GangScheduling struct {
	handle                  fwk.Handle
	compositePodGroupLister schedulinglisters.CompositePodGroupLister
	podGroupLister          schedulinglisters.PodGroupLister
	podGroupManager         fwk.PodGroupManager
	snapshotLister          fwk.SharedLister
}

var _ fwk.EnqueueExtensions = &GangScheduling{}
var _ fwk.PreEnqueuePlugin = &GangScheduling{}
var _ fwk.PermitPlugin = &GangScheduling{}
var _ framework.PlacementFeasiblePlugin = &GangScheduling{}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, fh fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	var compositePodGroupLister schedulinglisters.CompositePodGroupLister
	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
		compositePodGroupLister = fh.SharedInformerFactory().Scheduling().V1alpha3().CompositePodGroups().Lister()
	}
	return &GangScheduling{
		handle:                  fh,
		compositePodGroupLister: compositePodGroupLister,
		podGroupLister:          fh.SharedInformerFactory().Scheduling().V1alpha3().PodGroups().Lister(),
		podGroupManager:         fh.PodGroupManager(),
		snapshotLister:          fh.SnapshotSharedLister(),
	}, nil
}

// Name returns name of the plugin.
func (pl *GangScheduling) Name() string {
	return Name
}

// EventsToRegister returns the possible events that may make a Pod failed by this plugin schedulable.
func (pl *GangScheduling) EventsToRegister(_ context.Context) ([]fwk.ClusterEventWithHint, error) {
	events := []fwk.ClusterEventWithHint{
		// A new pod (either unscheduled or pre-bound) being added might be the one that completes a gang, meeting its MinCount requirement.
		// PodSchedulingGroup field is immutable, so there is no need to subscribe on Pod/Update event.
		{Event: fwk.ClusterEvent{Resource: fwk.UnscheduledPod, ActionType: fwk.Add}, QueueingHintFn: pl.isSchedulableAfterPodAdded},
		{Event: fwk.ClusterEvent{Resource: fwk.AssignedPod, ActionType: fwk.Add}, QueueingHintFn: pl.isSchedulableAfterPodAdded},
		// A PodGroup being added can be making a waiting gang schedulable.
		{Event: fwk.ClusterEvent{Resource: fwk.PodGroup, ActionType: fwk.Add}, QueueingHintFn: pl.isSchedulableAfterPodGroupAdded},
		// A PodGroup update to MinCount may make it schedulable
		{Event: fwk.ClusterEvent{Resource: fwk.PodGroup, ActionType: fwk.Update}, QueueingHintFn: pl.isSchedulableAfterPodGroupUpdated},
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
		// A CompositePodGroup being added can be making a waiting gang schedulable.
		events = append(events, fwk.ClusterEventWithHint{Event: fwk.ClusterEvent{Resource: fwk.CompositePodGroup, ActionType: fwk.Add}, QueueingHintFn: pl.isSchedulableAfterCompositePodGroupAdded})
		// A CompositePodGroup update can be making a waiting gang schedulable.
		events = append(events, fwk.ClusterEventWithHint{Event: fwk.ClusterEvent{Resource: fwk.CompositePodGroup, ActionType: fwk.Update}, QueueingHintFn: pl.isSchedulableAfterCompositePodGroupUpdated})
	}

	return events, nil
}

// isSchedulableAfterPodAdded checks whether a newly added pod (either unscheduled or pre-bound)
// could make a previously unschedulable pod schedulable by completing the gang's MinCount.
func (pl *GangScheduling) isSchedulableAfterPodAdded(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, addedPod, err := util.As[*v1.Pod](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if helper.MatchingSchedulingGroup(pod, addedPod) {
		return fwk.Queue, nil
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
		// Check if they belong to the same CPG hierarchy.
		rootPod, err := pl.getRootCPGNameForPod(pod)
		if err != nil {
			if apierrors.IsNotFound(err) {
				logger.V(5).Info("pod group not found for target pod, assuming it might belong to the same CPG", "pod", klog.KObj(pod))
				return fwk.Queue, nil
			}
			return fwk.QueueSkip, err
		}
		rootAddedPod, err := pl.getRootCPGNameForPod(addedPod)
		if err != nil {
			if apierrors.IsNotFound(err) {
				logger.V(5).Info("pod group not found for added pod, assuming it might belong to the same CPG", "addedPod", klog.KObj(addedPod))
				return fwk.Queue, nil
			}
			return fwk.QueueSkip, err
		}
		if rootPod != "" && rootPod == rootAddedPod {
			logger.V(5).Info("another pod was added and it matches the target pod's root CPG, which may make the pod schedulable",
				"pod", klog.KObj(pod), "rootCPG", rootPod, "addedPod", klog.KObj(addedPod))
			return fwk.Queue, nil
		}
	}

	logger.V(5).Info("another pod was added but it doesn't match the target pod's scheduling group or root CPG",
		"pod", klog.KObj(pod), "schedulingGroup", pod.Spec.SchedulingGroup, "addedPod", klog.KObj(addedPod), "addedPodSchedulingGroup", addedPod.Spec.SchedulingGroup)
	return fwk.QueueSkip, nil
}

// isSchedulableAfterPodGroupUpdated triggers re-enqueueing of the group's pods if the minCount requirement has decreased.
func (pl *GangScheduling) isSchedulableAfterPodGroupUpdated(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	oldPodGroup, newPodGroup, err := util.As[*schedulingapi.PodGroup](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	// Only consider updates for the PodGroup this pod belongs to
	if pod.Spec.SchedulingGroup == nil || pod.Namespace != newPodGroup.Namespace || *pod.Spec.SchedulingGroup.PodGroupName != newPodGroup.Name {
		logger.V(5).Info("pod group was updated but it doesn't match the target pod's scheduling group",
			"pod", klog.KObj(pod), "schedulingGroup", pod.Spec.SchedulingGroup, "updatedPodGroup", klog.KObj(newPodGroup))
		return fwk.QueueSkip, nil
	}

	oldPolicy := oldPodGroup.Spec.SchedulingPolicy
	newPolicy := newPodGroup.Spec.SchedulingPolicy

	// Non-gang policies should not be updated.
	if newPolicy.Gang == nil || oldPolicy.Gang == nil {
		logger.V(5).Info("pod group was updated but it's not a gang policy, this is unexpected, enqueuing pod", "pod", klog.KObj(pod), "podGroup", klog.KObj(newPodGroup))
		return fwk.Queue, nil
	}

	// If the gang scheduling policy minCount did not decrease, it will not make the waiting pods schedulable.
	if newPolicy.Gang.MinCount >= oldPolicy.Gang.MinCount {
		logger.V(5).Info("PodGroup minCount did not decrease, skipping", "pod", klog.KObj(pod), "podGroup", klog.KObj(newPodGroup), "oldMinCount", oldPolicy.Gang.MinCount, "newMinCount", newPolicy.Gang.MinCount)
		return fwk.QueueSkip, nil
	}

	logger.V(5).Info("pod group was updated and minCount decreased, enqueuing pod", "pod", klog.KObj(pod), "podGroup", klog.KObj(newPodGroup), "oldMinCount", oldPolicy.Gang.MinCount, "newMinCount", newPolicy.Gang.MinCount)
	return fwk.Queue, nil
}

func (pl *GangScheduling) isSchedulableAfterPodGroupAdded(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, addedPodGroup, err := util.As[*schedulingapi.PodGroup](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if pod.Spec.SchedulingGroup != nil && pod.Namespace == addedPodGroup.Namespace && *pod.Spec.SchedulingGroup.PodGroupName == addedPodGroup.Name {
		return fwk.Queue, nil
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
		// Check if they belong to the same CPG hierarchy.
		rootPod, err := pl.getRootCPGNameForPod(pod)
		if err != nil {
			if apierrors.IsNotFound(err) {
				logger.V(5).Info("pod group not found for target pod, assuming it might belong to the same CPG", "pod", klog.KObj(pod))
				return fwk.Queue, nil
			}
			return fwk.QueueSkip, err
		}
		var rootAddedPodGroup string
		if addedPodGroup.Spec.ParentCompositePodGroupName != nil {
			rootAddedPodGroup, err = pl.getRootCPGName(addedPodGroup.Namespace, *addedPodGroup.Spec.ParentCompositePodGroupName)
			if err != nil {
				if apierrors.IsNotFound(err) {
					logger.V(5).Info("root CPG not found for added pod group, assuming it might belong to the same CPG", "addedPodGroup", klog.KObj(addedPodGroup))
					return fwk.Queue, nil
				}
				return fwk.QueueSkip, err
			}
		}
		if rootPod != "" && rootPod == rootAddedPodGroup {
			logger.V(5).Info("pod group was added and it matches the target pod's root CPG, which may make the pod schedulable",
				"pod", klog.KObj(pod), "rootCPG", rootPod, "addedPodGroup", klog.KObj(addedPodGroup))
			return fwk.Queue, nil
		}
	}

	logger.V(5).Info("pod group was added but it doesn't match the target pod's scheduling group or root CPG",
		"pod", klog.KObj(pod), "schedulingGroup", pod.Spec.SchedulingGroup, "addedPodGroup", klog.KObj(addedPodGroup))
	return fwk.QueueSkip, nil
}

func (pl *GangScheduling) isSchedulableAfterCompositePodGroupAdded(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
		return fwk.QueueSkip, nil
	}

	_, addedCPG, err := util.As[*schedulingapi.CompositePodGroup](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if pod.Spec.SchedulingGroup == nil {
		return fwk.QueueSkip, nil
	}

	root1, err := pl.getRootCPGNameForPod(pod)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return fwk.Queue, nil
		}
		return fwk.QueueSkip, err
	}
	root2, err := pl.getRootCPGName(pod.Namespace, addedCPG.Name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return fwk.Queue, nil
		}
		return fwk.QueueSkip, err
	}
	if root1 != "" && root1 == root2 {
		return fwk.Queue, nil
	}

	return fwk.QueueSkip, nil
}

// isSchedulableAfterCompositePodGroupUpdated triggers re-enqueueing of the group's pods if the minGroupCount requirement has decreased.
func (pl *GangScheduling) isSchedulableAfterCompositePodGroupUpdated(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	if !utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
		return fwk.QueueSkip, nil
	}

	oldCPG, newCPG, err := util.As[*schedulingapi.CompositePodGroup](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if pod.Spec.SchedulingGroup == nil {
		return fwk.QueueSkip, nil
	}

	root1, err := pl.getRootCPGNameForPod(pod)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return fwk.Queue, nil
		}
		return fwk.QueueSkip, err
	}
	root2, err := pl.getRootCPGName(pod.Namespace, newCPG.Name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return fwk.Queue, nil
		}
		return fwk.QueueSkip, err
	}
	if root1 != "" && root1 == root2 {
		oldPolicy := oldCPG.Spec.SchedulingPolicy
		newPolicy := newCPG.Spec.SchedulingPolicy

		if newPolicy.Gang == nil || oldPolicy.Gang == nil {
			return fwk.Queue, nil
		}

		if newPolicy.Gang.MinGroupCount >= oldPolicy.Gang.MinGroupCount {
			return fwk.QueueSkip, nil
		}

		return fwk.Queue, nil
	}
	return fwk.QueueSkip, nil
}

// PreEnqueue checks if the pod belongs to a gang and, if so, whether the gang has met its MinCount of available pods.
// If not, the pod is rejected until more pods arrive.
func (pl *GangScheduling) PreEnqueue(ctx context.Context, pod *v1.Pod) *fwk.Status {
	if pod.Spec.SchedulingGroup == nil {
		return nil
	}

	namespace := pod.Namespace
	schedulingGroup := pod.Spec.SchedulingGroup

	podGroup, err := pl.podGroupManager.PodGroups().Get(namespace, *schedulingGroup.PodGroupName)
	if err != nil {
		// The pod is unschedulable until its PodGroup object is created.
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("waiting for pods's pod group %q to appear in scheduling queue", *schedulingGroup.PodGroupName))
	}

	policy := podGroup.Spec.SchedulingPolicy
	// This plugin only cares about pods with a Gang scheduling policy.
	if policy.Gang == nil {
		// But if the basic PodGroup is a member of a CPG hierarchy, we still need to check if the root CPG is ready.
		if podGroup.Spec.ParentCompositePodGroupName == nil {
			return nil
		}
		return pl.checkCPGHierarchyReadiness(namespace, *podGroup.Spec.ParentCompositePodGroupName)
	}

	podGroupState, err := pl.podGroupManager.PodGroupStates().Get(framework.PodGroupKeyType, namespace, *schedulingGroup.PodGroupName)
	if err != nil {
		return fwk.AsStatus(err)
	}
	allPodsCount := podGroupState.AllPodsCount()

	// Standalone pod group (no CPG parent).
	if podGroup.Spec.ParentCompositePodGroupName == nil {
		if allPodsCount < int(policy.Gang.MinCount) {
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for minCount pods from a gang to appear in scheduling queue")
		}
		return nil
	}

	// Find top level CPG in the hierarchy - it must be ready.
	return pl.checkCPGHierarchyReadiness(namespace, *podGroup.Spec.ParentCompositePodGroupName)
}

func (pl *GangScheduling) getRootCPGName(namespace, cpgName string) (string, error) {
	if pl.compositePodGroupLister == nil {
		return "", fmt.Errorf("CompositePodGroup lister is not available")
	}
	currentCPGName := cpgName
	for {
		cpgSpec, err := pl.compositePodGroupLister.CompositePodGroups(namespace).Get(currentCPGName)
		if err != nil {
			return "", err
		}
		if cpgSpec.Spec.ParentCompositePodGroupName == nil {
			break
		}
		currentCPGName = *cpgSpec.Spec.ParentCompositePodGroupName
	}
	return currentCPGName, nil
}

func (pl *GangScheduling) getRootCPGNameForPod(pod *v1.Pod) (string, error) {
	if pod.Spec.SchedulingGroup == nil {
		return "", nil
	}
	pg, err := pl.podGroupLister.PodGroups(pod.Namespace).Get(*pod.Spec.SchedulingGroup.PodGroupName)
	if err != nil {
		return "", err
	}
	if pg.Spec.ParentCompositePodGroupName == nil {
		return "", nil
	}
	return pl.getRootCPGName(pod.Namespace, *pg.Spec.ParentCompositePodGroupName)
}

// For every pod in a CPG we traverse the entire tree! This is highly inefficient but good enough for alpha.
// This results in O(#pods * #podgroups) complexity.
func (pl *GangScheduling) checkCPGHierarchyReadiness(namespace, startCPGName string) *fwk.Status {
	rootCPGName, err := pl.getRootCPGName(namespace, startCPGName)
	if err != nil {
		if apierrors.IsNotFound(err) {
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("waiting for composite pod group %q spec to appear", startCPGName))
		}
		return fwk.AsStatus(err)
	}

	if !pl.isCPGTreeReady(namespace, rootCPGName) {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("waiting for composite pod group %q tree to meet quorum", rootCPGName))
	}
	return nil
}

func (pl *GangScheduling) isCPGTreeReady(namespace, cpgName string) bool {
	cpgState, err := pl.podGroupManager.PodGroupStates().Get(framework.CompositePodGroupKeyType, namespace, cpgName)
	if err != nil {
		return false
	}

	cpgSpec, err := pl.compositePodGroupLister.CompositePodGroups(namespace).Get(cpgName)
	if err != nil {
		return false
	}
	minGroupCount := 1
	policy := cpgSpec.Spec.SchedulingPolicy
	if policy.Gang != nil {
		minGroupCount = int(policy.Gang.MinGroupCount)
	}

	successfulChildren := 0
	for _, childKey := range cpgState.GetChildren() {
		childType, _, childName := unpackChildKey(childKey)

		if childType == framework.CompositePodGroupKeyType {
			if pl.isCPGTreeReady(namespace, childName) {
				successfulChildren++
			}
		} else {
			if pl.isPGReadyForPreEnqueue(namespace, childName) {
				successfulChildren++
			}
		}
	}

	return successfulChildren >= minGroupCount
}

func (pl *GangScheduling) isPGReadyForPreEnqueue(namespace, pgName string) bool {
	pg, err := pl.podGroupLister.PodGroups(namespace).Get(pgName)
	if err != nil {
		return false
	}

	minCount := 1
	if pg.Spec.SchedulingPolicy.Gang != nil {
		minCount = int(pg.Spec.SchedulingPolicy.Gang.MinCount)
	}

	pgState, err := pl.podGroupManager.PodGroupStates().Get(framework.PodGroupKeyType, namespace, pgName)
	if err != nil {
		return false
	}

	return pgState.AllPodsCount() >= minCount
}

// Permit forces all pods in a gang to wait at this stage. Once the number of waiting (assumed) pods
// reaches the gang's MinCount, all pods in the gang are permitted to proceed to binding simultaneously.
func (pl *GangScheduling) Permit(ctx context.Context, state fwk.CycleState, pod *v1.Pod, nodeName string) (*fwk.Status, time.Duration) {
	if pod.Spec.SchedulingGroup == nil {
		return nil, 0
	}

	logger := klog.FromContext(ctx)
	namespace := pod.Namespace
	schedulingGroup := pod.Spec.SchedulingGroup

	podGroup, err := pl.snapshotLister.PodGroups().Get(namespace, *schedulingGroup.PodGroupName)
	if err != nil {
		// It's likely that the pod group was removed or another error happened.
		// Returning error to retry the Pod when the pod group is added again.
		return fwk.AsStatus(fmt.Errorf("failed to get podGroup %s/%s from snapshot: %w", namespace, *schedulingGroup.PodGroupName, err)), 0
	}

	// For pods that are part of a composite pod group hierarchy we just skip this plugin.
	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) && podGroup.Spec.ParentCompositePodGroupName != nil {
		return nil, 0
	}

	policy := podGroup.Spec.SchedulingPolicy
	// This plugin only cares about pods with a Gang scheduling policy.
	if policy.Gang == nil {
		return nil, 0
	}

	podGroupStateLister := pl.podGroupManager.PodGroupStates()
	podGroupState, err := podGroupStateLister.Get(framework.PodGroupKeyType, namespace, *schedulingGroup.PodGroupName)
	if err != nil {
		return fwk.AsStatus(err), 0
	}
	scheduledPodsCount := podGroupState.ScheduledPodsCount()
	if scheduledPodsCount < int(policy.Gang.MinCount) {
		// Activate unscheduled pods from this pod group in case they were waiting for this pod to be scheduled.
		unscheduledPods := podGroupState.UnscheduledPods()
		pl.handle.Activate(klog.FromContext(ctx), unscheduledPods)
		logger.V(4).Info("Quorum is not met for a gang. Waiting for another pod to allow", "pod", klog.KObj(pod), "schedulingGroup", schedulingGroup, "activatedPods", len(unscheduledPods))
		return fwk.NewStatus(fwk.Wait, "waiting for minCount pods from a gang to be scheduled"), permitTimeoutDuration
	}

	assumedPods := podGroupState.AssumedPods()
	logger.V(4).Info("Quorum is met for a gang. Allowing other pods from a gang waiting on permit", "pod", klog.KObj(pod), "schedulingGroup", schedulingGroup, "allowedPods", len(assumedPods))

	// The quorum is met. Allow this pod and signal all other waiting pods from the same gang to proceed.
	for podUID := range assumedPods {
		waitingPod := pl.handle.GetWaitingPod(podUID)
		if waitingPod != nil {
			waitingPod.Allow(Name)
		}
	}

	return nil, 0
}

const placementFeasibleStateKey = "PlacementFeasible" + Name

type placementFeasibleState struct {
	evaluated, succeeded int
}

func (s *placementFeasibleState) Clone() fwk.StateData {
	return &placementFeasibleState{
		evaluated: s.evaluated,
		succeeded: s.succeeded,
	}
}

func getPlacementFeasibleState(placementCycleState fwk.PlacementCycleState) *placementFeasibleState {
	state, err := placementCycleState.Read(placementFeasibleStateKey)
	if err != nil {
		state = &placementFeasibleState{}
		placementCycleState.Write(placementFeasibleStateKey, state)
	}
	return state.(*placementFeasibleState)
}

const placementFeasibleStatusesKey = "PlacementFeasible" + Name + "Statuses"

// PlacementFeasibleStatuses holds the status of the PlacementFeasible plugin for each pod group.
type PlacementFeasibleStatuses struct {
	Status map[string]*fwk.Status
}

func (s *PlacementFeasibleStatuses) Clone() fwk.StateData {
	res := make(map[string]*fwk.Status, len(s.Status))
	for k, v := range s.Status {
		res[k] = v
	}
	return &PlacementFeasibleStatuses{Status: res}
}

func getPlacementFeasibleStatuses(podGroupCycleState fwk.PodGroupCycleState) *PlacementFeasibleStatuses {
	state, err := podGroupCycleState.Read(placementFeasibleStatusesKey)
	if err != nil {
		state = &PlacementFeasibleStatuses{Status: make(map[string]*fwk.Status)}
		podGroupCycleState.Write(placementFeasibleStatusesKey, state)
	}
	return state.(*PlacementFeasibleStatuses)
}

// PlacementFeasible is responsible for enforcing the gang's MinCount constraint in the pod group scheduling cycle.
// The function will only return success once the gang's MinCount is satisfied or if the pod group is not using gang scheduling policy.
// In case there are not enough remaining pods to satisfy the gang's MinCount, it returns Unschedulable which will terminate the pod group scheduling cycle early.
func (pl *GangScheduling) PlacementFeasible(ctx context.Context, placementCycleState fwk.PlacementCycleState, podGroupInfo fwk.PodGroupInfo) *fwk.Status {
	statuses := getPlacementFeasibleStatuses(placementCycleState.GetPodGroupSchedulingCycle())
	status := pl.placementFeasible(ctx, placementCycleState, podGroupInfo, statuses)
	statuses.Status[podGroupInfo.GetKey()] = status
	return status
}

func (pl *GangScheduling) placementFeasible(ctx context.Context, placementCycleState fwk.PlacementCycleState, podGroupInfo fwk.PodGroupInfo, statuses *PlacementFeasibleStatuses) *fwk.Status {
	if podGroupInfo.GetType() == framework.PodGroupKeyType {
		return pl.placementFeasibleForPodGroup(ctx, placementCycleState, podGroupInfo)
	}

	cpgState, err := pl.snapshotLister.PodGroupStates().Get(podGroupInfo.GetType(), podGroupInfo.GetNamespace(), podGroupInfo.GetName())
	if err != nil {
		return fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("pod group state for %s %s of type %s", podGroupInfo.GetNamespace(), podGroupInfo.GetName(), podGroupInfo.GetType()))
	}

	scheduled := 0
	for _, childKey := range cpgState.GetChildren() {
		status, exists := statuses.Status[childKey]
		// If the child is not in the status map, it means its PodGroup was already scheduled.
		// This can happen if we partially schedule a CPG and then some more pods arrive.
		if !exists || status.IsSuccess() {
			scheduled++
		}
	}

	cpg, err := pl.compositePodGroupLister.CompositePodGroups(podGroupInfo.GetNamespace()).Get(podGroupInfo.GetName())
	if err != nil {
		return fwk.AsStatus(fmt.Errorf("failed to get composite pod group %s: %w", klog.KObj(podGroupInfo), err))
	}

	gangPolicy := cpg.Spec.SchedulingPolicy.Gang
	minCount := 1
	if gangPolicy != nil {
		minCount = int(gangPolicy.MinGroupCount)
	}

	if scheduled < minCount {
		return fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("minCount (%d) cannot be satisfied: %d scheduled", minCount, scheduled))
	}

	return nil
}

func (pl *GangScheduling) placementFeasibleForPodGroup(ctx context.Context, placementCycleState fwk.PlacementCycleState, podGroupInfo fwk.PodGroupInfo) *fwk.Status {
	pg := podGroupInfo.GetPodGroup()
	if pg.Spec.SchedulingPolicy.Gang == nil {
		return nil
	}
	podGroupState, err := pl.snapshotLister.PodGroupStates().Get(framework.PodGroupKeyType, podGroupInfo.GetNamespace(), podGroupInfo.GetName())
	if err != nil {
		return fwk.AsStatus(fmt.Errorf("failed to get podGroup state for podGroup %s to compute gang feasibility: %w", klog.KObj(pg), err))
	}

	// We need to keep track of how many pods have already been evaluated in the current PodGroup scheduling cycle.
	pgState := getPlacementFeasibleState(placementCycleState)
	pgState.evaluated++

	// remaining is the number of unscheduled pods that haven't been evaluated yet in the current PodGroup scheduling cycle.
	remaining := len(podGroupInfo.GetUnscheduledPods()) - pgState.evaluated

	// scheduled includes the pods that are assigned or assumed in the current PodGroup scheduling cycle.
	scheduled := podGroupState.ScheduledPodsCount()

	gangPolicy := pg.Spec.SchedulingPolicy.Gang
	minCount := 1
	if gangPolicy != nil {
		minCount = int(gangPolicy.MinCount)
	}

	if remaining+scheduled < minCount {
		// minCount can't be satisfied because there are not enough remaining pods.
		return fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("minCount (%d) cannot be satisfied: %d scheduled, %d remaining", minCount, scheduled, remaining))
	}

	if scheduled < minCount {
		// minCount might be satisfied once more remaining pods are evaluated.
		return fwk.NewStatus(fwk.Wait, fmt.Sprintf("minCount (%d) is not yet satisfied: %d scheduled, %d remaining", minCount, scheduled, remaining))
	}

	// minCount is satisfied.
	return nil
}

func unpackChildKey(childKey string) (string, string, string) {
	parts := strings.Split(childKey, "/")
	if len(parts) == 3 {
		return parts[0], parts[1], parts[2]
	}
	return framework.PodGroupKeyType, "", childKey
}
