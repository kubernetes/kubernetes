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
	"time"

	v1 "k8s.io/api/core/v1"
	schedulingv1alpha3 "k8s.io/api/scheduling/v1alpha3"
	schedulingapi "k8s.io/api/scheduling/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/klog/v2"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
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
	handle                     fwk.Handle
	podGroupManager            fwk.PodGroupManager
	snapshotLister             fwk.SharedLister
	isCompositePodGroupEnabled bool
}

var _ fwk.EnqueueExtensions = &GangScheduling{}
var _ fwk.PreEnqueuePlugin = &GangScheduling{}
var _ fwk.PermitPlugin = &GangScheduling{}
var _ framework.PlacementFeasiblePlugin = &GangScheduling{}

// New initializes a new plugin and returns it.
func New(_ context.Context, _ runtime.Object, fh fwk.Handle, fts feature.Features) (fwk.Plugin, error) {
	return &GangScheduling{
		handle:                     fh,
		podGroupManager:            fh.PodGroupManager(),
		snapshotLister:             fh.SnapshotSharedLister(),
		isCompositePodGroupEnabled: fts.EnableCompositePodGroup,
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

	if pl.isCompositePodGroupEnabled {
		// A CompositePodGroup being added can be making a waiting gang schedulable.
		events = append(events, fwk.ClusterEventWithHint{Event: fwk.ClusterEvent{Resource: fwk.CompositePodGroup, ActionType: fwk.Add}, QueueingHintFn: pl.isSchedulableAfterCompositePodGroupAdded})
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

	if pod.Spec.SchedulingGroup == nil || addedPod.Spec.SchedulingGroup == nil {
		return fwk.QueueSkip, nil
	}
	addedPodGroupKey := fwk.PodGroupKey(addedPod.Namespace, *addedPod.Spec.SchedulingGroup.PodGroupName)
	if pl.areSameHierarchy(logger, pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName, addedPodGroupKey) {
		return fwk.Queue, nil
	}
	return fwk.QueueSkip, nil
}

// isSchedulableAfterPodGroupUpdated triggers re-enqueueing of the group's pods if the minCount requirement has decreased.
func (pl *GangScheduling) isSchedulableAfterPodGroupUpdated(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	oldPodGroup, newPodGroup, err := util.As[*schedulingapi.PodGroup](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if pod.Spec.SchedulingGroup == nil {
		return fwk.QueueSkip, nil
	}
	updatedPodGroupKey := fwk.PodGroupKey(newPodGroup.Namespace, newPodGroup.Name)
	if !pl.areSameHierarchy(logger, pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName, updatedPodGroupKey) {
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

	if pod.Spec.SchedulingGroup == nil {
		return fwk.QueueSkip, nil
	}
	addedPodGroupKey := fwk.PodGroupKey(addedPodGroup.Namespace, addedPodGroup.Name)
	if pl.areSameHierarchy(logger, pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName, addedPodGroupKey) {
		return fwk.Queue, nil
	}
	return fwk.QueueSkip, nil
}

func (pl *GangScheduling) isSchedulableAfterCompositePodGroupAdded(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, addedCPG, err := util.As[*schedulingv1alpha3.CompositePodGroup](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if pod.Spec.SchedulingGroup == nil {
		return fwk.QueueSkip, nil
	}

	compositePodGroupKey := fwk.CompositePodGroupKey(addedCPG.Namespace, addedCPG.Name)
	if pl.areSameHierarchy(logger, pod.Namespace, *pod.Spec.SchedulingGroup.PodGroupName, compositePodGroupKey) {
		return fwk.Queue, nil
	}
	return fwk.QueueSkip, nil
}

// areSameHierarchy checks if the given pod group is in the same hierarchy as the target key.
// It retrieves the root composite pod group for both keys and compares them.
func (pl *GangScheduling) areSameHierarchy(logger klog.Logger, namespace, podGroupName string, targetKey fwk.EntityKey) bool {
	if !pl.isCompositePodGroupEnabled {
		return fwk.PodGroupKey(namespace, podGroupName) == targetKey
	}

	podGroupKey := fwk.PodGroupKey(namespace, podGroupName)
	root1, ok1, err := pl.podGroupManager.GetRootKeyForGroup(podGroupKey)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to get root key for key", "key", podGroupKey)
		return false
	}
	if !ok1 {
		return false
	}
	root2, ok2, err := pl.podGroupManager.GetRootKeyForGroup(targetKey)
	if err != nil {
		utilruntime.HandleErrorWithLogger(logger, err, "Failed to get root key for key", "key", targetKey)
		return false
	}
	if !ok2 {
		return false
	}
	return root1 == root2
}

// PreEnqueue checks if the pod belongs to a gang and, if so, whether the gang has met its MinCount of available pods.
// If not, the pod is rejected until more pods arrive.
func (pl *GangScheduling) PreEnqueue(ctx context.Context, pod *v1.Pod) *fwk.Status {
	if !pl.isCompositePodGroupEnabled {
		return pl.preEnqueueHierarchiesDisabled(pod)
	}
	return pl.preEnqueueWithHierarchies(pod)
}

// preEnqueueWithHierarchies checks if the pod belongs to a gang and, if so, whether the gang has met its MinCount of available pods.
// If not, the pod is rejected until more pods arrive.
func (pl *GangScheduling) preEnqueueWithHierarchies(pod *v1.Pod) *fwk.Status {
	if pod.Spec.SchedulingGroup == nil {
		return nil
	}

	snapshot, err := pl.handle.PodGroupManager().BuildHierarchySnapshotFromPod(pod)
	if err != nil {
		// Could not build snapshot (e.g. root PG not found). Treat as unschedulable.
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("failed to build hierarchy snapshot: %v", err))
	}

	namespace := pod.Namespace
	schedulingGroup := pod.Spec.SchedulingGroup

	podGroup, err := snapshot.PodGroups().Get(namespace, *schedulingGroup.PodGroupName)
	if err != nil {
		// The pod is unschedulable until its PodGroup object is created.
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("waiting for pods's pod group %q to appear in scheduling queue", *schedulingGroup.PodGroupName))
	}

	if podGroup.Spec.ParentCompositePodGroupName == nil || !pl.isCompositePodGroupEnabled {
		if pl.isPGReady(snapshot, namespace, podGroup.Name, func(s fwk.PodGroupState) int { return s.AllPodsCount() }) {
			return nil
		}
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for minCount pods from a gang to appear in scheduling queue")
	}

	return pl.checkCPGHierarchyReadiness(snapshot, namespace, *podGroup.Spec.ParentCompositePodGroupName, func(s fwk.PodGroupState) int { return s.AllPodsCount() })
}

// preEnqueueHierarchiesDisabled checks if the pod belongs to a gang and, if so, whether the gang has met its MinCount of available pods.
// If not, the pod is rejected until more pods arrive.
// The function should be used only when CompositePodGroup feature gate is disabled.
func (pl *GangScheduling) preEnqueueHierarchiesDisabled(pod *v1.Pod) *fwk.Status {
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
		return nil
	}

	podGroupState, err := pl.podGroupManager.PodGroupStates().Get(namespace, *schedulingGroup.PodGroupName)
	if err != nil {
		return fwk.AsStatus(err)
	}
	allPodsCount := podGroupState.AllPodsCount()
	if allPodsCount < int(policy.Gang.MinCount) {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, "waiting for minCount pods from a gang to appear in scheduling queue")
	}

	// The quorum is met, allow the pod to enter the scheduling queue.
	return nil
}

// checkCPGHierarchyReadiness checks if the Composite Pod Group hierarchy is ready for scheduling.
// It first retrieves the root composite pod group and then recursively traverses the entire Composite Pod Group hierarchy
// to determine if the hierarchy is ready for scheduling.
func (pl *GangScheduling) checkCPGHierarchyReadiness(snapshot fwk.PodGroupManager, namespace, startCPGName string, readinessCountFn func(fwk.PodGroupState) int) *fwk.Status {
	cpgKey := fwk.CompositePodGroupKey(namespace, startCPGName)
	rootKey, ok, err := snapshot.GetRootKeyForGroup(cpgKey)
	if err != nil {
		return fwk.AsStatus(err)
	}
	if !ok {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("failed to build hierarchy snapshot: composite pod group object not found in state for %s", cpgKey.String()))
	}

	if !pl.isCPGTreeReady(snapshot, rootKey.Namespace, rootKey.Name, readinessCountFn) {
		return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("waiting for composite pod group %q tree to meet quorum", rootKey.Name))
	}
	return nil
}

func (pl *GangScheduling) isCPGTreeReady(snapshot fwk.PodGroupManager, namespace, cpgName string, readinessCountFn func(fwk.PodGroupState) int) bool {
	cpgState, err := snapshot.CompositePodGroupStates().Get(namespace, cpgName)
	if err != nil {
		return false
	}

	cpgSpec, err := snapshot.CompositePodGroups().Get(namespace, cpgName)
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
		childType, _, childName := childKey.Type, childKey.Namespace, childKey.Name
		if childType == fwk.CompositePodGroupKeyType {
			if pl.isCPGTreeReady(snapshot, namespace, childName, readinessCountFn) {
				successfulChildren++
			}
		} else {
			if pl.isPGReady(snapshot, namespace, childName, readinessCountFn) {
				successfulChildren++
			}
		}
	}

	return successfulChildren >= minGroupCount
}

func (pl *GangScheduling) isPGReady(snapshot fwk.PodGroupManager, namespace, pgName string, readinessCountFn func(fwk.PodGroupState) int) bool {
	pg, err := snapshot.PodGroups().Get(namespace, pgName)
	if err != nil {
		return false
	}

	minCount := 1
	if pg.Spec.SchedulingPolicy.Gang != nil {
		minCount = int(pg.Spec.SchedulingPolicy.Gang.MinCount)
	}

	pgState, err := snapshot.PodGroupStates().Get(namespace, pgName)
	if err != nil {
		return false
	}

	return readinessCountFn(pgState) >= minCount
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

	var podGroup *schedulingapi.PodGroup
	var err error
	var snapshot fwk.PodGroupManager

	if pl.isCompositePodGroupEnabled {
		snapshot, err = pl.podGroupManager.BuildHierarchySnapshotFromPod(pod)
		if err != nil {
			return fwk.AsStatus(fmt.Errorf("failed to build hierarchy snapshot: %w", err)), 0
		}
		podGroup, err = snapshot.PodGroups().Get(namespace, *schedulingGroup.PodGroupName)
	} else {
		podGroup, err = pl.snapshotLister.PodGroups().Get(namespace, *schedulingGroup.PodGroupName)
		snapshot = pl.podGroupManager
	}

	if err != nil {
		return fwk.AsStatus(fmt.Errorf("failed to get podGroup %s/%s: %w", namespace, *schedulingGroup.PodGroupName, err)), 0
	}

	if pl.isCompositePodGroupEnabled && podGroup.Spec.ParentCompositePodGroupName != nil {
		return pl.permitPodForHierarchy(logger, snapshot, pod, namespace, *podGroup.Spec.ParentCompositePodGroupName)
	}

	podGroupState, err := snapshot.PodGroupStates().Get(namespace, podGroup.Name)
	if err != nil {
		return fwk.AsStatus(err), 0
	}

	return pl.permitPodGroup(logger, pod, podGroup, podGroupState)
}

func (pl *GangScheduling) permitPodGroup(logger klog.Logger, pod *v1.Pod, podGroup *schedulingapi.PodGroup, podGroupState fwk.PodGroupState) (*fwk.Status, time.Duration) {
	if podGroup.Spec.SchedulingPolicy.Gang == nil {
		return nil, 0
	}

	scheduledPodsCount := podGroupState.ScheduledPodsCount()
	if scheduledPodsCount < int(podGroup.Spec.SchedulingPolicy.Gang.MinCount) {
		// Activate unscheduled pods from this pod group in case they were waiting for this pod to be scheduled.
		unscheduledPods := podGroupState.UnscheduledPods()
		pl.handle.Activate(logger, unscheduledPods)
		logger.V(4).Info("Quorum is not met for a gang. Waiting for another pod to allow", "pod", klog.KObj(pod), "schedulingGroup", podGroup.Name, "activatedPods", len(unscheduledPods))
		return fwk.NewStatus(fwk.Wait, "waiting for minCount pods from a gang to be scheduled"), permitTimeoutDuration
	}

	assumedPods := podGroupState.AssumedPods()
	logger.V(4).Info("Quorum is met for a gang. Allowing other pods from a gang waiting on permit", "pod", klog.KObj(pod), "schedulingGroup", podGroup.Name, "allowedPods", len(assumedPods))

	// The quorum is met. Allow this pod and signal all other waiting pods from the same gang to proceed.
	for podUID := range assumedPods {
		waitingPod := pl.handle.GetWaitingPod(podUID)
		if waitingPod != nil {
			waitingPod.Allow(Name)
		}
	}

	return nil, 0
}

func (pl *GangScheduling) permitPodForHierarchy(logger klog.Logger, snapshot fwk.PodGroupManager, pod *v1.Pod, namespace string, startCPGName string) (*fwk.Status, time.Duration) {
	cpgKey := fwk.CompositePodGroupKey(namespace, startCPGName)
	rootKey, ok, err := snapshot.GetRootKeyForGroup(cpgKey)
	if err != nil {
		return fwk.AsStatus(err), 0
	}
	if !ok {
		return fwk.AsStatus(fmt.Errorf("failed to build hierarchy snapshot: composite pod group object not found in state for %s", cpgKey.String())), 0
	}

	if !pl.isCPGTreeReady(snapshot, rootKey.Namespace, rootKey.Name, func(s fwk.PodGroupState) int { return s.ScheduledPodsCount() }) {
		pl.activateUnscheduledPodsInHierarchy(logger, snapshot, rootKey.Namespace, rootKey.Name)
		logger.V(4).Info("Quorum is not met for a CPG hierarchy. Waiting for another pod to allow", "pod", klog.KObj(pod), "rootCPG", rootKey.Name)
		return fwk.NewStatus(fwk.Wait, fmt.Sprintf("waiting for composite pod group %q tree to meet quorum", rootKey.Name)), permitTimeoutDuration
	}

	pl.allowAssumedPodsInHierarchy(snapshot, rootKey.Namespace, rootKey.Name)
	logger.V(4).Info("Quorum is met for a CPG hierarchy. Allowing other pods from the hierarchy waiting on permit", "pod", klog.KObj(pod), "rootCPG", rootKey.Name)
	return nil, 0
}

func (pl *GangScheduling) activateUnscheduledPodsInHierarchy(logger klog.Logger, snapshot fwk.PodGroupManager, namespace, cpgName string) {
	cpgState, err := snapshot.CompositePodGroupStates().Get(namespace, cpgName)
	if err != nil {
		return
	}
	for _, childKey := range cpgState.GetChildren() {
		childType, _, childName := childKey.Type, childKey.Namespace, childKey.Name
		if childType == fwk.CompositePodGroupKeyType {
			pl.activateUnscheduledPodsInHierarchy(logger, snapshot, namespace, childName)
		} else {
			if pgState, err := snapshot.PodGroupStates().Get(namespace, childName); err == nil {
				unscheduledPods := pgState.UnscheduledPods()
				pl.handle.Activate(logger, unscheduledPods)
			}
		}
	}
}

func (pl *GangScheduling) allowAssumedPodsInHierarchy(snapshot fwk.PodGroupManager, namespace, cpgName string) {
	cpgState, err := snapshot.CompositePodGroupStates().Get(namespace, cpgName)
	if err != nil {
		return
	}
	for _, childKey := range cpgState.GetChildren() {
		childType, _, childName := childKey.Type, childKey.Namespace, childKey.Name
		if childType == fwk.CompositePodGroupKeyType {
			pl.allowAssumedPodsInHierarchy(snapshot, namespace, childName)
			continue
		}
		if pgState, err := snapshot.PodGroupStates().Get(namespace, childName); err == nil {
			assumedPods := pgState.AssumedPods()
			for podUID := range assumedPods {
				waitingPod := pl.handle.GetWaitingPod(podUID)
				if waitingPod != nil {
					waitingPod.Allow(Name)
				}
			}
		}
	}
}

// PlacementFeasible is responsible for enforcing the gang's MinCount constraint in the pod group scheduling cycle.
// The function will only return success once the gang's MinCount is satisfied or if the pod group is not using gang scheduling policy.
// In case there are not enough remaining pods to satisfy the gang's MinCount, it returns Unschedulable which will terminate the pod group scheduling cycle early.
func (pl *GangScheduling) PlacementFeasible(ctx context.Context, placementCycleState fwk.PlacementCycleState, podGroupInfo fwk.PodGroupInfo, args framework.PlacementProgress) *fwk.Status {
	minCount := getMinCount(podGroupInfo)
	remaining := args.Remaining
	scheduled := args.Scheduled

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

// getMinCount returns the min count for a pod group or a composite pod group. For basic groups it returns 1.
func getMinCount(podGroupInfo fwk.PodGroupInfo) int {
	if podGroupInfo.GetType() == fwk.CompositePodGroupKeyType {
		if podGroupInfo.GetCompositePodGroup().Spec.SchedulingPolicy.Gang == nil {
			return 1
		}
		return int(podGroupInfo.GetCompositePodGroup().Spec.SchedulingPolicy.Gang.MinGroupCount)
	}
	pg := podGroupInfo.GetPodGroup()
	if pg.Spec.SchedulingPolicy.Gang == nil {
		return 1
	}
	return int(pg.Spec.SchedulingPolicy.Gang.MinCount)
}
