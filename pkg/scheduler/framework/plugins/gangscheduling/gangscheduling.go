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
	schedulingapi "k8s.io/api/scheduling/v1alpha3"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
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
	podGroupLister          schedulinglisters.PodGroupLister
	compositePodGroupLister schedulinglisters.CompositePodGroupLister
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
		podGroupLister:          fh.SharedInformerFactory().Scheduling().V1alpha3().PodGroups().Lister(),
		compositePodGroupLister: compositePodGroupLister,
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
		// PodGroups are immutable, so there's no need to handle PodGroup/Update event.
		{Event: fwk.ClusterEvent{Resource: fwk.PodGroup, ActionType: fwk.Add}, QueueingHintFn: pl.isSchedulableAfterPodGroupAdded},
	}

	if utilfeature.DefaultFeatureGate.Enabled(features.CompositePodGroup) {
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

	if helper.MatchingSchedulingGroup(pod, addedPod) {
		return fwk.Queue, nil
	}

	// Check if they belong to the same CPG hierarchy.
	root1, err := pl.getRootCPGNameForPod(pod)
	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.V(5).Info("pod group not found for target pod, assuming it might belong to the same CPG", "pod", klog.KObj(pod))
			return fwk.Queue, nil
		}
		return fwk.QueueSkip, err
	}
	root2, err := pl.getRootCPGNameForPod(addedPod)
	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.V(5).Info("pod group not found for added pod, assuming it might belong to the same CPG", "addedPod", klog.KObj(addedPod))
			return fwk.Queue, nil
		}
		return fwk.QueueSkip, err
	}
	if root1 != "" && root1 == root2 {
		logger.V(5).Info("another pod was added and it matches the target pod's root CPG, which may make the pod schedulable",
			"pod", klog.KObj(pod), "rootCPG", root1, "addedPod", klog.KObj(addedPod))
		return fwk.Queue, nil
	}

	logger.V(5).Info("another pod was added but it doesn't match the target pod's scheduling group or root CPG",
		"pod", klog.KObj(pod), "schedulingGroup", pod.Spec.SchedulingGroup, "addedPod", klog.KObj(addedPod), "addedPodSchedulingGroup", addedPod.Spec.SchedulingGroup)
	return fwk.QueueSkip, nil
}

func (pl *GangScheduling) isSchedulableAfterPodGroupAdded(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
	_, addedPodGroup, err := util.As[*schedulingapi.PodGroup](oldObj, newObj)
	if err != nil {
		return fwk.Queue, err
	}

	if pod.Spec.SchedulingGroup != nil && pod.Namespace == addedPodGroup.Namespace && *pod.Spec.SchedulingGroup.PodGroupName == addedPodGroup.Name {
		return fwk.Queue, nil
	}

	// Check if they belong to the same CPG hierarchy.
	root1, err := pl.getRootCPGNameForPod(pod)
	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.V(5).Info("pod group not found for target pod, assuming it might belong to the same CPG", "pod", klog.KObj(pod))
			return fwk.Queue, nil
		}
		return fwk.QueueSkip, err
	}
	var root2 string
	if addedPodGroup.Spec.ParentCompositePodGroupName != nil {
		root2, err = pl.getRootCPGName(addedPodGroup.Namespace, *addedPodGroup.Spec.ParentCompositePodGroupName)
		if err != nil {
			if apierrors.IsNotFound(err) {
				logger.V(5).Info("root CPG not found for added pod group, assuming it might belong to the same CPG", "addedPodGroup", klog.KObj(addedPodGroup))
				return fwk.Queue, nil
			}
			return fwk.QueueSkip, err
		}
	}
	if root1 != "" && root1 == root2 {
		logger.V(5).Info("pod group was added and it matches the target pod's root CPG, which may make the pod schedulable",
			"pod", klog.KObj(pod), "rootCPG", root1, "addedPodGroup", klog.KObj(addedPodGroup))
		return fwk.Queue, nil
	}

	logger.V(5).Info("pod group was added but it doesn't match the target pod's scheduling group or root CPG",
		"pod", klog.KObj(pod), "schedulingGroup", pod.Spec.SchedulingGroup, "addedPodGroup", klog.KObj(addedPodGroup))
	return fwk.QueueSkip, nil
}

func (pl *GangScheduling) isSchedulableAfterCompositePodGroupAdded(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
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
			logger.V(5).Info("pod group not found for target pod, assuming it might belong to the same CPG", "pod", klog.KObj(pod))
			return fwk.Queue, nil
		}
		return fwk.QueueSkip, err
	}
	root2, err := pl.getRootCPGName(pod.Namespace, addedCPG.Name)
	if err != nil {
		if apierrors.IsNotFound(err) {
			logger.V(5).Info("root CPG not found for added CPG, assuming it might belong to the same CPG", "addedCPG", klog.KObj(addedCPG))
			return fwk.Queue, nil
		}
		return fwk.QueueSkip, err
	}
	if root1 != "" && root1 == root2 {
		logger.V(5).Info("composite pod group was added and it matches the target pod's root CPG, which may make the pod schedulable",
			"pod", klog.KObj(pod), "rootCPG", root1, "addedCPG", klog.KObj(addedCPG))
		return fwk.Queue, nil
	}

	logger.V(5).Info("composite pod group was added but it doesn't match the target pod's root CPG",
		"pod", klog.KObj(pod), "addedCPG", klog.KObj(addedCPG))
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

	podGroup, err := pl.podGroupLister.PodGroups(namespace).Get(*schedulingGroup.PodGroupName)
	if err != nil {
		if apierrors.IsNotFound(err) {
			// The pod is unschedulable until its PodGroup object is created.
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("waiting for pods's pod group %q to appear in scheduling queue", *schedulingGroup.PodGroupName))
		}
		klog.FromContext(ctx).Error(err, "Failed to get pod group", "pod", klog.KObj(pod), "schedulingGroup", schedulingGroup)
		return fwk.AsStatus(fmt.Errorf("failed to get pod group %s/%s", namespace, *schedulingGroup.PodGroupName))
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

	podGroupState, err := pl.podGroupManager.PodGroupStates().Get(namespace, *schedulingGroup.PodGroupName)
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

// This is the part I'm not proud of. For every pod in CPG we traverse the entire tree!
// This is highly inefficient. This results in O(n^2) complexity where n is the number of pods in the CPG.
// Let's keep it like this in alpha, in beta we will optimize it somehow :)
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
	cpgState, err := pl.podGroupManager.GetCompositePodGroupState(namespace, cpgName)
	if err != nil {
		return false
	}

	successfulChildren := 0

	// Check child PGs
	for _, pgName := range cpgState.GetChildrenPGs() {
		if pl.isPGReadyForPreEnqueue(namespace, pgName) {
			successfulChildren++
		}
	}

	// Check child CPGs
	for _, childCPGName := range cpgState.GetChildrenCPGs() {
		if pl.isCPGTreeReady(namespace, childCPGName) {
			successfulChildren++
		}
	}

	return successfulChildren >= cpgState.GetMinGroupCount()
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

	pgState, err := pl.podGroupManager.PodGroupStates().Get(namespace, pgName)
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

	podGroup, err := pl.podGroupLister.PodGroups(namespace).Get(*schedulingGroup.PodGroupName)
	if err != nil {
		// It's likely that the pod group was removed or another error happened.
		// Returning error to retry the Pod when the pod group is added again.
		return fwk.AsStatus(fmt.Errorf("failed to get podGroup %s/%s: %w", namespace, *schedulingGroup.PodGroupName, err)), 0
	}

	if podGroup.Spec.ParentCompositePodGroupName != nil {
		// We are skipping permit for pod groups that have a parent in CPG hierarchy.
		return nil, 0
	}

	policy := podGroup.Spec.SchedulingPolicy
	// This plugin only cares about pods with a Gang scheduling policy.
	if policy.Gang == nil {
		return nil, 0
	}

	podGroupStateLister := pl.podGroupManager.PodGroupStates()
	podGroupState, err := podGroupStateLister.Get(namespace, *schedulingGroup.PodGroupName)
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

func getPlacementFeasibleState(placementCycleState fwk.PodGroupCycleState) *placementFeasibleState {
	state, err := placementCycleState.Read(placementFeasibleStateKey)
	if err != nil {
		state = &placementFeasibleState{}
		placementCycleState.Write(placementFeasibleStateKey, state)
	}
	return state.(*placementFeasibleState)
}

var PlacementFeasibleResultsKey fwk.StateKey = "gangscheduling.io/placement-feasible-results"

type PlacementFeasibleResults struct {
	Results map[string]bool
}

func (s *PlacementFeasibleResults) Clone() fwk.StateData {
	res := make(map[string]bool, len(s.Results))
	for k, v := range s.Results {
		res[k] = v
	}
	return &PlacementFeasibleResults{Results: res}
}

func getPlacementFeasibleResults(cycleState *framework.CycleState) *PlacementFeasibleResults {
	state, err := cycleState.Read(PlacementFeasibleResultsKey)
	if err != nil {
		state = &PlacementFeasibleResults{Results: make(map[string]bool)}
		cycleState.Write(PlacementFeasibleResultsKey, state)
	}
	return state.(*PlacementFeasibleResults)
}

// PlacementFeasible is responsible for enforcing the gang's MinCount constraint in the pod group scheduling cycle.
// The function will only return success once the gang's MinCount is satisfied or if the pod group is not using gang scheduling policy.
// In case there are not enough remaining pods to satisfy the gang's MinCount, it returns UnschedulableAndUnresolvable which will terminate the pod group scheduling cycle early.
func (pl *GangScheduling) PlacementFeasible(ctx context.Context, placementCycleState fwk.PodGroupCycleState, entity framework.QueuedEntityInfo) *fwk.Status {
	switch t := entity.(type) {
	case *framework.QueuedPodGroupInfo:
		pg, err := pl.podGroupLister.PodGroups(t.GetNamespace()).Get(t.GetName())
		if err != nil {
			return fwk.AsStatus(fmt.Errorf("failed to get podGroup %s to compute gang feasibility: %w", klog.KObj(t), err))
		}

		gangPolicy := pg.Spec.SchedulingPolicy.Gang
		minCount := 1
		if gangPolicy != nil {
			minCount = int(gangPolicy.MinCount)
		}

		podGroupState, err := pl.snapshotLister.PodGroupStates().Get(t.GetNamespace(), t.GetName())
		if err != nil {
			return fwk.AsStatus(fmt.Errorf("failed to get podGroup state for podGroup %s to compute gang feasibility: %w", klog.KObj(pg), err))
		}

		// We need to keep track of how many pods have already been evaluated in the current PodGroup scheduling cycle.
		pgState := getPlacementFeasibleState(placementCycleState)
		pgState.evaluated++

		// remaining is the number of unscheduled pods that haven't been evaluated yet in the current PodGroup scheduling cycle.
		remaining := len(t.GetUnscheduledPods()) - pgState.evaluated

		// scheduled includes the pods that are assigned or assumed in the current PodGroup scheduling cycle.
		scheduled := podGroupState.ScheduledPodsCount()

		if remaining+scheduled < minCount {
			// minCount can't be satisfied because there are not enough remaining pods.
			return fwk.NewStatus(fwk.UnschedulableAndUnresolvable, fmt.Sprintf("minCount (%d) cannot be satisfied: %d scheduled, %d remaining", minCount, scheduled, remaining))
		}

		if scheduled < minCount {
			// minCount might be satisfied once more remaining pods are evaluated.
			return fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("minCount (%d) is not yet satisfied: %d scheduled, %d remaining", minCount, scheduled, remaining))
		}

		return nil

	case *framework.QueuedCompositePodGroupInfo:
		if pl.compositePodGroupLister == nil {
			return fwk.AsStatus(fmt.Errorf("CompositePodGroup lister is not available"))
		}
		cpgSpec, err := pl.compositePodGroupLister.CompositePodGroups(t.GetNamespace()).Get(t.GetName())
		if err != nil {
			return fwk.AsStatus(fmt.Errorf("failed to get composite pod group %s: %w", klog.KObj(t), err))
		}

		cycleState, ok := placementCycleState.(*framework.CycleState)
		if !ok {
			return fwk.NewStatus(fwk.Error, "failed to cast placementCycleState to *CycleState")
		}

		results := getPlacementFeasibleResults(cycleState)
		scheduledChildren := 0

		for _, childCPG := range t.ChildrenCPGs {
			key := fmt.Sprintf("%s/%s", childCPG.Namespace, childCPG.Name)
			if results.Results[key] {
				scheduledChildren++
			}
		}
		for _, pgInfo := range t.ChildrenPGs {
			// Check the number of scheduled pods of the PG in cache - when we do not have updates to PG we don't even run PlacementFeasible on it.
			// The best way is to check cache
			pgState, err := pl.snapshotLister.PodGroupStates().Get(pgInfo.Namespace, pgInfo.Name)
			if err == nil && pgState.ScheduledPodsCount() >= int(pgInfo.MinCount) {
				scheduledChildren++
			}
		}

		gangPolicy := cpgSpec.Spec.SchedulingPolicy.Gang
		if gangPolicy == nil {
			if scheduledChildren < 1 {
				return fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("Basic CPG %s/%s: at least 1 child must be scheduled, got 0", t.GetNamespace(), t.GetName()))
			}
			results.Results[fmt.Sprintf("%s/%s", t.GetNamespace(), t.GetName())] = true
			return nil
		}
		minGroupCount := int(gangPolicy.MinGroupCount)

		if scheduledChildren < minGroupCount {
			return fwk.NewStatus(fwk.Unschedulable, fmt.Sprintf("CPG %s/%s: minGroupCount %d not met, got %d", t.GetNamespace(), t.GetName(), minGroupCount, scheduledChildren))
		}

		results.Results[fmt.Sprintf("%s/%s", t.GetNamespace(), t.GetName())] = true
		return nil

	default:
		return fwk.NewStatus(fwk.Error, "unupported queuing type")
	}
}
