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

package allocation

import (
	"context"
	"fmt"
	"path/filepath"
	"slices"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/record"
	resourcehelper "k8s.io/component-helpers/resource"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/api/v1/resource"
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	"k8s.io/kubernetes/pkg/kubelet/cm/cpumanager"
	"k8s.io/kubernetes/pkg/kubelet/cm/memorymanager"
	"k8s.io/kubernetes/pkg/kubelet/config"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/status"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

// podStatusManagerStateFile is the file name where status manager stores its state
const (
	allocatedPodsStateFile = "allocated_pods_state"

	initialRetryDelay = 30 * time.Second
	retryDelay        = 3 * time.Minute

	TriggerReasonPodResized  = "pod_resized"
	TriggerReasonPodUpdated  = "pod_updated"
	TriggerReasonPodsAdded   = "pods_added"
	TriggerReasonPodsRemoved = "pods_removed"

	triggerReasonPeriodic = "periodic_retry"
)

// AllocationManager tracks pod resource allocations.
type Manager interface {
	// GetContainerResourceAllocation returns the AllocatedResources value for the container
	GetContainerResourceAllocation(podUID types.UID, containerName string) (v1.ResourceRequirements, bool)

	// GetPodLevelResourceAllocation returns the AllocatedResources value for the container
	GetPodLevelResourceAllocation(podUID types.UID) (*v1.ResourceRequirements, bool)

	// UpdatePodFromAllocation overwrites the pod spec with the allocation.
	// This function does a deep copy only if updates are needed.
	// Returns the updated (or original) pod, and whether there was an allocation stored.
	UpdatePodFromAllocation(pod *v1.Pod) (*v1.Pod, bool)

	// SetAllocatedResources checkpoints the resources allocated to a pod's containers.
	SetAllocatedResources(allocatedPod *v1.Pod) error

	// AddPodAdmitHandlers adds the admit handlers to the allocation manager.
	// TODO: See if we can remove this and just add them in the allocation manager constructor.
	AddPodAdmitHandlers(handlers lifecycle.PodAdmitHandlers)

	// SetContainerRuntime sets the allocation manager's container runtime.
	// TODO: See if we can remove this and just add it in the allocation manager constructor.
	SetContainerRuntime(runtime kubecontainer.Runtime)

	// AddPod checks if a pod can be admitted. If so, it admits the pod and updates the allocation.
	// The function returns a boolean value indicating whether the pod
	// can be admitted, a brief single-word reason and a message explaining why
	// the pod cannot be admitted.
	// allocatedPods should represent the pods that have already been admitted, along with their
	// admitted (allocated) resources.
	AddPod(activePods []*v1.Pod, pod *v1.Pod) (ok bool, reason, message string)

	// RemovePod removes any stored state for the given pod UID.
	RemovePod(uid types.UID)

	// RemoveOrphanedPods removes the stored state for any pods not included in the set of remaining pods.
	RemoveOrphanedPods(remainingPods sets.Set[types.UID])

	// Run starts the allocation manager. This is currently only used to handle periodic retry of
	// pending resizes.
	Run(ctx context.Context)

	// PushPendingResize queues a pod with a pending resize request for later reevaluation.
	PushPendingResize(uid types.UID)

	// HasPendingResizes returns whether there are currently any pending resizes.
	HasPendingResizes() bool

	// RetryPendingResizes retries all pending resizes.
	RetryPendingResizes(trigger string)
}

type manager struct {
	allocated state.State

	admitHandlers           lifecycle.PodAdmitHandlers
	containerRuntime        kubecontainer.Runtime
	statusManager           status.Manager
	sourcesReady            config.SourcesReady
	nodeConfig              cm.NodeConfig
	nodeAllocatableAbsolute v1.ResourceList

	ticker         *time.Ticker
	triggerPodSync func(pod *v1.Pod)
	getActivePods  func() []*v1.Pod
	getPodByUID    func(types.UID) (*v1.Pod, bool)

	allocationMutex        sync.Mutex
	podsWithPendingResizes []types.UID

	recorder record.EventRecorder
}

func NewManager(
	checkpointDirectory string,
	nodeConfig cm.NodeConfig,
	nodeAllocatableAbsolute v1.ResourceList,
	statusManager status.Manager,
	triggerPodSync func(pod *v1.Pod),
	getActivePods func() []*v1.Pod,
	getPodByUID func(types.UID) (*v1.Pod, bool),
	sourcesReady config.SourcesReady,
	recorder record.EventRecorder,
) Manager {
	// Use klog.TODO() because we currently do not have a proper logger to pass in.
	// Replace this with an appropriate logger when refactoring this function to accept a logger parameter.
	logger := klog.TODO()
	return &manager{
		allocated: newStateImpl(logger, checkpointDirectory, allocatedPodsStateFile),

		statusManager:           statusManager,
		admitHandlers:           lifecycle.PodAdmitHandlers{},
		sourcesReady:            sourcesReady,
		nodeConfig:              nodeConfig,
		nodeAllocatableAbsolute: nodeAllocatableAbsolute,

		ticker:         time.NewTicker(initialRetryDelay),
		triggerPodSync: triggerPodSync,
		getActivePods:  getActivePods,
		getPodByUID:    getPodByUID,
		recorder:       recorder,
	}
}

func newStateImpl(logger klog.Logger, checkpointDirectory, checkpointName string) state.State {
	if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		return state.NewNoopStateCheckpoint()
	}

	stateImpl, err := state.NewStateCheckpoint(checkpointDirectory, checkpointName)
	if err != nil {
		// This is a critical, non-recoverable failure.
		logger.Error(err, "Failed to initialize allocation checkpoint manager",
			"checkpointPath", filepath.Join(checkpointDirectory, checkpointName))
		panic(err)
	}

	return stateImpl
}

// NewInMemoryManager returns an allocation manager that doesn't persist state.
// For testing purposes only!
func NewInMemoryManager(
	nodeConfig cm.NodeConfig,
	nodeAllocatableAbsolute v1.ResourceList,
	statusManager status.Manager,
	triggerPodSync func(pod *v1.Pod),
	getActivePods func() []*v1.Pod,
	getPodByUID func(types.UID) (*v1.Pod, bool),
	sourcesReady config.SourcesReady,
	recorder record.EventRecorder,
) Manager {
	return &manager{
		allocated: state.NewStateMemory(nil),

		statusManager:           statusManager,
		admitHandlers:           lifecycle.PodAdmitHandlers{},
		sourcesReady:            sourcesReady,
		nodeConfig:              nodeConfig,
		nodeAllocatableAbsolute: nodeAllocatableAbsolute,

		ticker:         time.NewTicker(initialRetryDelay),
		triggerPodSync: triggerPodSync,
		getActivePods:  getActivePods,
		getPodByUID:    getPodByUID,
		recorder:       recorder,
	}
}

func (m *manager) Run(ctx context.Context) {
	// Start a goroutine to periodically check for pending resizes and process them if needed.
	go func() {
		logger := klog.FromContext(ctx)
		for {
			select {
			case <-m.ticker.C:
				successfulResizes := m.retryPendingResizes(logger, triggerReasonPeriodic)
				for _, po := range successfulResizes {
					logger.Info("Successfully retried resize after timeout", "pod", klog.KObj(po))
				}
			case <-ctx.Done():
				m.ticker.Stop()
				return
			}
		}
	}()
}

func (m *manager) RetryPendingResizes(trigger string) {
	// Use klog.TODO() because we currently do not have a proper logger to pass in.
	// Replace this with an appropriate logger when refactoring this function to accept a logger parameter.
	logger := klog.TODO()
	m.retryPendingResizes(logger, trigger)
}

func (m *manager) retryPendingResizes(logger klog.Logger, trigger string) []*v1.Pod {
	m.allocationMutex.Lock()
	defer m.allocationMutex.Unlock()

	if !m.sourcesReady.AllReady() {
		logger.V(4).Info("Skipping evaluation of pending resizes; sources are not ready")
		m.ticker.Reset(initialRetryDelay)
		return nil
	}

	m.ticker.Reset(retryDelay)

	var newPendingResizes []types.UID
	var successfulResizes []*v1.Pod

	// Retry all pending resizes.
	for _, uid := range m.podsWithPendingResizes {
		pod, found := m.getPodByUID(uid)
		if !found {
			logger.V(4).Info("Pod not found; removing from pending resizes", "podUID", uid)
			continue
		}

		oldResizeStatus := m.statusManager.GetPodResizeConditions(uid)
		isDeferred := m.statusManager.IsPodResizeDeferred(uid)

		resizeAllocated, err := m.handlePodResourcesResize(logger, pod)
		switch {
		case err != nil:
			logger.Error(err, "Failed to handle pod resources resize", "pod", klog.KObj(pod))
			newPendingResizes = append(newPendingResizes, uid)
		case m.statusManager.IsPodResizeDeferred(uid):
			logger.V(4).Info("Pod resize is deferred; will reevaluate later", "pod", klog.KObj(pod))
			newPendingResizes = append(newPendingResizes, uid)
		case m.statusManager.IsPodResizeInfeasible(uid):
			logger.V(4).Info("Pod resize is infeasible", "pod", klog.KObj(pod))
		default:
			logger.V(4).Info("Pod resize successfully allocated", "pod", klog.KObj(pod))
			successfulResizes = append(successfulResizes, pod)
			if isDeferred {
				metrics.PodDeferredAcceptedResizes.WithLabelValues(trigger).Inc()
			}
		}

		// If the pod resize status has changed, we need to update the pod status.
		newResizeStatus := m.statusManager.GetPodResizeConditions(uid)
		if resizeAllocated || !apiequality.Semantic.DeepEqual(oldResizeStatus, newResizeStatus) {
			m.triggerPodSync(pod)
		}
	}

	m.podsWithPendingResizes = newPendingResizes
	return successfulResizes
}

func (m *manager) PushPendingResize(uid types.UID) {
	// Use klog.TODO() because we currently do not have a proper logger to pass in.
	// Replace this with an appropriate logger when refactoring this function to accept a logger parameter.
	logger := klog.TODO()
	m.allocationMutex.Lock()
	defer m.allocationMutex.Unlock()

	for _, p := range m.podsWithPendingResizes {
		if p == uid {
			// Pod is already in the pending resizes queue.
			return
		}
	}

	// Add the pod to the pending resizes list and sort by priority.
	m.podsWithPendingResizes = append(m.podsWithPendingResizes, uid)
	m.sortPendingResizes(logger)
}

// sortPendingResizes sorts the list of pending resizes:
// - First, prioritizing resizes that do not increase requests.
// - Second, based on the pod's PriorityClass.
// - Third, based on the pod's QoS class.
// - Last, prioritizing resizes that have been in the deferred state the longest.
func (m *manager) sortPendingResizes(logger klog.Logger) {
	var pendingPods []*v1.Pod
	for _, uid := range m.podsWithPendingResizes {
		pod, found := m.getPodByUID(uid)
		if !found {
			logger.V(4).Info("Pod not found; removing from pending resizes", "podUID", uid)
			continue
		}
		pendingPods = append(pendingPods, pod)
	}

	slices.SortFunc(pendingPods, func(firstPod, secondPod *v1.Pod) int {
		// First, resizes that don't increase requests will be prioritized.
		// These resizes are expected to always succeed.
		firstPodIncreasing := m.isResizeIncreasingRequests(firstPod)
		secondPodIncreasing := m.isResizeIncreasingRequests(secondPod)
		if !firstPodIncreasing {
			return -1
		}
		if !secondPodIncreasing {
			return 1
		}

		// Second, pods with a higher PriorityClass will be prioritized.
		firstPodPriority := int32(0)
		if firstPod.Spec.Priority != nil {
			firstPodPriority = *firstPod.Spec.Priority
		}
		secondPodPriority := int32(0)
		if secondPod.Spec.Priority != nil {
			secondPodPriority = *secondPod.Spec.Priority
		}
		if firstPodPriority > secondPodPriority {
			return -1
		}
		if secondPodPriority > firstPodPriority {
			return 1
		}

		// Third, pods with a higher QoS class will be prioritized, where guaranteed > burstable.
		// Best effort pods don't have resource requests or limits, so we don't need to consider them here.
		firstPodQOS := v1qos.GetPodQOS(firstPod)
		secondPodQOS := v1qos.GetPodQOS(secondPod)
		if firstPodQOS == v1.PodQOSGuaranteed && secondPodQOS != v1.PodQOSGuaranteed {
			return -1
		}
		if secondPodQOS == v1.PodQOSGuaranteed && firstPodQOS != v1.PodQOSGuaranteed {
			return 1
		}

		// If all else is the same, resize requests that have been pending longer will be
		// evaluated first.
		var firstPodLastTransitionTime *metav1.Time
		firstPodResizeConditions := m.statusManager.GetPodResizeConditions(firstPod.UID)
		for _, c := range firstPodResizeConditions {
			if c.Type == v1.PodResizePending {
				firstPodLastTransitionTime = &c.LastTransitionTime
			}
		}
		var secondPodLastTransitionTime *metav1.Time
		secondPodResizeConditions := m.statusManager.GetPodResizeConditions(secondPod.UID)
		for _, c := range secondPodResizeConditions {
			if c.Type == v1.PodResizePending {
				secondPodLastTransitionTime = &c.LastTransitionTime
			}
		}
		if firstPodLastTransitionTime == nil {
			return 1
		}
		if secondPodLastTransitionTime == nil {
			return -1
		}
		if firstPodLastTransitionTime.Before(secondPodLastTransitionTime) {
			return -1
		}
		return 1
	})

	m.podsWithPendingResizes = make([]types.UID, len(pendingPods))
	for i, pod := range pendingPods {
		m.podsWithPendingResizes[i] = pod.UID
	}
}

// isResizeIncreasingRequests returns true if any of the resource requests are increasing.
func (m *manager) isResizeIncreasingRequests(pod *v1.Pod) bool {
	allocatedPod, updated := m.UpdatePodFromAllocation(pod)
	if !updated {
		return false
	}

	opts := resourcehelper.PodResourcesOptions{
		SkipPodLevelResources: !utilfeature.DefaultFeatureGate.Enabled(features.PodLevelResources),
	}
	oldRequest := resourcehelper.PodRequests(allocatedPod, opts)
	newRequest := resourcehelper.PodRequests(pod, opts)

	return newRequest.Memory().Cmp(*oldRequest.Memory()) > 0 ||
		newRequest.Cpu().Cmp(*oldRequest.Cpu()) > 0
}

func (m *manager) HasPendingResizes() bool {
	m.allocationMutex.Lock()
	defer m.allocationMutex.Unlock()

	return len(m.podsWithPendingResizes) > 0
}

// GetContainerResourceAllocation returns the last checkpointed AllocatedResources values
// If checkpoint manager has not been initialized, it returns nil, false
func (m *manager) GetContainerResourceAllocation(podUID types.UID, containerName string) (v1.ResourceRequirements, bool) {
	return m.allocated.GetContainerResources(podUID, containerName)
}

// GetPodLevelResourceAllocation returns the last checkpointed AllocatedResources values
// If checkpoint manager has not been initialized, it returns nil, false
func (m *manager) GetPodLevelResourceAllocation(podUID types.UID) (*v1.ResourceRequirements, bool) {
	return m.allocated.GetPodLevelResources(podUID)
}

// UpdatePodFromAllocation overwrites the pod spec with the allocation.
// This function does a deep copy only if updates are needed.
func (m *manager) UpdatePodFromAllocation(pod *v1.Pod) (*v1.Pod, bool) {
	if pod == nil {
		return pod, false
	}

	allocated, ok := m.allocated.GetPodResourceInfo(pod.UID)
	if !ok {
		return pod, false
	}

	return updatePodFromAllocation(pod, allocated)
}

func updatePodFromAllocation(pod *v1.Pod, allocated state.PodResourceInfo) (*v1.Pod, bool) {
	if pod == nil {
		return pod, false
	}

	updated := false
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodLevelResourcesVerticalScaling) {
		pAlloc := allocated.PodLevelResources
		if !apiequality.Semantic.DeepEqual(pod.Spec.Resources, pAlloc) {
			// Allocation differs from pod spec, retrieve the allocation
			pod = pod.DeepCopy()
			pod.Spec.Resources = pAlloc
			updated = true
		}
	}
	containerAlloc := func(c v1.Container) (v1.ResourceRequirements, bool) {
		if cAlloc, ok := allocated.ContainerResources[c.Name]; ok {
			if !apiequality.Semantic.DeepEqual(c.Resources, cAlloc) {
				// Allocation differs from pod spec, retrieve the allocation
				if !updated {
					// If this is the first update to be performed, copy the pod
					pod = pod.DeepCopy()
					updated = true
				}
				return cAlloc, true
			}
		}
		return v1.ResourceRequirements{}, false
	}

	for i, c := range pod.Spec.Containers {
		if cAlloc, found := containerAlloc(c); found {
			// Allocation differs from pod spec, update
			pod.Spec.Containers[i].Resources = cAlloc
		}
	}
	for i, c := range pod.Spec.InitContainers {
		if cAlloc, found := containerAlloc(c); found {
			// Allocation differs from pod spec, update
			pod.Spec.InitContainers[i].Resources = cAlloc
		}
	}
	return pod, updated
}

// SetAllocatedResources checkpoints the resources allocated to a pod's containers
func (m *manager) SetAllocatedResources(pod *v1.Pod) error {
	// Use klog.TODO() because we currently do not have a proper logger to pass in.
	// Replace this with an appropriate logger when refactoring this function to accept a logger parameter.
	logger := klog.TODO()
	return m.allocated.SetPodResourceInfo(logger, pod.UID, allocationFromPod(pod))
}

func allocationFromPod(pod *v1.Pod) state.PodResourceInfo {
	var podAlloc state.PodResourceInfo
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodLevelResourcesVerticalScaling) && pod.Spec.Resources != nil {
		podAlloc.PodLevelResources = pod.Spec.Resources.DeepCopy()
	}
	podAlloc.ContainerResources = make(map[string]v1.ResourceRequirements)
	for _, container := range pod.Spec.Containers {
		alloc := *container.Resources.DeepCopy()
		podAlloc.ContainerResources[container.Name] = alloc
	}

	for _, container := range pod.Spec.InitContainers {
		if podutil.IsRestartableInitContainer(&container) {
			alloc := *container.Resources.DeepCopy()
			podAlloc.ContainerResources[container.Name] = alloc
		}
	}

	return podAlloc
}

func (m *manager) AddPodAdmitHandlers(handlers lifecycle.PodAdmitHandlers) {
	for _, a := range handlers {
		m.admitHandlers.AddPodAdmitHandler(a)
	}
}

func (m *manager) SetContainerRuntime(runtime kubecontainer.Runtime) {
	m.containerRuntime = runtime
}

func (m *manager) AddPod(activePods []*v1.Pod, pod *v1.Pod) (bool, string, string) {
	// Use klog.TODO() because we currently do not have a proper logger to pass in.
	// Replace this with an appropriate logger when refactoring this function to accept a logger parameter.
	logger := klog.TODO()
	m.allocationMutex.Lock()
	defer m.allocationMutex.Unlock()

	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		// To handle kubelet restarts, test pod admissibility using AllocatedResources values
		// (for cpu & memory) from checkpoint store. If found, that is the source of truth.
		pod, _ = m.UpdatePodFromAllocation(pod)
	}

	// Check if we can admit the pod; if so, update the allocation.
	allocatedPods := m.getAllocatedPods(activePods)
	ok, reason, message := m.canAdmitPod(logger, allocatedPods, pod)

	if ok && utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		// Checkpoint the resource values at which the Pod has been admitted or resized.
		if err := m.SetAllocatedResources(pod); err != nil {
			// TODO(vinaykul,InPlacePodVerticalScaling): Can we recover from this in some way? Investigate
			logger.Error(err, "SetPodAllocation failed", "pod", klog.KObj(pod))
		}
	}

	return ok, reason, message
}

func (m *manager) RemovePod(uid types.UID) {
	// Use klog.TODO() because we currently do not have a proper logger to pass in.
	// Replace this with an appropriate logger when refactoring this function to accept a logger parameter.
	logger := klog.TODO()
	if err := m.allocated.RemovePod(uid); err != nil {
		// If the deletion fails, it will be retried by RemoveOrphanedPods, so we can safely ignore the error.
		logger.V(3).Error(err, "Failed to delete pod allocation", "podUID", uid)
	}
}

func (m *manager) RemoveOrphanedPods(remainingPods sets.Set[types.UID]) {
	m.allocated.RemoveOrphanedPods(remainingPods)
}

func (m *manager) handlePodResourcesResize(logger klog.Logger, pod *v1.Pod) (bool, error) {
	allocatedPod, updated := m.UpdatePodFromAllocation(pod)
	if !updated {
		// Desired resources == allocated resources. Pod allocation does not need to be updated.
		m.statusManager.ClearPodResizePendingCondition(pod.UID)
		return false, nil

	} else if resizable, msg, reason := IsInPlacePodVerticalScalingAllowed(pod); !resizable {
		// If there is a pending resize but the resize is not allowed, always use the allocated resources.
		metrics.PodInfeasibleResizes.WithLabelValues(reason).Inc()
		m.statusManager.SetPodResizePendingCondition(pod.UID, v1.PodReasonInfeasible, msg, pod.Generation)
		return false, nil

	} else if resizeNotAllowed, msg := disallowResizeForSwappableContainers(m.containerRuntime, pod, allocatedPod); resizeNotAllowed {
		// If this resize involve swap recalculation, set as infeasible, as IPPR with swap is not supported for beta.
		metrics.PodInfeasibleResizes.WithLabelValues("swap_limitation").Inc()
		m.statusManager.SetPodResizePendingCondition(pod.UID, v1.PodReasonInfeasible, msg, pod.Generation)
		return false, nil
	}

	if !apiequality.Semantic.DeepEqual(pod.Spec.Resources, allocatedPod.Spec.Resources) {
		if resizable, msg, reason := IsInPlacePodLevelResourcesVerticalScalingAllowed(pod); !resizable {
			// If there is a pending pod-level resources resize but the resize is not allowed, always use the allocated resources.
			metrics.PodInfeasibleResizes.WithLabelValues(reason).Inc()
			m.statusManager.SetPodResizePendingCondition(pod.UID, v1.PodReasonInfeasible, msg, pod.Generation)
			return false, nil
		}
	}

	// Desired resources != allocated resources. Can we update the allocation to the desired resources?
	fit, reason, message := m.canResizePod(logger, m.getAllocatedPods(m.getActivePods()), pod)
	if fit {
		// Update pod resource allocation checkpoint
		if err := m.SetAllocatedResources(pod); err != nil {
			return false, err
		}
		allocatedPod = pod
		m.statusManager.ClearPodResizePendingCondition(pod.UID)

		// Clear any errors that may have been surfaced from a previous resize and update the
		// generation of the resize in-progress condition.
		m.statusManager.ClearPodResizeInProgressCondition(pod.UID)
		m.statusManager.SetPodResizeInProgressCondition(pod.UID, "", "", pod.Generation)

		msg := events.PodResizeStartedMsg(allocatedPod, pod.Generation)
		m.recorder.Eventf(pod, v1.EventTypeNormal, events.ResizeStarted, msg)

		return true, nil
	}

	if reason != "" {
		if m.statusManager.SetPodResizePendingCondition(pod.UID, reason, message, pod.Generation) {
			eventType := events.ResizeDeferred
			if reason == v1.PodReasonInfeasible {
				eventType = events.ResizeInfeasible
			}
			msg := events.PodResizePendingMsg(pod, reason, message, pod.Generation)
			m.recorder.Eventf(pod, v1.EventTypeWarning, eventType, msg)
		}
	}

	return false, nil
}

func disallowResizeForSwappableContainers(runtime kubecontainer.Runtime, desiredPod, allocatedPod *v1.Pod) (bool, string) {
	if desiredPod == nil || allocatedPod == nil {
		return false, ""
	}
	restartableMemoryResizePolicy := func(resizePolicies []v1.ContainerResizePolicy) bool {
		for _, policy := range resizePolicies {
			if policy.ResourceName == v1.ResourceMemory {
				return policy.RestartPolicy == v1.RestartContainer
			}
		}
		return false
	}
	allocatedContainers := make(map[string]v1.Container)
	for _, container := range append(allocatedPod.Spec.Containers, allocatedPod.Spec.InitContainers...) {
		allocatedContainers[container.Name] = container
	}
	for _, desiredContainer := range append(desiredPod.Spec.Containers, desiredPod.Spec.InitContainers...) {
		allocatedContainer, ok := allocatedContainers[desiredContainer.Name]
		if !ok {
			continue
		}
		origMemRequest := desiredContainer.Resources.Requests[v1.ResourceMemory]
		newMemRequest := allocatedContainer.Resources.Requests[v1.ResourceMemory]
		if !origMemRequest.Equal(newMemRequest) && !restartableMemoryResizePolicy(allocatedContainer.ResizePolicy) {
			aSwapBehavior := runtime.GetContainerSwapBehavior(desiredPod, &desiredContainer)
			bSwapBehavior := runtime.GetContainerSwapBehavior(allocatedPod, &allocatedContainer)
			if aSwapBehavior != kubetypes.NoSwap || bSwapBehavior != kubetypes.NoSwap {
				return true, "In-place resize of containers with swap is not supported."
			}
		}
	}
	return false, ""
}

// canAdmitPod determines if a pod can be admitted, and gives a reason if it
// cannot. "pod" is new pod, while "pods" are all admitted pods
// The function returns a boolean value indicating whether the pod
// can be admitted, a brief single-word reason and a message explaining why
// the pod cannot be admitted.
// allocatedPods should represent the pods that have already been admitted, along with their
// admitted (allocated) resources.
func (m *manager) canAdmitPod(logger klog.Logger, allocatedPods []*v1.Pod, pod *v1.Pod) (bool, string, string) {
	// Filter out the pod being evaluated.
	allocatedPods = slices.DeleteFunc(allocatedPods, func(p *v1.Pod) bool { return p.UID == pod.UID })

	// If any handler rejects, the pod is rejected.
	attrs := &lifecycle.PodAdmitAttributes{Pod: pod, OtherPods: allocatedPods}
	for _, podAdmitHandler := range m.admitHandlers {
		if result := podAdmitHandler.Admit(attrs); !result.Admit {
			logger.Info("Pod admission denied", "podUID", attrs.Pod.UID, "pod", klog.KObj(attrs.Pod), "reason", result.Reason, "message", result.Message)
			return false, result.Reason, result.Message
		}
	}

	return true, "", ""
}

// canResizePod determines if the requested resize is currently feasible.
// pod should hold the desired (pre-allocated) spec.
// Returns true if the resize can proceed; returns a reason and message
// otherwise.
func (m *manager) canResizePod(logger klog.Logger, allocatedPods []*v1.Pod, pod *v1.Pod) (bool, string, string) {
	// TODO: Move this logic into a PodAdmitHandler by introducing an operation field to
	// lifecycle.PodAdmitAttributes, and combine canResizePod with canAdmitPod.
	if v1qos.GetPodQOS(pod) == v1.PodQOSGuaranteed {
		if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveCPUs) &&
			m.nodeConfig.CPUManagerPolicy == string(cpumanager.PolicyStatic) &&
			m.guaranteedPodResourceResizeRequired(pod, v1.ResourceCPU) {
			msg := fmt.Sprintf("Resize is infeasible for Guaranteed Pods alongside CPU Manager policy \"%s\"", string(cpumanager.PolicyStatic))
			logger.V(3).Info(msg, "pod", format.Pod(pod))
			metrics.PodInfeasibleResizes.WithLabelValues("guaranteed_pod_cpu_manager_static_policy").Inc()
			return false, v1.PodReasonInfeasible, msg
		}
		if utilfeature.DefaultFeatureGate.Enabled(features.MemoryManager) &&
			!utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingExclusiveMemory) &&
			m.nodeConfig.MemoryManagerPolicy == string(memorymanager.PolicyTypeStatic) &&
			m.guaranteedPodResourceResizeRequired(pod, v1.ResourceMemory) {
			msg := fmt.Sprintf("Resize is infeasible for Guaranteed Pods alongside Memory Manager policy \"%s\"", string(memorymanager.PolicyTypeStatic))
			logger.V(3).Info(msg, "pod", format.Pod(pod))
			metrics.PodInfeasibleResizes.WithLabelValues("guaranteed_pod_memory_manager_static_policy").Inc()
			return false, v1.PodReasonInfeasible, msg
		}
	}

	cpuAvailable := m.nodeAllocatableAbsolute.Cpu().MilliValue()
	memAvailable := m.nodeAllocatableAbsolute.Memory().Value()
	cpuRequests := resource.GetResourceRequest(pod, v1.ResourceCPU)
	memRequests := resource.GetResourceRequest(pod, v1.ResourceMemory)
	if cpuRequests > cpuAvailable || memRequests > memAvailable {
		var msg string
		if memRequests > memAvailable {
			msg = fmt.Sprintf("memory, requested: %d, capacity: %d", memRequests, memAvailable)
		} else {
			msg = fmt.Sprintf("cpu, requested: %d, capacity: %d", cpuRequests, cpuAvailable)
		}
		msg = "Node didn't have enough capacity: " + msg
		logger.V(3).Info(msg, "pod", klog.KObj(pod))
		metrics.PodInfeasibleResizes.WithLabelValues("insufficient_node_allocatable").Inc()
		return false, v1.PodReasonInfeasible, msg
	}

	if ok, failReason, failMessage := m.canAdmitPod(logger, allocatedPods, pod); !ok {
		// Log reason and return.
		logger.V(3).Info("Resize cannot be accommodated", "pod", klog.KObj(pod), "reason", failReason, "message", failMessage)
		return false, v1.PodReasonDeferred, failMessage
	}

	return true, "", ""
}

func (m *manager) guaranteedPodResourceResizeRequired(pod *v1.Pod, resourceName v1.ResourceName) bool {
	for container, containerType := range podutil.ContainerIter(&pod.Spec, podutil.InitContainers|podutil.Containers) {
		if !IsResizableContainer(container, containerType) {
			continue
		}
		requestedResources := container.Resources
		allocatedresources, _ := m.GetContainerResourceAllocation(pod.UID, container.Name)
		// For Guaranteed pods, requests must equal limits, so checking requests is sufficient.
		if !requestedResources.Requests[resourceName].Equal(allocatedresources.Requests[resourceName]) {
			return true
		}
	}
	return false
}

func (m *manager) getAllocatedPods(activePods []*v1.Pod) []*v1.Pod {
	if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		return activePods
	}

	allocatedPods := make([]*v1.Pod, len(activePods))
	for i, pod := range activePods {
		allocatedPods[i], _ = m.UpdatePodFromAllocation(pod)
	}
	return allocatedPods
}

func IsResizableContainer(container *v1.Container, containerType podutil.ContainerType) bool {
	switch containerType {
	case podutil.InitContainers:
		return podutil.IsRestartableInitContainer(container)
	case podutil.Containers:
		return true
	default:
		return false
	}
}
