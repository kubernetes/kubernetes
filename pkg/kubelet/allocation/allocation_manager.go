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
	v1qos "k8s.io/kubernetes/pkg/apis/core/v1/helper/qos"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/allocation/state"
	"k8s.io/kubernetes/pkg/kubelet/config"
	"k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	"k8s.io/kubernetes/pkg/kubelet/status"
)

// podStatusManagerStateFile is the file name where status manager stores its state
const (
	allocatedPodsStateFile = "allocated_pods_state"

	initialRetryDelay = 30 * time.Second
	retryDelay        = 3 * time.Minute

	TriggerReasonPodResized    = "pod_resized"
	TriggerReasonPodUpdated    = "pod_updated"
	TriggerReasonPodsAdded     = "pods_added"
	TriggerReasonPodsRemoved   = "pods_removed"
	TriggerReasonPodTerminated = "pod_terminated"

	triggerReasonPeriodic = "periodic_retry"
)

// AllocationManager tracks pod resource allocations.
type Manager interface {
	// GetContainerResourceAllocation returns the AllocatedResources value for the container
	GetContainerResourceAllocation(podUID types.UID, containerName string) (v1.ResourceRequirements, bool)

	// GetPodLevelResourceAllocation returns the AllocatedResources value for the container
	GetPodLevelResourceAllocation(podUID types.UID) (*v1.ResourceRequirements, bool)

	// UpdatePodFromAllocation returns the pod as it is allocated, and whether a change to a
	// field that is gated on allocation is pending. A deep copy of the pod is only returned if an allocation is pending;
	// otherwise, the pod is returned as-is.
	UpdatePodFromAllocation(pod *v1.Pod) (*v1.Pod, bool)

	// SetAllocatedResources checkpoints the allocation for a pod.
	SetAllocatedResources(logger klog.Logger, allocatedPod *v1.Pod) error

	// AddPodAdmitHandlers adds the admit handlers to the allocation manager.
	// TODO: See if we can remove this and just add them in the allocation manager constructor.
	AddPodAdmitHandlers(handlers lifecycle.PodAdmitHandlers)

	// AddPod checks if a pod can be admitted. If so, it admits the pod and updates the allocation.
	// The function returns a boolean value indicating whether the pod
	// can be admitted, a brief single-word reason and a message explaining why
	// the pod cannot be admitted.
	// allocatedPods should represent the pods that have already been admitted, along with their
	// admitted (allocated) resources.
	AddPod(ctx context.Context, activePods []*v1.Pod, pod *v1.Pod) (ok bool, reason, message string)

	// RemovePod removes any stored state for the given pod UID.
	RemovePod(logger klog.Logger, uid types.UID)

	// RemoveOrphanedPods removes the stored state for any pods not included in the set of remaining pods.
	RemoveOrphanedPods(remainingPods sets.Set[types.UID])

	// Run starts the allocation manager. This is currently only used to handle periodic retry of
	// pending resizes.
	Run(ctx context.Context)

	// PushPendingResize queues a pod with a pending resize request for later reevaluation.
	PushPendingResize(logger klog.Logger, uid types.UID)

	// HasPendingResizes returns whether there are currently any pending resizes.
	HasPendingResizes() bool

	// RetryPendingResizes retries all pending resizes.
	RetryPendingResizes(ctx context.Context, trigger string)

	// HasPodAllocatedResources returns whether a pod has been allocated resources.
	HasPodAllocatedResources(podUID types.UID) bool

	// GetAllocatedPods returns all active pods with their allocated resources.
	GetAllocatedPods() []*v1.Pod
}

type manager struct {
	allocated state.State

	admitHandlers lifecycle.PodAdmitHandlers
	statusManager status.Manager
	sourcesReady  config.SourcesReady

	ticker         *time.Ticker
	triggerPodSync func(context.Context, *v1.Pod)
	getActivePods  func() []*v1.Pod
	getPodByUID    func(types.UID) (*v1.Pod, bool)

	allocationMutex        sync.Mutex
	podsWithPendingResizes []types.UID

	recorder record.EventRecorderLogger
}

func NewManager(checkpointDirectory string,
	statusManager status.Manager,
	triggerPodSync func(context.Context, *v1.Pod),
	getActivePods func() []*v1.Pod,
	getPodByUID func(types.UID) (*v1.Pod, bool),
	sourcesReady config.SourcesReady,
	recorder record.EventRecorderLogger,
	logger klog.Logger,
) Manager {
	return &manager{
		allocated: newStateImpl(logger, checkpointDirectory, allocatedPodsStateFile),

		statusManager: statusManager,
		admitHandlers: lifecycle.PodAdmitHandlers{},
		sourcesReady:  sourcesReady,

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

	stateImpl, err := state.NewStateCheckpoint(logger, checkpointDirectory, checkpointName)
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
	logger klog.Logger,
	statusManager status.Manager,
	triggerPodSync func(context.Context, *v1.Pod),
	getActivePods func() []*v1.Pod,
	getPodByUID func(types.UID) (*v1.Pod, bool),
	sourcesReady config.SourcesReady,
	recorder record.EventRecorderLogger,
) Manager {
	return &manager{
		allocated: state.NewStateMemory(logger, nil),

		statusManager: statusManager,
		admitHandlers: lifecycle.PodAdmitHandlers{},
		sourcesReady:  sourcesReady,

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
				successfulResizes := m.retryPendingResizes(ctx, triggerReasonPeriodic)
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

func (m *manager) RetryPendingResizes(ctx context.Context, trigger string) {
	m.retryPendingResizes(ctx, trigger)
}

func (m *manager) retryPendingResizes(ctx context.Context, trigger string) []*v1.Pod {
	logger := klog.FromContext(ctx)
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

		resizeAllocated, err := m.handlePodResourcesResize(ctx, pod)
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
			m.triggerPodSync(ctx, pod)
		}
	}

	m.podsWithPendingResizes = newPendingResizes
	return successfulResizes
}

func (m *manager) PushPendingResize(logger klog.Logger, uid types.UID) {
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

// UpdatePodFromAllocation returns the pod as it is allocated, and whether a change to a
// field that is gated on allocation is pending. A deep copy of the pod is only returned if an allocation is pending;
// otherwise, the pod is returned as-is.
//
// Pod spec fields are either:
//   - passthrough: they take effect as soon as they change in the pod, e.g. tolerations.
//   - gated on allocation: they take their allocated value, so that a change only takes effect once it is
//     allocated (admitted and checkpointed) as a whole. Today the only gated fields that can change
//     are resources.
//   - immutable fields: they are classified as 'gated on allocation' too, since which value they take then does
//     not matter, and this requires fewer changes as we make more fields mutable over time.
//
// The allocated pod is the stored pod, with only the passthrough fields copied from the pod, so a
// change is pending when it differs from the pod. A new PodSpec or Container field has to be
// classified in TestUpdatePodFromAllocationFieldClassification, which makes it explicit which of the
// two it is.
//
// Pods migrated from a V1 checkpoint only have their resources until they are added again, so for
// them those resources are overlaid onto the pod instead, and only a change to them can be pending.
func (m *manager) UpdatePodFromAllocation(pod *v1.Pod) (*v1.Pod, bool) {
	if pod == nil {
		return pod, false
	}

	allocated, ok := m.allocated.GetPod(pod.UID)
	if !ok {
		return pod, false
	}

	// TODO: Remove when the V1 checkpoint format is no longer supported (see migrateV1ToV2).
	if isMigratedFromV1(allocated) {
		return updatePodFromAllocation(pod, allocated)
	}

	result := *pod
	result.Spec = allocated.Spec
	copyPassthroughFields(&result.Spec, &pod.Spec)
	// The passthrough fields have the same value in both specs by now, so only a gated field can
	// make them differ.
	pending := !apiequality.Semantic.DeepEqual(&pod.Spec, &result.Spec)
	if !pending {
		return pod, false
	}
	// The metadata, the status and the passthrough fields are still shared with the pod here. Copying
	// the whole result keeps it private however many passthrough fields are added.
	return result.DeepCopy(), true
}

// isMigratedFromV1 returns whether the allocated pod was rebuilt from a V1 checkpoint. Those only
// recorded resources, so the pod has no name, until it is replaced by a complete one when it is
// added again.
func isMigratedFromV1(allocated *v1.Pod) bool {
	return allocated.Name == ""
}

// copyPassthroughFields sets the fields of spec that take effect as soon as they change in the pod
// to their values in desired.
func copyPassthroughFields(spec, desired *v1.PodSpec) {
	spec.ActiveDeadlineSeconds = desired.ActiveDeadlineSeconds
	spec.Tolerations = desired.Tolerations
	spec.TerminationGracePeriodSeconds = desired.TerminationGracePeriodSeconds
	spec.EphemeralContainers = desired.EphemeralContainers
	// TODO: The image becomes gated with dynamic containers, at which point we
	// should stop copying the image here.
	copyImages(spec.Containers, desired.Containers)
	copyImages(spec.InitContainers, desired.InitContainers)
}

// copyImages sets the image of each container to the one of the container with the same name in desired.
func copyImages(containers, desired []v1.Container) {
	for i := range containers {
		for j := range desired {
			if desired[j].Name == containers[i].Name {
				containers[i].Image = desired[j].Image
				break
			}
		}
	}
}

// updatePodFromAllocation overlays the allocated resources onto the pod. It is only used for pods
// migrated from a V1 checkpoint, which have nothing else (see isMigratedFromV1).
func updatePodFromAllocation(pod *v1.Pod, allocated *v1.Pod) (*v1.Pod, bool) {
	if pod == nil {
		return pod, false
	}

	updated := false
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodLevelResourcesVerticalScaling) {
		pod, updated = updatePodLevelResourcesFromAllocation(pod, allocated)
	}
	pod, updated = updateContainerResourcesFromAllocation(pod, allocated, updated)
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingMemoryBackedVolumes) {
		pod, updated = updateEmptyDirVolumeLimitsFromAllocation(pod, allocated, updated)
	}

	return pod, updated
}

func updateContainerResourcesFromAllocation(pod *v1.Pod, allocated *v1.Pod, alreadyUpdated bool) (*v1.Pod, bool) {
	updated := alreadyUpdated
	containerAlloc := func(c v1.Container) (v1.ResourceRequirements, bool) {
		for ac := range podutil.ContainerIter(&allocated.Spec, podutil.AllContainers) {
			if ac.Name == c.Name && !apiequality.Semantic.DeepEqual(c.Resources, ac.Resources) {
				// Allocation differs from pod spec, retrieve the allocation
				if !updated {
					// If this is the first update to be performed, copy the pod
					pod = pod.DeepCopy()
					updated = true
				}
				return ac.Resources, true
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

func updatePodLevelResourcesFromAllocation(pod *v1.Pod, allocated *v1.Pod) (*v1.Pod, bool) {
	pAlloc := allocated.Spec.Resources
	if !apiequality.Semantic.DeepEqual(pod.Spec.Resources, pAlloc) {
		// Allocation differs from pod spec, retrieve the allocation
		pod = pod.DeepCopy()
		pod.Spec.Resources = pAlloc.DeepCopy()
		return pod, true
	}
	return pod, false
}

func updateEmptyDirVolumeLimitsFromAllocation(pod *v1.Pod, allocated *v1.Pod, alreadyUpdated bool) (*v1.Pod, bool) {
	updated := alreadyUpdated
	for i, vol := range pod.Spec.Volumes {
		if !VolHasMemoryBackedEmptyDirSizeLimit(&vol) {
			continue
		}
		for _, allocVol := range allocated.Spec.Volumes {
			if allocVol.Name != vol.Name || allocVol.EmptyDir == nil || allocVol.EmptyDir.SizeLimit == nil {
				continue
			}
			if alloc := allocVol.EmptyDir.SizeLimit; alloc.Cmp(*vol.EmptyDir.SizeLimit) != 0 {
				if !updated {
					pod = pod.DeepCopy()
					updated = true
				}
				allocCopy := alloc.DeepCopy()
				pod.Spec.Volumes[i].EmptyDir.SizeLimit = &allocCopy
			}
			break
		}
	}
	return pod, updated
}

// HasPodAllocatedResources returns whether a pod has been allocated resources.
func (m *manager) HasPodAllocatedResources(podUID types.UID) bool {
	return m.allocated.HasPod(podUID)
}

// SetAllocatedResources checkpoints the allocation for a pod
func (m *manager) SetAllocatedResources(logger klog.Logger, pod *v1.Pod) error {
	return m.allocated.SetPod(logger, pod)
}

func (m *manager) AddPodAdmitHandlers(handlers lifecycle.PodAdmitHandlers) {
	for _, a := range handlers {
		m.admitHandlers.AddPodAdmitHandler(a)
	}
}

func (m *manager) AddPod(ctx context.Context, activePods []*v1.Pod, pod *v1.Pod) (bool, string, string) {
	logger := klog.FromContext(ctx)
	m.allocationMutex.Lock()
	defer m.allocationMutex.Unlock()

	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		// To handle kubelet restarts, test pod admissibility using AllocatedResources values
		// (for cpu & memory) from checkpoint store. If found, that is the source of truth.
		pod, _ = m.UpdatePodFromAllocation(pod)
	}

	// Check if we can admit the pod; if so, update the allocation.
	allocatedPods := m.getAllocatedPods(activePods)
	ok, reason, message := m.canAdmitPod(ctx, allocatedPods, pod, lifecycle.AddOperation)

	if ok && utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		// Checkpoint the resource values at which the Pod has been admitted or resized.
		if err := m.SetAllocatedResources(logger, pod); err != nil {
			// TODO(vinaykul,InPlacePodVerticalScaling): Can we recover from this in some way? Investigate
			logger.Error(err, "SetPodAllocation failed", "pod", klog.KObj(pod))
		}
	}

	return ok, reason, message
}

func (m *manager) RemovePod(logger klog.Logger, uid types.UID) {
	if err := m.allocated.RemovePod(logger, uid); err != nil {
		// If the deletion fails, it will be retried by RemoveOrphanedPods, so we can safely ignore the error.
		logger.V(3).Info("Failed to delete pod allocation", "podUID", uid, "err", err)
	}
}

func (m *manager) RemoveOrphanedPods(remainingPods sets.Set[types.UID]) {
	m.allocated.RemoveOrphanedPods(remainingPods)
}

func (m *manager) handlePodResourcesResize(ctx context.Context, pod *v1.Pod) (bool, error) {
	logger := klog.FromContext(ctx)
	allocatedPod, updated := m.UpdatePodFromAllocation(pod)
	if !updated {
		// Desired resources == allocated resources. Pod allocation does not need to be updated.
		m.statusManager.ClearPodResizePendingCondition(pod.UID, metrics.DeferredResizeResolutionReverted)
		return false, nil
	}

	resizeRequested := isResizeRequested(pod, allocatedPod)
	if !resizeRequested {
		m.statusManager.ClearPodResizePendingCondition(pod.UID, metrics.DeferredResizeResolutionReverted)
	}

	// Desired pod != allocated pod. Can we update the allocation to the desired pod?
	fit, reason, message := m.canAdmitPod(ctx, m.getAllocatedPods(m.getActivePods()), pod, lifecycle.ResizeOperation)
	if fit {
		// Update pod resource allocation checkpoint
		if err := m.SetAllocatedResources(logger, pod); err != nil {
			return false, err
		}
		m.statusManager.ClearPodResizePendingCondition(pod.UID, metrics.DeferredResizeResolutionAccepted)

		if resizeRequested {
			// Clear any errors that may have been surfaced from a previous resize and update the
			// generation of the resize in-progress condition.
			m.statusManager.ClearPodResizeInProgressCondition(pod.UID)
			m.statusManager.SetPodResizeInProgressCondition(pod.UID, "", "", pod.Generation)

			msg := events.PodResizeStartedMsg(logger, pod, pod.Generation)
			m.recorder.WithLogger(logger).Eventf(pod, v1.EventTypeNormal, events.ResizeStarted, "%s", msg)
		}
		return true, nil
	}

	if reason != "" && resizeRequested {
		if m.statusManager.SetPodResizePendingCondition(pod.UID, reason, message, pod.Generation) {
			eventType := events.ResizeDeferred
			if reason == v1.PodReasonInfeasible {
				eventType = events.ResizeInfeasible
			}
			msg := events.PodResizePendingMsg(logger, pod, reason, message, pod.Generation)
			m.recorder.WithLogger(logger).Eventf(pod, v1.EventTypeWarning, eventType, "%s", msg)
		}
	}

	return false, nil
}

// isResizeRequested returns whether the desired pod asks for a change to the resources of its
// containers, to its pod-level resources, or to the size limit of one of its memory-backed emptyDir
// volumes. Any other field that is gated on allocation can differ too, but that is not a resize.
func isResizeRequested(desired, allocated *v1.Pod) bool {
	// TODO: Adjust this function when dynamic containers is implemented, to check for added/removed containers
	// that will also cause a resize.
	if !apiequality.Semantic.DeepEqual(desired.Spec.Resources, allocated.Spec.Resources) {
		return true
	}
	for c := range podutil.ContainerIter(&desired.Spec, podutil.InitContainers|podutil.Containers) {
		for ac := range podutil.ContainerIter(&allocated.Spec, podutil.AllContainers) {
			if ac.Name == c.Name && !apiequality.Semantic.DeepEqual(c.Resources, ac.Resources) {
				return true
			}
		}
	}
	for _, vol := range desired.Spec.Volumes {
		for _, allocVol := range allocated.Spec.Volumes {
			if allocVol.Name != vol.Name {
				continue
			}
			hasLimit, hasAllocLimit := VolHasMemoryBackedEmptyDirSizeLimit(&vol), VolHasMemoryBackedEmptyDirSizeLimit(&allocVol)
			if hasLimit != hasAllocLimit || (hasLimit && vol.EmptyDir.SizeLimit.Cmp(*allocVol.EmptyDir.SizeLimit) != 0) {
				return true
			}
			break
		}
	}
	return false
}

// canAdmitPod determines if a pod can be admitted, and gives a reason if it
// cannot. "pod" is new pod, while "pods" are all admitted pods
// The function returns a boolean value indicating whether the pod
// can be admitted, a brief single-word reason and a message explaining why
// the pod cannot be admitted.
// allocatedPods should represent the pods that have already been admitted, along with their
// admitted (allocated) resources.
func (m *manager) canAdmitPod(ctx context.Context, allocatedPods []*v1.Pod, pod *v1.Pod, operation lifecycle.Operation) (bool, string, string) {
	logger := klog.FromContext(ctx)
	// Filter out the pod being evaluated.
	allocatedPods = slices.DeleteFunc(allocatedPods, func(p *v1.Pod) bool { return p.UID == pod.UID })

	// If any handler rejects, the pod is rejected.
	attrs := &lifecycle.PodAdmitAttributes{Pod: pod, OtherPods: allocatedPods, Operation: operation}
	for _, podAdmitHandler := range m.admitHandlers {
		if result := podAdmitHandler.Admit(ctx, attrs); !result.Admit {
			logger.Info("Pod admission denied", "podUID", attrs.Pod.UID, "pod", klog.KObj(attrs.Pod), "reason", result.Reason, "message", result.Message, "operation", operation)
			return false, result.Reason, result.Message
		}
	}

	return true, "", ""
}

func (m *manager) getAllocatedPods(activePods []*v1.Pod) []*v1.Pod {
	if !utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		return activePods
	}

	allocatedPods := make([]*v1.Pod, 0, len(activePods))
	for _, pod := range activePods {
		// Filter out pods that don't yet have an allocation, which will filter pods that
		// are potentially going to be denied at admission.
		if m.HasPodAllocatedResources(pod.UID) {
			allocatedPod, _ := m.UpdatePodFromAllocation(pod)
			allocatedPods = append(allocatedPods, allocatedPod)
		}
	}
	return allocatedPods
}

func (m *manager) GetAllocatedPods() []*v1.Pod {
	return m.getAllocatedPods(m.getActivePods())
}

func IsResizableContainer(container *v1.Container, containerType podutil.ContainerType) bool {
	switch containerType {
	case podutil.InitContainers:
		return utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScalingInitContainers) || podutil.IsRestartableInitContainer(container)
	case podutil.Containers:
		return true
	default:
		return false
	}
}

func VolHasMemoryBackedEmptyDirSizeLimit(vol *v1.Volume) bool {
	return vol != nil && vol.EmptyDir != nil && vol.EmptyDir.Medium == v1.StorageMediumMemory && vol.EmptyDir.SizeLimit != nil && !vol.EmptyDir.SizeLimit.IsZero()
}
