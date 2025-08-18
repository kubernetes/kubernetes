/*
Copyright 2014 The Kubernetes Authors.

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

//go:generate mockery
package status

import (
	"context"
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	clientset "k8s.io/client-go/kubernetes"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/klog/v2"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/features"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	kubeutil "k8s.io/kubernetes/pkg/kubelet/util"
	statusutil "k8s.io/kubernetes/pkg/util/pod"
)

// A wrapper around v1.PodStatus that includes a version to enforce that stale pod statuses are
// not sent to the API server.
type versionedPodStatus struct {
	// version is a monotonically increasing version number (per pod).
	version uint64
	// Pod name & namespace, for sending updates to API server.
	podName      string
	podNamespace string
	// at is the time at which the most recent status update was detected
	at time.Time

	// True if the status is generated at the end of SyncTerminatedPod, or after it is completed.
	podIsFinished bool

	status v1.PodStatus
}

// Updates pod statuses in apiserver. Writes only when new status has changed.
// All methods are thread-safe.
type manager struct {
	kubeClient clientset.Interface
	podManager PodManager
	// Map from pod UID to sync status of the corresponding pod.
	podStatuses         map[types.UID]versionedPodStatus
	podResizeConditions map[types.UID]podResizeConditions
	podStatusesLock     sync.RWMutex
	podStatusChannel    chan struct{}
	// Map from (mirror) pod UID to latest status version successfully sent to the API server.
	// apiStatusVersions must only be accessed from the sync thread.
	apiStatusVersions map[kubetypes.MirrorPodUID]uint64
	podDeletionSafety PodDeletionSafetyProvider

	podStartupLatencyHelper PodStartupLatencyStateHelper
}

type podResizeConditions struct {
	PodResizePending    *v1.PodCondition
	PodResizeInProgress *v1.PodCondition
}

func (prc podResizeConditions) List() []*v1.PodCondition {
	var conditions []*v1.PodCondition
	if prc.PodResizePending != nil {
		conditions = append(conditions, prc.PodResizePending)
	}
	if prc.PodResizeInProgress != nil {
		conditions = append(conditions, prc.PodResizeInProgress)
	}
	return conditions
}

// PodManager is the subset of methods the manager needs to observe the actual state of the kubelet.
// See pkg/k8s.io/kubernetes/pkg/kubelet/pod.Manager for method godoc.
type PodManager interface {
	GetPodByUID(types.UID) (*v1.Pod, bool)
	GetMirrorPodByPod(*v1.Pod) (*v1.Pod, bool)
	TranslatePodUID(uid types.UID) kubetypes.ResolvedPodUID
	GetUIDTranslations() (podToMirror map[kubetypes.ResolvedPodUID]kubetypes.MirrorPodUID, mirrorToPod map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID)
}

// PodStatusProvider knows how to provide status for a pod. It is intended to be used by other components
// that need to introspect the authoritative status of a pod.  The PodStatusProvider represents the actual
// status of a running pod as the kubelet sees it.
type PodStatusProvider interface {
	// GetPodStatus returns the cached status for the provided pod UID, as well as whether it
	// was a cache hit.
	GetPodStatus(uid types.UID) (v1.PodStatus, bool)
}

// PodDeletionSafetyProvider provides guarantees that a pod can be safely deleted.
type PodDeletionSafetyProvider interface {
	// PodCouldHaveRunningContainers returns true if the pod could have running containers.
	PodCouldHaveRunningContainers(pod *v1.Pod) bool
}

type PodStartupLatencyStateHelper interface {
	RecordStatusUpdated(pod *v1.Pod)
	DeletePodStartupState(podUID types.UID)
}

// Manager is the Source of truth for kubelet pod status, and should be kept up-to-date with
// the latest v1.PodStatus. It also syncs updates back to the API server.
type Manager interface {
	PodStatusProvider

	// Start the API server status sync loop.
	Start(ctx context.Context)

	// SetPodStatus caches updates the cached status for the given pod, and triggers a status update.
	SetPodStatus(logger klog.Logger, pod *v1.Pod, status v1.PodStatus)

	// SetContainerReadiness updates the cached container status with the given readiness, and
	// triggers a status update.
	SetContainerReadiness(logger klog.Logger, podUID types.UID, containerID kubecontainer.ContainerID, ready bool)

	// SetContainerStartup updates the cached container status with the given startup, and
	// triggers a status update.
	SetContainerStartup(logger klog.Logger, podUID types.UID, containerID kubecontainer.ContainerID, started bool)

	// TerminatePod resets the container status for the provided pod to terminated and triggers
	// a status update.
	TerminatePod(logger klog.Logger, pod *v1.Pod)

	// RemoveOrphanedStatuses scans the status cache and removes any entries for pods not included in
	// the provided podUIDs.
	RemoveOrphanedStatuses(logger klog.Logger, podUIDs map[types.UID]bool)

	// GetPodResizeConditions returns cached PodStatus Resize conditions value
	GetPodResizeConditions(podUID types.UID) []*v1.PodCondition

	// SetPodResizePendingCondition caches the last PodResizePending condition for the pod.
	SetPodResizePendingCondition(podUID types.UID, reason, message string, observedGeneration int64)

	// SetPodResizeInProgressCondition caches the last PodResizeInProgress condition for the pod.
	// This function does not update observedGeneration if the condition already exists, nor does
	// it allow the reason or message to be cleared.
	SetPodResizeInProgressCondition(podUID types.UID, reason, message string, observedGeneration int64)

	// ClearPodResizePendingCondition clears the PodResizePending condition for the pod from the cache.
	ClearPodResizePendingCondition(podUID types.UID)

	// ClearPodResizeInProgressCondition clears the PodResizeInProgress condition for the pod from the cache.
	// Returns true if the condition was cleared, false if it was not set.
	ClearPodResizeInProgressCondition(podUID types.UID) bool

	// IsPodResizeDeferred returns true if the pod resize is currently deferred.
	IsPodResizeDeferred(podUID types.UID) bool

	// IsPodResizeInfeasible returns true if the pod resize is infeasible.
	IsPodResizeInfeasible(podUID types.UID) bool

	// BackfillPodResizeConditions backfills the status manager's resize conditions by reading them from the
	// provided pods' statuses.
	BackfillPodResizeConditions(pods []*v1.Pod)
}

const syncPeriod = 10 * time.Second

// NewManager returns a functional Manager.
func NewManager(kubeClient clientset.Interface, podManager PodManager, podDeletionSafety PodDeletionSafetyProvider, podStartupLatencyHelper PodStartupLatencyStateHelper) Manager {
	return &manager{
		kubeClient:              kubeClient,
		podManager:              podManager,
		podStatuses:             make(map[types.UID]versionedPodStatus),
		podResizeConditions:     make(map[types.UID]podResizeConditions),
		podStatusChannel:        make(chan struct{}, 1),
		apiStatusVersions:       make(map[kubetypes.MirrorPodUID]uint64),
		podDeletionSafety:       podDeletionSafety,
		podStartupLatencyHelper: podStartupLatencyHelper,
	}
}

// isPodStatusByKubeletEqual returns true if the given pod statuses are equal when non-kubelet-owned
// pod conditions are excluded.
// This method normalizes the status before comparing so as to make sure that meaningless
// changes will be ignored.
func isPodStatusByKubeletEqual(oldStatus, status *v1.PodStatus) bool {
	oldCopy := oldStatus.DeepCopy()

	newConditions := make(map[v1.PodConditionType]*v1.PodCondition, len(status.Conditions))
	oldConditions := make(map[v1.PodConditionType]*v1.PodCondition, len(oldStatus.Conditions))
	for _, c := range status.Conditions {
		if kubetypes.PodConditionByKubelet(c.Type) || kubetypes.PodConditionSharedByKubelet(c.Type) {
			newConditions[c.Type] = &c
		}
	}
	for _, c := range oldStatus.Conditions {
		if kubetypes.PodConditionByKubelet(c.Type) || kubetypes.PodConditionSharedByKubelet(c.Type) {
			oldConditions[c.Type] = &c
		}
	}

	if len(newConditions) != len(oldConditions) {
		return false
	}
	for _, newCondition := range newConditions {
		oldCondition := oldConditions[newCondition.Type]
		if oldCondition == nil || oldCondition.Status != newCondition.Status || oldCondition.Message != newCondition.Message || oldCondition.Reason != newCondition.Reason {
			return false
		}
	}

	oldCopy.Conditions = status.Conditions
	return apiequality.Semantic.DeepEqual(oldCopy, status)
}

func (m *manager) Start(ctx context.Context) {
	logger := klog.FromContext(ctx)
	// Don't start the status manager if we don't have a client. This will happen
	// on the master, where the kubelet is responsible for bootstrapping the pods
	// of the master components.
	if m.kubeClient == nil {
		logger.Info("Kubernetes client is nil, not starting status manager")
		return
	}

	logger.Info("Starting to sync pod status with apiserver")

	//nolint:staticcheck // SA1015 Ticker can leak since this is only called once and doesn't handle termination.
	syncTicker := time.NewTicker(syncPeriod).C

	// syncPod and syncBatch share the same go routine to avoid sync races.
	go wait.Forever(func() {
		for {
			select {
			case <-m.podStatusChannel:
				logger.V(4).Info("Syncing updated statuses")
				m.syncBatch(ctx, false)
			case <-syncTicker:
				logger.V(4).Info("Syncing all statuses")
				m.syncBatch(ctx, true)
			}
		}
	}, 0)
}

// GetPodResizeConditions returns the last cached ResizeStatus value.
func (m *manager) GetPodResizeConditions(podUID types.UID) []*v1.PodCondition {
	m.podStatusesLock.RLock()
	defer m.podStatusesLock.RUnlock()
	return m.podResizeConditions[podUID].List()
}

// SetPodResizePendingCondition caches the last PodResizePending condition for the pod.
func (m *manager) SetPodResizePendingCondition(podUID types.UID, reason, message string, observedGeneration int64) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()

	alreadyPending := m.podResizeConditions[podUID].PodResizePending != nil

	m.podResizeConditions[podUID] = podResizeConditions{
		PodResizePending:    updatedPodResizeCondition(v1.PodResizePending, m.podResizeConditions[podUID].PodResizePending, reason, message, observedGeneration),
		PodResizeInProgress: m.podResizeConditions[podUID].PodResizeInProgress,
	}

	if !alreadyPending {
		m.recordPendingResizeCount()
	}
}

// This function does not update observedGeneration if the condition already exists, nor does
// it allow the reason or message to be cleared.
func (m *manager) SetPodResizeInProgressCondition(podUID types.UID, reason, message string, observedGeneration int64) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()

	alreadyInProgress := m.podResizeConditions[podUID].PodResizeInProgress != nil

	if c := m.podResizeConditions[podUID].PodResizeInProgress; c != nil {
		// Preserve the old reason, message if they exist.
		if reason == "" && message == "" {
			reason = c.Reason
			message = c.Message
		}
		// The observedGeneration is always preserved. To update it, the caller must
		// clear the existing condition first.
		observedGeneration = c.ObservedGeneration
	}

	m.podResizeConditions[podUID] = podResizeConditions{
		PodResizeInProgress: updatedPodResizeCondition(v1.PodResizeInProgress, m.podResizeConditions[podUID].PodResizeInProgress, reason, message, observedGeneration),
		PodResizePending:    m.podResizeConditions[podUID].PodResizePending,
	}

	if !alreadyInProgress {
		m.recordInProgressResizeCount()
	}
}

// ClearPodResizePendingCondition clears the PodResizePending condition for the pod from the cache.
func (m *manager) ClearPodResizePendingCondition(podUID types.UID) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()

	if m.podResizeConditions[podUID].PodResizePending == nil {
		return
	}

	m.podResizeConditions[podUID] = podResizeConditions{
		PodResizeInProgress: m.podResizeConditions[podUID].PodResizeInProgress,
		PodResizePending:    nil,
	}

	m.recordPendingResizeCount()
}

// ClearPodResizeInProgressCondition clears the PodResizeInProgress condition for the pod from the cache.
// Returns true if the condition was cleared, false if it was not set.
func (m *manager) ClearPodResizeInProgressCondition(podUID types.UID) bool {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()

	if m.podResizeConditions[podUID].PodResizeInProgress == nil {
		return false
	}

	m.podResizeConditions[podUID] = podResizeConditions{
		PodResizePending:    m.podResizeConditions[podUID].PodResizePending,
		PodResizeInProgress: nil,
	}

	m.recordInProgressResizeCount()
	return true
}

func (m *manager) BackfillPodResizeConditions(pods []*v1.Pod) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()

	for _, pod := range pods {
		for _, c := range pod.Status.Conditions {
			switch c.Type {
			case v1.PodResizePending:
				newCondition := updatedPodResizeCondition(v1.PodResizePending, nil, c.Reason, c.Message, c.ObservedGeneration)
				oldConditions := m.podResizeConditions[pod.UID]

				m.podResizeConditions[pod.UID] = podResizeConditions{
					PodResizePending:    newCondition,
					PodResizeInProgress: oldConditions.PodResizeInProgress,
				}

			case v1.PodResizeInProgress:
				newCondition := updatedPodResizeCondition(v1.PodResizeInProgress, nil, c.Reason, c.Message, c.ObservedGeneration)
				oldConditions := m.podResizeConditions[pod.UID]

				m.podResizeConditions[pod.UID] = podResizeConditions{
					PodResizeInProgress: newCondition,
					PodResizePending:    oldConditions.PodResizePending,
				}
			}
		}
	}
	m.recordPendingResizeCount()
	m.recordInProgressResizeCount()
}

// IsPodResizeDeferred returns true if the pod resize is currently deferred.
func (m *manager) IsPodResizeDeferred(podUID types.UID) bool {
	m.podStatusesLock.RLock()
	defer m.podStatusesLock.RUnlock()

	return m.podResizeConditions[podUID].PodResizePending != nil && m.podResizeConditions[podUID].PodResizePending.Reason == v1.PodReasonDeferred
}

// IsPodResizeInfeasible returns true if the pod resize is currently infeasible.
func (m *manager) IsPodResizeInfeasible(podUID types.UID) bool {
	m.podStatusesLock.RLock()
	defer m.podStatusesLock.RUnlock()

	return m.podResizeConditions[podUID].PodResizePending != nil && m.podResizeConditions[podUID].PodResizePending.Reason == v1.PodReasonInfeasible
}

func (m *manager) GetPodStatus(uid types.UID) (v1.PodStatus, bool) {
	m.podStatusesLock.RLock()
	defer m.podStatusesLock.RUnlock()
	status, ok := m.podStatuses[types.UID(m.podManager.TranslatePodUID(uid))]
	return status.status, ok
}

func (m *manager) SetPodStatus(logger klog.Logger, pod *v1.Pod, status v1.PodStatus) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()

	// Make sure we're caching a deep copy.
	status = *status.DeepCopy()

	// Set the observedGeneration for this pod status.
	status.ObservedGeneration = podutil.CalculatePodStatusObservedGeneration(pod)

	// Force a status update if deletion timestamp is set. This is necessary
	// because if the pod is in the non-running state, the pod worker still
	// needs to be able to trigger an update and/or deletion.
	m.updateStatusInternal(logger, pod, status, pod.DeletionTimestamp != nil, false)
}

func (m *manager) SetContainerReadiness(logger klog.Logger, podUID types.UID, containerID kubecontainer.ContainerID, ready bool) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()

	pod, ok := m.podManager.GetPodByUID(podUID)
	if !ok {
		logger.V(4).Info("Pod has been deleted, no need to update readiness", "podUID", podUID)
		return
	}

	oldStatus, found := m.podStatuses[pod.UID]
	if !found {
		logger.Info("Container readiness changed before pod has synced",
			"pod", klog.KObj(pod),
			"containerID", containerID.String())
		return
	}

	// Find the container to update.
	containerStatus, _, ok := findContainerStatus(&oldStatus.status, containerID.String())
	if !ok {
		logger.Info("Container readiness changed for unknown container",
			"pod", klog.KObj(pod),
			"containerID", containerID.String())
		return
	}

	if containerStatus.Ready == ready {
		logger.V(4).Info("Container readiness unchanged",
			"ready", ready,
			"pod", klog.KObj(pod),
			"containerID", containerID.String())
		return
	}

	// Make sure we're not updating the cached version.
	status := *oldStatus.status.DeepCopy()
	containerStatus, _, _ = findContainerStatus(&status, containerID.String())
	containerStatus.Ready = ready

	// updateConditionFunc updates the corresponding type of condition
	updateConditionFunc := func(conditionType v1.PodConditionType, condition v1.PodCondition) {
		conditionIndex := -1
		for i, condition := range status.Conditions {
			if condition.Type == conditionType {
				conditionIndex = i
				break
			}
		}
		if conditionIndex != -1 {
			status.Conditions[conditionIndex] = condition
		} else {
			logger.Info("PodStatus missing condition type", "conditionType", conditionType, "status", status)
			status.Conditions = append(status.Conditions, condition)
		}
	}

	allContainerStatuses := append(status.InitContainerStatuses, status.ContainerStatuses...)
	updateConditionFunc(v1.PodReady, GeneratePodReadyCondition(pod, &oldStatus.status, status.Conditions, allContainerStatuses, status.Phase))
	updateConditionFunc(v1.ContainersReady, GenerateContainersReadyCondition(pod, &oldStatus.status, allContainerStatuses, status.Phase))
	m.updateStatusInternal(logger, pod, status, false, false)
}

func (m *manager) SetContainerStartup(logger klog.Logger, podUID types.UID, containerID kubecontainer.ContainerID, started bool) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()

	pod, ok := m.podManager.GetPodByUID(podUID)
	if !ok {
		logger.V(4).Info("Pod has been deleted, no need to update startup", "podUID", string(podUID))
		return
	}

	oldStatus, found := m.podStatuses[pod.UID]
	if !found {
		logger.Info("Container startup changed before pod has synced",
			"pod", klog.KObj(pod),
			"containerID", containerID.String())
		return
	}

	// Find the container to update.
	containerStatus, _, ok := findContainerStatus(&oldStatus.status, containerID.String())
	if !ok {
		logger.Info("Container startup changed for unknown container",
			"pod", klog.KObj(pod),
			"containerID", containerID.String())
		return
	}

	if containerStatus.Started != nil && *containerStatus.Started == started {
		logger.V(4).Info("Container startup unchanged",
			"pod", klog.KObj(pod),
			"containerID", containerID.String())
		return
	}

	// Make sure we're not updating the cached version.
	status := *oldStatus.status.DeepCopy()
	containerStatus, _, _ = findContainerStatus(&status, containerID.String())
	containerStatus.Started = &started

	m.updateStatusInternal(logger, pod, status, false, false)
}

func findContainerStatus(status *v1.PodStatus, containerID string) (containerStatus *v1.ContainerStatus, init bool, ok bool) {
	// Find the container to update.
	for i, c := range status.ContainerStatuses {
		if c.ContainerID == containerID {
			return &status.ContainerStatuses[i], false, true
		}
	}

	for i, c := range status.InitContainerStatuses {
		if c.ContainerID == containerID {
			return &status.InitContainerStatuses[i], true, true
		}
	}

	return nil, false, false

}

// TerminatePod ensures that the status of containers is properly defaulted at the end of the pod
// lifecycle. As the Kubelet must reconcile with the container runtime to observe container status
// there is always the possibility we are unable to retrieve one or more container statuses due to
// garbage collection, admin action, or loss of temporary data on a restart. This method ensures
// that any absent container status is treated as a failure so that we do not incorrectly describe
// the pod as successful. If we have not yet initialized the pod in the presence of init containers,
// the init container failure status is sufficient to describe the pod as failing, and we do not need
// to override waiting containers (unless there is evidence the pod previously started those containers).
// It also makes sure that pods are transitioned to a terminal phase (Failed or Succeeded) before
// their deletion.
func (m *manager) TerminatePod(logger klog.Logger, pod *v1.Pod) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()

	// ensure that all containers have a terminated state - because we do not know whether the container
	// was successful, always report an error
	oldStatus := &pod.Status
	cachedStatus, isCached := m.podStatuses[pod.UID]
	if isCached {
		oldStatus = &cachedStatus.status
	}
	status := *oldStatus.DeepCopy()

	// once a pod has initialized, any missing status is treated as a failure
	if hasPodInitialized(logger, pod) {
		for i := range status.ContainerStatuses {
			if status.ContainerStatuses[i].State.Terminated != nil {
				continue
			}
			status.ContainerStatuses[i].State = v1.ContainerState{
				Terminated: &v1.ContainerStateTerminated{
					Reason:   kubecontainer.ContainerReasonStatusUnknown,
					Message:  "The container could not be located when the pod was terminated",
					ExitCode: 137,
				},
			}
		}
	}

	// all but the final suffix of init containers which have no evidence of a container start are
	// marked as failed containers
	for i := range initializedContainers(status.InitContainerStatuses) {
		if status.InitContainerStatuses[i].State.Terminated != nil {
			continue
		}
		status.InitContainerStatuses[i].State = v1.ContainerState{
			Terminated: &v1.ContainerStateTerminated{
				Reason:   kubecontainer.ContainerReasonStatusUnknown,
				Message:  "The container could not be located when the pod was terminated",
				ExitCode: 137,
			},
		}
	}

	// Make sure all pods are transitioned to a terminal phase.
	// TODO(#116484): Also assign terminal phase to static an pods.
	if !kubetypes.IsStaticPod(pod) {
		switch status.Phase {
		case v1.PodSucceeded, v1.PodFailed:
			// do nothing, already terminal
		case v1.PodPending, v1.PodRunning:
			if status.Phase == v1.PodRunning && isCached {
				logger.Info("Terminal running pod should have already been marked as failed, programmer error", "pod", klog.KObj(pod), "podUID", pod.UID)
			}
			logger.V(3).Info("Marking terminal pod as failed", "oldPhase", status.Phase, "pod", klog.KObj(pod), "podUID", pod.UID)
			status.Phase = v1.PodFailed
		default:
			logger.Error(fmt.Errorf("unknown phase: %v", status.Phase), "Unknown phase, programmer error", "pod", klog.KObj(pod), "podUID", pod.UID)
			status.Phase = v1.PodFailed
		}
	}

	logger.V(5).Info("TerminatePod calling updateStatusInternal", "pod", klog.KObj(pod), "podUID", pod.UID)
	m.updateStatusInternal(logger, pod, status, true, true)
}

// hasPodInitialized returns true if the pod has no evidence of ever starting a regular container, which
// implies those containers should not be transitioned to terminated status.
func hasPodInitialized(logger klog.Logger, pod *v1.Pod) bool {
	// a pod without init containers is always initialized
	if len(pod.Spec.InitContainers) == 0 {
		return true
	}
	// if any container has ever moved out of waiting state, the pod has initialized
	for _, status := range pod.Status.ContainerStatuses {
		if status.LastTerminationState.Terminated != nil || status.State.Waiting == nil {
			return true
		}
	}
	// if the last init container has ever completed with a zero exit code, the pod is initialized
	if l := len(pod.Status.InitContainerStatuses); l > 0 {
		container, ok := kubeutil.GetContainerByIndex(pod.Spec.InitContainers, pod.Status.InitContainerStatuses, l-1)
		if !ok {
			logger.V(4).Info("Mismatch between pod spec and status, likely programmer error", "pod", klog.KObj(pod), "containerName", container.Name)
			return false
		}

		containerStatus := pod.Status.InitContainerStatuses[l-1]
		if podutil.IsRestartableInitContainer(&container) {
			if containerStatus.State.Running != nil &&
				containerStatus.Started != nil && *containerStatus.Started {
				return true
			}
		} else { // regular init container
			if state := containerStatus.LastTerminationState; state.Terminated != nil && state.Terminated.ExitCode == 0 {
				return true
			}
			if state := containerStatus.State; state.Terminated != nil && state.Terminated.ExitCode == 0 {
				return true
			}
		}
	}
	// otherwise the pod has no record of being initialized
	return false
}

// initializedContainers returns all status except for suffix of containers that are in Waiting
// state, which is the set of containers that have attempted to start at least once. If all containers
// are Waiting, the first container is always returned.
func initializedContainers(containers []v1.ContainerStatus) []v1.ContainerStatus {
	for i := len(containers) - 1; i >= 0; i-- {
		if containers[i].State.Waiting == nil || containers[i].LastTerminationState.Terminated != nil {
			return containers[0 : i+1]
		}
	}
	// always return at least one container
	if len(containers) > 0 {
		return containers[0:1]
	}
	return nil
}

// checkContainerStateTransition ensures that no container is trying to transition
// from a terminated to non-terminated state, which is illegal and indicates a
// logical error in the kubelet.
func checkContainerStateTransition(oldStatuses, newStatuses *v1.PodStatus, podSpec *v1.PodSpec) error {
	// If we should always restart, containers are allowed to leave the terminated state
	if podSpec.RestartPolicy == v1.RestartPolicyAlways {
		return nil
	}
	for _, oldStatus := range oldStatuses.ContainerStatuses {
		// Skip any container that wasn't terminated
		if oldStatus.State.Terminated == nil {
			continue
		}
		// Skip any container that failed but is allowed to restart
		if oldStatus.State.Terminated.ExitCode != 0 && podSpec.RestartPolicy == v1.RestartPolicyOnFailure {
			continue
		}
		// Skip any container that is allowed to restart by container restart policy
		if utilfeature.DefaultFeatureGate.Enabled(features.ContainerRestartRules) {
			restartable := false
			for _, container := range podSpec.Containers {
				if container.Name == oldStatus.Name && podutil.ContainerShouldRestart(container, *podSpec, oldStatus.State.Terminated.ExitCode) {
					restartable = true
				}
			}
			if restartable {
				continue
			}
		}
		for _, newStatus := range newStatuses.ContainerStatuses {
			if oldStatus.Name == newStatus.Name && newStatus.State.Terminated == nil {
				return fmt.Errorf("terminated container %v attempted illegal transition to non-terminated state", newStatus.Name)
			}
		}
	}

	for i, oldStatus := range oldStatuses.InitContainerStatuses {
		initContainer, ok := kubeutil.GetContainerByIndex(podSpec.InitContainers, oldStatuses.InitContainerStatuses, i)
		if !ok {
			return fmt.Errorf("found mismatch between pod spec and status, container: %v", oldStatus.Name)
		}
		// Skip any restartable init container as it always is allowed to restart
		if podutil.IsRestartableInitContainer(&initContainer) {
			continue
		}
		// Skip any container that wasn't terminated
		if oldStatus.State.Terminated == nil {
			continue
		}
		// Skip any container that failed but is allowed to restart
		if oldStatus.State.Terminated.ExitCode != 0 && podSpec.RestartPolicy == v1.RestartPolicyOnFailure {
			continue
		}
		// Skip any container that is allowed to restart by container restart policy
		if utilfeature.DefaultFeatureGate.Enabled(features.ContainerRestartRules) {
			restartable := false
			for _, container := range podSpec.InitContainers {
				if container.Name == oldStatus.Name && podutil.ContainerShouldRestart(container, *podSpec, oldStatus.State.Terminated.ExitCode) {
					restartable = true
				}
			}
			if restartable {
				continue
			}
		}
		for _, newStatus := range newStatuses.InitContainerStatuses {
			if oldStatus.Name == newStatus.Name && newStatus.State.Terminated == nil {
				return fmt.Errorf("terminated init container %v attempted illegal transition to non-terminated state", newStatus.Name)
			}
		}
	}
	return nil
}

// updateStatusInternal updates the internal status cache, and queues an update to the api server if
// necessary.
// This method IS NOT THREAD SAFE and must be called from a locked function.
func (m *manager) updateStatusInternal(logger klog.Logger, pod *v1.Pod, status v1.PodStatus, forceUpdate, podIsFinished bool) {
	var oldStatus v1.PodStatus
	cachedStatus, isCached := m.podStatuses[pod.UID]
	if isCached {
		oldStatus = cachedStatus.status
		// TODO(#116484): Also assign terminal phase to static pods.
		if !kubetypes.IsStaticPod(pod) {
			if cachedStatus.podIsFinished && !podIsFinished {
				logger.Info("Got unexpected podIsFinished=false, while podIsFinished=true in status cache, programmer error", "pod", klog.KObj(pod))
				podIsFinished = true
			}
		}
	} else if mirrorPod, ok := m.podManager.GetMirrorPodByPod(pod); ok {
		oldStatus = mirrorPod.Status
	} else {
		oldStatus = pod.Status
	}

	// Check for illegal state transition in containers
	if err := checkContainerStateTransition(&oldStatus, &status, &pod.Spec); err != nil {
		logger.Error(err, "Status update on pod aborted", "pod", klog.KObj(pod))
		return
	}

	// Set ContainersReadyCondition.LastTransitionTime.
	updateLastTransitionTime(&status, &oldStatus, v1.ContainersReady)

	// Set ReadyCondition.LastTransitionTime.
	updateLastTransitionTime(&status, &oldStatus, v1.PodReady)

	// Set InitializedCondition.LastTransitionTime.
	updateLastTransitionTime(&status, &oldStatus, v1.PodInitialized)

	// Set PodReadyToStartContainersCondition.LastTransitionTime.
	updateLastTransitionTime(&status, &oldStatus, v1.PodReadyToStartContainers)

	// Set PodScheduledCondition.LastTransitionTime.
	updateLastTransitionTime(&status, &oldStatus, v1.PodScheduled)

	// Set DisruptionTarget.LastTransitionTime.
	updateLastTransitionTime(&status, &oldStatus, v1.DisruptionTarget)

	// ensure that the start time does not change across updates.
	if oldStatus.StartTime != nil && !oldStatus.StartTime.IsZero() {
		status.StartTime = oldStatus.StartTime
	} else if status.StartTime.IsZero() {
		// if the status has no start time, we need to set an initial time
		now := metav1.Now()
		status.StartTime = &now
	}

	// prevent sending unnecessary patches
	if oldStatus.ObservedGeneration > status.ObservedGeneration {
		status.ObservedGeneration = oldStatus.ObservedGeneration
	}

	normalizeStatus(pod, &status)

	// Perform some more extensive logging of container termination state to assist in
	// debugging production races (generally not needed).
	if loggerV := logger.V(5); loggerV.Enabled() {
		var containers []string
		for _, s := range append(append([]v1.ContainerStatus(nil), status.InitContainerStatuses...), status.ContainerStatuses...) {
			var current, previous string
			switch {
			case s.State.Running != nil:
				current = "running"
			case s.State.Waiting != nil:
				current = "waiting"
			case s.State.Terminated != nil:
				current = fmt.Sprintf("terminated=%d", s.State.Terminated.ExitCode)
			default:
				current = "unknown"
			}
			switch {
			case s.LastTerminationState.Running != nil:
				previous = "running"
			case s.LastTerminationState.Waiting != nil:
				previous = "waiting"
			case s.LastTerminationState.Terminated != nil:
				previous = fmt.Sprintf("terminated=%d", s.LastTerminationState.Terminated.ExitCode)
			default:
				previous = "<none>"
			}
			containers = append(containers, fmt.Sprintf("(%s state=%s previous=%s)", s.Name, current, previous))
		}
		sort.Strings(containers)
		loggerV.Info("updateStatusInternal", "version", cachedStatus.version+1, "podIsFinished", podIsFinished, "pod", klog.KObj(pod), "podUID", pod.UID, "containers", strings.Join(containers, " "))
	}

	// The intent here is to prevent concurrent updates to a pod's status from
	// clobbering each other so the phase of a pod progresses monotonically.
	if isCached && isPodStatusByKubeletEqual(&cachedStatus.status, &status) && !forceUpdate {
		logger.V(3).Info("Ignoring same status for pod", "pod", klog.KObj(pod), "status", status)
		return
	}

	newStatus := versionedPodStatus{
		status:        status,
		version:       cachedStatus.version + 1,
		podName:       pod.Name,
		podNamespace:  pod.Namespace,
		podIsFinished: podIsFinished,
	}

	// Multiple status updates can be generated before we update the API server,
	// so we track the time from the first status update until we retire it to
	// the API.
	if cachedStatus.at.IsZero() {
		newStatus.at = time.Now()
	} else {
		newStatus.at = cachedStatus.at
	}

	m.podStatuses[pod.UID] = newStatus

	select {
	case m.podStatusChannel <- struct{}{}:
	default:
		// there's already a status update pending
	}
}

// updateLastTransitionTime updates the LastTransitionTime of a pod condition.
func updateLastTransitionTime(status, oldStatus *v1.PodStatus, conditionType v1.PodConditionType) {
	_, condition := podutil.GetPodCondition(status, conditionType)
	if condition == nil {
		return
	}
	// Need to set LastTransitionTime.
	lastTransitionTime := metav1.Now()
	_, oldCondition := podutil.GetPodCondition(oldStatus, conditionType)
	if oldCondition != nil && condition.Status == oldCondition.Status {
		lastTransitionTime = oldCondition.LastTransitionTime
	}
	condition.LastTransitionTime = lastTransitionTime
}

// deletePodStatus simply removes the given pod from the status cache.
func (m *manager) deletePodStatus(uid types.UID) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()
	delete(m.podStatuses, uid)
	m.podStartupLatencyHelper.DeletePodStartupState(uid)
	if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
		if _, exists := m.podResizeConditions[uid]; exists {
			delete(m.podResizeConditions, uid)
			m.recordInProgressResizeCount()
			m.recordPendingResizeCount()
		}
	}
}

// TODO(filipg): It'd be cleaner if we can do this without signal from user.
func (m *manager) RemoveOrphanedStatuses(logger klog.Logger, podUIDs map[types.UID]bool) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()
	for key := range m.podStatuses {
		if _, ok := podUIDs[key]; !ok {
			logger.V(5).Info("Removing pod from status map", "podUID", key)
			delete(m.podStatuses, key)
			if utilfeature.DefaultFeatureGate.Enabled(features.InPlacePodVerticalScaling) {
				if _, exists := m.podResizeConditions[key]; exists {
					delete(m.podResizeConditions, key)
					m.recordInProgressResizeCount()
					m.recordPendingResizeCount()
				}
			}
		}
	}
}

// syncBatch syncs pods statuses with the apiserver. Returns the number of syncs
// attempted for testing.
func (m *manager) syncBatch(ctx context.Context, all bool) int {
	logger := klog.FromContext(ctx)
	type podSync struct {
		podUID    types.UID
		statusUID kubetypes.MirrorPodUID
		status    versionedPodStatus
	}

	var updatedStatuses []podSync
	podToMirror, mirrorToPod := m.podManager.GetUIDTranslations()
	func() { // Critical section
		m.podStatusesLock.RLock()
		defer m.podStatusesLock.RUnlock()

		// Clean up orphaned versions.
		if all {
			for uid := range m.apiStatusVersions {
				_, hasPod := m.podStatuses[types.UID(uid)]
				_, hasMirror := mirrorToPod[uid]
				if !hasPod && !hasMirror {
					delete(m.apiStatusVersions, uid)
				}
			}
		}

		// Decide which pods need status updates.
		for uid, status := range m.podStatuses {
			// translate the pod UID (source) to the status UID (API pod) -
			// static pods are identified in source by pod UID but tracked in the
			// API via the uid of the mirror pod
			uidOfStatus := kubetypes.MirrorPodUID(uid)
			if mirrorUID, ok := podToMirror[kubetypes.ResolvedPodUID(uid)]; ok {
				if mirrorUID == "" {
					logger.V(5).Info("Static pod does not have a corresponding mirror pod; skipping",
						"podUID", uid,
						"pod", klog.KRef(status.podNamespace, status.podName))
					continue
				}
				uidOfStatus = mirrorUID
			}

			// if a new status update has been delivered, trigger an update, otherwise the
			// pod can wait for the next bulk check (which performs reconciliation as well)
			if !all {
				if m.apiStatusVersions[uidOfStatus] >= status.version {
					continue
				}
				updatedStatuses = append(updatedStatuses, podSync{uid, uidOfStatus, status})
				continue
			}

			// Ensure that any new status, or mismatched status, or pod that is ready for
			// deletion gets updated. If a status update fails we retry the next time any
			// other pod is updated.
			if m.needsUpdate(logger, types.UID(uidOfStatus), status) {
				updatedStatuses = append(updatedStatuses, podSync{uid, uidOfStatus, status})
			} else if m.needsReconcile(logger, uid, status.status) {
				// Delete the apiStatusVersions here to force an update on the pod status
				// In most cases the deleted apiStatusVersions here should be filled
				// soon after the following syncPod() [If the syncPod() sync an update
				// successfully].
				delete(m.apiStatusVersions, uidOfStatus)
				updatedStatuses = append(updatedStatuses, podSync{uid, uidOfStatus, status})
			}
		}
	}()

	for _, update := range updatedStatuses {
		logger.V(5).Info("Sync pod status", "podUID", update.podUID, "statusUID", update.statusUID, "version", update.status.version)
		m.syncPod(ctx, update.podUID, update.status)
	}

	return len(updatedStatuses)
}

// syncPod syncs the given status with the API server. The caller must not hold the status lock.
func (m *manager) syncPod(ctx context.Context, uid types.UID, status versionedPodStatus) {
	logger := klog.FromContext(ctx)
	// TODO: make me easier to express from client code
	pod, err := m.kubeClient.CoreV1().Pods(status.podNamespace).Get(ctx, status.podName, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		logger.V(3).Info("Pod does not exist on the server",
			"podUID", uid,
			"pod", klog.KRef(status.podNamespace, status.podName))
		// If the Pod is deleted the status will be cleared in
		// RemoveOrphanedStatuses, so we just ignore the update here.
		return
	}
	if err != nil {
		logger.Error(err, "Failed to get status for pod",
			"podUID", uid,
			"pod", klog.KRef(status.podNamespace, status.podName))
		return
	}

	translatedUID := m.podManager.TranslatePodUID(pod.UID)
	// Type convert original uid just for the purpose of comparison.
	if len(translatedUID) > 0 && translatedUID != kubetypes.ResolvedPodUID(uid) {
		logger.V(2).Info("Pod was deleted and then recreated, skipping status update",
			"pod", klog.KObj(pod),
			"oldPodUID", uid,
			"podUID", translatedUID)
		m.deletePodStatus(uid)
		return
	}

	mergedStatus := mergePodStatus(pod, pod.Status, status.status, m.podDeletionSafety.PodCouldHaveRunningContainers(pod))

	newPod, patchBytes, unchanged, err := statusutil.PatchPodStatus(ctx, m.kubeClient, pod.Namespace, pod.Name, pod.UID, pod.Status, mergedStatus)
	logger.V(3).Info("Patch status for pod", "pod", klog.KObj(pod), "podUID", uid, "patch", string(patchBytes))

	if err != nil {
		logger.Error(err, "Failed to update status for pod", "pod", klog.KObj(pod))
		return
	}
	if unchanged {
		logger.V(3).Info("Status for pod is up-to-date", "pod", klog.KObj(pod), "statusVersion", status.version)
	} else {
		logger.V(3).Info("Status for pod updated successfully", "pod", klog.KObj(pod), "statusVersion", status.version, "status", mergedStatus)
		pod = newPod
		// We pass a new object (result of API call which contains updated ResourceVersion)
		m.podStartupLatencyHelper.RecordStatusUpdated(pod)
	}

	// measure how long the status update took to propagate from generation to update on the server
	if status.at.IsZero() {
		logger.V(3).Info("Pod had no status time set", "pod", klog.KObj(pod), "podUID", uid, "version", status.version)
	} else {
		duration := time.Since(status.at).Truncate(time.Millisecond)
		metrics.PodStatusSyncDuration.Observe(duration.Seconds())
	}

	m.apiStatusVersions[kubetypes.MirrorPodUID(pod.UID)] = status.version

	// We don't handle graceful deletion of mirror pods.
	if m.canBeDeleted(logger, pod, status.status, status.podIsFinished) {
		deleteOptions := metav1.DeleteOptions{
			GracePeriodSeconds: new(int64),
			// Use the pod UID as the precondition for deletion to prevent deleting a
			// newly created pod with the same name and namespace.
			Preconditions: metav1.NewUIDPreconditions(string(pod.UID)),
		}
		err = m.kubeClient.CoreV1().Pods(pod.Namespace).Delete(ctx, pod.Name, deleteOptions)
		if err != nil {
			logger.Info("Failed to delete status for pod", "pod", klog.KObj(pod), "err", err)
			return
		}
		logger.V(3).Info("Pod fully terminated and removed from etcd", "pod", klog.KObj(pod))
		m.deletePodStatus(uid)
	}
}

// needsUpdate returns whether the status is stale for the given pod UID.
// This method is not thread safe, and must only be accessed by the sync thread.
func (m *manager) needsUpdate(logger klog.Logger, uid types.UID, status versionedPodStatus) bool {
	latest, ok := m.apiStatusVersions[kubetypes.MirrorPodUID(uid)]
	if !ok || latest < status.version {
		return true
	}
	pod, ok := m.podManager.GetPodByUID(uid)
	if !ok {
		return false
	}
	return m.canBeDeleted(logger, pod, status.status, status.podIsFinished)
}

func (m *manager) canBeDeleted(logger klog.Logger, pod *v1.Pod, status v1.PodStatus, podIsFinished bool) bool {
	if pod.DeletionTimestamp == nil || kubetypes.IsMirrorPod(pod) {
		return false
	}
	// Delay deletion of pods until the phase is terminal, based on pod.Status
	// which comes from pod manager.
	if !podutil.IsPodPhaseTerminal(pod.Status.Phase) {
		// For debugging purposes we also log the kubelet's local phase, when the deletion is delayed.
		logger.V(3).Info("Delaying pod deletion as the phase is non-terminal", "phase", pod.Status.Phase, "localPhase", status.Phase, "pod", klog.KObj(pod), "podUID", pod.UID)
		return false
	}
	// If this is an update completing pod termination then we know the pod termination is finished.
	if podIsFinished {
		logger.V(3).Info("The pod termination is finished as SyncTerminatedPod completes its execution", "phase", pod.Status.Phase, "localPhase", status.Phase, "pod", klog.KObj(pod), "podUID", pod.UID)
		return true
	}
	return false
}

// needsReconcile compares the given status with the status in the pod manager (which
// in fact comes from apiserver), returns whether the status needs to be reconciled with
// the apiserver. Now when pod status is inconsistent between apiserver and kubelet,
// kubelet should forcibly send an update to reconcile the inconsistence, because kubelet
// should be the source of truth of pod status.
// NOTE(random-liu): It's simpler to pass in mirror pod uid and get mirror pod by uid, but
// now the pod manager only supports getting mirror pod by static pod, so we have to pass
// static pod uid here.
// TODO(random-liu): Simplify the logic when mirror pod manager is added.
func (m *manager) needsReconcile(logger klog.Logger, uid types.UID, status v1.PodStatus) bool {
	// The pod could be a static pod, so we should translate first.
	pod, ok := m.podManager.GetPodByUID(uid)
	if !ok {
		logger.V(4).Info("Pod has been deleted, no need to reconcile", "podUID", uid)
		return false
	}
	// If the pod is a static pod, we should check its mirror pod, because only status in mirror pod is meaningful to us.
	if kubetypes.IsStaticPod(pod) {
		mirrorPod, ok := m.podManager.GetMirrorPodByPod(pod)
		if !ok {
			logger.V(4).Info("Static pod has no corresponding mirror pod, no need to reconcile", "pod", klog.KObj(pod))
			return false
		}
		pod = mirrorPod
	}

	podStatus := pod.Status.DeepCopy()
	normalizeStatus(pod, podStatus)

	if isPodStatusByKubeletEqual(podStatus, &status) {
		// If the status from the source is the same with the cached status,
		// reconcile is not needed. Just return.
		return false
	}
	logger.V(3).Info("Pod status is inconsistent with cached status for pod, a reconciliation should be triggered",
		"pod", klog.KObj(pod),
		"statusDiff", diff.Diff(podStatus, &status))

	return true
}

// normalizeStatus normalizes nanosecond precision timestamps in podStatus
// down to second precision (*RFC339NANO* -> *RFC3339*). This must be done
// before comparing podStatus to the status returned by apiserver because
// apiserver does not support RFC339NANO.
// Related issue #15262/PR #15263 to move apiserver to RFC339NANO is closed.
func normalizeStatus(pod *v1.Pod, status *v1.PodStatus) *v1.PodStatus {
	bytesPerStatus := kubecontainer.MaxPodTerminationMessageLogLength
	if containers := len(pod.Spec.Containers) + len(pod.Spec.InitContainers) + len(pod.Spec.EphemeralContainers); containers > 0 {
		bytesPerStatus = bytesPerStatus / containers
	}
	normalizeTimeStamp := func(t *metav1.Time) {
		*t = t.Rfc3339Copy()
	}
	normalizeContainerState := func(c *v1.ContainerState) {
		if c.Running != nil {
			normalizeTimeStamp(&c.Running.StartedAt)
		}
		if c.Terminated != nil {
			normalizeTimeStamp(&c.Terminated.StartedAt)
			normalizeTimeStamp(&c.Terminated.FinishedAt)
			if len(c.Terminated.Message) > bytesPerStatus {
				c.Terminated.Message = c.Terminated.Message[:bytesPerStatus]
			}
		}
	}

	if status.StartTime != nil {
		normalizeTimeStamp(status.StartTime)
	}
	for i := range status.Conditions {
		condition := &status.Conditions[i]
		normalizeTimeStamp(&condition.LastProbeTime)
		normalizeTimeStamp(&condition.LastTransitionTime)
	}

	normalizeContainerStatuses := func(containerStatuses []v1.ContainerStatus) {
		for i := range containerStatuses {
			cstatus := &containerStatuses[i]
			normalizeContainerState(&cstatus.State)
			normalizeContainerState(&cstatus.LastTerminationState)
		}
	}

	normalizeContainerStatuses(status.ContainerStatuses)
	sort.Sort(kubetypes.SortedContainerStatuses(status.ContainerStatuses))

	normalizeContainerStatuses(status.InitContainerStatuses)
	kubetypes.SortInitContainerStatuses(pod, status.InitContainerStatuses)

	normalizeContainerStatuses(status.EphemeralContainerStatuses)
	sort.Sort(kubetypes.SortedContainerStatuses(status.EphemeralContainerStatuses))

	return status
}

// mergePodStatus merges oldPodStatus and newPodStatus to preserve where pod conditions
// not owned by kubelet and to ensure terminal phase transition only happens after all
// running containers have terminated. This method does not modify the old status.
func mergePodStatus(pod *v1.Pod, oldPodStatus, newPodStatus v1.PodStatus, couldHaveRunningContainers bool) v1.PodStatus {
	podConditions := make([]v1.PodCondition, 0, len(oldPodStatus.Conditions)+len(newPodStatus.Conditions))

	for _, c := range oldPodStatus.Conditions {
		if !kubetypes.PodConditionByKubelet(c.Type) {
			podConditions = append(podConditions, c)
		}
	}

	transitioningToTerminalPhase := !podutil.IsPodPhaseTerminal(oldPodStatus.Phase) && podutil.IsPodPhaseTerminal(newPodStatus.Phase)

	for _, c := range newPodStatus.Conditions {
		if kubetypes.PodConditionByKubelet(c.Type) {
			podConditions = append(podConditions, c)
		} else if kubetypes.PodConditionSharedByKubelet(c.Type) {
			// we replace or append all the "shared by kubelet" conditions
			if c.Type == v1.DisruptionTarget {
				// guard the update of the DisruptionTarget condition with a check to ensure
				// it will only be sent once all containers have terminated and the phase
				// is terminal. This avoids sending an unnecessary patch request to add
				// the condition if the actual status phase transition is delayed.
				if transitioningToTerminalPhase && !couldHaveRunningContainers {
					// update the LastTransitionTime again here because the older transition
					// time set in updateStatusInternal is likely stale as sending of
					// the condition was delayed until all pod's containers have terminated.
					updateLastTransitionTime(&newPodStatus, &oldPodStatus, c.Type)
					if _, c := podutil.GetPodConditionFromList(newPodStatus.Conditions, c.Type); c != nil {
						// for shared conditions we update or append in podConditions
						podConditions = statusutil.ReplaceOrAppendPodCondition(podConditions, c)
					}
				}
			}
		}
	}
	newPodStatus.Conditions = podConditions

	// ResourceClaimStatuses is not owned and not modified by kubelet.
	newPodStatus.ResourceClaimStatuses = oldPodStatus.ResourceClaimStatuses
	// ExtendedResourceClaimStatus is not owned and not modified by kubelet.
	newPodStatus.ExtendedResourceClaimStatus = oldPodStatus.ExtendedResourceClaimStatus

	// Delay transitioning a pod to a terminal status unless the pod is actually terminal.
	// The Kubelet should never transition a pod to terminal status that could have running
	// containers and thus actively be leveraging exclusive resources. Note that resources
	// like volumes are reconciled by a subsystem in the Kubelet and will converge if a new
	// pod reuses an exclusive resource (unmount -> free -> mount), which means we do not
	// need wait for those resources to be detached by the Kubelet. In general, resources
	// the Kubelet exclusively owns must be released prior to a pod being reported terminal,
	// while resources that have participanting components above the API use the pod's
	// transition to a terminal phase (or full deletion) to release those resources.
	if transitioningToTerminalPhase {
		if couldHaveRunningContainers {
			newPodStatus.Phase = oldPodStatus.Phase
			newPodStatus.Reason = oldPodStatus.Reason
			newPodStatus.Message = oldPodStatus.Message
		}
	}

	// If the new phase is terminal, explicitly set the ready condition to false for v1.PodReady and v1.ContainersReady.
	// It may take some time for kubelet to reconcile the ready condition, so explicitly set ready conditions to false if the phase is terminal.
	// This is done to ensure kubelet does not report a status update with terminal pod phase and ready=true.
	// See https://issues.k8s.io/108594 for more details.
	if podutil.IsPodPhaseTerminal(newPodStatus.Phase) {
		if podutil.IsPodReadyConditionTrue(newPodStatus) || podutil.IsContainersReadyConditionTrue(newPodStatus) {
			containersReadyCondition := generateContainersReadyConditionForTerminalPhase(pod, &oldPodStatus, newPodStatus.Phase)
			podutil.UpdatePodCondition(&newPodStatus, &containersReadyCondition)

			podReadyCondition := generatePodReadyConditionForTerminalPhase(pod, &oldPodStatus, newPodStatus.Phase)
			podutil.UpdatePodCondition(&newPodStatus, &podReadyCondition)
		}
	}

	return newPodStatus
}

// NeedToReconcilePodReadiness returns if the pod "Ready" condition need to be reconcile
func NeedToReconcilePodReadiness(pod *v1.Pod) bool {
	if len(pod.Spec.ReadinessGates) == 0 {
		return false
	}
	podReadyCondition := GeneratePodReadyCondition(pod, &pod.Status, pod.Status.Conditions, pod.Status.ContainerStatuses, pod.Status.Phase)
	i, curCondition := podutil.GetPodConditionFromList(pod.Status.Conditions, v1.PodReady)
	// Only reconcile if "Ready" condition is present and Status or Message is not expected
	if i >= 0 && (curCondition.Status != podReadyCondition.Status || curCondition.Message != podReadyCondition.Message) {
		return true
	}
	return false
}

func updatedPodResizeCondition(conditionType v1.PodConditionType, oldCondition *v1.PodCondition, reason, message string, observedGeneration int64) *v1.PodCondition {
	now := metav1.NewTime(time.Now())
	var lastTransitionTime metav1.Time
	if oldCondition == nil || oldCondition.Reason != reason {
		lastTransitionTime = now
	} else {
		lastTransitionTime = oldCondition.LastTransitionTime
	}

	return &v1.PodCondition{
		Type:               conditionType,
		Status:             v1.ConditionTrue,
		LastProbeTime:      now,
		LastTransitionTime: lastTransitionTime,
		ObservedGeneration: observedGeneration,
		Reason:             reason,
		Message:            message,
	}
}

// recordPendingResizeCount sets the pending resize metric.
func (m *manager) recordPendingResizeCount() {
	pendingResizeCount := make(map[string]int)
	for _, conditions := range m.podResizeConditions {
		if conditions.PodResizePending != nil {
			pendingResizeCount[strings.ToLower(conditions.PodResizePending.Reason)]++
		}
	}

	metrics.PodPendingResizes.Reset()
	for reason, count := range pendingResizeCount {
		metrics.PodPendingResizes.WithLabelValues(reason).Set(float64(count))
	}
}

// recordInProgressResize sets the in-progress resize metric.
func (m *manager) recordInProgressResizeCount() {
	inProgressResizeCount := 0
	for _, conditions := range m.podResizeConditions {
		if conditions.PodResizeInProgress != nil {
			inProgressResizeCount++
		}
	}
	metrics.PodInProgressResizes.Set(float64(inProgressResizeCount))
}
