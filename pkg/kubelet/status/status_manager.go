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

package status

import (
	"sort"
	"sync"
	"time"

	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/diff"
	"k8s.io/kubernetes/pkg/util/wait"
)

// A wrapper around api.PodStatus that includes a version to enforce that stale pod statuses are
// not sent to the API server.
type versionedPodStatus struct {
	status api.PodStatus
	// Monotonically increasing version number (per pod).
	version uint64
	// Pod name & namespace, for sending updates to API server.
	podName      string
	podNamespace string
}

type podStatusSyncRequest struct {
	podUID types.UID
	status versionedPodStatus
}

// Updates pod statuses in apiserver. Writes only when new status has changed.
// All methods are thread-safe.
type manager struct {
	kubeClient clientset.Interface
	podManager kubepod.Manager
	// Map from pod UID to sync status of the corresponding pod.
	podStatuses      map[types.UID]versionedPodStatus
	podStatusesLock  sync.RWMutex
	podStatusChannel chan podStatusSyncRequest
	// Map from (mirror) pod UID to latest status version successfully sent to the API server.
	// apiStatusVersions must only be accessed from the sync thread.
	apiStatusVersions map[types.UID]uint64
}

// PodStatusProvider knows how to provide status for a pod.  It's intended to be used by other components
// that need to introspect status.
type PodStatusProvider interface {
	// GetPodStatus returns the cached status for the provided pod UID, as well as whether it
	// was a cache hit.
	GetPodStatus(uid types.UID) (api.PodStatus, bool)
}

// Manager is the Source of truth for kubelet pod status, and should be kept up-to-date with
// the latest api.PodStatus. It also syncs updates back to the API server.
type Manager interface {
	PodStatusProvider

	// Start the API server status sync loop.
	Start()

	// SetPodStatus caches updates the cached status for the given pod, and triggers a status update.
	SetPodStatus(pod *api.Pod, status api.PodStatus)

	// SetContainerReadiness updates the cached container status with the given readiness, and
	// triggers a status update.
	SetContainerReadiness(podUID types.UID, containerID kubecontainer.ContainerID, ready bool)

	// TerminatePod resets the container status for the provided pod to terminated and triggers
	// a status update.
	TerminatePod(pod *api.Pod)

	// RemoveOrphanedStatuses scans the status cache and removes any entries for pods not included in
	// the provided podUIDs.
	RemoveOrphanedStatuses(podUIDs map[types.UID]bool)
}

const syncPeriod = 10 * time.Second

func NewManager(kubeClient clientset.Interface, podManager kubepod.Manager) Manager {
	return &manager{
		kubeClient:        kubeClient,
		podManager:        podManager,
		podStatuses:       make(map[types.UID]versionedPodStatus),
		podStatusChannel:  make(chan podStatusSyncRequest, 1000), // Buffer up to 1000 statuses
		apiStatusVersions: make(map[types.UID]uint64),
	}
}

// isStatusEqual returns true if the given pod statuses are equal, false otherwise.
// This method normalizes the status before comparing so as to make sure that meaningless
// changes will be ignored.
func isStatusEqual(oldStatus, status *api.PodStatus) bool {
	return api.Semantic.DeepEqual(status, oldStatus)
}

func (m *manager) Start() {
	// Don't start the status manager if we don't have a client. This will happen
	// on the master, where the kubelet is responsible for bootstrapping the pods
	// of the master components.
	if m.kubeClient == nil {
		glog.Infof("Kubernetes client is nil, not starting status manager.")
		return
	}

	glog.Info("Starting to sync pod status with apiserver")
	syncTicker := time.Tick(syncPeriod)
	// syncPod and syncBatch share the same go routine to avoid sync races.
	go wait.Forever(func() {
		select {
		case syncRequest := <-m.podStatusChannel:
			m.syncPod(syncRequest.podUID, syncRequest.status)
		case <-syncTicker:
			m.syncBatch()
		}
	}, 0)
}

func (m *manager) GetPodStatus(uid types.UID) (api.PodStatus, bool) {
	m.podStatusesLock.RLock()
	defer m.podStatusesLock.RUnlock()
	status, ok := m.podStatuses[m.podManager.TranslatePodUID(uid)]
	return status.status, ok
}

func (m *manager) SetPodStatus(pod *api.Pod, status api.PodStatus) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()
	// Make sure we're caching a deep copy.
	status, err := copyStatus(&status)
	if err != nil {
		return
	}
	// Force a status update if deletion timestamp is set. This is necessary
	// because if the pod is in the non-running state, the pod worker still
	// needs to be able to trigger an update and/or deletion.
	m.updateStatusInternal(pod, status, pod.DeletionTimestamp != nil)
}

func (m *manager) SetContainerReadiness(podUID types.UID, containerID kubecontainer.ContainerID, ready bool) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()

	pod, ok := m.podManager.GetPodByUID(podUID)
	if !ok {
		glog.V(4).Infof("Pod %q has been deleted, no need to update readiness", string(podUID))
		return
	}

	oldStatus, found := m.podStatuses[pod.UID]
	if !found {
		glog.Warningf("Container readiness changed before pod has synced: %q - %q",
			format.Pod(pod), containerID.String())
		return
	}

	// Find the container to update.
	containerStatus, _, ok := findContainerStatus(&oldStatus.status, containerID.String())
	if !ok {
		glog.Warningf("Container readiness changed for unknown container: %q - %q",
			format.Pod(pod), containerID.String())
		return
	}

	if containerStatus.Ready == ready {
		glog.V(4).Infof("Container readiness unchanged (%v): %q - %q", ready,
			format.Pod(pod), containerID.String())
		return
	}

	// Make sure we're not updating the cached version.
	status, err := copyStatus(&oldStatus.status)
	if err != nil {
		return
	}
	containerStatus, _, _ = findContainerStatus(&status, containerID.String())
	containerStatus.Ready = ready

	// Update pod condition.
	readyConditionIndex := -1
	for i, condition := range status.Conditions {
		if condition.Type == api.PodReady {
			readyConditionIndex = i
			break
		}
	}
	readyCondition := GeneratePodReadyCondition(&pod.Spec, status.ContainerStatuses, status.Phase)
	if readyConditionIndex != -1 {
		status.Conditions[readyConditionIndex] = readyCondition
	} else {
		glog.Warningf("PodStatus missing PodReady condition: %+v", status)
		status.Conditions = append(status.Conditions, readyCondition)
	}

	m.updateStatusInternal(pod, status, false)
}

func findContainerStatus(status *api.PodStatus, containerID string) (containerStatus *api.ContainerStatus, init bool, ok bool) {
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

func (m *manager) TerminatePod(pod *api.Pod) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()
	oldStatus := &pod.Status
	if cachedStatus, ok := m.podStatuses[pod.UID]; ok {
		oldStatus = &cachedStatus.status
	}
	status, err := copyStatus(oldStatus)
	if err != nil {
		return
	}
	for i := range status.ContainerStatuses {
		status.ContainerStatuses[i].State = api.ContainerState{
			Terminated: &api.ContainerStateTerminated{},
		}
	}
	for i := range status.InitContainerStatuses {
		status.InitContainerStatuses[i].State = api.ContainerState{
			Terminated: &api.ContainerStateTerminated{},
		}
	}
	m.updateStatusInternal(pod, pod.Status, true)
}

// updateStatusInternal updates the internal status cache, and queues an update to the api server if
// necessary. Returns whether an update was triggered.
// This method IS NOT THREAD SAFE and must be called from a locked function.
func (m *manager) updateStatusInternal(pod *api.Pod, status api.PodStatus, forceUpdate bool) bool {
	var oldStatus api.PodStatus
	cachedStatus, isCached := m.podStatuses[pod.UID]
	if isCached {
		oldStatus = cachedStatus.status
	} else if mirrorPod, ok := m.podManager.GetMirrorPodByPod(pod); ok {
		oldStatus = mirrorPod.Status
	} else {
		oldStatus = pod.Status
	}

	// Set ReadyCondition.LastTransitionTime.
	if _, readyCondition := api.GetPodCondition(&status, api.PodReady); readyCondition != nil {
		// Need to set LastTransitionTime.
		lastTransitionTime := unversioned.Now()
		_, oldReadyCondition := api.GetPodCondition(&oldStatus, api.PodReady)
		if oldReadyCondition != nil && readyCondition.Status == oldReadyCondition.Status {
			lastTransitionTime = oldReadyCondition.LastTransitionTime
		}
		readyCondition.LastTransitionTime = lastTransitionTime
	}

	// Set InitializedCondition.LastTransitionTime.
	if _, initCondition := api.GetPodCondition(&status, api.PodInitialized); initCondition != nil {
		// Need to set LastTransitionTime.
		lastTransitionTime := unversioned.Now()
		_, oldInitCondition := api.GetPodCondition(&oldStatus, api.PodInitialized)
		if oldInitCondition != nil && initCondition.Status == oldInitCondition.Status {
			lastTransitionTime = oldInitCondition.LastTransitionTime
		}
		initCondition.LastTransitionTime = lastTransitionTime
	}

	// ensure that the start time does not change across updates.
	if oldStatus.StartTime != nil && !oldStatus.StartTime.IsZero() {
		status.StartTime = oldStatus.StartTime
	} else if status.StartTime.IsZero() {
		// if the status has no start time, we need to set an initial time
		now := unversioned.Now()
		status.StartTime = &now
	}

	normalizeStatus(pod, &status)
	// The intent here is to prevent concurrent updates to a pod's status from
	// clobbering each other so the phase of a pod progresses monotonically.
	if isCached && isStatusEqual(&cachedStatus.status, &status) && !forceUpdate {
		glog.V(3).Infof("Ignoring same status for pod %q, status: %+v", format.Pod(pod), status)
		return false // No new status.
	}

	newStatus := versionedPodStatus{
		status:       status,
		version:      cachedStatus.version + 1,
		podName:      pod.Name,
		podNamespace: pod.Namespace,
	}
	m.podStatuses[pod.UID] = newStatus

	select {
	case m.podStatusChannel <- podStatusSyncRequest{pod.UID, newStatus}:
		return true
	default:
		// Let the periodic syncBatch handle the update if the channel is full.
		// We can't block, since we hold the mutex lock.
		glog.V(4).Infof("Skpping the status update for pod %q for now because the channel is full; status: %+v",
			format.Pod(pod), status)
		return false
	}
}

// deletePodStatus simply removes the given pod from the status cache.
func (m *manager) deletePodStatus(uid types.UID) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()
	delete(m.podStatuses, uid)
}

// TODO(filipg): It'd be cleaner if we can do this without signal from user.
func (m *manager) RemoveOrphanedStatuses(podUIDs map[types.UID]bool) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()
	for key := range m.podStatuses {
		if _, ok := podUIDs[key]; !ok {
			glog.V(5).Infof("Removing %q from status map.", key)
			delete(m.podStatuses, key)
		}
	}
}

// syncBatch syncs pods statuses with the apiserver.
func (m *manager) syncBatch() {
	var updatedStatuses []podStatusSyncRequest
	podToMirror, mirrorToPod := m.podManager.GetUIDTranslations()
	func() { // Critical section
		m.podStatusesLock.RLock()
		defer m.podStatusesLock.RUnlock()

		// Clean up orphaned versions.
		for uid := range m.apiStatusVersions {
			_, hasPod := m.podStatuses[uid]
			_, hasMirror := mirrorToPod[uid]
			if !hasPod && !hasMirror {
				delete(m.apiStatusVersions, uid)
			}
		}

		for uid, status := range m.podStatuses {
			syncedUID := uid
			if mirrorUID, ok := podToMirror[uid]; ok {
				syncedUID = mirrorUID
			}
			if m.needsUpdate(syncedUID, status) {
				updatedStatuses = append(updatedStatuses, podStatusSyncRequest{uid, status})
			} else if m.needsReconcile(uid, status.status) {
				// Delete the apiStatusVersions here to force an update on the pod status
				// In most cases the deleted apiStatusVersions here should be filled
				// soon after the following syncPod() [If the syncPod() sync an update
				// successfully].
				delete(m.apiStatusVersions, syncedUID)
				updatedStatuses = append(updatedStatuses, podStatusSyncRequest{uid, status})
			}
		}
	}()

	for _, update := range updatedStatuses {
		m.syncPod(update.podUID, update.status)
	}
}

// syncPod syncs the given status with the API server. The caller must not hold the lock.
func (m *manager) syncPod(uid types.UID, status versionedPodStatus) {
	if !m.needsUpdate(uid, status) {
		glog.V(1).Infof("Status for pod %q is up-to-date; skipping", uid)
		return
	}

	// TODO: make me easier to express from client code
	pod, err := m.kubeClient.Core().Pods(status.podNamespace).Get(status.podName)
	if errors.IsNotFound(err) {
		glog.V(3).Infof("Pod %q (%s) does not exist on the server", status.podName, uid)
		// If the Pod is deleted the status will be cleared in
		// RemoveOrphanedStatuses, so we just ignore the update here.
		return
	}
	if err == nil {
		translatedUID := m.podManager.TranslatePodUID(pod.UID)
		if len(translatedUID) > 0 && translatedUID != uid {
			glog.V(3).Infof("Pod %q was deleted and then recreated, skipping status update", format.Pod(pod))
			m.deletePodStatus(uid)
			return
		}
		pod.Status = status.status
		// TODO: handle conflict as a retry, make that easier too.
		pod, err = m.kubeClient.Core().Pods(pod.Namespace).UpdateStatus(pod)
		if err == nil {
			glog.V(3).Infof("Status for pod %q updated successfully: %+v", format.Pod(pod), status)
			m.apiStatusVersions[pod.UID] = status.version
			if kubepod.IsMirrorPod(pod) {
				// We don't handle graceful deletion of mirror pods.
				return
			}
			if pod.DeletionTimestamp == nil {
				return
			}
			if !notRunning(pod.Status.ContainerStatuses) {
				glog.V(3).Infof("Pod %q is terminated, but some containers are still running", format.Pod(pod))
				return
			}
			deleteOptions := api.NewDeleteOptions(0)
			// Use the pod UID as the precondition for deletion to prevent deleting a newly created pod with the same name and namespace.
			deleteOptions.Preconditions = api.NewUIDPreconditions(string(pod.UID))
			if err := m.kubeClient.Core().Pods(pod.Namespace).Delete(pod.Name, deleteOptions); err == nil {
				glog.V(3).Infof("Pod %q fully terminated and removed from etcd", format.Pod(pod))
				m.deletePodStatus(uid)
				return
			}
		}
	}

	// We failed to update status, wait for periodic sync to retry.
	glog.Warningf("Failed to update status for pod %q: %v", format.Pod(pod), err)
}

// needsUpdate returns whether the status is stale for the given pod UID.
// This method is not thread safe, and most only be accessed by the sync thread.
func (m *manager) needsUpdate(uid types.UID, status versionedPodStatus) bool {
	latest, ok := m.apiStatusVersions[uid]
	return !ok || latest < status.version
}

// needsReconcile compares the given status with the status in the pod manager (which
// in fact comes from apiserver), returns whether the status needs to be reconciled with
// the apiserver. Now when pod status is inconsistent between apiserver and kubelet,
// kubelet should forcibly send an update to reconclie the inconsistence, because kubelet
// should be the source of truth of pod status.
// NOTE(random-liu): It's simpler to pass in mirror pod uid and get mirror pod by uid, but
// now the pod manager only supports getting mirror pod by static pod, so we have to pass
// static pod uid here.
// TODO(random-liu): Simplify the logic when mirror pod manager is added.
func (m *manager) needsReconcile(uid types.UID, status api.PodStatus) bool {
	// The pod could be a static pod, so we should translate first.
	pod, ok := m.podManager.GetPodByUID(uid)
	if !ok {
		glog.V(4).Infof("Pod %q has been deleted, no need to reconcile", string(uid))
		return false
	}
	// If the pod is a static pod, we should check its mirror pod, because only status in mirror pod is meaningful to us.
	if kubepod.IsStaticPod(pod) {
		mirrorPod, ok := m.podManager.GetMirrorPodByPod(pod)
		if !ok {
			glog.V(4).Infof("Static pod %q has no corresponding mirror pod, no need to reconcile", format.Pod(pod))
			return false
		}
		pod = mirrorPod
	}

	podStatus, err := copyStatus(&pod.Status)
	if err != nil {
		return false
	}
	normalizeStatus(pod, &podStatus)

	if isStatusEqual(&podStatus, &status) {
		// If the status from the source is the same with the cached status,
		// reconcile is not needed. Just return.
		return false
	}
	glog.V(3).Infof("Pod status is inconsistent with cached status for pod %q, a reconciliation should be triggered:\n %+v", format.Pod(pod),
		diff.ObjectDiff(podStatus, status))

	return true
}

// We add this function, because apiserver only supports *RFC3339* now, which means that the timestamp returned by
// apiserver has no nanosecond information. However, the timestamp returned by unversioned.Now() contains nanosecond,
// so when we do comparison between status from apiserver and cached status, isStatusEqual() will always return false.
// There is related issue #15262 and PR #15263 about this.
// In fact, the best way to solve this is to do it on api side. However, for now, we normalize the status locally in
// kubelet temporarily.
// TODO(random-liu): Remove timestamp related logic after apiserver supports nanosecond or makes it consistent.
func normalizeStatus(pod *api.Pod, status *api.PodStatus) *api.PodStatus {
	normalizeTimeStamp := func(t *unversioned.Time) {
		*t = t.Rfc3339Copy()
	}
	normalizeContainerState := func(c *api.ContainerState) {
		if c.Running != nil {
			normalizeTimeStamp(&c.Running.StartedAt)
		}
		if c.Terminated != nil {
			normalizeTimeStamp(&c.Terminated.StartedAt)
			normalizeTimeStamp(&c.Terminated.FinishedAt)
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

	// update container statuses
	for i := range status.ContainerStatuses {
		cstatus := &status.ContainerStatuses[i]
		normalizeContainerState(&cstatus.State)
		normalizeContainerState(&cstatus.LastTerminationState)
	}
	// Sort the container statuses, so that the order won't affect the result of comparison
	sort.Sort(kubetypes.SortedContainerStatuses(status.ContainerStatuses))

	// update init container statuses
	for i := range status.InitContainerStatuses {
		cstatus := &status.InitContainerStatuses[i]
		normalizeContainerState(&cstatus.State)
		normalizeContainerState(&cstatus.LastTerminationState)
	}
	// Sort the container statuses, so that the order won't affect the result of comparison
	kubetypes.SortInitContainerStatuses(pod, status.InitContainerStatuses)
	return status
}

// notRunning returns true if every status is terminated or waiting, or the status list
// is empty.
func notRunning(statuses []api.ContainerStatus) bool {
	for _, status := range statuses {
		if status.State.Terminated == nil && status.State.Waiting == nil {
			return false
		}
	}
	return true
}

func copyStatus(source *api.PodStatus) (api.PodStatus, error) {
	clone, err := api.Scheme.DeepCopy(source)
	if err != nil {
		glog.Errorf("Failed to clone status %+v: %v", source, err)
		return api.PodStatus{}, err
	}
	status := *clone.(*api.PodStatus)
	return status, nil
}
