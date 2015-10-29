/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"reflect"
	"sort"
	"sync"
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	kubeletutil "k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
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
	kubeClient client.Interface
	// Map from pod UID to sync status of the corresponding pod.
	podStatuses      map[types.UID]versionedPodStatus
	podStatusesLock  sync.RWMutex
	podStatusChannel chan podStatusSyncRequest
	// Map from pod UID to latest status version successfully sent to the API server.
	// apiStatusVersions must only be accessed from the sync thread.
	apiStatusVersions map[types.UID]uint64
}

// status.Manager is the Source of truth for kubelet pod status, and should be kept up-to-date with
// the latest api.PodStatus. It also syncs updates back to the API server.
type Manager interface {
	// Start the API server status sync loop.
	Start()

	// GetPodStatus returns the cached status for the provided pod UID, as well as whether it
	// was a cache hit.
	GetPodStatus(uid types.UID) (api.PodStatus, bool)

	// SetPodStatus caches updates the cached status for the given pod, and triggers a status update.
	SetPodStatus(pod *api.Pod, status api.PodStatus)

	// TerminatePods resets the container status for the provided pods to terminated and triggers
	// a status update. This function may not enqueue all the provided pods, in which case it will
	// return false
	TerminatePods(pods []*api.Pod) bool

	// RemoveOrphanedStatuses scans the status cache and removes any entries for pods not included in
	// the provided podUIDs.
	RemoveOrphanedStatuses(podUIDs map[types.UID]bool)
}

const syncPeriod = 10 * time.Second

func NewManager(kubeClient client.Interface) Manager {
	return &manager{
		kubeClient:        kubeClient,
		podStatuses:       make(map[types.UID]versionedPodStatus),
		podStatusChannel:  make(chan podStatusSyncRequest, 1000), // Buffer up to 1000 statuses
		apiStatusVersions: make(map[types.UID]uint64),
	}
}

// isStatusEqual returns true if the given pod statuses are equal, false otherwise.
// This method sorts container statuses so order does not affect equality.
func isStatusEqual(oldStatus, status *api.PodStatus) bool {
	sort.Sort(kubetypes.SortedContainerStatuses(status.ContainerStatuses))
	sort.Sort(kubetypes.SortedContainerStatuses(oldStatus.ContainerStatuses))

	// TODO: More sophisticated equality checking.
	return reflect.DeepEqual(status, oldStatus)
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
	go util.Forever(func() {
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
	status, ok := m.podStatuses[uid]
	return status.status, ok
}

func (m *manager) SetPodStatus(pod *api.Pod, status api.PodStatus) {
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()
	oldStatus, found := m.podStatuses[pod.UID]

	// ensure that the start time does not change across updates.
	if found && oldStatus.status.StartTime != nil {
		status.StartTime = oldStatus.status.StartTime
	}

	// Set ReadyCondition.LastTransitionTime.
	// Note we cannot do this while generating the status since we do not have oldStatus
	// at that time for mirror pods.
	if readyCondition := api.GetPodReadyCondition(status); readyCondition != nil {
		// Need to set LastTransitionTime.
		lastTransitionTime := unversioned.Now()
		if found {
			oldReadyCondition := api.GetPodReadyCondition(oldStatus.status)
			if oldReadyCondition != nil && readyCondition.Status == oldReadyCondition.Status {
				lastTransitionTime = oldReadyCondition.LastTransitionTime
			}
		}
		readyCondition.LastTransitionTime = lastTransitionTime
	}

	// if the status has no start time, we need to set an initial time
	// TODO(yujuhong): Consider setting StartTime when generating the pod
	// status instead, which would allow manager to become a simple cache
	// again.
	if status.StartTime.IsZero() {
		if pod.Status.StartTime.IsZero() {
			// the pod did not have a previously recorded value so set to now
			now := unversioned.Now()
			status.StartTime = &now
		} else {
			// the pod had a recorded value, but the kubelet restarted so we need to rebuild cache
			// based on last observed value
			status.StartTime = pod.Status.StartTime
		}
	}

	newStatus := m.updateStatusInternal(pod, status)
	if newStatus != nil {
		select {
		case m.podStatusChannel <- podStatusSyncRequest{pod.UID, *newStatus}:
		default:
			// Let the periodic syncBatch handle the update if the channel is full.
			// We can't block, since we hold the mutex lock.
		}
	}
}

func (m *manager) TerminatePods(pods []*api.Pod) bool {
	sent := true
	m.podStatusesLock.Lock()
	defer m.podStatusesLock.Unlock()
	for _, pod := range pods {
		for i := range pod.Status.ContainerStatuses {
			pod.Status.ContainerStatuses[i].State = api.ContainerState{
				Terminated: &api.ContainerStateTerminated{},
			}
		}
		newStatus := m.updateStatusInternal(pod, pod.Status)
		if newStatus != nil {
			select {
			case m.podStatusChannel <- podStatusSyncRequest{pod.UID, *newStatus}:
			default:
				sent = false
				glog.V(4).Infof("Termination notice for %q was dropped because the status channel is full", kubeletutil.FormatPodName(pod))
			}
		} else {
			sent = false
		}
	}
	return sent
}

// updateStatusInternal updates the internal status cache, and returns a versioned status if an
// update is necessary. This method IS NOT THREAD SAFE and must be called from a locked function.
func (m *manager) updateStatusInternal(pod *api.Pod, status api.PodStatus) *versionedPodStatus {
	// The intent here is to prevent concurrent updates to a pod's status from
	// clobbering each other so the phase of a pod progresses monotonically.
	oldStatus, found := m.podStatuses[pod.UID]
	if !found || !isStatusEqual(&oldStatus.status, &status) || pod.DeletionTimestamp != nil {
		newStatus := versionedPodStatus{
			status:       status,
			version:      oldStatus.version + 1,
			podName:      pod.Name,
			podNamespace: pod.Namespace,
		}
		m.podStatuses[pod.UID] = newStatus
		return &newStatus
	} else {
		glog.V(3).Infof("Ignoring same status for pod %q, status: %+v", kubeletutil.FormatPodName(pod), status)
		return nil // No new status.
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
	func() { // Critical section
		m.podStatusesLock.RLock()
		defer m.podStatusesLock.RUnlock()

		// Clean up orphaned versions.
		for uid := range m.apiStatusVersions {
			if _, ok := m.podStatuses[uid]; !ok {
				delete(m.apiStatusVersions, uid)
			}
		}

		for uid, status := range m.podStatuses {
			if m.needsUpdate(uid, status) {
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
		glog.Warningf("Status is up-to-date; skipping: %q %+v", uid, status)
		return
	}

	// TODO: make me easier to express from client code
	pod, err := m.kubeClient.Pods(status.podNamespace).Get(status.podName)
	if errors.IsNotFound(err) {
		glog.V(3).Infof("Pod %q was deleted on the server", status.podName)
		m.deletePodStatus(uid)
		return
	}
	if err == nil {
		if len(pod.UID) > 0 && pod.UID != uid {
			glog.V(3).Infof("Pod %q was deleted and then recreated, skipping status update",
				kubeletutil.FormatPodName(pod))
			m.deletePodStatus(uid)
			return
		}
		pod.Status = status.status
		// TODO: handle conflict as a retry, make that easier too.
		pod, err = m.kubeClient.Pods(pod.Namespace).UpdateStatus(pod)
		if err == nil {
			glog.V(3).Infof("Status for pod %q updated successfully", kubeletutil.FormatPodName(pod))
			m.apiStatusVersions[uid] = status.version

			if pod.DeletionTimestamp == nil {
				return
			}
			if !notRunning(pod.Status.ContainerStatuses) {
				glog.V(3).Infof("Pod %q is terminated, but some containers are still running", pod.Name)
				return
			}
			if err := m.kubeClient.Pods(pod.Namespace).Delete(pod.Name, api.NewDeleteOptions(0)); err == nil {
				glog.V(3).Infof("Pod %q fully terminated and removed from etcd", pod.Name)
				m.deletePodStatus(pod.UID)
				return
			}
		}
	}

	// We failed to update status, wait for periodic sync to retry.
	glog.Warningf("Failed to updated status for pod %q: %v", kubeletutil.FormatPodName(pod), err)
}

// needsUpdate returns whether the status is stale for the given pod UID.
// This method is not thread safe, and most only be accessed by the sync thread.
func (m *manager) needsUpdate(uid types.UID, status versionedPodStatus) bool {
	latest, ok := m.apiStatusVersions[uid]
	return !ok || latest < status.version
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
