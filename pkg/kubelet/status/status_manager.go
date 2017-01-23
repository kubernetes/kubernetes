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

	"k8s.io/kubernetes/pkg/client/clientset_generated/clientset"

	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubepod "k8s.io/kubernetes/pkg/kubelet/pod"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
)

// Updates pod statuses in apiserver. Writes only when new status has changed.
// All methods are thread-safe.
type manager struct {
	kubeClient clientset.Interface
	podManager kubepod.Manager
	// The podStatusChannel preserves the order of status updates to the api server
	podStatusChannel chan types.UID
	// updateSetLock and updateSet enforce that if a pod is updated more than once
	// between syncs with the api server, we will only update the api server once
	updateSetLock sync.Mutex
	// The updateSet contains the UIDs of pods that have outstanding changes to be sent to the server
	updateSet map[types.UID]struct{}
}

// PodStatusProvider knows how to provide status for a pod.  It's intended to be used by other components
// that need to introspect status.
type PodStatusProvider interface {
	// GetPodStatus returns the cached status for the provided pod UID, as well as whether it
	// was a cache hit.
	GetPodStatus(uid types.UID) (v1.PodStatus, bool)
}

// Manager is the Source of truth for kubelet pod status, and should be kept up-to-date with
// the latest v1.PodStatus. It also syncs updates back to the API server.
type Manager interface {
	PodStatusProvider

	// Start the API server status sync loop.
	Start()

	// AddPod adds a pod
	AddPod(pod *v1.Pod)

	// SetPodStatus caches updates the cached status for the given pod, and triggers a status update.
	SetPodStatus(pod *v1.Pod, status v1.PodStatus)

	// SetContainerReadiness updates the cached container status with the given readiness, and
	// triggers a status update.
	SetContainerReadiness(podUID types.UID, containerID kubecontainer.ContainerID, ready bool)

	// ReconcilePod updates the v1 server with the latest status for that pod in order to reconcile differences between
	// the kubelet and the v1 server
	ReconcilePod(uid types.UID)

	// TerminatePod resets the container status for the provided pod to terminated and triggers
	// a status update.
	TerminatePod(pod *v1.Pod)
}

const (
	syncPeriod  = 10 * time.Second
	retryPeriod = 10 * time.Second
)

func NewManager(kubeClient clientset.Interface, podManager kubepod.Manager) Manager {
	return &manager{
		kubeClient:       kubeClient,
		podManager:       podManager,
		podStatusChannel: make(chan types.UID, 1000), // Buffer up to 1000 uids to update
		updateSet:        make(map[types.UID]struct{}),
	}
}

// isStatusEqual returns true if the given pod statuses are equal, false otherwise.
// This method normalizes the status before comparing so as to make sure that meaningless
// changes will be ignored.
func isStatusEqual(oldStatus, status *v1.PodStatus) bool {
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
	go func() {
		for update := range m.podStatusChannel {
			// if it needs to be updated
			if needsUpdate := m.needsUpdate(update); needsUpdate {
				// if the update needs to be retried
				if !m.syncPod(update) {
					// retry the update after a delay
					go func() {
						time.Sleep(retryPeriod)
						m.enqueueUpdate(update, nil)
					}()
				}
			} else {
				glog.Infof("Pod %v was already updated, skipping... ", update)
			}
		}
	}()
}

func (m *manager) AddPod(pod *v1.Pod) {
	m.podManager.AddPod(pod)

	// when adding a mirror pod, update the status of the corresponding static pod
	_, mirrorToPod := m.podManager.GetUIDTranslations()
	if p, ok := mirrorToPod[pod.UID]; ok {
		m.enqueueUpdate(p, nil)
	}
}

func (m *manager) GetPodStatus(uid types.UID) (v1.PodStatus, bool) {
	pod, ok := m.podManager.GetPodByUID(m.podManager.TranslatePodUID(uid))
	if ok {
		return pod.Status, true
	}
	return v1.PodStatus{}, false
}

func (m *manager) SetPodStatus(pod *v1.Pod, status v1.PodStatus) {
	m.podManager.UpdatePodSafe(pod.UID, func(inputPod *v1.Pod) {
		// Make sure we're caching a deep copy.
		status, err := copyStatus(&status)
		if err != nil {
			return
		}
		// Force a status update if deletion timestamp is set. This is necessary
		// because if the pod is in the non-running state, the pod worker still
		// needs to be able to trigger an update and/or deletion.
		m.updateStatusInternal(inputPod, status, pod.DeletionTimestamp != nil)
	})
}

func (m *manager) SetContainerReadiness(podUID types.UID, containerID kubecontainer.ContainerID, ready bool) {
	m.podManager.UpdatePodSafe(podUID, func(pod *v1.Pod) {
		// Find the container to update.
		containerStatus, _, ok := findContainerStatus(&pod.Status, containerID.String())
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
		status, err := copyStatus(&pod.Status)
		if err != nil {
			return
		}
		containerStatus, _, _ = findContainerStatus(&status, containerID.String())
		containerStatus.Ready = ready

		// Update pod condition.
		readyConditionIndex := -1
		for i, condition := range status.Conditions {
			if condition.Type == v1.PodReady {
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
	})
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

func (m *manager) ReconcilePod(uid types.UID) {
	// queue the uid for an upate to the api server
	m.enqueueUpdate(uid, nil)
}

func (m *manager) TerminatePod(pod *v1.Pod) {
	m.podManager.UpdatePodSafe(pod.UID, func(inputPod *v1.Pod) {
		newStatus, err := copyStatus(&inputPod.Status)
		if err != nil {
			return
		}
		for i := range newStatus.ContainerStatuses {
			newStatus.ContainerStatuses[i].State = v1.ContainerState{
				Terminated: &v1.ContainerStateTerminated{},
			}
		}
		for i := range newStatus.InitContainerStatuses {
			newStatus.InitContainerStatuses[i].State = v1.ContainerState{
				Terminated: &v1.ContainerStateTerminated{},
			}
		}
		m.updateStatusInternal(inputPod, newStatus, true)
	})
}

// updateStatusInternal updates the pod Manager's status, and queues an update to the api server if necessary.
func (m *manager) updateStatusInternal(pod *v1.Pod, status v1.PodStatus, forceUpdate bool) {
	// Set ReadyCondition.LastTransitionTime.
	if _, readyCondition := v1.GetPodCondition(&status, v1.PodReady); readyCondition != nil {
		// Need to set LastTransitionTime.
		lastTransitionTime := metav1.Now()
		_, oldReadyCondition := v1.GetPodCondition(&pod.Status, v1.PodReady)
		if oldReadyCondition != nil && readyCondition.Status == oldReadyCondition.Status {
			lastTransitionTime = oldReadyCondition.LastTransitionTime
		}
		readyCondition.LastTransitionTime = lastTransitionTime
	}

	// Set InitializedCondition.LastTransitionTime.
	if _, initCondition := v1.GetPodCondition(&status, v1.PodInitialized); initCondition != nil {
		// Need to set LastTransitionTime.
		lastTransitionTime := metav1.Now()
		_, oldInitCondition := v1.GetPodCondition(&pod.Status, v1.PodInitialized)
		if oldInitCondition != nil && initCondition.Status == oldInitCondition.Status {
			lastTransitionTime = oldInitCondition.LastTransitionTime
		}
		initCondition.LastTransitionTime = lastTransitionTime
	}

	// ensure that the start time does not change across updates.
	if pod.Status.StartTime != nil && !pod.Status.StartTime.IsZero() {
		status.StartTime = pod.Status.StartTime
	} else if status.StartTime.IsZero() {
		// if the status has no start time, we need to set an initial time
		now := metav1.Now()
		status.StartTime = &now
	}

	normalizeStatus(pod, &status)
	// The intent here is to prevent concurrent updates to a pod's status from
	// clobbering each other so the phase of a pod progresses monotonically.
	if isStatusEqual(&pod.Status, &status) && !forceUpdate {
		glog.V(3).Infof("Ignoring same status for pod %q, status: %+v", format.Pod(pod), status)
		return
	}

	m.enqueueUpdate(pod.UID, func() { pod.Status = status })
}

// syncPod syncs the given status with the API server. The caller must not hold the lock.
// returns true if the update succeeded or is not neede; returns false if syncpod should be retried
func (m *manager) syncPod(updateUID types.UID) bool {
	updatedPod, found := m.podManager.GetPodByUID(updateUID)
	if !found {
		glog.V(3).Infof("pod %v not found. Skipping update", updateUID)
		return true
	}
	// TODO: make me easier to express from client code
	pod, err := m.kubeClient.Core().Pods(updatedPod.Namespace).Get(updatedPod.Name, metav1.GetOptions{})
	if errors.IsNotFound(err) {
		glog.V(3).Infof("Pod %q (%s) does not exist on the server", updatedPod.Name, updatedPod.UID)
		return true
	}
	if err != nil {
		glog.Warningf("Failed to get status for pod %q from api server: %v", format.Pod(pod), err)
		return false
	}

	translatedUID := m.podManager.TranslatePodUID(pod.UID)
	if len(translatedUID) > 0 && translatedUID != updatedPod.UID {
		glog.V(2).Infof("Pod %q was deleted and then recreated, skipping status update; old UID %q, new UID %q", format.Pod(pod), updatedPod.UID, translatedUID)
		m.podManager.DeletePod(pod)
		return true
	}
	pod.Status = updatedPod.Status
	if err := podutil.SetInitContainersStatusesAnnotations(pod); err != nil {
		glog.Error(err)
	}
	// TODO: handle conflict as a retry, make that easier too.
	pod, err = m.kubeClient.Core().Pods(pod.Namespace).UpdateStatus(pod)
	if err != nil {
		glog.Warningf("Failed to update status for pod %q from api server: %v", format.Pod(pod), err)
		return false
	}
	glog.V(3).Infof("Status for pod %q updated successfully: %+v", format.Pod(pod), updatedPod.Status)

	// we dont handle graceful deletion of mirror pods
	if !kubepod.IsMirrorPod(pod) && pod.DeletionTimestamp != nil {
		if !notRunning(pod.Status.ContainerStatuses) {
			glog.V(3).Infof("Pod %q is terminated, but some containers are still running", format.Pod(pod))
			return false
		}
		deleteOptions := v1.NewDeleteOptions(0)
		// Use the pod UID as the precondition for deletion to prevent deleting a newly created pod with the same name and namespace.
		deleteOptions.Preconditions = v1.NewUIDPreconditions(string(pod.UID))
		glog.V(2).Infof("Removing Pod %q from etcd", format.Pod(pod))
		if err = m.kubeClient.Core().Pods(pod.Namespace).Delete(pod.Name, deleteOptions); err != nil {
			glog.Warningf("Failed to delete pod %q from api server: %v", format.Pod(pod), err)
			return false
		}
		glog.V(3).Infof("Pod %q fully terminated and removed from etcd", format.Pod(pod))
		m.podManager.DeletePod(pod)
	}
	return true
}

// enqueueUpdate adds the update to the update channel without blocking.
// returns true if it was successfully added to the queue
// this should be called while holding the lock
// onSuccess should not block, and is optional.
func (m *manager) enqueueUpdate(update types.UID, onSuccess func()) bool {
	m.updateSetLock.Lock()
	defer m.updateSetLock.Unlock()
	// Dont enqueue an update if an update to that uid is already queued.
	if _, ok := m.updateSet[update]; !ok {
		select {
		case m.podStatusChannel <- update:
			m.updateSet[update] = struct{}{}
		default:
			glog.V(4).Infof("Skpping the status update for pod %q for now because the channel is full", update)
			return false
		}
	}
	if onSuccess != nil {
		onSuccess()
	}
	return true
}

// needsUpdate returns whether the pod in question has not been updated since it was added to the buffer
// if it has not, then it needs to be updated
func (m *manager) needsUpdate(uid types.UID) bool {
	m.updateSetLock.Lock()
	defer m.updateSetLock.Unlock()
	if _, shouldUpdate := m.updateSet[uid]; shouldUpdate {
		delete(m.updateSet, uid)
		return true
	}
	// it was already updated by a different update.
	return false
}

// We add this function, because apiserver only supports *RFC3339* now, which means that the timestamp returned by
// apiserver has no nanosecond information. However, the timestamp returned by metav1.Now() contains nanosecond,
// so when we do comparison between status from apiserver and cached status, isStatusEqual() will always return false.
// There is related issue #15262 and PR #15263 about this.
// In fact, the best way to solve this is to do it on api side. However, for now, we normalize the status locally in
// kubelet temporarily.
// TODO(random-liu): Remove timestamp related logic after apiserver supports nanosecond or makes it consistent.
func normalizeStatus(pod *v1.Pod, status *v1.PodStatus) {
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
}

// notRunning returns true if every status is terminated or waiting, or the status list
// is empty.
func notRunning(statuses []v1.ContainerStatus) bool {
	for _, status := range statuses {
		if status.State.Terminated == nil && status.State.Waiting == nil {
			return false
		}
	}
	return true
}

func copyStatus(source *v1.PodStatus) (v1.PodStatus, error) {
	clone, err := api.Scheme.DeepCopy(source)
	if err != nil {
		glog.Errorf("Failed to clone status %+v: %v", source, err)
		return v1.PodStatus{}, err
	}
	status := *clone.(*v1.PodStatus)
	return status, nil
}
