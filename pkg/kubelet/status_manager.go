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

package kubelet

import (
	"fmt"
	"reflect"
	"sort"
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	kubeletTypes "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type podStatusSyncRequest struct {
	pod    *api.Pod
	status api.PodStatus
}

// Updates pod statuses in apiserver. Writes only when new status has changed.
// All methods are thread-safe.
type statusManager struct {
	kubeClient client.Interface
	// Map from pod full name to sync status of the corresponding pod.
	podStatusesLock  sync.RWMutex
	podStatuses      map[string]api.PodStatus
	podStatusChannel chan podStatusSyncRequest
}

func newStatusManager(kubeClient client.Interface) *statusManager {
	return &statusManager{
		kubeClient:       kubeClient,
		podStatuses:      make(map[string]api.PodStatus),
		podStatusChannel: make(chan podStatusSyncRequest, 1000), // Buffer up to 1000 statuses
	}
}

// isStatusEqual returns true if the given pod statuses are equal, false otherwise.
// This method sorts container statuses so order does not affect equality.
func isStatusEqual(oldStatus, status *api.PodStatus) bool {
	sort.Sort(kubeletTypes.SortedContainerStatuses(status.ContainerStatuses))
	sort.Sort(kubeletTypes.SortedContainerStatuses(oldStatus.ContainerStatuses))

	// TODO: More sophisticated equality checking.
	return reflect.DeepEqual(status, oldStatus)
}

func (s *statusManager) Start() {
	// Don't start the status manager if we don't have a client. This will happen
	// on the master, where the kubelet is responsible for bootstrapping the pods
	// of the master components.
	if s.kubeClient == nil {
		glog.Infof("Kubernetes client is nil, not starting status manager.")
		return
	}
	// syncBatch blocks when no updates are available, we can run it in a tight loop.
	glog.Info("Starting to sync pod status with apiserver")
	go util.Forever(func() {
		err := s.syncBatch()
		if err != nil {
			glog.Warningf("Failed to updated pod status: %v", err)
		}
	}, 0)
}

func (s *statusManager) GetPodStatus(podFullName string) (api.PodStatus, bool) {
	s.podStatusesLock.RLock()
	defer s.podStatusesLock.RUnlock()
	status, ok := s.podStatuses[podFullName]
	return status, ok
}

func (s *statusManager) SetPodStatus(pod *api.Pod, status api.PodStatus) {
	podFullName := kubecontainer.GetPodFullName(pod)
	s.podStatusesLock.Lock()
	defer s.podStatusesLock.Unlock()
	oldStatus, found := s.podStatuses[podFullName]

	// ensure that the start time does not change across updates.
	if found && oldStatus.StartTime != nil {
		status.StartTime = oldStatus.StartTime
	}

	// if the status has no start time, we need to set an initial time
	// TODO(yujuhong): Consider setting StartTime when generating the pod
	// status instead, which would allow statusManager to become a simple cache
	// again.
	if status.StartTime.IsZero() {
		if pod.Status.StartTime.IsZero() {
			// the pod did not have a previously recorded value so set to now
			now := util.Now()
			status.StartTime = &now
		} else {
			// the pod had a recorded value, but the kubelet restarted so we need to rebuild cache
			// based on last observed value
			status.StartTime = pod.Status.StartTime
		}
	}

	// TODO: Holding a lock during blocking operations is dangerous. Refactor so this isn't necessary.
	// The intent here is to prevent concurrent updates to a pod's status from
	// clobbering each other so the phase of a pod progresses monotonically.
	// Currently this routine is not called for the same pod from multiple
	// workers and/or the kubelet but dropping the lock before sending the
	// status down the channel feels like an easy way to get a bullet in foot.
	if !found || !isStatusEqual(&oldStatus, &status) {
		s.podStatuses[podFullName] = status
		s.podStatusChannel <- podStatusSyncRequest{pod, status}
	} else {
		glog.V(3).Infof("Ignoring same pod status for %q - old: %+v new: %+v", podFullName, oldStatus, status)
	}
}

func (s *statusManager) DeletePodStatus(podFullName string) {
	s.podStatusesLock.Lock()
	defer s.podStatusesLock.Unlock()
	delete(s.podStatuses, podFullName)
}

// TODO(filipg): It'd be cleaner if we can do this without signal from user.
func (s *statusManager) RemoveOrphanedStatuses(podFullNames map[string]bool) {
	s.podStatusesLock.Lock()
	defer s.podStatusesLock.Unlock()
	for key := range s.podStatuses {
		if _, ok := podFullNames[key]; !ok {
			glog.V(5).Infof("Removing %q from status map.", key)
			delete(s.podStatuses, key)
		}
	}
}

// syncBatch syncs pods statuses with the apiserver.
func (s *statusManager) syncBatch() error {
	syncRequest := <-s.podStatusChannel
	pod := syncRequest.pod
	podFullName := kubecontainer.GetPodFullName(pod)
	status := syncRequest.status

	var err error
	statusPod := &api.Pod{
		ObjectMeta: pod.ObjectMeta,
	}
	// TODO: make me easier to express from client code
	statusPod, err = s.kubeClient.Pods(statusPod.Namespace).Get(statusPod.Name)
	if err == nil {
		statusPod.Status = status
		_, err = s.kubeClient.Pods(pod.Namespace).UpdateStatus(statusPod)
		// TODO: handle conflict as a retry, make that easier too.
		if err == nil {
			glog.V(3).Infof("Status for pod %q updated successfully", pod.Name)
			return nil
		}
	}

	// We failed to update status. In order to make sure we retry next time
	// we delete cached value. This may result in an additional update, but
	// this is ok.
	// Doing this synchronously will lead to a deadlock if the podStatusChannel
	// is full, and the pod worker holding the lock is waiting on this method
	// to clear the channel. Even if this delete never runs subsequent container
	// changes on the node should trigger updates.
	go s.DeletePodStatus(podFullName)
	return fmt.Errorf("error updating status for pod %q: %v", pod.Name, err)
}
