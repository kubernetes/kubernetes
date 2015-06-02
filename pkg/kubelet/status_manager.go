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
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
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

func (s *statusManager) Start() {
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

	if !found || !reflect.DeepEqual(oldStatus, status) {
		s.podStatuses[podFullName] = status
		s.podStatusChannel <- podStatusSyncRequest{pod, status}
	} else {
		glog.V(3).Infof("Ignoring same pod status for %s - old: %s new: %s", podFullName, oldStatus, status)
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
	if errors.IsNotFound(err) {
		glog.V(3).Infof("Pod %q was deleted on the server", pod.Name)
		return nil
	}
	if err == nil {
		statusPod.Status = status
		// TODO: handle conflict as a retry, make that easier too.
		statusPod, err = s.kubeClient.Pods(pod.Namespace).UpdateStatus(statusPod)
		if err == nil {
			glog.V(3).Infof("Status for pod %q updated successfully", pod.Name)

			if statusPod.DeletionTimestamp == nil || !allTerminated(statusPod.Status.ContainerStatuses) {
				return nil
			}
			if err := s.kubeClient.Pods(statusPod.Namespace).Delete(statusPod.Name, api.NewDeleteOptions(0)); err == nil {
				glog.V(3).Infof("Pod %q fully terminated and removed from etcd", statusPod.Name)
				return nil
			}
		}
	}

	// We failed to update status. In order to make sure we retry next time
	// we delete cached value. This may result in an additional update, but
	// this is ok.
	s.DeletePodStatus(podFullName)
	return fmt.Errorf("error updating status for pod %q: %v", pod.Name, err)
}

// allTerminated returns true if every status is terminated, or the status list
// is empty.
func allTerminated(statuses []api.ContainerStatus) bool {
	for _, status := range statuses {
		if status.State.Terminated == nil {
			return false
		}
	}
	return true
}
