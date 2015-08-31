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

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	kubeletTypes "k8s.io/kubernetes/pkg/kubelet/types"
	kubeletUtil "k8s.io/kubernetes/pkg/kubelet/util"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util"
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
	podStatuses      map[types.UID]api.PodStatus
	podStatusChannel chan podStatusSyncRequest
}

func newStatusManager(kubeClient client.Interface) *statusManager {
	return &statusManager{
		kubeClient:       kubeClient,
		podStatuses:      make(map[types.UID]api.PodStatus),
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
	go util.Until(func() {
		err := s.syncBatch()
		if err != nil {
			glog.Warningf("Failed to updated pod status: %v", err)
		}
	}, 0, util.NeverStop)
}

func (s *statusManager) GetPodStatus(uid types.UID) (api.PodStatus, bool) {
	s.podStatusesLock.RLock()
	defer s.podStatusesLock.RUnlock()
	status, ok := s.podStatuses[uid]
	return status, ok
}

func (s *statusManager) SetPodStatus(pod *api.Pod, status api.PodStatus) {
	s.podStatusesLock.Lock()
	defer s.podStatusesLock.Unlock()
	oldStatus, found := s.podStatuses[pod.UID]

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
	if !found || !isStatusEqual(&oldStatus, &status) || pod.DeletionTimestamp != nil {
		s.podStatuses[pod.UID] = status
		s.podStatusChannel <- podStatusSyncRequest{pod, status}
	} else {
		glog.V(3).Infof("Ignoring same status for pod %q, status: %+v", kubeletUtil.FormatPodName(pod), status)
	}
}

// TerminatePods resets the container status for the provided pods to terminated and triggers
// a status update. This function may not enqueue all the provided pods, in which case it will
// return false
func (s *statusManager) TerminatePods(pods []*api.Pod) bool {
	sent := true
	s.podStatusesLock.Lock()
	defer s.podStatusesLock.Unlock()
	for _, pod := range pods {
		for i := range pod.Status.ContainerStatuses {
			pod.Status.ContainerStatuses[i].State = api.ContainerState{
				Terminated: &api.ContainerStateTerminated{},
			}
		}
		select {
		case s.podStatusChannel <- podStatusSyncRequest{pod, pod.Status}:
		default:
			sent = false
			glog.V(4).Infof("Termination notice for %q was dropped because the status channel is full", kubeletUtil.FormatPodName(pod))
		}
	}
	return sent
}

func (s *statusManager) DeletePodStatus(uid types.UID) {
	s.podStatusesLock.Lock()
	defer s.podStatusesLock.Unlock()
	delete(s.podStatuses, uid)
}

// TODO(filipg): It'd be cleaner if we can do this without signal from user.
func (s *statusManager) RemoveOrphanedStatuses(podUIDs map[types.UID]bool) {
	s.podStatusesLock.Lock()
	defer s.podStatusesLock.Unlock()
	for key := range s.podStatuses {
		if _, ok := podUIDs[key]; !ok {
			glog.V(5).Infof("Removing %q from status map.", key)
			delete(s.podStatuses, key)
		}
	}
}

// syncBatch syncs pods statuses with the apiserver.
func (s *statusManager) syncBatch() error {
	syncRequest := <-s.podStatusChannel
	pod := syncRequest.pod
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
		if len(pod.UID) > 0 && statusPod.UID != pod.UID {
			glog.V(3).Infof("Pod %q was deleted and then recreated, skipping status update", kubeletUtil.FormatPodName(pod))
			return nil
		}
		statusPod.Status = status
		// TODO: handle conflict as a retry, make that easier too.
		statusPod, err = s.kubeClient.Pods(pod.Namespace).UpdateStatus(statusPod)
		if err == nil {
			glog.V(3).Infof("Status for pod %q updated successfully", kubeletUtil.FormatPodName(pod))

			if pod.DeletionTimestamp == nil {
				return nil
			}
			if !notRunning(pod.Status.ContainerStatuses) {
				glog.V(3).Infof("Pod %q is terminated, but some pods are still running", pod.Name)
				return nil
			}
			if err := s.kubeClient.Pods(statusPod.Namespace).Delete(statusPod.Name, api.NewDeleteOptions(0)); err == nil {
				glog.V(3).Infof("Pod %q fully terminated and removed from etcd", statusPod.Name)
				s.DeletePodStatus(pod.UID)
				return nil
			}
		}
	}

	// We failed to update status. In order to make sure we retry next time
	// we delete cached value. This may result in an additional update, but
	// this is ok.
	// Doing this synchronously will lead to a deadlock if the podStatusChannel
	// is full, and the pod worker holding the lock is waiting on this method
	// to clear the channel. Even if this delete never runs subsequent container
	// changes on the node should trigger updates.
	go s.DeletePodStatus(pod.UID)
	return fmt.Errorf("error updating status for pod %q: %v", kubeletUtil.FormatPodName(pod), err)
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
