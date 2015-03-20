/*
Copyright 2014 Google Inc. All rights reserved.

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
	"reflect"
	"sync"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

type podStatusSyncRequest struct {
	podFullName string
	status      api.PodStatus
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
	// We can run SyncBatch() often because it will block until we have some updates to send.
	go util.Forever(s.SyncBatch, 0)
}

func (s *statusManager) GetPodStatus(podFullName string) (api.PodStatus, bool) {
	s.podStatusesLock.RLock()
	defer s.podStatusesLock.RUnlock()
	status, ok := s.podStatuses[podFullName]
	return status, ok
}

func (s *statusManager) SetPodStatus(podFullName string, status api.PodStatus) {
	s.podStatusesLock.Lock()
	defer s.podStatusesLock.Unlock()
	oldStatus, found := s.podStatuses[podFullName]
	if !found || !reflect.DeepEqual(oldStatus, status) {
		s.podStatuses[podFullName] = status
		s.podStatusChannel <- podStatusSyncRequest{podFullName, status}
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

// SyncBatch syncs pods statuses with the apiserver. It will loop until channel
// s.podStatusChannel is empty for at least 1s.
func (s *statusManager) SyncBatch() {
	for {
		select {
		case syncRequest := <-s.podStatusChannel:
			podFullName := syncRequest.podFullName
			status := syncRequest.status
			glog.V(3).Infof("Syncing status for %s", podFullName)
			name, namespace, err := ParsePodFullName(podFullName)
			if err != nil {
				glog.Warningf("Cannot parse pod full name %q: %s", podFullName, err)
			}
			_, err = s.kubeClient.Pods(namespace).UpdateStatus(name, &status)
			if err != nil {
				// We failed to update status. In order to make sure we retry next time
				// we delete cached value. This may result in an additional update, but
				// this is ok.
				s.DeletePodStatus(podFullName)
				glog.Warningf("Error updating status for pod %q: %v", name, err)
			} else {
				glog.V(3).Infof("Status for pod %q updated successfully", name)
			}
		case <-time.After(1 * time.Second):
			return
		}
	}
}
