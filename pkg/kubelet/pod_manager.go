/*
Copyright 2015 Google Inc. All rights reserved.

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
	"sync"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/metrics"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/golang/glog"
)

type podManager interface {
	UpdatePods(u PodUpdate, podSyncTypes map[types.UID]metrics.SyncPodType)
	GetPods() ([]api.Pod, mirrorPods)
	GetPodByName(namespace, name string) (*api.Pod, bool)
	GetPodByFullName(podFullName string) (*api.Pod, bool)
	TranslatePodUID(uid types.UID) types.UID
	DeleteOrphanedMirrorPods(mirrorPods *mirrorPods)
	SetPods(pods []api.Pod)
	mirrorManager
}

type basicPodManager struct {
	// Protects all internal pod storage/mappings.
	lock sync.RWMutex
	pods []api.Pod
	// Record the set of mirror pods (see mirror_manager.go for more details);
	// similar to pods, this is not immutable and is protected by the same podLock.
	// Note that basicPodManager.pods do not contain mirror pods as they are
	// filtered out beforehand.
	mirrorPods mirrorPods

	// A mirror pod manager which provides helper functions.
	mirrorManager mirrorManager
}

func newBasicPodManager(apiserverClient client.Interface) *basicPodManager {
	podManager := &basicPodManager{}
	podManager.mirrorManager = newBasicMirrorManager(apiserverClient)
	podManager.mirrorPods = *newMirrorPods()
	podManager.pods = []api.Pod{}
	return podManager
}

// This method is used only for testing to quickly set the internal pods.
func (self *basicPodManager) SetPods(pods []api.Pod) {
	self.pods, self.mirrorPods = filterAndCategorizePods(pods)
}

// Update the internal pods with those provided by the update.
// Records new and updated pods in newPods and updatedPods.
func (self *basicPodManager) UpdatePods(u PodUpdate, podSyncTypes map[types.UID]metrics.SyncPodType) {
	self.lock.Lock()
	defer self.lock.Unlock()
	switch u.Op {
	case SET:
		glog.V(3).Infof("SET: Containers changed")
		newPods, newMirrorPods := filterAndCategorizePods(u.Pods)

		// Store the new pods. Don't worry about filtering host ports since those
		// pods will never be looked up.
		existingPods := make(map[types.UID]struct{})
		for i := range self.pods {
			existingPods[self.pods[i].UID] = struct{}{}
		}
		for _, pod := range newPods {
			if _, ok := existingPods[pod.UID]; !ok {
				podSyncTypes[pod.UID] = metrics.SyncPodCreate
			}
		}
		// Actually update the pods.
		self.pods = newPods
		self.mirrorPods = newMirrorPods
	case UPDATE:
		glog.V(3).Infof("Update: Containers changed")

		// Store the updated pods. Don't worry about filtering host ports since those
		// pods will never be looked up.
		for i := range u.Pods {
			podSyncTypes[u.Pods[i].UID] = metrics.SyncPodUpdate
		}
		allPods := updatePods(u.Pods, self.pods)
		self.pods, self.mirrorPods = filterAndCategorizePods(allPods)
	default:
		panic("syncLoop does not support incremental changes")
	}

	// Mark all remaining pods as sync.
	for i := range self.pods {
		if _, ok := podSyncTypes[self.pods[i].UID]; !ok {
			podSyncTypes[u.Pods[i].UID] = metrics.SyncPodSync
		}
	}
}

func updatePods(changed []api.Pod, current []api.Pod) []api.Pod {
	updated := []api.Pod{}
	m := map[types.UID]*api.Pod{}
	for i := range changed {
		pod := &changed[i]
		m[pod.UID] = pod
	}

	for i := range current {
		pod := &current[i]
		if m[pod.UID] != nil {
			updated = append(updated, *m[pod.UID])
			glog.V(4).Infof("pod with UID: %q has a new spec %+v", pod.UID, *m[pod.UID])
		} else {
			updated = append(updated, *pod)
			glog.V(4).Infof("pod with UID: %q stay with the same spec %+v", pod.UID, *pod)
		}
	}

	return updated
}

// GetPods returns all pods bound to the kubelet and their spec, and the mirror
// pod map.
func (self *basicPodManager) GetPods() ([]api.Pod, mirrorPods) {
	self.lock.RLock()
	defer self.lock.RUnlock()
	return append([]api.Pod{}, self.pods...), self.mirrorPods
}

// GetPodByName provides the first pod that matches namespace and name, as well
// as whether the pod was found.
func (self *basicPodManager) GetPodByName(namespace, name string) (*api.Pod, bool) {
	self.lock.RLock()
	defer self.lock.RUnlock()
	for i := range self.pods {
		pod := self.pods[i]
		if pod.Namespace == namespace && pod.Name == name {
			return &pod, true
		}
	}
	return nil, false
}

func (self *basicPodManager) GetPodByFullName(podFullName string) (*api.Pod, bool) {
	name, namespace, err := ParsePodFullName(podFullName)
	if err != nil {
		return nil, false
	}
	return self.GetPodByName(namespace, name)
}

// If the UID belongs to a mirror pod, maps it to the UID of its static pod.
// Otherwise, return the original UID. All public-facing functions should
// perform this translation for UIDs because user may provide a mirror pod UID,
// which is not recognized by internal Kubelet functions.
func (self *basicPodManager) TranslatePodUID(uid types.UID) types.UID {
	if uid == "" {
		return uid
	}

	self.lock.RLock()
	defer self.lock.RUnlock()
	staticUID, ok := self.mirrorPods.GetStaticUID(uid)
	if ok {
		return staticUID
	} else {
		return uid
	}
}

// Delete all orphaned mirror pods. This method doesn't acquire the lock
// because it assumes the a copy of the mirrorPod is passed as an argument.
func (self *basicPodManager) DeleteOrphanedMirrorPods(mirrorPods *mirrorPods) {
	podFullNames := mirrorPods.GetOrphanedMirrorPodNames()
	for _, podFullName := range podFullNames {
		self.mirrorManager.DeleteMirrorPod(podFullName)
	}
}

func (self *basicPodManager) CreateMirrorPod(pod api.Pod, hostname string) error {
	return self.mirrorManager.CreateMirrorPod(pod, hostname)
}

func (self *basicPodManager) DeleteMirrorPod(podFullName string) error {
	return self.mirrorManager.DeleteMirrorPod(podFullName)
}
