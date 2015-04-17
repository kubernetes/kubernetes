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
	kubecontainer "github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/container"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/kubelet/metrics"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/golang/glog"
)

// Pod manager stores and manages access to the pods.
//
// Kubelet discovers pod updates from 3 sources: file, http, and apiserver.
// Pods from non-apiserver sources are called static pods, and API server is
// not aware of the existence of static pods. In order to monitor the status of
// such pods, kubelet creates a mirror pod for each static pod via the API
// server.
//
// A mirror pod has the same pod full name (name and namespace) as its static
// counterpart (albeit different metadata such as UID, etc). By leveraging the
// fact that kubelet reports the pod status using the pod full name, the status
// of the mirror pod always reflects the actual status of the static pod.
// When a static pod gets deleted, the associated orphaned mirror pod will
// also be removed.

type podManager interface {
	GetPods() []*api.Pod
	GetPodByFullName(podFullName string) (*api.Pod, bool)
	GetPodByName(namespace, name string) (*api.Pod, bool)
	GetPodsAndMirrorMap() ([]*api.Pod, map[string]api.Pod)
	SetPods(pods []*api.Pod)
	UpdatePods(u PodUpdate, podSyncTypes map[types.UID]metrics.SyncPodType)
	DeleteOrphanedMirrorPods()
	TranslatePodUID(uid types.UID) types.UID
	IsMirrorPodOf(mirrorPod, pod *api.Pod) bool
	mirrorClient
}

// All maps in basicPodManager should be set by calling UpdatePods();
// individual arrays/maps are not immutable and no other methods should attempt
// to modify them.
type basicPodManager struct {
	// Protects all internal maps.
	lock sync.RWMutex

	// Regular pods indexed by UID.
	podByUID map[types.UID]*api.Pod
	// Mirror pods indexed by UID.
	mirrorPodByUID map[types.UID]*api.Pod

	// Pods indexed by full name for easy access.
	podByFullName       map[string]*api.Pod
	mirrorPodByFullName map[string]*api.Pod

	// A mirror pod client to create/delete mirror pods.
	mirrorClient
}

func newBasicPodManager(apiserverClient client.Interface) *basicPodManager {
	pm := &basicPodManager{}
	pm.mirrorClient = newBasicMirrorClient(apiserverClient)
	pm.SetPods(nil)
	return pm
}

// Update the internal pods with those provided by the update.
func (self *basicPodManager) UpdatePods(u PodUpdate, podSyncTypes map[types.UID]metrics.SyncPodType) {
	self.lock.Lock()
	defer self.lock.Unlock()
	switch u.Op {
	case SET:
		glog.V(3).Infof("SET: Containers changed")
		// Store the new pods. Don't worry about filtering host ports since those
		// pods will never be looked up.
		existingPods := make(map[types.UID]struct{})
		for uid := range self.podByUID {
			existingPods[uid] = struct{}{}
		}

		// Update the internal pods.
		self.setPods(u.Pods)

		for uid := range self.podByUID {
			if _, ok := existingPods[uid]; !ok {
				podSyncTypes[uid] = metrics.SyncPodCreate
			}
		}
	case UPDATE:
		glog.V(3).Infof("Update: Containers changed")

		// Store the updated pods. Don't worry about filtering host ports since those
		// pods will never be looked up.
		for i := range u.Pods {
			podSyncTypes[u.Pods[i].UID] = metrics.SyncPodUpdate
		}
		allPods := applyUpdates(u.Pods, self.getAllPods())
		self.setPods(allPods)
	default:
		panic("syncLoop does not support incremental changes")
	}

	// Mark all remaining pods as sync.
	for uid := range self.podByUID {
		if _, ok := podSyncTypes[uid]; !ok {
			podSyncTypes[uid] = metrics.SyncPodSync
		}
	}
}

// Set the internal pods based on the new pods.
func (self *basicPodManager) SetPods(newPods []*api.Pod) {
	self.lock.Lock()
	defer self.lock.Unlock()
	self.setPods(newPods)
}

func (self *basicPodManager) setPods(newPods []*api.Pod) {
	podByUID := make(map[types.UID]*api.Pod)
	mirrorPodByUID := make(map[types.UID]*api.Pod)
	podByFullName := make(map[string]*api.Pod)
	mirrorPodByFullName := make(map[string]*api.Pod)

	for _, pod := range newPods {
		podFullName := kubecontainer.GetPodFullName(pod)
		if isMirrorPod(pod) {
			mirrorPodByUID[pod.UID] = pod
			mirrorPodByFullName[podFullName] = pod
		} else {
			podByUID[pod.UID] = pod
			podByFullName[podFullName] = pod
		}
	}

	self.podByUID = podByUID
	self.podByFullName = podByFullName
	self.mirrorPodByUID = mirrorPodByUID
	self.mirrorPodByFullName = mirrorPodByFullName
}

func applyUpdates(changed []*api.Pod, current []*api.Pod) []*api.Pod {
	updated := []*api.Pod{}
	m := map[types.UID]*api.Pod{}
	for _, pod := range changed {
		m[pod.UID] = pod
	}

	for _, pod := range current {
		if m[pod.UID] != nil {
			updated = append(updated, m[pod.UID])
			glog.V(4).Infof("pod with UID: %q has a new spec %+v", pod.UID, *m[pod.UID])
		} else {
			updated = append(updated, pod)
			glog.V(4).Infof("pod with UID: %q stay with the same spec %+v", pod.UID, *pod)
		}
	}

	return updated
}

// GetPods returns the regular pods bound to the kubelet and their spec.
func (self *basicPodManager) GetPods() []*api.Pod {
	self.lock.RLock()
	defer self.lock.RUnlock()
	return podsMapToPods(self.podByUID)
}

// Returns all pods (including mirror pods).
func (self *basicPodManager) getAllPods() []*api.Pod {
	return append(podsMapToPods(self.podByUID), podsMapToPods(self.mirrorPodByUID)...)
}

// GetPodsAndMirrorMap returns the a copy of the regular pods and the mirror
// pods indexed by full name.
func (self *basicPodManager) GetPodsAndMirrorMap() ([]*api.Pod, map[string]api.Pod) {
	self.lock.RLock()
	defer self.lock.RUnlock()
	mirrorPods := make(map[string]api.Pod)
	for key, pod := range self.mirrorPodByFullName {
		mirrorPods[key] = *pod
	}
	return podsMapToPods(self.podByUID), mirrorPods
}

// GetPodByName provides the (non-mirror) pod that matches namespace and name,
// as well as whether the pod was found.
func (self *basicPodManager) GetPodByName(namespace, name string) (*api.Pod, bool) {
	podFullName := kubecontainer.BuildPodFullName(name, namespace)
	return self.GetPodByFullName(podFullName)
}

// GetPodByName returns the (non-mirror) pod that matches full name, as well as
// whether the pod was found.
func (self *basicPodManager) GetPodByFullName(podFullName string) (*api.Pod, bool) {
	self.lock.RLock()
	defer self.lock.RUnlock()
	pod, ok := self.podByFullName[podFullName]
	return pod, ok
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
	if mirrorPod, ok := self.mirrorPodByUID[uid]; ok {
		podFullName := kubecontainer.GetPodFullName(mirrorPod)
		if pod, ok := self.podByFullName[podFullName]; ok {
			return pod.UID
		}
	}
	return uid
}

func (self *basicPodManager) getOrphanedMirrorPodNames() []string {
	self.lock.RLock()
	defer self.lock.RUnlock()
	var podFullNames []string
	for podFullName := range self.mirrorPodByFullName {
		if _, ok := self.podByFullName[podFullName]; !ok {
			podFullNames = append(podFullNames, podFullName)
		}
	}
	return podFullNames
}

// Delete all mirror pods which do not have associated static pods. This method
// sends deletion requets to the API server, but does NOT modify the internal
// pod storage in basicPodManager.
func (self *basicPodManager) DeleteOrphanedMirrorPods() {
	podFullNames := self.getOrphanedMirrorPodNames()
	for _, podFullName := range podFullNames {
		self.mirrorClient.DeleteMirrorPod(podFullName)
	}
}

// Returns true if mirrorPod is a correct representation of pod; false otherwise.
func (self *basicPodManager) IsMirrorPodOf(mirrorPod, pod *api.Pod) bool {
	// Check name and namespace first.
	if pod.Name != mirrorPod.Name || pod.Namespace != mirrorPod.Namespace {
		return false
	}
	return api.Semantic.DeepEqual(&pod.Spec, &mirrorPod.Spec)
}

func podsMapToPods(UIDMap map[types.UID]*api.Pod) []*api.Pod {
	pods := make([]*api.Pod, 0, len(UIDMap))
	for _, pod := range UIDMap {
		pods = append(pods, pod)
	}
	return pods
}
