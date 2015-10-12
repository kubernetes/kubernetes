/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package pod

import (
	"sync"

	"k8s.io/kubernetes/pkg/api"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
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

type Manager interface {
	GetPods() []*api.Pod
	GetPodByFullName(podFullName string) (*api.Pod, bool)
	GetPodByName(namespace, name string) (*api.Pod, bool)
	GetPodByMirrorPod(*api.Pod) (*api.Pod, bool)
	GetMirrorPodByPod(*api.Pod) (*api.Pod, bool)
	GetPodsAndMirrorPods() ([]*api.Pod, []*api.Pod)

	// SetPods replaces the internal pods with the new pods.
	// It is currently only used for testing.
	SetPods(pods []*api.Pod)

	// Methods that modify a single pod.
	AddPod(pod *api.Pod)
	UpdatePod(pod *api.Pod)
	DeletePod(pod *api.Pod)

	DeleteOrphanedMirrorPods()
	TranslatePodUID(uid types.UID) types.UID
	IsMirrorPodOf(mirrorPod, pod *api.Pod) bool
	MirrorClient
}

// All maps in basicManager should be set by calling UpdatePods();
// individual arrays/maps are not immutable and no other methods should attempt
// to modify them.
type basicManager struct {
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
	MirrorClient
}

func NewBasicPodManager(client MirrorClient) Manager {
	pm := &basicManager{}
	pm.MirrorClient = client
	pm.SetPods(nil)
	return pm
}

// Set the internal pods based on the new pods.
func (pm *basicManager) SetPods(newPods []*api.Pod) {
	pm.lock.Lock()
	defer pm.lock.Unlock()
	pm.setPods(newPods)
}

func (pm *basicManager) setPods(newPods []*api.Pod) {
	podByUID := make(map[types.UID]*api.Pod)
	mirrorPodByUID := make(map[types.UID]*api.Pod)
	podByFullName := make(map[string]*api.Pod)
	mirrorPodByFullName := make(map[string]*api.Pod)

	for _, pod := range newPods {
		podFullName := kubecontainer.GetPodFullName(pod)
		if IsMirrorPod(pod) {
			mirrorPodByUID[pod.UID] = pod
			mirrorPodByFullName[podFullName] = pod
		} else {
			podByUID[pod.UID] = pod
			podByFullName[podFullName] = pod
		}
	}

	pm.podByUID = podByUID
	pm.podByFullName = podByFullName
	pm.mirrorPodByUID = mirrorPodByUID
	pm.mirrorPodByFullName = mirrorPodByFullName
}

func (pm *basicManager) AddPod(pod *api.Pod) {
	pm.UpdatePod(pod)
}

func (pm *basicManager) UpdatePod(pod *api.Pod) {
	pm.lock.Lock()
	defer pm.lock.Unlock()
	podFullName := kubecontainer.GetPodFullName(pod)
	if IsMirrorPod(pod) {
		pm.mirrorPodByUID[pod.UID] = pod
		pm.mirrorPodByFullName[podFullName] = pod
	} else {
		pm.podByUID[pod.UID] = pod
		pm.podByFullName[podFullName] = pod
	}
}

func (pm *basicManager) DeletePod(pod *api.Pod) {
	pm.lock.Lock()
	defer pm.lock.Unlock()
	podFullName := kubecontainer.GetPodFullName(pod)
	if IsMirrorPod(pod) {
		delete(pm.mirrorPodByUID, pod.UID)
		delete(pm.mirrorPodByFullName, podFullName)
	} else {
		delete(pm.podByUID, pod.UID)
		delete(pm.podByFullName, podFullName)
	}
}

// GetPods returns the regular pods bound to the kubelet and their spec.
func (pm *basicManager) GetPods() []*api.Pod {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	return podsMapToPods(pm.podByUID)
}

// GetPodsAndMirrorPods returns the both regular and mirror pods.
func (pm *basicManager) GetPodsAndMirrorPods() ([]*api.Pod, []*api.Pod) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	pods := podsMapToPods(pm.podByUID)
	mirrorPods := podsMapToPods(pm.mirrorPodByUID)
	return pods, mirrorPods
}

// Returns all pods (including mirror pods).
func (pm *basicManager) getAllPods() []*api.Pod {
	return append(podsMapToPods(pm.podByUID), podsMapToPods(pm.mirrorPodByUID)...)
}

// GetPodByName provides the (non-mirror) pod that matches namespace and name,
// as well as whether the pod was found.
func (pm *basicManager) GetPodByName(namespace, name string) (*api.Pod, bool) {
	podFullName := kubecontainer.BuildPodFullName(name, namespace)
	return pm.GetPodByFullName(podFullName)
}

// GetPodByName returns the (non-mirror) pod that matches full name, as well as
// whether the pod was found.
func (pm *basicManager) GetPodByFullName(podFullName string) (*api.Pod, bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	pod, ok := pm.podByFullName[podFullName]
	return pod, ok
}

// If the UID belongs to a mirror pod, maps it to the UID of its static pod.
// Otherwise, return the original UID. All public-facing functions should
// perform this translation for UIDs because user may provide a mirror pod UID,
// which is not recognized by internal Kubelet functions.
func (pm *basicManager) TranslatePodUID(uid types.UID) types.UID {
	if uid == "" {
		return uid
	}

	pm.lock.RLock()
	defer pm.lock.RUnlock()
	if mirrorPod, ok := pm.mirrorPodByUID[uid]; ok {
		podFullName := kubecontainer.GetPodFullName(mirrorPod)
		if pod, ok := pm.podByFullName[podFullName]; ok {
			return pod.UID
		}
	}
	return uid
}

func (pm *basicManager) getOrphanedMirrorPodNames() []string {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	var podFullNames []string
	for podFullName := range pm.mirrorPodByFullName {
		if _, ok := pm.podByFullName[podFullName]; !ok {
			podFullNames = append(podFullNames, podFullName)
		}
	}
	return podFullNames
}

// Delete all mirror pods which do not have associated static pods. This method
// sends deletion requets to the API server, but does NOT modify the internal
// pod storage in basicManager.
func (pm *basicManager) DeleteOrphanedMirrorPods() {
	podFullNames := pm.getOrphanedMirrorPodNames()
	for _, podFullName := range podFullNames {
		pm.MirrorClient.DeleteMirrorPod(podFullName)
	}
}

// Returns true if mirrorPod is a correct representation of pod; false otherwise.
func (pm *basicManager) IsMirrorPodOf(mirrorPod, pod *api.Pod) bool {
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

func (pm *basicManager) GetMirrorPodByPod(pod *api.Pod) (*api.Pod, bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	mirrorPod, ok := pm.mirrorPodByFullName[kubecontainer.GetPodFullName(pod)]
	return mirrorPod, ok
}

func (pm *basicManager) GetPodByMirrorPod(mirrorPod *api.Pod) (*api.Pod, bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	pod, ok := pm.podByFullName[kubecontainer.GetPodFullName(mirrorPod)]
	return pod, ok
}
