/*
Copyright 2015 The Kubernetes Authors.

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

// Manager stores and manages access to pods, maintaining the mappings
// between static pods and mirror pods.
//
// The kubelet discovers pod updates from 3 sources: file, http, and
// apiserver. Pods from non-apiserver sources are called static pods, and API
// server is not aware of the existence of static pods. In order to monitor
// the status of such pods, the kubelet creates a mirror pod for each static
// pod via the API server.
//
// A mirror pod has the same pod full name (name and namespace) as its static
// counterpart (albeit different metadata such as UID, etc). By leveraging the
// fact that the kubelet reports the pod status using the pod full name, the
// status of the mirror pod always reflects the actual status of the static
// pod. When a static pod gets deleted, the associated orphaned mirror pod
// will also be removed.
type Manager interface {
	// GetPods returns the regular pods bound to the kubelet and their spec.
	GetPods() []*api.Pod
	// GetPodByName returns the (non-mirror) pod that matches full name, as well as
	// whether the pod was found.
	GetPodByFullName(podFullName string) (*api.Pod, bool)
	// GetPodByName provides the (non-mirror) pod that matches namespace and
	// name, as well as whether the pod was found.
	GetPodByName(namespace, name string) (*api.Pod, bool)
	// GetPodByUID provides the (non-mirror) pod that matches pod UID, as well as
	// whether the pod is found.
	GetPodByUID(types.UID) (*api.Pod, bool)
	// GetPodByMirrorPod returns the static pod for the given mirror pod and
	// whether it was known to the pod manger.
	GetPodByMirrorPod(*api.Pod) (*api.Pod, bool)
	// GetMirrorPodByPod returns the mirror pod for the given static pod and
	// whether it was known to the pod manager.
	GetMirrorPodByPod(*api.Pod) (*api.Pod, bool)
	// GetPodsAndMirrorPods returns the both regular and mirror pods.
	GetPodsAndMirrorPods() ([]*api.Pod, []*api.Pod)
	// SetPods replaces the internal pods with the new pods.
	// It is currently only used for testing.
	SetPods(pods []*api.Pod)
	// AddPod adds the given pod to the manager.
	AddPod(pod *api.Pod)
	// UpdatePod updates the given pod in the manager.
	UpdatePod(pod *api.Pod)
	// DeletePod deletes the given pod from the manager.  For mirror pods,
	// this means deleting the mappings related to mirror pods.  For non-
	// mirror pods, this means deleting from indexes for all non-mirror pods.
	DeletePod(pod *api.Pod)
	// DeleteOrphanedMirrorPods deletes all mirror pods which do not have
	// associated static pods. This method sends deletion requests to the API
	// server, but does NOT modify the internal pod storage in basicManager.
	DeleteOrphanedMirrorPods()
	// TranslatePodUID returns the UID which is the mirror pod or static pod
	// of the pod with the given UID.  If the UID belongs to a mirror pod,
	// returns the UID of its static pod.  If the UID belongs to a static pod,
	// returns the UID of its mirror pod.  Otherwise, returns the original UID.
	//
	// All public-facing functions should perform this translation for UIDs
	// because user may provide a mirror pod UID, which is not recognized by
	// internal Kubelet functions.
	TranslatePodUID(uid types.UID) types.UID
	// GetUIDTranslations returns the mappings of static pod UIDs to mirror pod
	// UIDs and mirror pod UIDs to static pod UIDs.
	GetUIDTranslations() (podToMirror, mirrorToPod map[types.UID]types.UID)
	// IsMirrorPodOf returns true if mirrorPod is a correct representation of
	// pod; false otherwise.
	IsMirrorPodOf(mirrorPod, pod *api.Pod) bool

	MirrorClient
}

// basicManager is a functional Manger.
//
// All fields in basicManager are read-only and are updated calling SetPods,
// AddPod, UpdatePod, or DeletePod.
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

	// Mirror pod UID to pod UID map.
	translationByUID map[types.UID]types.UID

	// A mirror pod client to create/delete mirror pods.
	MirrorClient
}

// NewBasicPodManager returns a functional Manager.
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

	pm.podByUID = make(map[types.UID]*api.Pod)
	pm.podByFullName = make(map[string]*api.Pod)
	pm.mirrorPodByUID = make(map[types.UID]*api.Pod)
	pm.mirrorPodByFullName = make(map[string]*api.Pod)
	pm.translationByUID = make(map[types.UID]types.UID)

	pm.updatePodsInternal(newPods...)
}

func (pm *basicManager) AddPod(pod *api.Pod) {
	pm.UpdatePod(pod)
}

func (pm *basicManager) UpdatePod(pod *api.Pod) {
	pm.lock.Lock()
	defer pm.lock.Unlock()
	pm.updatePodsInternal(pod)
}

// updatePodsInternal replaces the given pods in the current state of the
// manager, updating the various indices.  The caller is assumed to hold the
// lock.
func (pm *basicManager) updatePodsInternal(pods ...*api.Pod) {
	for _, pod := range pods {
		podFullName := kubecontainer.GetPodFullName(pod)
		if IsMirrorPod(pod) {
			pm.mirrorPodByUID[pod.UID] = pod
			pm.mirrorPodByFullName[podFullName] = pod
			if p, ok := pm.podByFullName[podFullName]; ok {
				pm.translationByUID[pod.UID] = p.UID
			}
		} else {
			pm.podByUID[pod.UID] = pod
			pm.podByFullName[podFullName] = pod
			if mirror, ok := pm.mirrorPodByFullName[podFullName]; ok {
				pm.translationByUID[mirror.UID] = pod.UID
			}
		}
	}
}

func (pm *basicManager) DeletePod(pod *api.Pod) {
	pm.lock.Lock()
	defer pm.lock.Unlock()
	podFullName := kubecontainer.GetPodFullName(pod)
	if IsMirrorPod(pod) {
		delete(pm.mirrorPodByUID, pod.UID)
		delete(pm.mirrorPodByFullName, podFullName)
		delete(pm.translationByUID, pod.UID)
	} else {
		delete(pm.podByUID, pod.UID)
		delete(pm.podByFullName, podFullName)
	}
}

func (pm *basicManager) GetPods() []*api.Pod {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	return podsMapToPods(pm.podByUID)
}

func (pm *basicManager) GetPodsAndMirrorPods() ([]*api.Pod, []*api.Pod) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	pods := podsMapToPods(pm.podByUID)
	mirrorPods := podsMapToPods(pm.mirrorPodByUID)
	return pods, mirrorPods
}

func (pm *basicManager) GetPodByUID(uid types.UID) (*api.Pod, bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	pod, ok := pm.podByUID[uid]
	return pod, ok
}

func (pm *basicManager) GetPodByName(namespace, name string) (*api.Pod, bool) {
	podFullName := kubecontainer.BuildPodFullName(name, namespace)
	return pm.GetPodByFullName(podFullName)
}

func (pm *basicManager) GetPodByFullName(podFullName string) (*api.Pod, bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	pod, ok := pm.podByFullName[podFullName]
	return pod, ok
}

func (pm *basicManager) TranslatePodUID(uid types.UID) types.UID {
	if uid == "" {
		return uid
	}

	pm.lock.RLock()
	defer pm.lock.RUnlock()
	if translated, ok := pm.translationByUID[uid]; ok {
		return translated
	}
	return uid
}

func (pm *basicManager) GetUIDTranslations() (podToMirror, mirrorToPod map[types.UID]types.UID) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()

	podToMirror = make(map[types.UID]types.UID, len(pm.translationByUID))
	mirrorToPod = make(map[types.UID]types.UID, len(pm.translationByUID))
	for k, v := range pm.translationByUID {
		mirrorToPod[k] = v
		podToMirror[v] = k
	}
	return podToMirror, mirrorToPod
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

func (pm *basicManager) DeleteOrphanedMirrorPods() {
	podFullNames := pm.getOrphanedMirrorPodNames()
	for _, podFullName := range podFullNames {
		pm.MirrorClient.DeleteMirrorPod(podFullName)
	}
}

func (pm *basicManager) IsMirrorPodOf(mirrorPod, pod *api.Pod) bool {
	// Check name and namespace first.
	if pod.Name != mirrorPod.Name || pod.Namespace != mirrorPod.Namespace {
		return false
	}
	hash, ok := getHashFromMirrorPod(mirrorPod)
	if !ok {
		return false
	}
	return hash == getPodHash(pod)
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
