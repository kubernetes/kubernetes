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

//go:generate mockgen -destination=testing/mock_manager.go -package=testing -build_flags=-mod=mod . Manager
package pod

import (
	"sync"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/kubelet/metrics"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
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
	// GetPodByFullName returns the (non-mirror) pod that matches full name, as well as
	// whether the pod was found.
	GetPodByFullName(podFullName string) (*v1.Pod, bool)
	// GetPodByName provides the (non-mirror) pod that matches namespace and
	// name, as well as whether the pod was found.
	GetPodByName(namespace, name string) (*v1.Pod, bool)
	// GetPodByUID provides the (non-mirror) pod that matches pod UID, as well as
	// whether the pod is found.
	GetPodByUID(types.UID) (*v1.Pod, bool)
	// GetPodByMirrorPod returns the static pod for the given mirror pod and
	// whether it was known to the pod manager.
	GetPodByMirrorPod(*v1.Pod) (*v1.Pod, bool)
	// GetMirrorPodByPod returns the mirror pod for the given static pod and
	// whether it was known to the pod manager.
	GetMirrorPodByPod(*v1.Pod) (*v1.Pod, bool)
	// GetPodAndMirrorPod returns the complement for a pod - if a pod was provided
	// and a mirror pod can be found, return it. If a mirror pod is provided and
	// the pod can be found, return it and true for wasMirror.
	GetPodAndMirrorPod(*v1.Pod) (pod, mirrorPod *v1.Pod, wasMirror bool)

	// GetPods returns the regular pods bound to the kubelet and their spec.
	GetPods() []*v1.Pod

	// GetPodsAndMirrorPods returns the set of pods, the set of mirror pods, and
	// the pod fullnames of any orphaned mirror pods.
	GetPodsAndMirrorPods() (allPods []*v1.Pod, allMirrorPods []*v1.Pod, orphanedMirrorPodFullnames []string)

	// SetPods replaces the internal pods with the new pods.
	// It is currently only used for testing.
	SetPods(pods []*v1.Pod)
	// AddPod adds the given pod to the manager.
	AddPod(pod *v1.Pod)
	// UpdatePod updates the given pod in the manager.
	UpdatePod(pod *v1.Pod)
	// RemovePod deletes the given pod from the manager.  For mirror pods,
	// this means deleting the mappings related to mirror pods.  For non-
	// mirror pods, this means deleting from indexes for all non-mirror pods.
	RemovePod(pod *v1.Pod)

	// TranslatePodUID returns the actual UID of a pod. If the UID belongs to
	// a mirror pod, returns the UID of its static pod. Otherwise, returns the
	// original UID.
	//
	// All public-facing functions should perform this translation for UIDs
	// because user may provide a mirror pod UID, which is not recognized by
	// internal Kubelet functions.
	TranslatePodUID(uid types.UID) kubetypes.ResolvedPodUID
	// GetUIDTranslations returns the mappings of static pod UIDs to mirror pod
	// UIDs and mirror pod UIDs to static pod UIDs.
	GetUIDTranslations() (podToMirror map[kubetypes.ResolvedPodUID]kubetypes.MirrorPodUID, mirrorToPod map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID)
}

// basicManager is a functional Manager.
//
// All fields in basicManager are read-only and are updated calling SetPods,
// AddPod, UpdatePod, or RemovePod.
type basicManager struct {
	// Protects all internal maps.
	lock sync.RWMutex

	// Regular pods indexed by UID.
	podByUID map[kubetypes.ResolvedPodUID]*v1.Pod
	// Mirror pods indexed by UID.
	mirrorPodByUID map[kubetypes.MirrorPodUID]*v1.Pod

	// Pods indexed by full name for easy access.
	podByFullName       map[string]*v1.Pod
	mirrorPodByFullName map[string]*v1.Pod

	// Mirror pod UID to pod UID map.
	translationByUID map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID
}

// NewBasicPodManager returns a functional Manager.
func NewBasicPodManager() Manager {
	pm := &basicManager{}
	pm.SetPods(nil)
	return pm
}

// Set the internal pods based on the new pods.
func (pm *basicManager) SetPods(newPods []*v1.Pod) {
	pm.lock.Lock()
	defer pm.lock.Unlock()

	pm.podByUID = make(map[kubetypes.ResolvedPodUID]*v1.Pod)
	pm.podByFullName = make(map[string]*v1.Pod)
	pm.mirrorPodByUID = make(map[kubetypes.MirrorPodUID]*v1.Pod)
	pm.mirrorPodByFullName = make(map[string]*v1.Pod)
	pm.translationByUID = make(map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID)

	pm.updatePodsInternal(newPods...)
}

func (pm *basicManager) AddPod(pod *v1.Pod) {
	pm.UpdatePod(pod)
}

func (pm *basicManager) UpdatePod(pod *v1.Pod) {
	pm.lock.Lock()
	defer pm.lock.Unlock()
	pm.updatePodsInternal(pod)
}

// updateMetrics updates the metrics surfaced by the pod manager.
// oldPod or newPod may be nil to signify creation or deletion.
func updateMetrics(oldPod, newPod *v1.Pod) {
	var numEC int
	if oldPod != nil {
		numEC -= len(oldPod.Spec.EphemeralContainers)
	}
	if newPod != nil {
		numEC += len(newPod.Spec.EphemeralContainers)
	}
	if numEC != 0 {
		metrics.ManagedEphemeralContainers.Add(float64(numEC))
	}
}

// updatePodsInternal replaces the given pods in the current state of the
// manager, updating the various indices. The caller is assumed to hold the
// lock.
func (pm *basicManager) updatePodsInternal(pods ...*v1.Pod) {
	for _, pod := range pods {
		podFullName := kubecontainer.GetPodFullName(pod)
		// This logic relies on a static pod and its mirror to have the same name.
		// It is safe to type convert here due to the IsMirrorPod guard.
		if kubetypes.IsMirrorPod(pod) {
			mirrorPodUID := kubetypes.MirrorPodUID(pod.UID)
			pm.mirrorPodByUID[mirrorPodUID] = pod
			pm.mirrorPodByFullName[podFullName] = pod
			if p, ok := pm.podByFullName[podFullName]; ok {
				pm.translationByUID[mirrorPodUID] = kubetypes.ResolvedPodUID(p.UID)
			}
		} else {
			resolvedPodUID := kubetypes.ResolvedPodUID(pod.UID)
			updateMetrics(pm.podByUID[resolvedPodUID], pod)
			pm.podByUID[resolvedPodUID] = pod
			pm.podByFullName[podFullName] = pod
			if mirror, ok := pm.mirrorPodByFullName[podFullName]; ok {
				pm.translationByUID[kubetypes.MirrorPodUID(mirror.UID)] = resolvedPodUID
			}
		}
	}
}

func (pm *basicManager) RemovePod(pod *v1.Pod) {
	updateMetrics(pod, nil)
	pm.lock.Lock()
	defer pm.lock.Unlock()
	podFullName := kubecontainer.GetPodFullName(pod)
	// It is safe to type convert here due to the IsMirrorPod guard.
	if kubetypes.IsMirrorPod(pod) {
		mirrorPodUID := kubetypes.MirrorPodUID(pod.UID)
		delete(pm.mirrorPodByUID, mirrorPodUID)
		delete(pm.mirrorPodByFullName, podFullName)
		delete(pm.translationByUID, mirrorPodUID)
	} else {
		delete(pm.podByUID, kubetypes.ResolvedPodUID(pod.UID))
		delete(pm.podByFullName, podFullName)
	}
}

func (pm *basicManager) GetPods() []*v1.Pod {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	return podsMapToPods(pm.podByUID)
}

func (pm *basicManager) GetPodsAndMirrorPods() (allPods []*v1.Pod, allMirrorPods []*v1.Pod, orphanedMirrorPodFullnames []string) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	allPods = podsMapToPods(pm.podByUID)
	allMirrorPods = mirrorPodsMapToMirrorPods(pm.mirrorPodByUID)

	for podFullName := range pm.mirrorPodByFullName {
		if _, ok := pm.podByFullName[podFullName]; !ok {
			orphanedMirrorPodFullnames = append(orphanedMirrorPodFullnames, podFullName)
		}
	}
	return allPods, allMirrorPods, orphanedMirrorPodFullnames
}

func (pm *basicManager) GetPodByUID(uid types.UID) (*v1.Pod, bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	pod, ok := pm.podByUID[kubetypes.ResolvedPodUID(uid)] // Safe conversion, map only holds non-mirrors.
	return pod, ok
}

func (pm *basicManager) GetPodByName(namespace, name string) (*v1.Pod, bool) {
	podFullName := kubecontainer.BuildPodFullName(name, namespace)
	return pm.GetPodByFullName(podFullName)
}

func (pm *basicManager) GetPodByFullName(podFullName string) (*v1.Pod, bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	pod, ok := pm.podByFullName[podFullName]
	return pod, ok
}

func (pm *basicManager) TranslatePodUID(uid types.UID) kubetypes.ResolvedPodUID {
	// It is safe to type convert to a resolved UID because type conversion is idempotent.
	if uid == "" {
		return kubetypes.ResolvedPodUID(uid)
	}

	pm.lock.RLock()
	defer pm.lock.RUnlock()
	if translated, ok := pm.translationByUID[kubetypes.MirrorPodUID(uid)]; ok {
		return translated
	}
	return kubetypes.ResolvedPodUID(uid)
}

func (pm *basicManager) GetUIDTranslations() (podToMirror map[kubetypes.ResolvedPodUID]kubetypes.MirrorPodUID,
	mirrorToPod map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()

	podToMirror = make(map[kubetypes.ResolvedPodUID]kubetypes.MirrorPodUID, len(pm.translationByUID))
	mirrorToPod = make(map[kubetypes.MirrorPodUID]kubetypes.ResolvedPodUID, len(pm.translationByUID))
	// Insert empty translation mapping for all static pods.
	for uid, pod := range pm.podByUID {
		if !kubetypes.IsStaticPod(pod) {
			continue
		}
		podToMirror[uid] = ""
	}
	// Fill in translations. Notice that if there is no mirror pod for a
	// static pod, its uid will be translated into empty string "". This
	// is WAI, from the caller side we can know that the static pod doesn't
	// have a corresponding mirror pod instead of using static pod uid directly.
	for k, v := range pm.translationByUID {
		mirrorToPod[k] = v
		podToMirror[v] = k
	}
	return podToMirror, mirrorToPod
}

// IsMirrorPodOf returns true if pod and mirrorPod are associated with each other.
func IsMirrorPodOf(mirrorPod, pod *v1.Pod) bool {
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

func podsMapToPods(UIDMap map[kubetypes.ResolvedPodUID]*v1.Pod) []*v1.Pod {
	pods := make([]*v1.Pod, 0, len(UIDMap))
	for _, pod := range UIDMap {
		pods = append(pods, pod)
	}
	return pods
}

func mirrorPodsMapToMirrorPods(UIDMap map[kubetypes.MirrorPodUID]*v1.Pod) []*v1.Pod {
	pods := make([]*v1.Pod, 0, len(UIDMap))
	for _, pod := range UIDMap {
		pods = append(pods, pod)
	}
	return pods
}

func (pm *basicManager) GetMirrorPodByPod(pod *v1.Pod) (*v1.Pod, bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	mirrorPod, ok := pm.mirrorPodByFullName[kubecontainer.GetPodFullName(pod)]
	return mirrorPod, ok
}

func (pm *basicManager) GetPodByMirrorPod(mirrorPod *v1.Pod) (*v1.Pod, bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	pod, ok := pm.podByFullName[kubecontainer.GetPodFullName(mirrorPod)]
	return pod, ok
}

func (pm *basicManager) GetPodAndMirrorPod(aPod *v1.Pod) (pod, mirrorPod *v1.Pod, wasMirror bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()

	fullName := kubecontainer.GetPodFullName(aPod)
	if kubetypes.IsMirrorPod(aPod) {
		return pm.podByFullName[fullName], aPod, true
	}
	return aPod, pm.mirrorPodByFullName[fullName], false

}
