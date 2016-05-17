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
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	"k8s.io/kubernetes/pkg/types"
)

// Pod manager stores and manages access to the pods.
type Manager interface {
	GetPods() []*api.Pod
	GetPodByFullName(podFullName string) (*api.Pod, bool)
	GetPodByName(namespace, name string) (*api.Pod, bool)
	GetPodByUID(types.UID) (*api.Pod, bool)

	// SetPods replaces the internal pods with the new pods.
	// It is currently only used for testing.
	SetPods(pods []*api.Pod)

	// Methods that modify a single pod.
	AddPod(pod *api.Pod)
	UpdatePod(pod *api.Pod)
	DeletePod(pod *api.Pod)

	// Mirror pod manager interface
	MirrorPodManager
}

// All maps in basicManager should be set by calling UpdatePods();
// individual arrays/maps are not immutable and no other methods should attempt
// to modify them.
type basicManager struct {
	// Protects all internal maps.
	lock sync.RWMutex

	// Regular pods indexed by UID.
	podByUID map[types.UID]*api.Pod

	// Pods indexed by full name for easy access.
	podByFullName map[string]*api.Pod

	// Mirror pod manager. Pod manager itself should have access to the management
	// function of mirror pod manager so as to set static pod correctly.
	*mirrorPodManager
}

func NewBasicPodManager(apiserverClient clientset.Interface) Manager {
	pm := &basicManager{mirrorPodManager: newMirrorPodManager(apiserverClient)}
	pm.SetPods(nil)
	return pm
}

// Set the internal pods based on the new pods.
func (pm *basicManager) SetPods(newPods []*api.Pod) {
	pm.lock.Lock()
	defer pm.lock.Unlock()

	pm.podByUID = make(map[types.UID]*api.Pod)
	pm.podByFullName = make(map[string]*api.Pod)

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

func (pm *basicManager) updatePodsInternal(pods ...*api.Pod) {
	for _, pod := range pods {
		if IsStaticPod(pod) && pm.mirrorPodManager != nil {
			pm.addStaticPod(pod)
		}
		podFullName := kubecontainer.GetPodFullName(pod)
		pm.podByUID[pod.UID] = pod
		pm.podByFullName[podFullName] = pod
	}
}

func (pm *basicManager) DeletePod(pod *api.Pod) {
	pm.lock.Lock()
	defer pm.lock.Unlock()
	if IsStaticPod(pod) && pm.mirrorPodManager != nil {
		pm.deleteStaticPod(pod)
	}
	podFullName := kubecontainer.GetPodFullName(pod)
	delete(pm.podByUID, pod.UID)
	delete(pm.podByFullName, podFullName)
}

// GetPods returns the regular pods bound to the kubelet and their spec.
func (pm *basicManager) GetPods() []*api.Pod {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	return podsMapToPods(pm.podByUID)
}

// GetPodByUID provides the pod that matches pod UID, as well as
// whether the pod is found.
func (pm *basicManager) GetPodByUID(uid types.UID) (*api.Pod, bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	pod, ok := pm.podByUID[uid]
	return pod, ok
}

// GetPodByName provides the pod that matches namespace and name,
// as well as whether the pod was found.
func (pm *basicManager) GetPodByName(namespace, name string) (*api.Pod, bool) {
	podFullName := kubecontainer.BuildPodFullName(name, namespace)
	return pm.GetPodByFullName(podFullName)
}

// GetPodByName returns the pod that matches full name, as well as
// whether the pod was found.
func (pm *basicManager) GetPodByFullName(podFullName string) (*api.Pod, bool) {
	pm.lock.RLock()
	defer pm.lock.RUnlock()
	pod, ok := pm.podByFullName[podFullName]
	return pod, ok
}

func podsMapToPods(UIDMap map[types.UID]*api.Pod) []*api.Pod {
	pods := make([]*api.Pod, 0, len(UIDMap))
	for _, pod := range UIDMap {
		pods = append(pods, pod)
	}
	return pods
}
