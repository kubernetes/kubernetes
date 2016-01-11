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
	"time"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
	"k8s.io/kubernetes/pkg/kubelet/util/format"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/wait"
)

// MirrorPodManager stores and manages all the mirror pods.
//
// Kubelet discovers pod updates from 3 sources: file, http, and apiserver.
// Pods from non-apiserver sources are called static pods, and apiserver is
// not aware of the existence of static pods. In order to monitor the status of
// such pods, MirrorPodManager creates a mirror pod for each static pod via the
// apiserver.
//
// A mirror pod has the same pod full name (name and namespace) as its static
// counterpart (albeit different metadata such as UID, etc). By leveraging the
// fact that kubelet reports the pod status using the pod full name, the status
// of the mirror pod always reflects the actual status of the static pod.
// When a static pod gets deleted, the associated orphaned mirror pod will
// also be removed.
//
// The mirror pod manager is composed of staticPods store, mirrorPods store and a syncer.
// As is shown in the figure below, mirrorPods stores the mirror pods from apiserver. staticPods
// stores the static pods from other sources. The syncer periodically checks mirrorPods and staticPods,
// if they are not match, it will create and delete mirror pods accordingly.
//     file/http ---> staticPods --->
//                                   syncer
//     apiserver ---> mirrorPods --->
type MirrorPodManager interface {
	// Start starts the mirror pod manager.
	Start()

	// AddMirrorPod adds a mirror pod, mirror pod will be automatically recreated when needed
	AddMirrorPod(mirrorPod *api.Pod)
	// DeleteMirrorPod deletes a mirror pod, pod will be automatically recreated when needed
	DeleteMirrorPod(mirrorPod *api.Pod)

	// GetMirrorPod gets current mirror pod by name and namespace, if there is no mirror pod now, return nil
	GetMirrorPod(name, namespace string) *api.Pod
	// GetMirrorPodByUID gets current mirror pod by mirror pod uid, if there is no corresponding mirror pod now,
	// return nil.
	// Note that mirror pod manager only holds the current mirror pod, so when passing in the uid of old
	// mirror pod, the function will also return nil. This make sure that we won't work with old mirror pod.
	GetMirrorPodByUID(uid types.UID) *api.Pod
	// GetMirrorOfStaticPod gets current mirror pod by static pod, if there is no mirror pod now (it hasn't been
	// created or recreated), return nil.
	GetMirrorOfStaticPod(staticPod *api.Pod) *api.Pod

	// TranslatePodUID translates the pod uid if its a mirror pod. If the UID belongs to a mirror pod, maps it to
	// the UID of its static pod. Otherwise, return the original UID. All public-facing functions should perform
	// this translation for UIDs because user may provide a mirror pod UID, which is not recognized by internal
	// Kubelet functions.
	TranslatePodUID(uid types.UID) types.UID
}

type mirrorPods struct {
	// Mirror pods indexed by full name
	pods map[string]*api.Pod
	// Mirror pods indexed by UID
	upods map[types.UID]*api.Pod
}

func newMirrorPods() mirrorPods {
	return mirrorPods{
		pods:  make(map[string]*api.Pod),
		upods: make(map[types.UID]*api.Pod),
	}
}

func (mp *mirrorPods) getPod(key string) *api.Pod {
	return mp.pods[key]
}

func (mp *mirrorPods) getPodByUID(uid types.UID) *api.Pod {
	return mp.upods[uid]
}

func (mp *mirrorPods) getPods() map[string]*api.Pod {
	return mp.pods
}

// Notice that we made the assumption that REMOVE of old mirror pod must comes before ADD of new mirror pod,
// so there won't be pods with different uids but the same full name in upods.
func (mp *mirrorPods) setPod(key string, pod *api.Pod) {
	mp.pods[key] = pod
	mp.upods[pod.UID] = pod
}

func (mp *mirrorPods) removePod(key string) {
	if pod, ok := mp.pods[key]; ok {
		delete(mp.upods, pod.UID)
		delete(mp.pods, key)
	}
}

type mirrorPodManager struct {
	// Protects all the data in mirrorPodManager
	lock sync.RWMutex
	// Static pods indexed by full name
	staticPods map[string]*api.Pod
	// Mirror pods from api server
	mirrorPods mirrorPods

	// The channel receiving mirror pod to be created
	mirrorPodCreationChannel chan *api.Pod
	// The channel receiving mirror pod to be deleted
	mirrorPodDeletionChannel chan *api.Pod

	// apiserver client
	apiserverClient clientset.Interface
}

const syncPeriod = 5 * time.Second

func newMirrorPodManager(apiserverClient clientset.Interface) *mirrorPodManager {
	return &mirrorPodManager{
		staticPods:      make(map[string]*api.Pod),
		mirrorPods:      newMirrorPods(),
		apiserverClient: apiserverClient,
		// There shouldn't be too many mirror pod changes, set buffer size to 100
		mirrorPodCreationChannel: make(chan *api.Pod, 100),
		mirrorPodDeletionChannel: make(chan *api.Pod, 100),
	}
}

func (m *mirrorPodManager) Start() {
	// Don't start mirror pod manager goroutine if we don't have a client. This will
	// happen on the master, where the kubelet is responsible for bootstrapping the
	// pods of the master components.
	if m.apiserverClient == nil {
		glog.Infof("Apiserver client is nil, not starting mirror pod manager.")
		return
	}

	glog.Info("Starting mirror pod manager syncing loop")
	syncTicker := time.Tick(syncPeriod)
	go wait.Forever(func() {
		select {
		case pod := <-m.mirrorPodCreationChannel:
			m.handleMirrorPodCreation(pod)
		case pod := <-m.mirrorPodDeletionChannel:
			m.handleMirrorPodDeletion(pod)
		case <-syncTicker:
			m.handleCleanup()
		}
	}, 0)
}

// Add and Delete a static pod, and the corresponding mirror pod will be created, recreated
// and deleted. These two functions should be only used in pod manager now.
// We'll always create mirror pod when addStaticPod(), the caller should make sure that the static pod
// is really changed. Usually kubelet config will guarantee that.
func (m *mirrorPodManager) addStaticPod(staticPod *api.Pod) {
	m.lock.Lock()
	defer m.lock.Unlock()
	m.staticPods[getKey(staticPod)] = staticPod
	glog.V(2).Infof("Static pod %q is added or updated, create new mirror pod", format.Pod(staticPod))
	m.createMirrorPod(staticPod)
}

func (m *mirrorPodManager) deleteStaticPod(staticPod *api.Pod) {
	m.lock.Lock()
	defer m.lock.Unlock()
	key := getKey(staticPod)
	if _, found := m.staticPods[key]; !found {
		glog.V(4).Infof("The static pod %q is not found", format.Pod(staticPod))
		return
	}
	glog.V(2).Infof("Delete static pod %q, delete the mirror pod", format.Pod(staticPod))
	delete(m.staticPods, key)
	// In fact using static pod directly is also ok, but it's more intuitive making sure that
	// only mirror pod is passing into deleteMirrorPod() and finally handleMirrorPodDeletion()
	m.deleteMirrorPod(newMirrorPod(staticPod))
}

func (m *mirrorPodManager) AddMirrorPod(mirrorPod *api.Pod) {
	m.lock.Lock()
	defer m.lock.Unlock()
	// Invalid mirror pod will be recreated in periodic cleanup if needed
	m.mirrorPods.setPod(getKey(mirrorPod), mirrorPod)
}

func (m *mirrorPodManager) DeleteMirrorPod(mirrorPod *api.Pod) {
	m.lock.Lock()
	defer m.lock.Unlock()
	// Removed mirror pod will be recreated in periodic cleanup if needed
	m.mirrorPods.removePod(getKey(mirrorPod))
}

func (m *mirrorPodManager) GetMirrorPod(name, namespace string) *api.Pod {
	m.lock.RLock()
	defer m.lock.RUnlock()
	return m.mirrorPods.getPod(getKeyByName(name, namespace))
}

func (m *mirrorPodManager) GetMirrorPodByUID(uid types.UID) *api.Pod {
	m.lock.RLock()
	defer m.lock.RUnlock()
	return m.mirrorPods.getPodByUID(uid)
}

func (m *mirrorPodManager) GetMirrorOfStaticPod(staticPod *api.Pod) *api.Pod {
	m.lock.RLock()
	defer m.lock.RUnlock()
	return m.mirrorPods.getPod(getKey(staticPod))
}

func (m *mirrorPodManager) TranslatePodUID(uid types.UID) types.UID {
	m.lock.RLock()
	defer m.lock.RUnlock()
	mirrorPod := m.mirrorPods.getPodByUID(uid)
	if mirrorPod != nil {
		if staticPod, found := m.staticPods[getKey(mirrorPod)]; found {
			return staticPod.UID
		}
	}
	return uid
}

func IsStaticPod(pod *api.Pod) bool {
	source, err := kubetypes.GetPodSource(pod)
	return err == nil && source != kubetypes.ApiserverSource
}

func IsMirrorPod(pod *api.Pod) bool {
	_, ok := pod.Annotations[kubetypes.ConfigMirrorAnnotationKey]
	return ok
}

// Create or recreate mirror pod passed in
func (m *mirrorPodManager) handleMirrorPodCreation(mirrorPod *api.Pod) {
	// There is one case will cause the mirror pod manager to keep recreating mirror pod:
	//  * Recreate a mirror pod m1, because the mirror pod now m0 is invalid.
	//  * Receive the REMOVE message of m0 and clear the corresponding mirror pod cache.
	//  * Can't find corresponding mirror pod in cleanup, recreate a new one m2.
	//  * Receive the ADD and REMOVE message of m1 and clear the corresponding mirror pod cache.
	//  * Can't find corresponding mirror pod in cleanup, recreate a new one m3
	//  * ...
	// So we should make sure the old mirror pod is indeed invalid so as to avoid recreation loop.
	oldMirrorPod, err := m.apiserverClient.Core().Pods(mirrorPod.Namespace).Get(mirrorPod.Name)
	if err == nil {
		m.lock.RLock()
		defer m.lock.RUnlock()
		if isMirrorPodOf(oldMirrorPod, m.staticPods[getKey(mirrorPod)]) {
			return
		}
	}
	if err == nil || !errors.IsNotFound(err) {
		// Delete old mirror pod first
		if !m.handleMirrorPodDeletion(oldMirrorPod) {
			// If the old mirror pod is failed to be deleted, just return, it will be recreated
			// again in next periodic cleanup.
			return
		}
	}
	// Create new mirror pod
	glog.V(4).Infof("Creating a mirror pod %q", format.Pod(mirrorPod))
	_, err = m.apiserverClient.Core().Pods(mirrorPod.Namespace).Create(mirrorPod)
	if err != nil {
		// If the mirror pod is failed to be created, just return, it will be created again
		// in next periodic cleanup
		glog.Errorf("Failed creating mirror pod %q: %v", format.Pod(mirrorPod), err)
		return
	}
}

func (m *mirrorPodManager) handleMirrorPodDeletion(mirrorPod *api.Pod) bool {
	glog.V(4).Infof("Deleting mirror pod %q", format.Pod(mirrorPod))
	namespace := mirrorPod.Namespace
	name := mirrorPod.Name
	if err := m.apiserverClient.Core().Pods(namespace).Delete(name, api.NewDeleteOptions(0)); err != nil {
		// * If the mirror pod is not found on apiserver, report an error and proceeding
		// * If the mirror pod is failed to be deleted for other reasons, the mirror pod
		// will be deleted again in next periodic cleanup.
		if !errors.IsNotFound(err) {
			glog.Errorf("Failed deleting mirror pod %q: %v", format.Pod(mirrorPod), err)
			return false
		}
		glog.V(4).Infof("Failed deleting mirror pod %q: %v", format.Pod(mirrorPod), err)
	}
	return true
}

// Periodic cleanup function.
func (m *mirrorPodManager) handleCleanup() {
	m.lock.Lock()
	defer m.lock.Unlock()
	// Ensure there is a corresponding mirror pod for each static pod.
	for key, staticPod := range m.staticPods {
		mirrorPod := m.mirrorPods.getPod(key)
		if !isMirrorPodOf(mirrorPod, staticPod) {
			m.createMirrorPod(staticPod)
		}
	}
	// Delete orphaned mirror pods
	for key, mirrorPod := range m.mirrorPods.getPods() {
		_, found := m.staticPods[key]
		if !found {
			m.deleteMirrorPod(mirrorPod)
		}
	}
}

// Send a message to manager goroutine to create new mirror pod
func (m *mirrorPodManager) createMirrorPod(staticPod *api.Pod) {
	mirrorPod := newMirrorPod(staticPod)
	select {
	case m.mirrorPodCreationChannel <- mirrorPod:
	default:
		// Not block if channel is full, let the next periodic cleanup handle the creation
	}
}

func (m *mirrorPodManager) deleteMirrorPod(mirrorPod *api.Pod) {
	select {
	case m.mirrorPodDeletionChannel <- mirrorPod:
	default:
		// Not block if channel is full, let the next periodic cleanup handle the deletion
	}
}

// Create mirror pod from a static pod
func newMirrorPod(pod *api.Pod) *api.Pod {
	// Make a copy of the pod.
	mirrorPod := *pod
	mirrorPod.Annotations = make(map[string]string)

	for k, v := range pod.Annotations {
		mirrorPod.Annotations[k] = v
	}
	mirrorPod.Annotations[kubetypes.ConfigMirrorAnnotationKey] = getStaticPodHash(pod)
	return &mirrorPod
}

func getMirrorPodHash(pod *api.Pod) string {
	// The annotation exists for all mirror pods
	return pod.Annotations[kubetypes.ConfigMirrorAnnotationKey]
}

func getStaticPodHash(pod *api.Pod) string {
	// The annotation exists for all static pods
	return pod.Annotations[kubetypes.ConfigHashAnnotationKey]
}

// Ideally we should check whether someone else changed the mirror pod. However, apiserver itself
// also apply defaults on mirror pod, we can not distinguish whether it is changed by apiserver or
// other ones (see #8683). For now, we only check:
// 1) Whether mirrorPod or staticPod is nil.
// 2) Whether the hash annotation of mirror pod is the same with static pod.
// 3) Whether the mirror pod is marked to be deleted.
func isMirrorPodOf(mirrorPod, staticPod *api.Pod) bool {
	if mirrorPod == nil || staticPod == nil {
		return mirrorPod == nil && staticPod == nil
	}
	mirrorHash := getMirrorPodHash(mirrorPod)
	staticHash := getStaticPodHash(staticPod)
	return mirrorHash == staticHash && mirrorPod.DeletionTimestamp == nil
}

func getKey(pod *api.Pod) string {
	return kubecontainer.GetPodFullName(pod)
}

func getKeyByName(name, namespace string) string {
	return kubecontainer.BuildPodFullName(name, namespace)
}
