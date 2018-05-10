/*
Copyright 2017 The Kubernetes Authors.

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

package configmap

import (
	"sync"

	"k8s.io/api/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
)

type Manager interface {
	// Get configmap by configmap namespace and name.
	GetConfigMap(namespace, name string) (*v1.ConfigMap, error)

	// WARNING: Register/UnregisterPod functions should be efficient,
	// i.e. should not block on network operations.

	// RegisterPod registers all configmaps from a given pod.
	RegisterPod(pod *v1.Pod)

	// UnregisterPod unregisters configmaps from a given pod that are not
	// used by any other registered pod.
	UnregisterPod(pod *v1.Pod)
}

type objectKey struct {
	namespace string
	name      string
}

// simpleConfigMapManager implements ConfigMap Manager interface with
// simple operations to apiserver.
type simpleConfigMapManager struct {
	kubeClient clientset.Interface
}

func NewSimpleConfigMapManager(kubeClient clientset.Interface) Manager {
	return &simpleConfigMapManager{kubeClient: kubeClient}
}

func (s *simpleConfigMapManager) GetConfigMap(namespace, name string) (*v1.ConfigMap, error) {
	return s.kubeClient.CoreV1().ConfigMaps(namespace).Get(name, metav1.GetOptions{})
}

func (s *simpleConfigMapManager) RegisterPod(pod *v1.Pod) {
}

func (s *simpleConfigMapManager) UnregisterPod(pod *v1.Pod) {
}

// store is the interface for a configmap cache that
// can be used be cacheBasedConfigMapManager.
type store interface {
	// Add a configmap to the store.
	// Note that multiple additions to the store has to be allowed
	// in the implementations and effectively treated as refcounted.
	Add(namespace, name string)
	// Delete a configmap from the store.
	// Note that configmap should be deleted only when there was a
	// corresponding Delete call for each of Add calls (effectively
	// when refcount was reduced to zero).
	Delete(namespace, name string)
	// get a configmap from a store.
	Get(namespace, name string) (*v1.ConfigMap, error)
}

// cachingBasedConfigMapManager keeps a store with configmaps necessary
// for registered pods. Different implementations of the store
// may result in different semantics for freshness of configmaps
// (e.g. ttl-based implementation vs watch-based implementation).
type cacheBasedConfigMapManager struct {
	configMapStore store

	lock           sync.Mutex
	registeredPods map[objectKey]*v1.Pod
}

func newCacheBasedConfigMapManager(configMapStore store) Manager {
	csm := &cacheBasedConfigMapManager{
		configMapStore: configMapStore,
		registeredPods: make(map[objectKey]*v1.Pod),
	}
	return csm
}

func (c *cacheBasedConfigMapManager) GetConfigMap(namespace, name string) (*v1.ConfigMap, error) {
	return c.configMapStore.Get(namespace, name)
}

func getConfigMapNames(pod *v1.Pod) sets.String {
	result := sets.NewString()
	podutil.VisitPodConfigmapNames(pod, func(name string) bool {
		result.Insert(name)
		return true
	})
	return result
}

func (c *cacheBasedConfigMapManager) RegisterPod(pod *v1.Pod) {
	names := getConfigMapNames(pod)
	c.lock.Lock()
	defer c.lock.Unlock()
	for name := range names {
		c.configMapStore.Add(pod.Namespace, name)
	}
	var prev *v1.Pod
	key := objectKey{namespace: pod.Namespace, name: pod.Name}
	prev = c.registeredPods[key]
	c.registeredPods[key] = pod
	if prev != nil {
		for name := range getConfigMapNames(prev) {
			// On an update, the .Add() call above will have re-incremented the
			// ref count of any existing items, so any configmaps that are in both
			// names and prev need to have their ref counts decremented. Any that
			// are only in prev need to be completely removed. This unconditional
			// call takes care of both cases.
			c.configMapStore.Delete(prev.Namespace, name)
		}
	}
}

func (c *cacheBasedConfigMapManager) UnregisterPod(pod *v1.Pod) {
	var prev *v1.Pod
	key := objectKey{namespace: pod.Namespace, name: pod.Name}
	c.lock.Lock()
	defer c.lock.Unlock()
	prev = c.registeredPods[key]
	delete(c.registeredPods, key)
	if prev != nil {
		for name := range getConfigMapNames(prev) {
			c.configMapStore.Delete(prev.Namespace, name)
		}
	}
}
