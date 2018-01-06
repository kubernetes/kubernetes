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
	"fmt"
	"strconv"
	"sync"
	"time"

	"k8s.io/api/core/v1"
	storageetcd "k8s.io/apiserver/pkg/storage/etcd"
	clientset "k8s.io/client-go/kubernetes"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/pkg/kubelet/util"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"
)

const (
	defaultTTL = time.Minute
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

type GetObjectTTLFunc func() (time.Duration, bool)

type objectKey struct {
	namespace string
	name      string
}

// configMapStoreItems is a single item stored in configMapStore.
type configMapStoreItem struct {
	refCount  int
	configMap *configMapData
}

type configMapData struct {
	sync.Mutex

	configMap      *v1.ConfigMap
	err            error
	lastUpdateTime time.Time
}

// configMapStore is a local cache of configmaps.
type configMapStore struct {
	kubeClient clientset.Interface
	clock      clock.Clock

	lock  sync.Mutex
	items map[objectKey]*configMapStoreItem

	defaultTTL time.Duration
	getTTL     GetObjectTTLFunc
}

func newConfigMapStore(kubeClient clientset.Interface, clock clock.Clock, getTTL GetObjectTTLFunc, ttl time.Duration) *configMapStore {
	return &configMapStore{
		kubeClient: kubeClient,
		clock:      clock,
		items:      make(map[objectKey]*configMapStoreItem),
		defaultTTL: ttl,
		getTTL:     getTTL,
	}
}

func isConfigMapOlder(newConfigMap, oldConfigMap *v1.ConfigMap) bool {
	if newConfigMap == nil || oldConfigMap == nil {
		return false
	}
	newVersion, _ := storageetcd.Versioner.ObjectResourceVersion(newConfigMap)
	oldVersion, _ := storageetcd.Versioner.ObjectResourceVersion(oldConfigMap)
	return newVersion < oldVersion
}

func (s *configMapStore) Add(namespace, name string) {
	key := objectKey{namespace: namespace, name: name}

	// Add is called from RegisterPod, thus it needs to be efficient.
	// Thus Add() is only increasing refCount and generation of a given configmap.
	// Then Get() is responsible for fetching if needed.
	s.lock.Lock()
	defer s.lock.Unlock()
	item, exists := s.items[key]
	if !exists {
		item = &configMapStoreItem{
			refCount:  0,
			configMap: &configMapData{},
		}
		s.items[key] = item
	}

	item.refCount++
	// This will trigger fetch on the next Get() operation.
	item.configMap = nil
}

func (s *configMapStore) Delete(namespace, name string) {
	key := objectKey{namespace: namespace, name: name}

	s.lock.Lock()
	defer s.lock.Unlock()
	if item, ok := s.items[key]; ok {
		item.refCount--
		if item.refCount == 0 {
			delete(s.items, key)
		}
	}
}

func GetObjectTTLFromNodeFunc(getNode func() (*v1.Node, error)) GetObjectTTLFunc {
	return func() (time.Duration, bool) {
		node, err := getNode()
		if err != nil {
			return time.Duration(0), false
		}
		if node != nil && node.Annotations != nil {
			if value, ok := node.Annotations[v1.ObjectTTLAnnotationKey]; ok {
				if intValue, err := strconv.Atoi(value); err == nil {
					return time.Duration(intValue) * time.Second, true
				}
			}
		}
		return time.Duration(0), false
	}
}

func (s *configMapStore) isConfigMapFresh(data *configMapData) bool {
	configMapTTL := s.defaultTTL
	if ttl, ok := s.getTTL(); ok {
		configMapTTL = ttl
	}
	return s.clock.Now().Before(data.lastUpdateTime.Add(configMapTTL))
}

func (s *configMapStore) Get(namespace, name string) (*v1.ConfigMap, error) {
	key := objectKey{namespace: namespace, name: name}

	data := func() *configMapData {
		s.lock.Lock()
		defer s.lock.Unlock()
		item, exists := s.items[key]
		if !exists {
			return nil
		}
		if item.configMap == nil {
			item.configMap = &configMapData{}
		}
		return item.configMap
	}()
	if data == nil {
		return nil, fmt.Errorf("configmap %q/%q not registered", namespace, name)
	}

	// After updating data in configMapStore, lock the data, fetch configMap if
	// needed and return data.
	data.Lock()
	defer data.Unlock()
	if data.err != nil || !s.isConfigMapFresh(data) {
		opts := metav1.GetOptions{}
		if data.configMap != nil && data.err == nil {
			// This is just a periodic refresh of a configmap we successfully fetched previously.
			// In this case, server data from apiserver cache to reduce the load on both
			// etcd and apiserver (the cache is eventually consistent).
			util.FromApiserverCache(&opts)
		}
		configMap, err := s.kubeClient.CoreV1().ConfigMaps(namespace).Get(name, opts)
		if err != nil && !apierrors.IsNotFound(err) && data.configMap == nil && data.err == nil {
			// Couldn't fetch the latest configmap, but there is no cached data to return.
			// Return the fetch result instead.
			return configMap, err
		}
		if (err == nil && !isConfigMapOlder(configMap, data.configMap)) || apierrors.IsNotFound(err) {
			// If the fetch succeeded with a newer version of the configmap, or if the
			// configmap could not be found in the apiserver, update the cached data to
			// reflect the current status.
			data.configMap = configMap
			data.err = err
			data.lastUpdateTime = s.clock.Now()
		}
	}
	return data.configMap, data.err
}

// cachingConfigMapManager keeps a cache of all configmaps necessary for registered pods.
// It implements the following logic:
// - whenever a pod is created or updated, the cached versions of all its configmaps
//   are invalidated
// - every GetConfigMap() call tries to fetch the value from local cache; if it is
//   not there, invalidated or too old, we fetch it from apiserver and refresh the
//   value in cache; otherwise it is just fetched from cache
type cachingConfigMapManager struct {
	configMapStore *configMapStore

	lock           sync.Mutex
	registeredPods map[objectKey]*v1.Pod
}

func NewCachingConfigMapManager(kubeClient clientset.Interface, getTTL GetObjectTTLFunc) Manager {
	csm := &cachingConfigMapManager{
		configMapStore: newConfigMapStore(kubeClient, clock.RealClock{}, getTTL, defaultTTL),
		registeredPods: make(map[objectKey]*v1.Pod),
	}
	return csm
}

func (c *cachingConfigMapManager) GetConfigMap(namespace, name string) (*v1.ConfigMap, error) {
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

func (c *cachingConfigMapManager) RegisterPod(pod *v1.Pod) {
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

func (c *cachingConfigMapManager) UnregisterPod(pod *v1.Pod) {
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
