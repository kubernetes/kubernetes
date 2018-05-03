/*
Copyright 2018 The Kubernetes Authors.

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

package vsphere

import (
	"fmt"
	"sync"

	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
)

const (
	// Since node name can be of length 253 and uuid is of length 36
	// given 1MB data limit of configmap, we can allow upto 3600 entries.
	maxDataEntries = 3600
)

type nodeUUID struct {
	uuid    string
	deleted bool
}

type nodeUUIDCache struct {
	nodeUUIDMap   map[string]*nodeUUID
	kubeClient    clientset.Interface
	configMapName string
	namespace     string
	cacheLock     sync.RWMutex
}

func newNodeUUIDCache(configMapName, namespace string) *nodeUUIDCache {
	cache := &nodeUUIDCache{configMapName: configMapName, namespace: namespace}
	cache.nodeUUIDMap = map[string]*nodeUUID{}
	return cache
}

func (cache *nodeUUIDCache) addNode(nodeName, uuid string) error {
	cache.cacheLock.Lock()
	defer cache.cacheLock.Unlock()
	configMap, err := cache.loadFromConfigMap()
	if err != nil {
		return fmt.Errorf("error loading uuid in configmap %s with %v", cache.configMapName, err)
	}
	cache.nodeUUIDMap[nodeName] = &nodeUUID{uuid: uuid}
	configMap.Data = cache.getUUIData()
	err = cache.updateConfigMap(configMap)
	if err != nil {
		return fmt.Errorf("error updating uuid in configmap %s with %v", cache.configMapName, err)
	}
	return nil
}

func (cache *nodeUUIDCache) markDelete(nodeName, uuid string) error {
	cache.cacheLock.Lock()
	defer cache.cacheLock.Unlock()
	_, err := cache.loadFromConfigMap()
	if err != nil {
		return fmt.Errorf("error loading uuid in configmap %s with %v", cache.configMapName, err)
	}
	cache.nodeUUIDMap[nodeName] = &nodeUUID{uuid: uuid, deleted: true}
	return nil
}

func (cache *nodeUUIDCache) getVMUUID(nodeName string) (string, error) {
	cache.cacheLock.Lock()
	defer cache.cacheLock.Unlock()
	if vmUUID, ok := cache.nodeUUIDMap[nodeName]; ok {
		return vmUUID.uuid, nil
	}

	_, err := cache.loadFromConfigMap()
	if err != nil {
		return "", fmt.Errorf("error loading uuid in configmap %s with %v", cache.configMapName, err)
	}
	if vmUUID, ok := cache.nodeUUIDMap[nodeName]; ok {
		return vmUUID.uuid, nil
	}
	return "", fmt.Errorf("error finding uuid for %s", nodeName)
}

func (cache *nodeUUIDCache) loadFromConfigMap() (*v1.ConfigMap, error) {
	if cache.kubeClient == nil {
		return nil, fmt.Errorf("kubeClient is not initialized yet")
	}
	var configMap *v1.ConfigMap
	var err error
	configMap, err = cache.kubeClient.CoreV1().ConfigMaps(cache.namespace).Get(cache.configMapName, metav1.GetOptions{})

	if err != nil {
		if !apierrors.IsNotFound(err) {
			return nil, fmt.Errorf("error loading vsphere-uuid configmap %s with %v", cache.configMapName, err)
		}
		initializeErr := cache.initializeConfigMap()
		if initializeErr != nil {
			return nil, initializeErr
		}
		configMap, err = cache.kubeClient.CoreV1().ConfigMaps(cache.namespace).Get(cache.configMapName, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("error loading vsphere-uuid configmap %s with %v", cache.configMapName, err)
		}

	}
	configMapData := configMap.Data
	for k, v := range configMapData {
		if _, ok := cache.nodeUUIDMap[k]; !ok {
			// add all new entries loaded from configmap as not deleted
			cache.nodeUUIDMap[k] = &nodeUUID{uuid: v}
		}
	}
	return configMap, nil
}

func (cache *nodeUUIDCache) initializeConfigMap() error {
	if cache.kubeClient == nil {
		return fmt.Errorf("kubeClient is not initialized yet")
	}
	_, err := cache.kubeClient.CoreV1().ConfigMaps(cache.namespace).Create(&v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: cache.configMapName,
		},
		Data: map[string]string{},
	})
	if err != nil {
		return fmt.Errorf("failed to initialize vsphere-uuid configmap %s with %v", cache.configMapName, err)
	}
	return nil
}

func (cache *nodeUUIDCache) updateConfigMap(configMap *v1.ConfigMap) error {
	_, err := cache.kubeClient.CoreV1().ConfigMaps(cache.namespace).Update(configMap)
	if err != nil {
		glog.Errorf("error while updating vm uuid configmap. err: %+v", err)
		return err
	}
	return nil
}

func (cache *nodeUUIDCache) getUUIData() map[string]string {
	// if we have too many entries we should prune not deleted node entries
	if len(cache.nodeUUIDMap) > maxDataEntries {
		cache.pruneCache()
	}
	result := make(map[string]string, len(cache.nodeUUIDMap))
	for k, v := range cache.nodeUUIDMap {
		// Even after pruning if cache still has more than maxDataEntries
		// then we just pick first N values. Not ideal - we are limited
		// by size limit of configmaps.
		if len(result) > maxDataEntries {
			return result
		}
		result[k] = v.uuid
	}
	return result
}

func (cache *nodeUUIDCache) pruneCache() {
	for k, v := range cache.nodeUUIDMap {
		// if node is not deleted then we can still fetch node's UUID from node object itself
		if !v.deleted {
			delete(cache.nodeUUIDMap, k)
		}
		if len(cache.nodeUUIDMap) < maxDataEntries {
			break
		}
	}
}
