/*
Copyright 2016 The Kubernetes Authors.

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

	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"

	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// ReasonCache stores the failure reason of the latest container start
// in a string, keyed by pod UID and container name. The goal is to
// propagate this reason to the container status.
type ReasonCache struct {
	lock  sync.RWMutex
	cache map[types.UID]map[string]*ReasonItem
}

// ReasonItem is the cached item in ReasonCache
type ReasonItem struct {
	Err     error
	Message string
}

// NewReasonCache creates an instance of 'ReasonCache'.
func NewReasonCache() *ReasonCache {
	return &ReasonCache{
		cache: make(map[types.UID]map[string]*ReasonItem),
	}
}

// add adds error reason into the cache
func (c *ReasonCache) add(uid types.UID, name string, reason error, message string) {
	c.lock.Lock()
	defer c.lock.Unlock()

	if c.cache[uid] == nil {
		c.cache[uid] = make(map[string]*ReasonItem)
	}
	c.cache[uid][name] = &ReasonItem{Err: reason, Message: message}
}

// Update updates the reason cache with the SyncPodResult. Only SyncResult with
// StartContainer action will change the cache.
func (c *ReasonCache) Update(uid types.UID, result kubecontainer.PodSyncResult) {
	for _, r := range result.SyncResults {
		if r.Action != kubecontainer.StartContainer {
			continue
		}
		name := r.Target.(string)
		if r.Error != nil {
			c.add(uid, name, r.Error, r.Message)
		} else {
			c.Remove(uid, name)
		}
	}
}

// Remove removes error reason from the cache
func (c *ReasonCache) Remove(uid types.UID, name string) {
	c.lock.Lock()
	defer c.lock.Unlock()

	if podCache, ok := c.cache[uid]; ok {
		delete(podCache, name)
		if len(podCache) == 0 {
			delete(c.cache, uid)
		}
	}
}

// RemovePod removes all error reasons for a specific pod from the cache
func (c *ReasonCache) RemovePod(uid types.UID) {
	c.lock.Lock()
	defer c.lock.Unlock()
	delete(c.cache, uid)
}

// CleanupOrphanedPods removes error reasons for any pod not in the provided active set.
func (c *ReasonCache) CleanupOrphanedPods(activePods sets.Set[types.UID]) {
	c.lock.Lock()
	defer c.lock.Unlock()
	for uid := range c.cache {
		if !activePods.Has(uid) {
			delete(c.cache, uid)
		}
	}
}

// Get gets error reason from the cache. The return values are error reason, error message and
// whether an error reason is found in the cache. If no error reason is found, empty string will
// be returned for error reason and error message.
func (c *ReasonCache) Get(uid types.UID, name string) (*ReasonItem, bool) {
	c.lock.RLock()
	defer c.lock.RUnlock()

	if podCache, ok := c.cache[uid]; ok {
		if info, ok := podCache[name]; ok {
			// Return a copy so the caller can't mutate the cache
			infoCopy := *info
			return &infoCopy, true
		}
	}
	return nil, false
}
