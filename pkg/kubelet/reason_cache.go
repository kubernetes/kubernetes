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
	"fmt"
	"sync"

	"github.com/golang/groupcache/lru"
	"k8s.io/apimachinery/pkg/types"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

// ReasonCache stores the failure reason of the latest container start
// in a string, keyed by <pod_UID>_<container_name>. The goal is to
// propagate this reason to the container status. This endeavor is
// "best-effort" for two reasons:
//   1. The cache is not persisted.
//   2. We use an LRU cache to avoid extra garbage collection work. This
//      means that some entries may be recycled before a pod has been
//      deleted.
// TODO(random-liu): Use more reliable cache which could collect garbage of failed pod.
// TODO(random-liu): Move reason cache to somewhere better.
type ReasonCache struct {
	lock  sync.Mutex
	cache *lru.Cache
}

// reasonInfo is the cached item in ReasonCache
type reasonInfo struct {
	reason  error
	message string
}

// maxReasonCacheEntries is the cache entry number in lru cache. 1000 is a proper number
// for our 100 pods per node target. If we support more pods per node in the future, we
// may want to increase the number.
const maxReasonCacheEntries = 1000

func NewReasonCache() *ReasonCache {
	return &ReasonCache{cache: lru.New(maxReasonCacheEntries)}
}

func (c *ReasonCache) composeKey(uid types.UID, name string) string {
	return fmt.Sprintf("%s_%s", uid, name)
}

// add adds error reason into the cache
func (c *ReasonCache) add(uid types.UID, name string, reason error, message string) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.Add(c.composeKey(uid, name), reasonInfo{reason, message})
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
	c.cache.Remove(c.composeKey(uid, name))
}

// Get gets error reason from the cache. The return values are error reason, error message and
// whether an error reason is found in the cache. If no error reason is found, empty string will
// be returned for error reason and error message.
func (c *ReasonCache) Get(uid types.UID, name string) (error, string, bool) {
	c.lock.Lock()
	defer c.lock.Unlock()
	value, ok := c.cache.Get(c.composeKey(uid, name))
	if !ok {
		return nil, "", ok
	}
	info := value.(reasonInfo)
	return info.reason, info.message, ok
}
