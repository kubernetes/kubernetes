/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

package dockertools

import (
	"fmt"
	"sync"

	"github.com/golang/groupcache/lru"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
)

type VersionCache struct {
	lock  sync.RWMutex
	cache *lru.Cache
}

// versionInfo caches api version and daemon version.
type versionInfo struct {
	apiVersion kubecontainer.Version
	version    kubecontainer.Version
}

const maxVersionCacheEntries = 1000

func NewVersionCache() *VersionCache {
	return &VersionCache{cache: lru.New(maxVersionCacheEntries)}
}

// Update updates cached versionInfo by using a unique string (e.g. machineInfo) as the key.
func (c *VersionCache) Update(key string, apiVersion kubecontainer.Version, version kubecontainer.Version) {
	c.lock.Lock()
	defer c.lock.Unlock()
	c.cache.Add(key, versionInfo{apiVersion, version})
}

// Get gets cached versionInfo by using a unique string (e.g. machineInfo) as the key.
// It returns apiVersion first and followed by daemon version.
func (c *VersionCache) Get(key string) (kubecontainer.Version, kubecontainer.Version, error) {
	c.lock.RLock()
	defer c.lock.RUnlock()
	value, ok := c.cache.Get(key)
	if !ok {
		return nil, nil, fmt.Errorf("Failed to get version info from cache by key: ", key)
	}
	versions := value.(versionInfo)
	return versions.apiVersion, versions.version, nil
}
