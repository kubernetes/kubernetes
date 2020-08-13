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

package container

// TestRuntimeCache embeds runtimeCache with some additional methods for testing.
// It must be declared in the container package to have visibility to runtimeCache.
// It cannot be in a "..._test.go" file in order for runtime_cache_test.go to have cross-package visibility to it.
// (cross-package declarations in test files cannot be used from dot imports if this package is vendored)
type TestRuntimeCache struct {
	runtimeCache
}

// UpdateCacheWithLock updates the cache with the lock.
func (r *TestRuntimeCache) UpdateCacheWithLock() error {
	r.Lock()
	defer r.Unlock()
	return r.updateCache()
}

// GetCachedPods returns the cached pods.
func (r *TestRuntimeCache) GetCachedPods() []*Pod {
	r.Lock()
	defer r.Unlock()
	return r.pods
}

// NewTestRuntimeCache creates a new instance of TestRuntimeCache.
func NewTestRuntimeCache(getter podsGetter) *TestRuntimeCache {
	return &TestRuntimeCache{
		runtimeCache: runtimeCache{
			getter: getter,
		},
	}
}
