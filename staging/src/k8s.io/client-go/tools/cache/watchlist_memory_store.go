/*
Copyright 2026 The Kubernetes Authors.

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

package cache

import (
	"k8s.io/apimachinery/pkg/api/meta"
)

// watchListMemoryOptimizedStore reuses cached objects when the incoming watch-list
// event matches an existing cache object by key and resourceVersion.
type watchListMemoryOptimizedStore struct {
	Store       // embedded delegate
	clientStore Store
	keyFunc     KeyFunc
}

func newWatchListMemoryOptimizedStore(delegate Store, clientStore Store, keyFunc KeyFunc) Store {
	// keyFunc should match the delegate store's keying function.
	if clientStore == nil {
		return delegate
	}
	return &watchListMemoryOptimizedStore{
		Store:       delegate,
		clientStore: clientStore,
		keyFunc:     keyFunc,
	}
}

func (s *watchListMemoryOptimizedStore) Add(obj interface{}) error {
	return s.Store.Add(s.maybeReuseObject(obj))
}

func (s *watchListMemoryOptimizedStore) Update(obj interface{}) error {
	return s.Store.Update(s.maybeReuseObject(obj))
}

func (s *watchListMemoryOptimizedStore) maybeReuseObject(obj interface{}) interface{} {
	key, err := s.keyFunc(obj)
	if err != nil {
		return obj
	}
	cached, exists, err := s.clientStore.GetByKey(key)
	if err != nil || !exists {
		return obj
	}
	if sameResourceVersion(cached, obj) {
		return cached
	}
	return obj
}

func sameResourceVersion(a, b interface{}) bool {
	aMeta, err := meta.Accessor(a)
	if err != nil {
		return false
	}
	bMeta, err := meta.Accessor(b)
	if err != nil {
		return false
	}
	return aMeta.GetResourceVersion() == bMeta.GetResourceVersion()
}
