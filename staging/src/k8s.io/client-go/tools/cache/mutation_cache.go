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

package cache

import (
	"fmt"
	"strconv"
	"sync"
	"time"

	"k8s.io/klog/v2"

	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/runtime"
	utilcache "k8s.io/apimachinery/pkg/util/cache"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
)

// MutationCache is able to take the result of update operations and stores them in an LRU
// that can be used to provide a more current view of a requested object.  It requires interpreting
// resourceVersions for comparisons.
// Implementations must be thread-safe.
// TODO find a way to layer this into an informer/lister
type MutationCache interface {
	GetByKey(key string) (interface{}, bool, error)
	ByIndex(indexName, indexKey string) ([]interface{}, error)
	Mutation(interface{})
}

// ResourceVersionComparator is able to compare object versions.
type ResourceVersionComparator interface {
	CompareResourceVersion(lhs, rhs runtime.Object) int
}

// NewIntegerResourceVersionMutationCache returns a MutationCache that understands how to
// deal with objects that have a resource version that:
//
//   - is an integer
//   - increases when updated
//   - is comparable across the same resource in a namespace
//
// Most backends will have these semantics. Indexer may be nil. ttl controls how long an item
// remains in the mutation cache before it is removed.
//
// If includeAdds is true, objects in the mutation cache will be returned even if they don't exist
// in the underlying store. This is only safe if your use of the cache can handle mutation entries
// remaining in the cache for up to ttl when mutations and deletes occur very closely in time.
func NewIntegerResourceVersionMutationCache(backingCache Store, indexer Indexer, ttl time.Duration, includeAdds bool) MutationCache {
	return &mutationCache{
		backingCache:  backingCache,
		indexer:       indexer,
		mutationCache: utilcache.NewLRUExpireCache(100),
		comparator:    etcdObjectVersioner{},
		ttl:           ttl,
		includeAdds:   includeAdds,
	}
}

// mutationCache doesn't guarantee that it returns values added via Mutation since they can page out and
// since you can't distinguish between, "didn't observe create" and "was deleted after create",
// if the key is missing from the backing cache, we always return it as missing
type mutationCache struct {
	lock          sync.Mutex
	backingCache  Store
	indexer       Indexer
	mutationCache *utilcache.LRUExpireCache
	includeAdds   bool
	ttl           time.Duration

	comparator ResourceVersionComparator
}

// GetByKey is never guaranteed to return back the value set in Mutation.  It could be paged out, it could
// be older than another copy, the backingCache may be more recent or, you might have written twice into the same key.
// You get a value that was valid at some snapshot of time and will always return the newer of backingCache and mutationCache.
func (c *mutationCache) GetByKey(key string) (interface{}, bool, error) {
	c.lock.Lock()
	defer c.lock.Unlock()

	obj, exists, err := c.backingCache.GetByKey(key)
	if err != nil {
		return nil, false, err
	}
	if !exists {
		if !c.includeAdds {
			// we can't distinguish between, "didn't observe create" and "was deleted after create", so
			// if the key is missing, we always return it as missing
			return nil, false, nil
		}
		obj, exists = c.mutationCache.Get(key)
		if !exists {
			return nil, false, nil
		}
	}
	objRuntime, ok := obj.(runtime.Object)
	if !ok {
		return obj, true, nil
	}
	return c.newerObject(key, objRuntime), true, nil
}

// ByIndex returns the newer objects that match the provided index and indexer key.
// Will return an error if no indexer was provided.
func (c *mutationCache) ByIndex(name string, indexKey string) ([]interface{}, error) {
	c.lock.Lock()
	defer c.lock.Unlock()
	if c.indexer == nil {
		return nil, fmt.Errorf("no indexer has been provided to the mutation cache")
	}
	keys, err := c.indexer.IndexKeys(name, indexKey)
	if err != nil {
		return nil, err
	}
	var items []interface{}
	keySet := sets.NewString()
	for _, key := range keys {
		keySet.Insert(key)
		obj, exists, err := c.indexer.GetByKey(key)
		if err != nil {
			return nil, err
		}
		if !exists {
			continue
		}
		if objRuntime, ok := obj.(runtime.Object); ok {
			items = append(items, c.newerObject(key, objRuntime))
		} else {
			items = append(items, obj)
		}
	}

	if c.includeAdds {
		fn := c.indexer.GetIndexers()[name]
		// Keys() is returned oldest to newest, so full traversal does not alter the LRU behavior
		for _, key := range c.mutationCache.Keys() {
			updated, ok := c.mutationCache.Get(key)
			if !ok {
				continue
			}
			if keySet.Has(key.(string)) {
				continue
			}
			elements, err := fn(updated)
			if err != nil {
				klog.V(4).Infof("Unable to calculate an index entry for mutation cache entry %s: %v", key, err)
				continue
			}
			for _, inIndex := range elements {
				if inIndex != indexKey {
					continue
				}
				items = append(items, updated)
				break
			}
		}
	}

	return items, nil
}

// newerObject checks the mutation cache for a newer object and returns one if found. If the
// mutated object is older than the backing object, it is removed from the  Must be
// called while the lock is held.
func (c *mutationCache) newerObject(key string, backing runtime.Object) runtime.Object {
	mutatedObj, exists := c.mutationCache.Get(key)
	if !exists {
		return backing
	}
	mutatedObjRuntime, ok := mutatedObj.(runtime.Object)
	if !ok {
		return backing
	}
	if c.comparator.CompareResourceVersion(backing, mutatedObjRuntime) >= 0 {
		c.mutationCache.Remove(key)
		return backing
	}
	return mutatedObjRuntime
}

// Mutation adds a change to the cache that can be returned in GetByKey if it is newer than the backingCache
// copy.  If you call Mutation twice with the same object on different threads, one will win, but its not defined
// which one.  This doesn't affect correctness, since the GetByKey guaranteed of "later of these two caches" is
// preserved, but you may not get the version of the object you want.  The object you get is only guaranteed to
// "one that was valid at some point in time", not "the one that I want".
func (c *mutationCache) Mutation(obj interface{}) {
	c.lock.Lock()
	defer c.lock.Unlock()

	key, err := DeletionHandlingMetaNamespaceKeyFunc(obj)
	if err != nil {
		// this is a "nice to have", so failures shouldn't do anything weird
		utilruntime.HandleError(err)
		return
	}

	if objRuntime, ok := obj.(runtime.Object); ok {
		if mutatedObj, exists := c.mutationCache.Get(key); exists {
			if mutatedObjRuntime, ok := mutatedObj.(runtime.Object); ok {
				if c.comparator.CompareResourceVersion(objRuntime, mutatedObjRuntime) < 0 {
					return
				}
			}
		}
	}
	c.mutationCache.Add(key, obj, c.ttl)
}

// etcdObjectVersioner implements versioning and extracting etcd node information
// for objects that have an embedded ObjectMeta or ListMeta field.
type etcdObjectVersioner struct{}

// ObjectResourceVersion implements Versioner
func (a etcdObjectVersioner) ObjectResourceVersion(obj runtime.Object) (uint64, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return 0, err
	}
	version := accessor.GetResourceVersion()
	if len(version) == 0 {
		return 0, nil
	}
	return strconv.ParseUint(version, 10, 64)
}

// CompareResourceVersion compares etcd resource versions.  Outside this API they are all strings,
// but etcd resource versions are special, they're actually ints, so we can easily compare them.
func (a etcdObjectVersioner) CompareResourceVersion(lhs, rhs runtime.Object) int {
	lhsVersion, err := a.ObjectResourceVersion(lhs)
	if err != nil {
		// coder error
		panic(err)
	}
	rhsVersion, err := a.ObjectResourceVersion(rhs)
	if err != nil {
		// coder error
		panic(err)
	}

	if lhsVersion == rhsVersion {
		return 0
	}
	if lhsVersion < rhsVersion {
		return -1
	}

	return 1
}
