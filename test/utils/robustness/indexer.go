/*
Copyright The Kubernetes Authors.

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

package robustness

import (
	"context"

	"k8s.io/client-go/tools/cache"
)

// Cache faults. Each implements CacheFault and is valid only at cache.* points.

// StaleRead simulates cache sync lag: the lookup behaves as if the object is not
// present yet (GetByKey -> missing, List/ByIndex -> empty, last-synced RV -> "1").
type StaleRead struct{}

func (StaleRead) ApplyCache() CacheVerdict { return CacheVerdict{Kind: CacheStaleRead} }

// FaultInjectingIndexer wraps cache.Indexer and intercepts local cache lookups.
type FaultInjectingIndexer struct {
	cache.Indexer
	registry *FaultRegistry
	name     string // Optional cache identifier (e.g. "pod-informer")
}

// NewFaultInjectingIndexer creates a wrapped cache.Indexer hooked to the registry.
func NewFaultInjectingIndexer(realIndexer cache.Indexer, registry *FaultRegistry, name string) cache.Indexer {
	return &FaultInjectingIndexer{
		Indexer:  realIndexer,
		registry: registry,
		name:     name,
	}
}

func (i *FaultInjectingIndexer) GetByKey(key string) (interface{}, bool, error) {
	v := i.registry.ResolveCache(context.Background(), CacheFacts{Cache: i.name, Op: "get", Key: key})
	if v.Kind == CacheStaleRead {
		return nil, false, nil // simulate sync lag: object appears missing
	}
	return i.Indexer.GetByKey(key)
}

func (i *FaultInjectingIndexer) Get(obj interface{}) (interface{}, bool, error) {
	key, err := cache.MetaNamespaceKeyFunc(obj)
	if err != nil {
		return nil, false, err
	}
	return i.GetByKey(key)
}

func (i *FaultInjectingIndexer) List() []interface{} {
	v := i.registry.ResolveCache(context.Background(), CacheFacts{Cache: i.name, Op: "list"})
	if v.Kind == CacheStaleRead {
		return nil // empty list simulates stale cache lag
	}
	return i.Indexer.List()
}

func (i *FaultInjectingIndexer) ByIndex(indexName, indexedValue string) ([]interface{}, error) {
	v := i.registry.ResolveCache(context.Background(), CacheFacts{Cache: i.name, Op: "by-index", Key: indexName + "/" + indexedValue})
	if v.Kind == CacheStaleRead {
		return nil, nil // empty result simulates a stale index
	}
	return i.Indexer.ByIndex(indexName, indexedValue)
}

func (i *FaultInjectingIndexer) LastStoreSyncResourceVersion() string {
	v := i.registry.ResolveCache(context.Background(), CacheFacts{Cache: i.name, Op: "last-sync-rv"})
	if v.Kind == CacheStaleRead {
		return "1" // a valid but very old resource version
	}
	return i.Indexer.LastStoreSyncResourceVersion()
}
