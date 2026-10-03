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

// StaleRead simulates cache sync lag: the lookup behaves as if the object is not
// present yet (GetByKey -> missing, List/ByIndex -> empty, last-synced RV -> "1").
type StaleRead struct{}

func (StaleRead) ApplyCache() bool { return true }

// FaultInjectingIndexer wraps cache.Indexer and intercepts local cache lookups.
type FaultInjectingIndexer struct {
	cache.Indexer
	registry *FaultRegistry
	name     string // plural resource name, e.g. "pods"
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
	if i.registry.ResolveCache(context.Background(), CacheFacts{Cache: i.name, Op: "get", Key: key}) {
		return nil, false, nil
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
	if i.registry.ResolveCache(context.Background(), CacheFacts{Cache: i.name, Op: "list"}) {
		return nil
	}
	return i.Indexer.List()
}

func (i *FaultInjectingIndexer) ByIndex(indexName, indexedValue string) ([]interface{}, error) {
	if i.registry.ResolveCache(context.Background(), CacheFacts{Cache: i.name, Op: "by-index", Key: indexName + "/" + indexedValue}) {
		return nil, nil
	}
	return i.Indexer.ByIndex(indexName, indexedValue)
}

func (i *FaultInjectingIndexer) LastStoreSyncResourceVersion() string {
	if i.registry.ResolveCache(context.Background(), CacheFacts{Cache: i.name, Op: "last-sync-rv"}) {
		return "1"
	}
	return i.Indexer.LastStoreSyncResourceVersion()
}
