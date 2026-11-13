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

// StaleRead simulates cache sync lag: every read behaves as if the cache were
// empty and the last-synced RV is "1".
type StaleRead struct{}

func (StaleRead) ApplyCache() bool { return true }

// FaultInjectingIndexer wraps cache.Indexer and intercepts every read path the
// generated listers use, including the namespace index.
type FaultInjectingIndexer struct {
	cache.Indexer
	ctx      context.Context // bounds BlockingFault actions
	registry *FaultRegistry
	name     string // plural resource name, e.g. "pods"
}

// NewFaultInjectingIndexer creates a wrapped cache.Indexer hooked to the registry.
func NewFaultInjectingIndexer(ctx context.Context, realIndexer cache.Indexer, registry *FaultRegistry, name string) cache.Indexer {
	return &FaultInjectingIndexer{
		Indexer:  realIndexer,
		ctx:      ctx,
		registry: registry,
		name:     name,
	}
}

func (i *FaultInjectingIndexer) stale(op, key string) bool {
	return i.registry.ResolveCache(i.ctx, CacheFacts{Cache: i.name, Op: op, Key: key})
}

func (i *FaultInjectingIndexer) GetByKey(key string) (interface{}, bool, error) {
	if i.stale("get", key) {
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
	if i.stale("list", "") {
		return nil
	}
	return i.Indexer.List()
}

func (i *FaultInjectingIndexer) ListKeys() []string {
	if i.stale("list-keys", "") {
		return nil
	}
	return i.Indexer.ListKeys()
}

// Index backs cache.ListAllByNamespace, i.e. every namespaced lister.
func (i *FaultInjectingIndexer) Index(indexName string, obj interface{}) ([]interface{}, error) {
	if i.stale("index", indexName) {
		return nil, nil
	}
	return i.Indexer.Index(indexName, obj)
}

func (i *FaultInjectingIndexer) IndexKeys(indexName, indexedValue string) ([]string, error) {
	if i.stale("index-keys", indexName+"/"+indexedValue) {
		return nil, nil
	}
	return i.Indexer.IndexKeys(indexName, indexedValue)
}

func (i *FaultInjectingIndexer) ListIndexFuncValues(indexName string) []string {
	if i.stale("index-values", indexName) {
		return nil
	}
	return i.Indexer.ListIndexFuncValues(indexName)
}

func (i *FaultInjectingIndexer) ByIndex(indexName, indexedValue string) ([]interface{}, error) {
	if i.stale("by-index", indexName+"/"+indexedValue) {
		return nil, nil
	}
	return i.Indexer.ByIndex(indexName, indexedValue)
}

func (i *FaultInjectingIndexer) LastStoreSyncResourceVersion() string {
	if i.stale("last-sync-rv", "") {
		return "1"
	}
	return i.Indexer.LastStoreSyncResourceVersion()
}
