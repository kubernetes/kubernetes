/*
Copyright 2025 The Kubernetes Authors.

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

package etcd3

import (
	"context"
	"sync"

	"go.etcd.io/etcd/api/v3/mvccpb"
	"k8s.io/apiserver/pkg/storage"
)

type keysFunc func(context.Context) ([]string, error)

func newStatsCache(getKeys keysFunc) *statsCache {
	sc := &statsCache{
		getKeys: getKeys,
		keys:    make(map[string]sizeRevision),
	}
	return sc
}

// statsCache efficiently estimates the average object size
// based on the last observed state of individual keys.
// By plugging statsCache into GetList and Watch functions,
// a fairly accurate estimate of object sizes can be maintained
// without additional requests to the underlying storage.
// To handle potential out-of-order or incomplete data,
// it uses a per-key revision to identify the newer state.
// This approach may leak keys if delete events are not observed,
// thus we run a background goroutine to periodically cleanup keys if needed.
type statsCache struct {
	getKeys keysFunc

	lock sync.Mutex
	keys map[string]sizeRevision
}

type sizeRevision struct {
	sizeBytes int64
	revision  int64
}

func (sc *statsCache) Stats(ctx context.Context) (storage.Stats, error) {
	keys, err := sc.getKeys(ctx)
	if err != nil {
		return storage.Stats{}, err
	}
	stats := storage.Stats{
		ObjectCount: int64(len(keys)),
	}
	sc.lock.Lock()
	defer sc.lock.Unlock()
	sc.cleanKeys(keys)
	if len(sc.keys) != 0 {
		stats.EstimatedAverageObjectSizeBytes = sc.keySizes() / int64(len(sc.keys))
	}
	return stats, nil
}

func (sc *statsCache) cleanKeys(keepKeys []string) {
	newKeys := make(map[string]sizeRevision, len(keepKeys))
	for _, key := range keepKeys {
		keySizeRevision, ok := sc.keys[key]
		if !ok {
			continue
		}
		newKeys[key] = keySizeRevision
	}
	sc.keys = newKeys
}

func (sc *statsCache) keySizes() (totalSize int64) {
	for _, sizeRevision := range sc.keys {
		totalSize += sizeRevision.sizeBytes
	}
	return totalSize
}

func (sc *statsCache) Update(kvs []*mvccpb.KeyValue) {
	sc.lock.Lock()
	defer sc.lock.Unlock()
	for _, kv := range kvs {
		sc.updateKey(kv)
	}
}

func (sc *statsCache) UpdateKey(kv *mvccpb.KeyValue) {
	sc.lock.Lock()
	defer sc.lock.Unlock()

	sc.updateKey(kv)
}

func (sc *statsCache) updateKey(kv *mvccpb.KeyValue) {
	key := string(kv.Key)
	keySizeRevision := sc.keys[key]
	if keySizeRevision.revision >= kv.ModRevision {
		return
	}

	sc.keys[key] = sizeRevision{
		sizeBytes: int64(len(kv.Value)),
		revision:  kv.ModRevision,
	}
}

func (sc *statsCache) DeleteKey(kv *mvccpb.KeyValue) {
	sc.lock.Lock()
	defer sc.lock.Unlock()

	key := string(kv.Key)
	keySizeRevision := sc.keys[key]
	if keySizeRevision.revision >= kv.ModRevision {
		return
	}

	delete(sc.keys, key)
}
