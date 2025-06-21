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
	"sync"

	"go.etcd.io/etcd/api/v3/mvccpb"
)

func newSizeCache() *sizeCache {
	return &sizeCache{
		perKey: make(map[string]sizeRevision),
	}
}

// sizeCache efficiently estimates the average object size based on the last observed state of individual keys.
// By plugging sizeCache into GetList and Watch functions, a fairly accurate estimate of object sizes can be maintained
// without additional requests to the underlying storage.
// To handle potential out-of-order or incomplete data, it uses a per-key revision to identify the newer state.
// This approach may lead to key leakage if delete events are not observed, thus AverageObjectSize function should be called periodically with the current list of keys.
type sizeCache struct {
	lock sync.Mutex

	perKey map[string]sizeRevision
}

type sizeRevision struct {
	sizeBytes int64
	revision  int64
}

// AverageObjectSize returns the average size of objects observed by cache.
// To prevent leakage and staleness, the caller should provide the current list of keys.
// To avoid allocating dedicated slice, kvs argument should be set to results of Range request with clientv3.WithKeysOnly().
func (ss *sizeCache) AverageObjectSize(kvs []*mvccpb.KeyValue) int64 {
	ss.lock.Lock()
	defer ss.lock.Unlock()

	totalSize := ss.sizeKeysAndCleanOthers(kvs)

	if len(ss.perKey) == 0 {
		return 0
	}
	return totalSize / int64(len(ss.perKey))
}

func (ss *sizeCache) sizeKeysAndCleanOthers(keysOnly []*mvccpb.KeyValue) (totalSize int64) {
	newKeys := make(map[string]sizeRevision, len(keysOnly))
	for _, kvs := range keysOnly {
		key := string(kvs.Key)
		keySizeRevision, ok := ss.perKey[key]
		if !ok {
			continue
		}
		newKeys[key] = keySizeRevision
		totalSize += keySizeRevision.sizeBytes
	}
	ss.perKey = newKeys
	return totalSize
}

func (ss *sizeCache) AddOrUpdate(key string, revision, size int64) {
	ss.lock.Lock()
	defer ss.lock.Unlock()

	keySizeRevision := ss.perKey[key]
	if keySizeRevision.revision >= revision {
		return
	}

	ss.perKey[key] = sizeRevision{
		sizeBytes: size,
		revision:  revision,
	}
}

func (ss *sizeCache) Delete(key string, revision int64) {
	ss.lock.Lock()
	defer ss.lock.Unlock()

	keySizeRevision := ss.perKey[key]
	if keySizeRevision.revision >= revision {
		return
	}

	delete(ss.perKey, key)
}
