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
	"errors"
	"strings"
	"sync"

	"sync/atomic"
	"time"

	"go.etcd.io/etcd/api/v3/mvccpb"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/klog/v2"
)

const sizerRefreshInterval = time.Minute

func newStatsCache(prefix string, getKeys storage.KeysFunc) *statsCache {
	if prefix[len(prefix)-1] != '/' {
		prefix += "/"
	}
	sc := &statsCache{
		prefix:  prefix,
		getKeys: getKeys,
		stop:    make(chan struct{}),
		keys:    make(map[string]sizeRevision),
	}
	sc.wg.Add(1)
	go func() {
		defer sc.wg.Done()
		sc.run()
	}()
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
	prefix         string
	stop           chan struct{}
	wg             sync.WaitGroup
	lastKeyCleanup atomic.Pointer[time.Time]

	getKeysLock sync.Mutex
	getKeys     storage.KeysFunc

	keysLock sync.Mutex
	keys     map[string]sizeRevision
}

type sizeRevision struct {
	sizeBytes int64
	revision  int64
}

var errStatsDisabled = errors.New("key size stats disabled")

func (sc *statsCache) Stats(ctx context.Context) (storage.Stats, error) {
	keys, err := sc.GetKeys(ctx)
	if err != nil {
		return storage.Stats{}, err
	}
	stats := storage.Stats{
		ObjectCount: int64(len(keys)),
	}
	sc.keysLock.Lock()
	defer sc.keysLock.Unlock()
	sc.cleanKeys(keys)
	if len(sc.keys) != 0 {
		stats.EstimatedAverageObjectSizeBytes = sc.keySizes() / int64(len(sc.keys))
	}
	return stats, nil
}

func (sc *statsCache) GetKeys(ctx context.Context) ([]string, error) {
	sc.getKeysLock.Lock()
	getKeys := sc.getKeys
	sc.getKeysLock.Unlock()

	if getKeys == nil {
		return nil, errStatsDisabled
	}
	// Don't execute getKeys under lock.
	return getKeys(ctx)
}

func (sc *statsCache) SetKeysFunc(keys storage.KeysFunc) {
	sc.getKeysLock.Lock()
	defer sc.getKeysLock.Unlock()
	sc.getKeys = keys
}

func (sc *statsCache) Close() {
	close(sc.stop)
	sc.wg.Wait()
}

func (sc *statsCache) run() {
	jitter := 0.5 // Period between [interval, interval * (1.0 + jitter)]
	sliding := true
	// wait.JitterUntilWithContext starts work immediately, so wait first.
	select {
	case <-time.After(wait.Jitter(sizerRefreshInterval, jitter)):
	case <-sc.stop:
	}
	wait.JitterUntilWithContext(wait.ContextForChannel(sc.stop), sc.cleanKeysIfNeeded, sizerRefreshInterval, jitter, sliding)
}

func (sc *statsCache) cleanKeysIfNeeded(ctx context.Context) {
	lastKeyCleanup := sc.lastKeyCleanup.Load()
	if lastKeyCleanup != nil && time.Since(*lastKeyCleanup) < sizerRefreshInterval {
		return
	}
	keys, err := sc.GetKeys(ctx)
	if err != nil {
		if !errors.Is(err, errStatsDisabled) {
			klog.InfoS("Error getting keys", "err", err)
		}
		return
	}
	sc.keysLock.Lock()
	defer sc.keysLock.Unlock()
	sc.cleanKeys(keys)
}

func (sc *statsCache) cleanKeys(keepKeys []string) {
	newKeys := make(map[string]sizeRevision, len(keepKeys))
	for _, key := range keepKeys {
		// Handle cacher keys not having prefix.
		if !strings.HasPrefix(key, sc.prefix) {
			startIndex := 0
			if key[0] == '/' {
				startIndex = 1
			}
			key = sc.prefix + key[startIndex:]
		}
		keySizeRevision, ok := sc.keys[key]
		if !ok {
			continue
		}
		newKeys[key] = keySizeRevision
	}
	sc.keys = newKeys
	now := time.Now()
	sc.lastKeyCleanup.Store(&now)
}

func (sc *statsCache) keySizes() (totalSize int64) {
	for _, sizeRevision := range sc.keys {
		totalSize += sizeRevision.sizeBytes
	}
	return totalSize
}

func (sc *statsCache) Update(kvs []*mvccpb.KeyValue) {
	sc.keysLock.Lock()
	defer sc.keysLock.Unlock()
	for _, kv := range kvs {
		sc.updateKey(kv)
	}
}

func (sc *statsCache) UpdateKey(kv *mvccpb.KeyValue) {
	sc.keysLock.Lock()
	defer sc.keysLock.Unlock()

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
	sc.keysLock.Lock()
	defer sc.keysLock.Unlock()

	key := string(kv.Key)
	keySizeRevision := sc.keys[key]
	if keySizeRevision.revision >= kv.ModRevision {
		return
	}

	delete(sc.keys, key)
}
