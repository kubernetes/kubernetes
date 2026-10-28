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

package testing

import (
	"context"
	"sync"
	"sync/atomic"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/kubernetes"
)

type KVRecorder struct {
	clientv3.KV

	reads       uint64
	streamReads uint64
	lists       *KubernetesRecorder
}

func NewKVRecorder(kv clientv3.KV, lists *KubernetesRecorder) *KVRecorder {
	return &KVRecorder{KV: kv, lists: lists}
}

func (r *KVRecorder) Get(ctx context.Context, key string, opts ...clientv3.OpOption) (*clientv3.GetResponse, error) {
	atomic.AddUint64(&r.reads, 1)
	return r.KV.Get(ctx, key, opts...)
}

func (r *KVRecorder) GetReadsAndReset() uint64 {
	return atomic.SwapUint64(&r.reads, 0)
}

func (r *KVRecorder) GetStream(ctx context.Context, key string, opts ...clientv3.OpOption) (clientv3.GetStreamChan, error) {
	atomic.AddUint64(&r.streamReads, 1)
	if r.lists != nil {
		op := clientv3.OpGet(key, opts...)
		r.lists.record(ctx, RecordedList{Key: key, Revision: op.Rev(), Limit: op.Limit()})
	}
	return r.KV.GetStream(ctx, key, opts...)
}

func (r *KVRecorder) GetStreamReadsAndReset() uint64 {
	return atomic.SwapUint64(&r.streamReads, 0)
}

type KubernetesRequestStats struct {
	Lists     []RecordedList
	Gets      uint64
	Writes    uint64
	Conflicts uint64
}

type KubernetesRecorder struct {
	kubernetes.Interface

	mux         sync.Mutex
	statsPerKey map[string]KubernetesRequestStats
}

func NewKubernetesRecorder(client kubernetes.Interface) *KubernetesRecorder {
	return &KubernetesRecorder{
		statsPerKey: make(map[string]KubernetesRequestStats),
		Interface:   client,
	}
}

func (r *KubernetesRecorder) Get(ctx context.Context, key string, opts kubernetes.GetOptions) (kubernetes.GetResponse, error) {
	r.recordStats(ctx, KubernetesRequestStats{Gets: 1})
	return r.Interface.Get(ctx, key, opts)
}

func (r *KubernetesRecorder) OptimisticPut(ctx context.Context, key string, value []byte, expectedRevision int64, opts kubernetes.PutOptions) (kubernetes.PutResponse, error) {
	resp, err := r.Interface.OptimisticPut(ctx, key, value, expectedRevision, opts)
	stats := KubernetesRequestStats{Writes: 1}
	if err == nil && !resp.Succeeded {
		stats.Conflicts = 1
	}
	r.recordStats(ctx, stats)
	return resp, err
}

func (r *KubernetesRecorder) OptimisticDelete(ctx context.Context, key string, expectedRevision int64, opts kubernetes.DeleteOptions) (kubernetes.DeleteResponse, error) {
	resp, err := r.Interface.OptimisticDelete(ctx, key, expectedRevision, opts)
	stats := KubernetesRequestStats{Writes: 1}
	if err == nil && !resp.Succeeded {
		stats.Conflicts = 1
	}
	r.recordStats(ctx, stats)
	return resp, err
}

func (r *KubernetesRecorder) recordStats(ctx context.Context, delta KubernetesRequestStats) {
	recorderKey, ok := ctx.Value(RecorderContextKey).(string)
	if !ok {
		return
	}
	r.mux.Lock()
	defer r.mux.Unlock()
	s := r.statsPerKey[recorderKey]
	s.Gets += delta.Gets
	s.Writes += delta.Writes
	s.Conflicts += delta.Conflicts
	s.Lists = append(s.Lists, delta.Lists...)
	r.statsPerKey[recorderKey] = s
}

func (r *KubernetesRecorder) StatsForKey(key string) KubernetesRequestStats {
	r.mux.Lock()
	defer r.mux.Unlock()
	return r.statsPerKey[key]
}

func (r *KubernetesRecorder) List(ctx context.Context, key string, opts kubernetes.ListOptions) (kubernetes.ListResponse, error) {
	// Continue, when set, is where the range actually starts (see kubernetes.Client.List), so fold it into Key.
	rangeStart := key
	if opts.Continue != "" {
		rangeStart = opts.Continue
	}
	r.record(ctx, RecordedList{Key: rangeStart, Revision: opts.Revision, Limit: opts.Limit})
	return r.Interface.List(ctx, key, opts)
}

func (r *KubernetesRecorder) record(ctx context.Context, list RecordedList) {
	r.recordStats(ctx, KubernetesRequestStats{Lists: []RecordedList{list}})
}

func (r *KubernetesRecorder) ListRequestForKey(key string) []RecordedList {
	return r.StatsForKey(key).Lists
}

// RecordedList is a list request captured by the recorder. Paged and streamed
// reads both fold their resume point into Key.
type RecordedList struct {
	Key      string
	Revision int64
	Limit    int64
}

var RecorderContextKey recorderKeyType

type recorderKeyType struct{}
