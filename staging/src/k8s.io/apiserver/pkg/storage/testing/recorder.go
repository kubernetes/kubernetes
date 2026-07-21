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

type KubernetesRecorder struct {
	kubernetes.Interface

	mux         sync.Mutex
	listsPerKey map[string][]RecordedList
}

func NewKubernetesRecorder(client kubernetes.Interface) *KubernetesRecorder {
	return &KubernetesRecorder{
		listsPerKey: make(map[string][]RecordedList),
		Interface:   client,
	}
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
	recorderKey, ok := ctx.Value(RecorderContextKey).(string)
	if !ok {
		return
	}
	r.mux.Lock()
	defer r.mux.Unlock()
	r.listsPerKey[recorderKey] = append(r.listsPerKey[recorderKey], list)
}

func (r *KubernetesRecorder) ListRequestForKey(key string) []RecordedList {
	r.mux.Lock()
	defer r.mux.Unlock()
	return r.listsPerKey[key]
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
