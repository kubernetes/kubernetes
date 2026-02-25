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

	reads uint64
}

func NewKVRecorder(kv clientv3.KV) *KVRecorder {
	return &KVRecorder{KV: kv}
}

func (r *KVRecorder) Get(ctx context.Context, key string, opts ...clientv3.OpOption) (*clientv3.GetResponse, error) {
	atomic.AddUint64(&r.reads, 1)
	return r.KV.Get(ctx, key, opts...)
}

func (r *KVRecorder) GetReadsAndReset() uint64 {
	return atomic.SwapUint64(&r.reads, 0)
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
	recorderKey, ok := ctx.Value(RecorderContextKey).(string)
	if ok {
		r.mux.Lock()
		r.listsPerKey[recorderKey] = append(r.listsPerKey[recorderKey], RecordedList{Key: key, ListOptions: opts})
		r.mux.Unlock()
	}
	return r.Interface.List(ctx, key, opts)
}

func (r *KubernetesRecorder) ListRequestForKey(key string) []RecordedList {
	r.mux.Lock()
	defer r.mux.Unlock()
	return r.listsPerKey[key]
}

type RecordedList struct {
	Key string
	kubernetes.ListOptions
}

var RecorderContextKey recorderKeyType

type recorderKeyType struct{}
