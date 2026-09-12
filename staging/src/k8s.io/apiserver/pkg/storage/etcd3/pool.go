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

package etcd3

import (
	"context"
	"sync/atomic"

	clientv3 "go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/client/v3/kubernetes"
)

// WrapWithConnectionPool decorates the primary kubernetes client's interfaces
// to round-robin requests across a pool of independent connections.
func WrapWithConnectionPool(primary *kubernetes.Client, pool []*kubernetes.Client) {
	if len(pool) <= 1 {
		return
	}
	primary.KV = &roundRobinKV{pool: pool}
	primary.Kubernetes = &roundRobinKubernetes{pool: pool}
}

type roundRobinKV struct {
	pool    []*kubernetes.Client
	counter uint64
}

func (rr *roundRobinKV) getKV() clientv3.KV {
	idx := atomic.AddUint64(&rr.counter, 1) % uint64(len(rr.pool))
	return rr.pool[idx].KV
}

func (rr *roundRobinKV) Put(ctx context.Context, key, val string, opts ...clientv3.OpOption) (*clientv3.PutResponse, error) {
	return rr.getKV().Put(ctx, key, val, opts...)
}

func (rr *roundRobinKV) Get(ctx context.Context, key string, opts ...clientv3.OpOption) (*clientv3.GetResponse, error) {
	return rr.getKV().Get(ctx, key, opts...)
}

func (rr *roundRobinKV) Delete(ctx context.Context, key string, opts ...clientv3.OpOption) (*clientv3.DeleteResponse, error) {
	return rr.getKV().Delete(ctx, key, opts...)
}

func (rr *roundRobinKV) Compact(ctx context.Context, rev int64, opts ...clientv3.CompactOption) (*clientv3.CompactResponse, error) {
	return rr.getKV().Compact(ctx, rev, opts...)
}

func (rr *roundRobinKV) Txn(ctx context.Context) clientv3.Txn {
	return rr.getKV().Txn(ctx)
}

func (rr *roundRobinKV) Do(ctx context.Context, op clientv3.Op) (clientv3.OpResponse, error) {
	return rr.getKV().Do(ctx, op)
}

func (rr *roundRobinKV) GetStream(ctx context.Context, key string, opts ...clientv3.OpOption) (clientv3.GetStreamChan, error) {
	return rr.getKV().GetStream(ctx, key, opts...)
}

type roundRobinKubernetes struct {
	pool    []*kubernetes.Client
	counter uint64
}

func (rr *roundRobinKubernetes) getKube() kubernetes.Interface {
	idx := atomic.AddUint64(&rr.counter, 1) % uint64(len(rr.pool))
	return rr.pool[idx].Kubernetes
}

func (rr *roundRobinKubernetes) Get(ctx context.Context, key string, opts kubernetes.GetOptions) (kubernetes.GetResponse, error) {
	return rr.getKube().Get(ctx, key, opts)
}

func (rr *roundRobinKubernetes) List(ctx context.Context, prefix string, opts kubernetes.ListOptions) (kubernetes.ListResponse, error) {
	return rr.getKube().List(ctx, prefix, opts)
}

func (rr *roundRobinKubernetes) Count(ctx context.Context, prefix string, opts kubernetes.CountOptions) (int64, error) {
	return rr.getKube().Count(ctx, prefix, opts)
}

func (rr *roundRobinKubernetes) OptimisticPut(ctx context.Context, key string, value []byte, expectedRevision int64, opts kubernetes.PutOptions) (kubernetes.PutResponse, error) {
	return rr.getKube().OptimisticPut(ctx, key, value, expectedRevision, opts)
}

func (rr *roundRobinKubernetes) OptimisticDelete(ctx context.Context, key string, expectedRevision int64, opts kubernetes.DeleteOptions) (kubernetes.DeleteResponse, error) {
	return rr.getKube().OptimisticDelete(ctx, key, expectedRevision, opts)
}
