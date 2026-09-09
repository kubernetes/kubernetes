// Copyright 2024 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kubernetes

import (
	"context"
	"fmt"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/api/v3/mvccpb"
	clientv3 "go.etcd.io/etcd/client/v3"
)

// New creates Client from config.
// Caller is responsible to call Close() to clean up client.
func New(cfg clientv3.Config) (*Client, error) {
	c, err := clientv3.New(cfg)
	if err != nil {
		return nil, err
	}
	kc := &Client{
		Client: c,
	}
	kc.Kubernetes = kc
	return kc, nil
}

type Client struct {
	*clientv3.Client
	Kubernetes Interface
}

var _ Interface = (*Client)(nil)

func (k Client) Get(ctx context.Context, key string, opts GetOptions) (resp GetResponse, err error) {
	rangeResp, err := k.KV.Get(ctx, key, clientv3.WithRev(opts.Revision), clientv3.WithLimit(1))
	if err != nil {
		return resp, err
	}
	resp.Revision = rangeResp.Header.Revision
	if len(rangeResp.Kvs) == 1 {
		resp.KV = rangeResp.Kvs[0]
	}
	return resp, nil
}

func (k Client) List(ctx context.Context, prefix string, opts ListOptions) (resp ListResponse, err error) {
	rangeStart := prefix
	if opts.Continue != "" {
		rangeStart = opts.Continue
	}
	rangeEnd := clientv3.GetPrefixRangeEnd(prefix)
	rangeResp, err := k.KV.Get(ctx, rangeStart, clientv3.WithRange(rangeEnd), clientv3.WithLimit(opts.Limit), clientv3.WithRev(opts.Revision))
	if err != nil {
		return resp, err
	}
	resp.Kvs = rangeResp.Kvs
	resp.Count = rangeResp.Count
	resp.Revision = rangeResp.Header.Revision
	return resp, nil
}

func (k Client) Count(ctx context.Context, prefix string, _ CountOptions) (int64, error) {
	resp, err := k.KV.Get(ctx, prefix, clientv3.WithPrefix(), clientv3.WithCountOnly())
	if err != nil {
		return 0, err
	}
	return resp.Count, nil
}

func (k Client) OptimisticPut(ctx context.Context, key string, value []byte, expectedRevision int64, opts PutOptions) (resp PutResponse, err error) {
	txn := k.KV.Txn(ctx).If(
		clientv3.Compare(clientv3.ModRevision(key), "=", expectedRevision),
	).Then(
		clientv3.OpPut(key, string(value), clientv3.WithLease(opts.LeaseID)),
	)

	if opts.GetOnFailure {
		txn = txn.Else(clientv3.OpGet(key))
	}

	txnResp, err := txn.Commit()
	if err != nil {
		return resp, err
	}
	resp.Succeeded = txnResp.Succeeded
	resp.Revision = txnResp.Header.Revision
	if opts.GetOnFailure && !txnResp.Succeeded {
		if len(txnResp.Responses) == 0 {
			return resp, fmt.Errorf("invalid OptimisticPut response: %v", txnResp.Responses)
		}
		resp.KV = kvFromTxnResponse(txnResp.Responses[0])
	}
	return resp, nil
}

func (k Client) OptimisticDelete(ctx context.Context, key string, expectedRevision int64, opts DeleteOptions) (resp DeleteResponse, err error) {
	txn := k.KV.Txn(ctx).If(
		clientv3.Compare(clientv3.ModRevision(key), "=", expectedRevision),
	).Then(
		clientv3.OpDelete(key),
	)
	if opts.GetOnFailure {
		txn = txn.Else(clientv3.OpGet(key))
	}
	txnResp, err := txn.Commit()
	if err != nil {
		return resp, err
	}
	resp.Succeeded = txnResp.Succeeded
	resp.Revision = txnResp.Header.Revision
	if opts.GetOnFailure && !txnResp.Succeeded {
		resp.KV = kvFromTxnResponse(txnResp.Responses[0])
	}
	return resp, nil
}

func kvFromTxnResponse(resp *pb.ResponseOp) *mvccpb.KeyValue {
	getResponse := resp.GetResponseRange()
	if len(getResponse.Kvs) == 1 {
		return getResponse.Kvs[0]
	}
	return nil
}
