// Copyright 2017 The etcd Authors
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

package namespace

import (
	"github.com/coreos/etcd/clientv3"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"

	"golang.org/x/net/context"
)

type kvPrefix struct {
	clientv3.KV
	pfx string
}

// NewKV wraps a KV instance so that all requests
// are prefixed with a given string.
func NewKV(kv clientv3.KV, prefix string) clientv3.KV {
	return &kvPrefix{kv, prefix}
}

func (kv *kvPrefix) Put(ctx context.Context, key, val string, opts ...clientv3.OpOption) (*clientv3.PutResponse, error) {
	if len(key) == 0 {
		return nil, rpctypes.ErrEmptyKey
	}
	op := kv.prefixOp(clientv3.OpPut(key, val, opts...))
	r, err := kv.KV.Do(ctx, op)
	if err != nil {
		return nil, err
	}
	put := r.Put()
	kv.unprefixPutResponse(put)
	return put, nil
}

func (kv *kvPrefix) Get(ctx context.Context, key string, opts ...clientv3.OpOption) (*clientv3.GetResponse, error) {
	if len(key) == 0 {
		return nil, rpctypes.ErrEmptyKey
	}
	r, err := kv.KV.Do(ctx, kv.prefixOp(clientv3.OpGet(key, opts...)))
	if err != nil {
		return nil, err
	}
	get := r.Get()
	kv.unprefixGetResponse(get)
	return get, nil
}

func (kv *kvPrefix) Delete(ctx context.Context, key string, opts ...clientv3.OpOption) (*clientv3.DeleteResponse, error) {
	if len(key) == 0 {
		return nil, rpctypes.ErrEmptyKey
	}
	r, err := kv.KV.Do(ctx, kv.prefixOp(clientv3.OpDelete(key, opts...)))
	if err != nil {
		return nil, err
	}
	del := r.Del()
	kv.unprefixDeleteResponse(del)
	return del, nil
}

func (kv *kvPrefix) Do(ctx context.Context, op clientv3.Op) (clientv3.OpResponse, error) {
	if len(op.KeyBytes()) == 0 {
		return clientv3.OpResponse{}, rpctypes.ErrEmptyKey
	}
	r, err := kv.KV.Do(ctx, kv.prefixOp(op))
	if err != nil {
		return r, err
	}
	switch {
	case r.Get() != nil:
		kv.unprefixGetResponse(r.Get())
	case r.Put() != nil:
		kv.unprefixPutResponse(r.Put())
	case r.Del() != nil:
		kv.unprefixDeleteResponse(r.Del())
	}
	return r, nil
}

type txnPrefix struct {
	clientv3.Txn
	kv *kvPrefix
}

func (kv *kvPrefix) Txn(ctx context.Context) clientv3.Txn {
	return &txnPrefix{kv.KV.Txn(ctx), kv}
}

func (txn *txnPrefix) If(cs ...clientv3.Cmp) clientv3.Txn {
	newCmps := make([]clientv3.Cmp, len(cs))
	for i := range cs {
		newCmps[i] = cs[i]
		pfxKey, _ := txn.kv.prefixInterval(cs[i].KeyBytes(), nil)
		newCmps[i].WithKeyBytes(pfxKey)
	}
	txn.Txn = txn.Txn.If(newCmps...)
	return txn
}

func (txn *txnPrefix) Then(ops ...clientv3.Op) clientv3.Txn {
	newOps := make([]clientv3.Op, len(ops))
	for i := range ops {
		newOps[i] = txn.kv.prefixOp(ops[i])
	}
	txn.Txn = txn.Txn.Then(newOps...)
	return txn
}

func (txn *txnPrefix) Else(ops ...clientv3.Op) clientv3.Txn {
	newOps := make([]clientv3.Op, len(ops))
	for i := range ops {
		newOps[i] = txn.kv.prefixOp(ops[i])
	}
	txn.Txn = txn.Txn.Else(newOps...)
	return txn
}

func (txn *txnPrefix) Commit() (*clientv3.TxnResponse, error) {
	resp, err := txn.Txn.Commit()
	if err != nil {
		return nil, err
	}
	txn.kv.unprefixTxnResponse(resp)
	return resp, nil
}

func (kv *kvPrefix) prefixOp(op clientv3.Op) clientv3.Op {
	begin, end := kv.prefixInterval(op.KeyBytes(), op.RangeBytes())
	op.WithKeyBytes(begin)
	op.WithRangeBytes(end)
	return op
}

func (kv *kvPrefix) unprefixGetResponse(resp *clientv3.GetResponse) {
	for i := range resp.Kvs {
		resp.Kvs[i].Key = resp.Kvs[i].Key[len(kv.pfx):]
	}
}

func (kv *kvPrefix) unprefixPutResponse(resp *clientv3.PutResponse) {
	if resp.PrevKv != nil {
		resp.PrevKv.Key = resp.PrevKv.Key[len(kv.pfx):]
	}
}

func (kv *kvPrefix) unprefixDeleteResponse(resp *clientv3.DeleteResponse) {
	for i := range resp.PrevKvs {
		resp.PrevKvs[i].Key = resp.PrevKvs[i].Key[len(kv.pfx):]
	}
}

func (kv *kvPrefix) unprefixTxnResponse(resp *clientv3.TxnResponse) {
	for _, r := range resp.Responses {
		switch tv := r.Response.(type) {
		case *pb.ResponseOp_ResponseRange:
			if tv.ResponseRange != nil {
				kv.unprefixGetResponse((*clientv3.GetResponse)(tv.ResponseRange))
			}
		case *pb.ResponseOp_ResponsePut:
			if tv.ResponsePut != nil {
				kv.unprefixPutResponse((*clientv3.PutResponse)(tv.ResponsePut))
			}
		case *pb.ResponseOp_ResponseDeleteRange:
			if tv.ResponseDeleteRange != nil {
				kv.unprefixDeleteResponse((*clientv3.DeleteResponse)(tv.ResponseDeleteRange))
			}
		default:
		}
	}
}

func (p *kvPrefix) prefixInterval(key, end []byte) (pfxKey []byte, pfxEnd []byte) {
	return prefixInterval(p.pfx, key, end)
}
