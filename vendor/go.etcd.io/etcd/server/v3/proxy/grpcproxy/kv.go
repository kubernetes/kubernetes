// Copyright 2016 The etcd Authors
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

package grpcproxy

import (
	"context"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/client/v3"
	"go.etcd.io/etcd/server/v3/proxy/grpcproxy/cache"
)

type kvProxy struct {
	kv    clientv3.KV
	cache cache.Cache
}

func NewKvProxy(c *clientv3.Client) (pb.KVServer, <-chan struct{}) {
	kv := &kvProxy{
		kv:    c.KV,
		cache: cache.NewCache(cache.DefaultMaxEntries),
	}
	donec := make(chan struct{})
	close(donec)
	return kv, donec
}

func (p *kvProxy) Range(ctx context.Context, r *pb.RangeRequest) (*pb.RangeResponse, error) {
	if r.Serializable {
		resp, err := p.cache.Get(r)
		switch err {
		case nil:
			cacheHits.Inc()
			return resp, nil
		case cache.ErrCompacted:
			cacheHits.Inc()
			return nil, err
		}

		cachedMisses.Inc()
	}

	resp, err := p.kv.Do(ctx, RangeRequestToOp(r))
	if err != nil {
		return nil, err
	}

	// cache linearizable as serializable
	req := *r
	req.Serializable = true
	gresp := (*pb.RangeResponse)(resp.Get())
	p.cache.Add(&req, gresp)
	cacheKeys.Set(float64(p.cache.Size()))

	return gresp, nil
}

func (p *kvProxy) Put(ctx context.Context, r *pb.PutRequest) (*pb.PutResponse, error) {
	p.cache.Invalidate(r.Key, nil)
	cacheKeys.Set(float64(p.cache.Size()))

	resp, err := p.kv.Do(ctx, PutRequestToOp(r))
	return (*pb.PutResponse)(resp.Put()), err
}

func (p *kvProxy) DeleteRange(ctx context.Context, r *pb.DeleteRangeRequest) (*pb.DeleteRangeResponse, error) {
	p.cache.Invalidate(r.Key, r.RangeEnd)
	cacheKeys.Set(float64(p.cache.Size()))

	resp, err := p.kv.Do(ctx, DelRequestToOp(r))
	return (*pb.DeleteRangeResponse)(resp.Del()), err
}

func (p *kvProxy) txnToCache(reqs []*pb.RequestOp, resps []*pb.ResponseOp) {
	for i := range resps {
		switch tv := resps[i].Response.(type) {
		case *pb.ResponseOp_ResponsePut:
			p.cache.Invalidate(reqs[i].GetRequestPut().Key, nil)
		case *pb.ResponseOp_ResponseDeleteRange:
			rdr := reqs[i].GetRequestDeleteRange()
			p.cache.Invalidate(rdr.Key, rdr.RangeEnd)
		case *pb.ResponseOp_ResponseRange:
			req := *(reqs[i].GetRequestRange())
			req.Serializable = true
			p.cache.Add(&req, tv.ResponseRange)
		}
	}
}

func (p *kvProxy) Txn(ctx context.Context, r *pb.TxnRequest) (*pb.TxnResponse, error) {
	op := TxnRequestToOp(r)
	opResp, err := p.kv.Do(ctx, op)
	if err != nil {
		return nil, err
	}
	resp := opResp.Txn()

	// txn may claim an outdated key is updated; be safe and invalidate
	for _, cmp := range r.Compare {
		p.cache.Invalidate(cmp.Key, cmp.RangeEnd)
	}
	// update any fetched keys
	if resp.Succeeded {
		p.txnToCache(r.Success, resp.Responses)
	} else {
		p.txnToCache(r.Failure, resp.Responses)
	}

	cacheKeys.Set(float64(p.cache.Size()))

	return (*pb.TxnResponse)(resp), nil
}

func (p *kvProxy) Compact(ctx context.Context, r *pb.CompactionRequest) (*pb.CompactionResponse, error) {
	var opts []clientv3.CompactOption
	if r.Physical {
		opts = append(opts, clientv3.WithCompactPhysical())
	}

	resp, err := p.kv.Compact(ctx, r.Revision, opts...)
	if err == nil {
		p.cache.Compact(r.Revision)
	}

	cacheKeys.Set(float64(p.cache.Size()))

	return (*pb.CompactionResponse)(resp), err
}

func requestOpToOp(union *pb.RequestOp) clientv3.Op {
	switch tv := union.Request.(type) {
	case *pb.RequestOp_RequestRange:
		if tv.RequestRange != nil {
			return RangeRequestToOp(tv.RequestRange)
		}
	case *pb.RequestOp_RequestPut:
		if tv.RequestPut != nil {
			return PutRequestToOp(tv.RequestPut)
		}
	case *pb.RequestOp_RequestDeleteRange:
		if tv.RequestDeleteRange != nil {
			return DelRequestToOp(tv.RequestDeleteRange)
		}
	case *pb.RequestOp_RequestTxn:
		if tv.RequestTxn != nil {
			return TxnRequestToOp(tv.RequestTxn)
		}
	}
	panic("unknown request")
}

func RangeRequestToOp(r *pb.RangeRequest) clientv3.Op {
	opts := []clientv3.OpOption{}
	if len(r.RangeEnd) != 0 {
		opts = append(opts, clientv3.WithRange(string(r.RangeEnd)))
	}
	opts = append(opts, clientv3.WithRev(r.Revision))
	opts = append(opts, clientv3.WithLimit(r.Limit))
	opts = append(opts, clientv3.WithSort(
		clientv3.SortTarget(r.SortTarget),
		clientv3.SortOrder(r.SortOrder)),
	)
	opts = append(opts, clientv3.WithMaxCreateRev(r.MaxCreateRevision))
	opts = append(opts, clientv3.WithMinCreateRev(r.MinCreateRevision))
	opts = append(opts, clientv3.WithMaxModRev(r.MaxModRevision))
	opts = append(opts, clientv3.WithMinModRev(r.MinModRevision))
	if r.CountOnly {
		opts = append(opts, clientv3.WithCountOnly())
	}
	if r.KeysOnly {
		opts = append(opts, clientv3.WithKeysOnly())
	}
	if r.Serializable {
		opts = append(opts, clientv3.WithSerializable())
	}

	return clientv3.OpGet(string(r.Key), opts...)
}

func PutRequestToOp(r *pb.PutRequest) clientv3.Op {
	opts := []clientv3.OpOption{}
	opts = append(opts, clientv3.WithLease(clientv3.LeaseID(r.Lease)))
	if r.IgnoreValue {
		opts = append(opts, clientv3.WithIgnoreValue())
	}
	if r.IgnoreLease {
		opts = append(opts, clientv3.WithIgnoreLease())
	}
	if r.PrevKv {
		opts = append(opts, clientv3.WithPrevKV())
	}
	return clientv3.OpPut(string(r.Key), string(r.Value), opts...)
}

func DelRequestToOp(r *pb.DeleteRangeRequest) clientv3.Op {
	opts := []clientv3.OpOption{}
	if len(r.RangeEnd) != 0 {
		opts = append(opts, clientv3.WithRange(string(r.RangeEnd)))
	}
	if r.PrevKv {
		opts = append(opts, clientv3.WithPrevKV())
	}
	return clientv3.OpDelete(string(r.Key), opts...)
}

func TxnRequestToOp(r *pb.TxnRequest) clientv3.Op {
	cmps := make([]clientv3.Cmp, len(r.Compare))
	thenops := make([]clientv3.Op, len(r.Success))
	elseops := make([]clientv3.Op, len(r.Failure))
	for i := range r.Compare {
		cmps[i] = (clientv3.Cmp)(*r.Compare[i])
	}
	for i := range r.Success {
		thenops[i] = requestOpToOp(r.Success[i])
	}
	for i := range r.Failure {
		elseops[i] = requestOpToOp(r.Failure[i])
	}
	return clientv3.OpTxn(cmps, thenops, elseops)
}
