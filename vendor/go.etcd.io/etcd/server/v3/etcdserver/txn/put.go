// Copyright 2025 The etcd Authors
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

package txn

import (
	"context"

	"go.uber.org/zap"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/pkg/v3/traceutil"
	"go.etcd.io/etcd/server/v3/etcdserver/errors"
	"go.etcd.io/etcd/server/v3/lease"
	"go.etcd.io/etcd/server/v3/storage/mvcc"
)

func Put(ctx context.Context, lg *zap.Logger, lessor lease.Lessor, kv mvcc.KV, p *pb.PutRequest) (resp *pb.PutResponse, trace *traceutil.Trace, err error) {
	ctx, trace = traceutil.EnsureTrace(ctx, lg, "put",
		traceutil.Field{Key: "key", Value: string(p.Key)},
		traceutil.Field{Key: "req_size", Value: p.Size()},
	)
	err = checkLease(lessor, p)
	if err != nil {
		return nil, trace, err
	}
	txnWrite := kv.Write(trace)
	defer txnWrite.End()
	prevKV, err := checkAndGetPrevKV(trace, txnWrite, p)
	if err != nil {
		return nil, trace, err
	}
	return put(ctx, txnWrite, p, prevKV), trace, nil
}

func put(ctx context.Context, txnWrite mvcc.TxnWrite, p *pb.PutRequest, prevKV *mvcc.RangeResult) *pb.PutResponse {
	trace := traceutil.Get(ctx)
	resp := &pb.PutResponse{}
	resp.Header = &pb.ResponseHeader{}
	val, leaseID := p.Value, lease.LeaseID(p.Lease)

	if p.IgnoreValue {
		val = prevKV.KVs[0].Value
	}
	if p.IgnoreLease {
		leaseID = lease.LeaseID(prevKV.KVs[0].Lease)
	}
	if p.PrevKv {
		if prevKV != nil && len(prevKV.KVs) != 0 {
			resp.PrevKv = &prevKV.KVs[0]
		}
	}

	resp.Header.Revision = txnWrite.Put(p.Key, val, leaseID)
	trace.AddField(traceutil.Field{Key: "response_revision", Value: resp.Header.Revision})
	return resp
}

func checkPut(trace *traceutil.Trace, txnWrite mvcc.ReadView, lessor lease.Lessor, p *pb.PutRequest) error {
	err := checkLease(lessor, p)
	if err != nil {
		return err
	}
	_, err = checkAndGetPrevKV(trace, txnWrite, p)
	return err
}

func checkLease(lessor lease.Lessor, p *pb.PutRequest) error {
	leaseID := lease.LeaseID(p.Lease)
	if leaseID != lease.NoLease {
		if l := lessor.Lookup(leaseID); l == nil {
			return lease.ErrLeaseNotFound
		}
	}
	return nil
}

func checkAndGetPrevKV(trace *traceutil.Trace, txnWrite mvcc.ReadView, p *pb.PutRequest) (prevKV *mvcc.RangeResult, err error) {
	prevKV, err = getPrevKV(trace, txnWrite, p)
	if err != nil {
		return nil, err
	}
	if p.IgnoreValue || p.IgnoreLease {
		if prevKV == nil || len(prevKV.KVs) == 0 {
			// ignore_{lease,value} flag expects previous key-value pair
			return nil, errors.ErrKeyNotFound
		}
	}
	return prevKV, nil
}

func getPrevKV(trace *traceutil.Trace, txnWrite mvcc.ReadView, p *pb.PutRequest) (prevKV *mvcc.RangeResult, err error) {
	if p.IgnoreValue || p.IgnoreLease || p.PrevKv {
		trace.StepWithFunction(func() {
			prevKV, err = txnWrite.Range(context.TODO(), p.Key, nil, mvcc.RangeOptions{})
		}, "get previous kv pair")

		if err != nil {
			return nil, err
		}
	}
	return prevKV, nil
}
