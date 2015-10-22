// Copyright 2015 CoreOS, Inc.
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

package etcdserver

import (
	"bytes"

	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	dstorage "github.com/coreos/etcd/storage"
	"github.com/gogo/protobuf/proto"
	"golang.org/x/net/context"
)

type V3DemoServer interface {
	V3DemoDo(ctx context.Context, r pb.InternalRaftRequest) proto.Message
}

func (s *EtcdServer) V3DemoDo(ctx context.Context, r pb.InternalRaftRequest) proto.Message {
	switch {
	case r.Range != nil:
		return doRange(s.kv, r.Range)
	case r.Put != nil:
		return doPut(s.kv, r.Put)
	case r.DeleteRange != nil:
		return doDeleteRange(s.kv, r.DeleteRange)
	case r.Txn != nil:
		return doTxn(s.kv, r.Txn)
	default:
		panic("not implemented")
	}
}

func doPut(kv dstorage.KV, p *pb.PutRequest) *pb.PutResponse {
	resp := &pb.PutResponse{}
	resp.Header = &pb.ResponseHeader{}
	rev := kv.Put(p.Key, p.Value)
	resp.Header.Revision = rev
	return resp
}

func doRange(kv dstorage.KV, r *pb.RangeRequest) *pb.RangeResponse {
	resp := &pb.RangeResponse{}
	resp.Header = &pb.ResponseHeader{}
	kvs, rev, err := kv.Range(r.Key, r.RangeEnd, r.Limit, 0)
	if err != nil {
		panic("not handled error")
	}

	resp.Header.Revision = rev
	for i := range kvs {
		resp.Kvs = append(resp.Kvs, &kvs[i])
	}
	return resp
}

func doDeleteRange(kv dstorage.KV, dr *pb.DeleteRangeRequest) *pb.DeleteRangeResponse {
	resp := &pb.DeleteRangeResponse{}
	resp.Header = &pb.ResponseHeader{}
	_, rev := kv.DeleteRange(dr.Key, dr.RangeEnd)
	resp.Header.Revision = rev
	return resp
}

func doTxn(kv dstorage.KV, rt *pb.TxnRequest) *pb.TxnResponse {
	var revision int64

	ok := true
	for _, c := range rt.Compare {
		if revision, ok = doCompare(kv, c); !ok {
			break
		}
	}

	var reqs []*pb.RequestUnion
	if ok {
		reqs = rt.Success
	} else {
		reqs = rt.Failure
	}
	resps := make([]*pb.ResponseUnion, len(reqs))
	for i := range reqs {
		resps[i] = doUnion(kv, reqs[i])
	}
	if len(resps) != 0 {
		revision += 1
	}

	txnResp := &pb.TxnResponse{}
	txnResp.Header = &pb.ResponseHeader{}
	txnResp.Header.Revision = revision
	txnResp.Responses = resps
	txnResp.Succeeded = ok
	return txnResp
}

func doUnion(kv dstorage.KV, union *pb.RequestUnion) *pb.ResponseUnion {
	switch {
	case union.RequestRange != nil:
		return &pb.ResponseUnion{ResponseRange: doRange(kv, union.RequestRange)}
	case union.RequestPut != nil:
		return &pb.ResponseUnion{ResponsePut: doPut(kv, union.RequestPut)}
	case union.RequestDeleteRange != nil:
		return &pb.ResponseUnion{ResponseDeleteRange: doDeleteRange(kv, union.RequestDeleteRange)}
	default:
		// empty union
		return nil
	}
}

func doCompare(kv dstorage.KV, c *pb.Compare) (int64, bool) {
	ckvs, rev, err := kv.Range(c.Key, nil, 1, 0)
	if err != nil {
		return rev, false
	}

	ckv := ckvs[0]

	// -1 is less, 0 is equal, 1 is greater
	var result int
	switch c.Target {
	case pb.Compare_VALUE:
		result = bytes.Compare(ckv.Value, c.Value)
	case pb.Compare_CREATE:
		result = compareInt64(ckv.CreateRevision, c.CreateRevision)
	case pb.Compare_MOD:
		result = compareInt64(ckv.ModRevision, c.ModRevision)
	case pb.Compare_VERSION:
		result = compareInt64(ckv.Version, c.Version)
	}

	switch c.Result {
	case pb.Compare_EQUAL:
		if result != 0 {
			return rev, false
		}
	case pb.Compare_GREATER:
		if result != 1 {
			return rev, false
		}
	case pb.Compare_LESS:
		if result != -1 {
			return rev, false
		}
	}
	return rev, true
}

func compareInt64(a, b int64) int {
	switch {
	case a < b:
		return -1
	case a > b:
		return 1
	default:
		return 0
	}
}
