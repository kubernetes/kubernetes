// Copyright 2015 The etcd Authors
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

// Package v3rpc implements etcd v3 RPC system based on gRPC.
package v3rpc

import (
	"sort"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/pkg/capnslog"
	"golang.org/x/net/context"
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "etcdserver/api/v3rpc")

	// Max operations per txn list. For example, Txn.Success can have at most 128 operations,
	// and Txn.Failure can have at most 128 operations.
	MaxOpsPerTxn = 128
)

type kvServer struct {
	hdr header
	kv  etcdserver.RaftKV
}

func NewKVServer(s *etcdserver.EtcdServer) pb.KVServer {
	return &kvServer{hdr: newHeader(s), kv: s}
}

func (s *kvServer) Range(ctx context.Context, r *pb.RangeRequest) (*pb.RangeResponse, error) {
	if err := checkRangeRequest(r); err != nil {
		return nil, err
	}

	resp, err := s.kv.Range(ctx, r)
	if err != nil {
		return nil, togRPCError(err)
	}

	if resp.Header == nil {
		plog.Panic("unexpected nil resp.Header")
	}
	s.hdr.fill(resp.Header)
	return resp, nil
}

func (s *kvServer) Put(ctx context.Context, r *pb.PutRequest) (*pb.PutResponse, error) {
	if err := checkPutRequest(r); err != nil {
		return nil, err
	}

	resp, err := s.kv.Put(ctx, r)
	if err != nil {
		return nil, togRPCError(err)
	}

	if resp.Header == nil {
		plog.Panic("unexpected nil resp.Header")
	}
	s.hdr.fill(resp.Header)
	return resp, nil
}

func (s *kvServer) DeleteRange(ctx context.Context, r *pb.DeleteRangeRequest) (*pb.DeleteRangeResponse, error) {
	if err := checkDeleteRequest(r); err != nil {
		return nil, err
	}

	resp, err := s.kv.DeleteRange(ctx, r)
	if err != nil {
		return nil, togRPCError(err)
	}

	if resp.Header == nil {
		plog.Panic("unexpected nil resp.Header")
	}
	s.hdr.fill(resp.Header)
	return resp, nil
}

func (s *kvServer) Txn(ctx context.Context, r *pb.TxnRequest) (*pb.TxnResponse, error) {
	if err := checkTxnRequest(r); err != nil {
		return nil, err
	}

	resp, err := s.kv.Txn(ctx, r)
	if err != nil {
		return nil, togRPCError(err)
	}

	if resp.Header == nil {
		plog.Panic("unexpected nil resp.Header")
	}
	s.hdr.fill(resp.Header)
	return resp, nil
}

func (s *kvServer) Compact(ctx context.Context, r *pb.CompactionRequest) (*pb.CompactionResponse, error) {
	resp, err := s.kv.Compact(ctx, r)
	if err != nil {
		return nil, togRPCError(err)
	}

	if resp.Header == nil {
		plog.Panic("unexpected nil resp.Header")
	}
	s.hdr.fill(resp.Header)
	return resp, nil
}

func checkRangeRequest(r *pb.RangeRequest) error {
	if len(r.Key) == 0 {
		return rpctypes.ErrGRPCEmptyKey
	}
	return nil
}

func checkPutRequest(r *pb.PutRequest) error {
	if len(r.Key) == 0 {
		return rpctypes.ErrGRPCEmptyKey
	}
	return nil
}

func checkDeleteRequest(r *pb.DeleteRangeRequest) error {
	if len(r.Key) == 0 {
		return rpctypes.ErrGRPCEmptyKey
	}
	return nil
}

func checkTxnRequest(r *pb.TxnRequest) error {
	if len(r.Compare) > MaxOpsPerTxn || len(r.Success) > MaxOpsPerTxn || len(r.Failure) > MaxOpsPerTxn {
		return rpctypes.ErrGRPCTooManyOps
	}

	for _, c := range r.Compare {
		if len(c.Key) == 0 {
			return rpctypes.ErrGRPCEmptyKey
		}
	}

	for _, u := range r.Success {
		if err := checkRequestOp(u); err != nil {
			return err
		}
	}
	if err := checkRequestDupKeys(r.Success); err != nil {
		return err
	}

	for _, u := range r.Failure {
		if err := checkRequestOp(u); err != nil {
			return err
		}
	}
	return checkRequestDupKeys(r.Failure)
}

// checkRequestDupKeys gives rpctypes.ErrGRPCDuplicateKey if the same key is modified twice
func checkRequestDupKeys(reqs []*pb.RequestOp) error {
	// check put overlap
	keys := make(map[string]struct{})
	for _, requ := range reqs {
		tv, ok := requ.Request.(*pb.RequestOp_RequestPut)
		if !ok {
			continue
		}
		preq := tv.RequestPut
		if preq == nil {
			continue
		}
		if _, ok := keys[string(preq.Key)]; ok {
			return rpctypes.ErrGRPCDuplicateKey
		}
		keys[string(preq.Key)] = struct{}{}
	}

	// no need to check deletes if no puts; delete overlaps are permitted
	if len(keys) == 0 {
		return nil
	}

	// sort keys for range checking
	sortedKeys := []string{}
	for k := range keys {
		sortedKeys = append(sortedKeys, k)
	}
	sort.Strings(sortedKeys)

	// check put overlap with deletes
	for _, requ := range reqs {
		tv, ok := requ.Request.(*pb.RequestOp_RequestDeleteRange)
		if !ok {
			continue
		}
		dreq := tv.RequestDeleteRange
		if dreq == nil {
			continue
		}
		if dreq.RangeEnd == nil {
			if _, found := keys[string(dreq.Key)]; found {
				return rpctypes.ErrGRPCDuplicateKey
			}
		} else {
			lo := sort.SearchStrings(sortedKeys, string(dreq.Key))
			hi := sort.SearchStrings(sortedKeys, string(dreq.RangeEnd))
			if lo != hi {
				// element between lo and hi => overlap
				return rpctypes.ErrGRPCDuplicateKey
			}
		}
	}

	return nil
}

func checkRequestOp(u *pb.RequestOp) error {
	// TODO: ensure only one of the field is set.
	switch uv := u.Request.(type) {
	case *pb.RequestOp_RequestRange:
		if uv.RequestRange != nil {
			return checkRangeRequest(uv.RequestRange)
		}
	case *pb.RequestOp_RequestPut:
		if uv.RequestPut != nil {
			return checkPutRequest(uv.RequestPut)
		}
	case *pb.RequestOp_RequestDeleteRange:
		if uv.RequestDeleteRange != nil {
			return checkDeleteRequest(uv.RequestDeleteRange)
		}
	default:
		// empty op
		return nil
	}
	return nil
}
