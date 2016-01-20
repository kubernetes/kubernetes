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

package v3rpc

import (
	"github.com/coreos/etcd/etcdserver"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"golang.org/x/net/context"
)

type handler struct {
	server etcdserver.V3DemoServer
}

func New(s etcdserver.V3DemoServer) pb.EtcdServer {
	return &handler{s}
}

func (h *handler) Range(ctx context.Context, r *pb.RangeRequest) (*pb.RangeResponse, error) {
	resp := h.server.V3DemoDo(ctx, pb.InternalRaftRequest{Range: r})
	return resp.(*pb.RangeResponse), nil
}

func (h *handler) Put(ctx context.Context, r *pb.PutRequest) (*pb.PutResponse, error) {
	resp := h.server.V3DemoDo(ctx, pb.InternalRaftRequest{Put: r})
	return resp.(*pb.PutResponse), nil
}

func (h *handler) DeleteRange(ctx context.Context, r *pb.DeleteRangeRequest) (*pb.DeleteRangeResponse, error) {
	resp := h.server.V3DemoDo(ctx, pb.InternalRaftRequest{DeleteRange: r})
	return resp.(*pb.DeleteRangeResponse), nil
}

func (h *handler) Txn(ctx context.Context, r *pb.TxnRequest) (*pb.TxnResponse, error) {
	resp := h.server.V3DemoDo(ctx, pb.InternalRaftRequest{Txn: r})
	return resp.(*pb.TxnResponse), nil
}

func (h *handler) Compact(ctx context.Context, r *pb.CompactionRequest) (*pb.CompactionResponse, error) {
	panic("not implemented")
}
