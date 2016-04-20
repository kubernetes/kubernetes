// Copyright 2016 CoreOS, Inc.
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
	"github.com/coreos/etcd/storage/backend"
	"github.com/coreos/etcd/version"
	"golang.org/x/net/context"
)

type BackendGetter interface {
	Backend() backend.Backend
}

type Alarmer interface {
	Alarm(ctx context.Context, ar *pb.AlarmRequest) (*pb.AlarmResponse, error)
}

type maintenanceServer struct {
	bg  BackendGetter
	a   Alarmer
	hdr header
}

func NewMaintenanceServer(s *etcdserver.EtcdServer) pb.MaintenanceServer {
	return &maintenanceServer{bg: s, a: s, hdr: newHeader(s)}
}

func (ms *maintenanceServer) Defragment(ctx context.Context, sr *pb.DefragmentRequest) (*pb.DefragmentResponse, error) {
	plog.Noticef("starting to defragment the storage backend...")
	err := ms.bg.Backend().Defrag()
	if err != nil {
		plog.Errorf("failed to deframent the storage backend (%v)", err)
		return nil, err
	}
	plog.Noticef("finished defragmenting the storage backend")
	return &pb.DefragmentResponse{}, nil
}

func (ms *maintenanceServer) Hash(ctx context.Context, r *pb.HashRequest) (*pb.HashResponse, error) {
	h, err := ms.bg.Backend().Hash()
	if err != nil {
		return nil, togRPCError(err)
	}
	resp := &pb.HashResponse{Header: &pb.ResponseHeader{Revision: ms.hdr.rev()}, Hash: h}
	ms.hdr.fill(resp.Header)
	return resp, nil
}

func (ms *maintenanceServer) Alarm(ctx context.Context, ar *pb.AlarmRequest) (*pb.AlarmResponse, error) {
	return ms.a.Alarm(ctx, ar)
}

func (ms *maintenanceServer) Status(ctx context.Context, ar *pb.StatusRequest) (*pb.StatusResponse, error) {
	resp := &pb.StatusResponse{Header: &pb.ResponseHeader{Revision: ms.hdr.rev()}, Version: version.Version}
	ms.hdr.fill(resp.Header)
	return resp, nil
}
