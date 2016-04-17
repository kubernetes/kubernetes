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
	"time"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/etcdserver/membership"
	"github.com/coreos/etcd/pkg/types"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

type ClusterServer struct {
	cluster   api.Cluster
	server    etcdserver.Server
	raftTimer etcdserver.RaftTimer
}

func NewClusterServer(s *etcdserver.EtcdServer) *ClusterServer {
	return &ClusterServer{
		cluster:   s.Cluster(),
		server:    s,
		raftTimer: s,
	}
}

func (cs *ClusterServer) MemberAdd(ctx context.Context, r *pb.MemberAddRequest) (*pb.MemberAddResponse, error) {
	urls, err := types.NewURLs(r.PeerURLs)
	if err != nil {
		return nil, rpctypes.ErrMemberBadURLs
	}

	now := time.Now()
	m := membership.NewMember("", urls, "", &now)
	err = cs.server.AddMember(ctx, *m)
	switch {
	case err == membership.ErrIDExists:
		return nil, rpctypes.ErrMemberExist
	case err == membership.ErrPeerURLexists:
		return nil, rpctypes.ErrPeerURLExist
	case err != nil:
		return nil, grpc.Errorf(codes.Internal, err.Error())
	}

	return &pb.MemberAddResponse{
		Header: cs.header(),
		Member: &pb.Member{ID: uint64(m.ID), IsLeader: m.ID == cs.server.Leader(), PeerURLs: m.PeerURLs},
	}, nil
}

func (cs *ClusterServer) MemberRemove(ctx context.Context, r *pb.MemberRemoveRequest) (*pb.MemberRemoveResponse, error) {
	err := cs.server.RemoveMember(ctx, r.ID)
	switch {
	case err == membership.ErrIDRemoved:
		fallthrough
	case err == membership.ErrIDNotFound:
		return nil, rpctypes.ErrMemberNotFound
	case err != nil:
		return nil, grpc.Errorf(codes.Internal, err.Error())
	}

	return &pb.MemberRemoveResponse{Header: cs.header()}, nil
}

func (cs *ClusterServer) MemberUpdate(ctx context.Context, r *pb.MemberUpdateRequest) (*pb.MemberUpdateResponse, error) {
	m := membership.Member{
		ID:             types.ID(r.ID),
		RaftAttributes: membership.RaftAttributes{PeerURLs: r.PeerURLs},
	}
	err := cs.server.UpdateMember(ctx, m)
	switch {
	case err == membership.ErrPeerURLexists:
		return nil, rpctypes.ErrPeerURLExist
	case err == membership.ErrIDNotFound:
		return nil, rpctypes.ErrMemberNotFound
	case err != nil:
		return nil, grpc.Errorf(codes.Internal, err.Error())
	}

	return &pb.MemberUpdateResponse{Header: cs.header()}, nil
}

func (cs *ClusterServer) MemberList(ctx context.Context, r *pb.MemberListRequest) (*pb.MemberListResponse, error) {
	membs := cs.cluster.Members()

	protoMembs := make([]*pb.Member, len(membs))
	for i := range membs {
		protoMembs[i] = &pb.Member{
			Name:       membs[i].Name,
			ID:         uint64(membs[i].ID),
			IsLeader:   membs[i].ID == cs.server.Leader(),
			PeerURLs:   membs[i].PeerURLs,
			ClientURLs: membs[i].ClientURLs,
		}
	}

	return &pb.MemberListResponse{Header: cs.header(), Members: protoMembs}, nil
}

func (cs *ClusterServer) header() *pb.ResponseHeader {
	return &pb.ResponseHeader{ClusterId: uint64(cs.cluster.ID()), MemberId: uint64(cs.server.ID()), RaftTerm: cs.raftTimer.Term()}
}
