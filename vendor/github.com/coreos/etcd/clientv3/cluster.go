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

package clientv3

import (
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"golang.org/x/net/context"
	"google.golang.org/grpc"
)

type (
	Member               pb.Member
	MemberListResponse   pb.MemberListResponse
	MemberAddResponse    pb.MemberAddResponse
	MemberRemoveResponse pb.MemberRemoveResponse
	MemberUpdateResponse pb.MemberUpdateResponse
)

type Cluster interface {
	// MemberList lists the current cluster membership.
	MemberList(ctx context.Context) (*MemberListResponse, error)

	// MemberAdd adds a new member into the cluster.
	MemberAdd(ctx context.Context, peerAddrs []string) (*MemberAddResponse, error)

	// MemberRemove removes an existing member from the cluster.
	MemberRemove(ctx context.Context, id uint64) (*MemberRemoveResponse, error)

	// MemberUpdate updates the peer addresses of the member.
	MemberUpdate(ctx context.Context, id uint64, peerAddrs []string) (*MemberUpdateResponse, error)
}

type cluster struct {
	remote pb.ClusterClient
}

func NewCluster(c *Client) Cluster {
	return &cluster{remote: RetryClusterClient(c)}
}

func (c *cluster) MemberAdd(ctx context.Context, peerAddrs []string) (*MemberAddResponse, error) {
	r := &pb.MemberAddRequest{PeerURLs: peerAddrs}
	resp, err := c.remote.MemberAdd(ctx, r)
	if err == nil {
		return (*MemberAddResponse)(resp), nil
	}
	if isHaltErr(ctx, err) {
		return nil, toErr(ctx, err)
	}
	return nil, toErr(ctx, err)
}

func (c *cluster) MemberRemove(ctx context.Context, id uint64) (*MemberRemoveResponse, error) {
	r := &pb.MemberRemoveRequest{ID: id}
	resp, err := c.remote.MemberRemove(ctx, r)
	if err == nil {
		return (*MemberRemoveResponse)(resp), nil
	}
	if isHaltErr(ctx, err) {
		return nil, toErr(ctx, err)
	}
	return nil, toErr(ctx, err)
}

func (c *cluster) MemberUpdate(ctx context.Context, id uint64, peerAddrs []string) (*MemberUpdateResponse, error) {
	// it is safe to retry on update.
	for {
		r := &pb.MemberUpdateRequest{ID: id, PeerURLs: peerAddrs}
		resp, err := c.remote.MemberUpdate(ctx, r, grpc.FailFast(false))
		if err == nil {
			return (*MemberUpdateResponse)(resp), nil
		}
		if isHaltErr(ctx, err) {
			return nil, toErr(ctx, err)
		}
	}
}

func (c *cluster) MemberList(ctx context.Context) (*MemberListResponse, error) {
	// it is safe to retry on list.
	for {
		resp, err := c.remote.MemberList(ctx, &pb.MemberListRequest{}, grpc.FailFast(false))
		if err == nil {
			return (*MemberListResponse)(resp), nil
		}
		if isHaltErr(ctx, err) {
			return nil, toErr(ctx, err)
		}
	}
}
