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
	"github.com/coreos/etcd/clientv3"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"

	"golang.org/x/net/context"
)

type clusterProxy struct {
	client *clientv3.Client
}

func NewClusterProxy(c *clientv3.Client) pb.ClusterServer {
	return &clusterProxy{
		client: c,
	}
}

func (cp *clusterProxy) MemberAdd(ctx context.Context, r *pb.MemberAddRequest) (*pb.MemberAddResponse, error) {
	conn := cp.client.ActiveConnection()
	return pb.NewClusterClient(conn).MemberAdd(ctx, r)
}

func (cp *clusterProxy) MemberRemove(ctx context.Context, r *pb.MemberRemoveRequest) (*pb.MemberRemoveResponse, error) {
	conn := cp.client.ActiveConnection()
	return pb.NewClusterClient(conn).MemberRemove(ctx, r)
}

func (cp *clusterProxy) MemberUpdate(ctx context.Context, r *pb.MemberUpdateRequest) (*pb.MemberUpdateResponse, error) {
	conn := cp.client.ActiveConnection()
	return pb.NewClusterClient(conn).MemberUpdate(ctx, r)
}

func (cp *clusterProxy) MemberList(ctx context.Context, r *pb.MemberListRequest) (*pb.MemberListResponse, error) {
	conn := cp.client.ActiveConnection()
	return pb.NewClusterClient(conn).MemberList(ctx, r)
}
