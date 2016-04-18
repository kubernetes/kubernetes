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
	"io"

	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/lease"
	"golang.org/x/net/context"
)

type LeaseServer struct {
	le etcdserver.Lessor
}

func NewLeaseServer(le etcdserver.Lessor) pb.LeaseServer {
	return &LeaseServer{le: le}
}

func (ls *LeaseServer) LeaseGrant(ctx context.Context, cr *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error) {
	resp, err := ls.le.LeaseGrant(ctx, cr)
	if err == lease.ErrLeaseExists {
		return nil, rpctypes.ErrLeaseExist
	}
	return resp, err
}

func (ls *LeaseServer) LeaseRevoke(ctx context.Context, rr *pb.LeaseRevokeRequest) (*pb.LeaseRevokeResponse, error) {
	r, err := ls.le.LeaseRevoke(ctx, rr)
	if err != nil {
		return nil, rpctypes.ErrLeaseNotFound
	}
	return r, nil
}

func (ls *LeaseServer) LeaseKeepAlive(stream pb.Lease_LeaseKeepAliveServer) error {
	for {
		req, err := stream.Recv()
		if err == io.EOF {
			return nil
		}
		if err != nil {
			return err
		}

		ttl, err := ls.le.LeaseRenew(lease.LeaseID(req.ID))
		if err == lease.ErrLeaseNotFound {
			return rpctypes.ErrLeaseNotFound
		}

		if err != nil && err != lease.ErrLeaseNotFound {
			return err
		}

		resp := &pb.LeaseKeepAliveResponse{ID: req.ID, TTL: ttl}
		err = stream.Send(resp)
		if err != nil {
			return err
		}
	}
}
