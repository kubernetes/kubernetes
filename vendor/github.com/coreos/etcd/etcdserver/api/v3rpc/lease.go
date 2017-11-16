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

package v3rpc

import (
	"io"

	"github.com/coreos/etcd/etcdserver"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/lease"
	"golang.org/x/net/context"
)

type LeaseServer struct {
	hdr header
	le  etcdserver.Lessor
}

func NewLeaseServer(s *etcdserver.EtcdServer) pb.LeaseServer {
	return &LeaseServer{le: s, hdr: newHeader(s)}
}

func (ls *LeaseServer) LeaseGrant(ctx context.Context, cr *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error) {
	resp, err := ls.le.LeaseGrant(ctx, cr)

	if err != nil {
		return nil, togRPCError(err)
	}
	ls.hdr.fill(resp.Header)
	return resp, nil
}

func (ls *LeaseServer) LeaseRevoke(ctx context.Context, rr *pb.LeaseRevokeRequest) (*pb.LeaseRevokeResponse, error) {
	resp, err := ls.le.LeaseRevoke(ctx, rr)
	if err != nil {
		return nil, togRPCError(err)
	}
	ls.hdr.fill(resp.Header)
	return resp, nil
}

func (ls *LeaseServer) LeaseTimeToLive(ctx context.Context, rr *pb.LeaseTimeToLiveRequest) (*pb.LeaseTimeToLiveResponse, error) {
	resp, err := ls.le.LeaseTimeToLive(ctx, rr)
	if err != nil {
		return nil, togRPCError(err)
	}
	ls.hdr.fill(resp.Header)
	return resp, nil
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

		// Create header before we sent out the renew request.
		// This can make sure that the revision is strictly smaller or equal to
		// when the keepalive happened at the local server (when the local server is the leader)
		// or remote leader.
		// Without this, a lease might be revoked at rev 3 but client can see the keepalive succeeded
		// at rev 4.
		resp := &pb.LeaseKeepAliveResponse{ID: req.ID, Header: &pb.ResponseHeader{}}
		ls.hdr.fill(resp.Header)

		ttl, err := ls.le.LeaseRenew(stream.Context(), lease.LeaseID(req.ID))
		if err == lease.ErrLeaseNotFound {
			err = nil
			ttl = 0
		}

		if err != nil {
			return togRPCError(err)
		}

		resp.TTL = ttl
		err = stream.Send(resp)
		if err != nil {
			return err
		}
	}
}
