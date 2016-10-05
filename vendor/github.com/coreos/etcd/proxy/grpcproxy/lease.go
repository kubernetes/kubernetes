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
	"golang.org/x/net/context"

	"github.com/coreos/etcd/clientv3"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
)

type leaseProxy struct {
	client *clientv3.Client
}

func NewLeaseProxy(c *clientv3.Client) pb.LeaseServer {
	return &leaseProxy{
		client: c,
	}
}

func (lp *leaseProxy) LeaseGrant(ctx context.Context, cr *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error) {
	conn := lp.client.ActiveConnection()
	return pb.NewLeaseClient(conn).LeaseGrant(ctx, cr)
}

func (lp *leaseProxy) LeaseRevoke(ctx context.Context, rr *pb.LeaseRevokeRequest) (*pb.LeaseRevokeResponse, error) {
	conn := lp.client.ActiveConnection()
	return pb.NewLeaseClient(conn).LeaseRevoke(ctx, rr)
}

func (lp *leaseProxy) LeaseTimeToLive(ctx context.Context, rr *pb.LeaseTimeToLiveRequest) (*pb.LeaseTimeToLiveResponse, error) {
	conn := lp.client.ActiveConnection()
	return pb.NewLeaseClient(conn).LeaseTimeToLive(ctx, rr)
}

func (lp *leaseProxy) LeaseKeepAlive(stream pb.Lease_LeaseKeepAliveServer) error {
	conn := lp.client.ActiveConnection()
	ctx, cancel := context.WithCancel(stream.Context())
	lc, err := pb.NewLeaseClient(conn).LeaseKeepAlive(ctx)
	if err != nil {
		cancel()
		return err
	}

	go func() {
		// Cancel the context attached to lc to unblock lc.Recv when
		// this routine returns on error.
		defer cancel()

		for {
			// stream.Recv will be unblock when the loop in the parent routine
			// returns on error.
			rr, err := stream.Recv()
			if err != nil {
				return
			}
			err = lc.Send(rr)
			if err != nil {
				return
			}
		}
	}()

	for {
		rr, err := lc.Recv()
		if err != nil {
			return err
		}
		err = stream.Send(rr)
		if err != nil {
			return err
		}
	}
}
