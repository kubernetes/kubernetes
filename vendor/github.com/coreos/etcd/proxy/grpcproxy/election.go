// Copyright 2017 The etcd Lockors
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
	"github.com/coreos/etcd/etcdserver/api/v3election/v3electionpb"
)

type electionProxy struct {
	client *clientv3.Client
}

func NewElectionProxy(client *clientv3.Client) v3electionpb.ElectionServer {
	return &electionProxy{client: client}
}

func (ep *electionProxy) Campaign(ctx context.Context, req *v3electionpb.CampaignRequest) (*v3electionpb.CampaignResponse, error) {
	return v3electionpb.NewElectionClient(ep.client.ActiveConnection()).Campaign(ctx, req)
}

func (ep *electionProxy) Proclaim(ctx context.Context, req *v3electionpb.ProclaimRequest) (*v3electionpb.ProclaimResponse, error) {
	return v3electionpb.NewElectionClient(ep.client.ActiveConnection()).Proclaim(ctx, req)
}

func (ep *electionProxy) Leader(ctx context.Context, req *v3electionpb.LeaderRequest) (*v3electionpb.LeaderResponse, error) {
	return v3electionpb.NewElectionClient(ep.client.ActiveConnection()).Leader(ctx, req)
}

func (ep *electionProxy) Observe(req *v3electionpb.LeaderRequest, s v3electionpb.Election_ObserveServer) error {
	conn := ep.client.ActiveConnection()
	ctx, cancel := context.WithCancel(s.Context())
	defer cancel()
	sc, err := v3electionpb.NewElectionClient(conn).Observe(ctx, req)
	if err != nil {
		return err
	}
	for {
		rr, err := sc.Recv()
		if err != nil {
			return err
		}
		if err = s.Send(rr); err != nil {
			return err
		}
	}
}

func (ep *electionProxy) Resign(ctx context.Context, req *v3electionpb.ResignRequest) (*v3electionpb.ResignResponse, error) {
	return v3electionpb.NewElectionClient(ep.client.ActiveConnection()).Resign(ctx, req)
}
