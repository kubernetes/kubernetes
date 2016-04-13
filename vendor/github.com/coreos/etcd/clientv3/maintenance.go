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

package clientv3

import (
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"golang.org/x/net/context"
)

type (
	DefragmentResponse pb.DefragmentResponse
)

type Maintenance interface {
	// Defragment defragments storage backend of the etcd member with given endpoint.
	// Defragment is only needed when deleting a large number of keys and want to reclaim
	// the resources.
	// Defragment is an expensive operation. User should avoid defragmenting multiple members
	// at the same time.
	// To defragment multiple members in the cluster, user need to call defragment multiple
	// times with different endpoints.
	Defragment(ctx context.Context, endpoint string) (*DefragmentResponse, error)
}

type maintenance struct {
	c *Client
}

func (m *maintenance) Defragment(ctx context.Context, endpoint string) (*DefragmentResponse, error) {
	conn, err := m.c.Dial(endpoint)
	if err != nil {
		return nil, err
	}
	remote := pb.NewMaintenanceClient(conn)
	resp, err := remote.Defragment(ctx, &pb.DefragmentRequest{})
	if err != nil {
		return nil, err
	}
	return (*DefragmentResponse)(resp), nil
}
