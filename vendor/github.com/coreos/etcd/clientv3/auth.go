// Copyright 2016 Nippon Telegraph and Telephone Corporation.
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
	AuthEnableResponse pb.AuthEnableResponse
)

type Auth interface {
	// AuthEnable enables auth of a etcd cluster.
	AuthEnable(ctx context.Context) (*AuthEnableResponse, error)
}

type auth struct {
	c *Client

	conn   *grpc.ClientConn // conn in-use
	remote pb.AuthClient
}

func NewAuth(c *Client) Auth {
	conn := c.ActiveConnection()
	return &auth{
		conn:   c.ActiveConnection(),
		remote: pb.NewAuthClient(conn),
		c:      c,
	}
}

func (auth *auth) AuthEnable(ctx context.Context) (*AuthEnableResponse, error) {
	resp, err := auth.remote.AuthEnable(ctx, &pb.AuthEnableRequest{})
	return (*AuthEnableResponse)(resp), err
}
