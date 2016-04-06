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
	AuthEnableResponse             pb.AuthEnableResponse
	AuthUserAddResponse            pb.AuthUserAddResponse
	AuthUserDeleteResponse         pb.AuthUserDeleteResponse
	AuthUserChangePasswordResponse pb.AuthUserChangePasswordResponse
	AuthRoleAddResponse            pb.AuthRoleAddResponse
)

type Auth interface {
	// AuthEnable enables auth of an etcd cluster.
	AuthEnable(ctx context.Context) (*AuthEnableResponse, error)

	// UserAdd adds a new user to an etcd cluster.
	UserAdd(ctx context.Context, name string, password string) (*AuthUserAddResponse, error)

	// UserDelete deletes a user from an etcd cluster.
	UserDelete(ctx context.Context, name string) (*AuthUserDeleteResponse, error)

	// UserChangePassword changes a password of a user.
	UserChangePassword(ctx context.Context, name string, password string) (*AuthUserChangePasswordResponse, error)

	// RoleAdd adds a new user to an etcd cluster.
	RoleAdd(ctx context.Context, name string) (*AuthRoleAddResponse, error)
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

func (auth *auth) UserAdd(ctx context.Context, name string, password string) (*AuthUserAddResponse, error) {
	resp, err := auth.remote.UserAdd(ctx, &pb.AuthUserAddRequest{Name: name, Password: password})
	return (*AuthUserAddResponse)(resp), err
}

func (auth *auth) UserDelete(ctx context.Context, name string) (*AuthUserDeleteResponse, error) {
	resp, err := auth.remote.UserDelete(ctx, &pb.AuthUserDeleteRequest{Name: name})
	return (*AuthUserDeleteResponse)(resp), err
}

func (auth *auth) UserChangePassword(ctx context.Context, name string, password string) (*AuthUserChangePasswordResponse, error) {
	resp, err := auth.remote.UserChangePassword(ctx, &pb.AuthUserChangePasswordRequest{Name: name, Password: password})
	return (*AuthUserChangePasswordResponse)(resp), err
}

func (auth *auth) RoleAdd(ctx context.Context, name string) (*AuthRoleAddResponse, error) {
	resp, err := auth.remote.RoleAdd(ctx, &pb.AuthRoleAddRequest{Name: name})
	return (*AuthRoleAddResponse)(resp), err
}
