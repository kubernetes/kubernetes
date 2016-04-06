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

package v3rpc

import (
	"github.com/coreos/etcd/etcdserver"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"golang.org/x/net/context"
)

type AuthServer struct {
	authenticator etcdserver.Authenticator
}

func NewAuthServer(s *etcdserver.EtcdServer) *AuthServer {
	return &AuthServer{authenticator: s}
}

func (as *AuthServer) AuthEnable(ctx context.Context, r *pb.AuthEnableRequest) (*pb.AuthEnableResponse, error) {
	resp, err := as.authenticator.AuthEnable(ctx, r)
	if err != nil {
		return nil, togRPCError(err)
	}
	return resp, nil
}

func (as *AuthServer) AuthDisable(ctx context.Context, r *pb.AuthDisableRequest) (*pb.AuthDisableResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) Authenticate(ctx context.Context, r *pb.AuthenticateRequest) (*pb.AuthenticateResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) RoleAdd(ctx context.Context, r *pb.AuthRoleAddRequest) (*pb.AuthRoleAddResponse, error) {
	resp, err := as.authenticator.RoleAdd(ctx, r)
	if err != nil {
		return nil, togRPCError(err)
	}
	return resp, nil
}

func (as *AuthServer) RoleDelete(ctx context.Context, r *pb.AuthRoleDeleteRequest) (*pb.AuthRoleDeleteResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) RoleGet(ctx context.Context, r *pb.AuthRoleGetRequest) (*pb.AuthRoleGetResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) RoleRevoke(ctx context.Context, r *pb.AuthRoleRevokeRequest) (*pb.AuthRoleRevokeResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) RoleGrant(ctx context.Context, r *pb.AuthRoleGrantRequest) (*pb.AuthRoleGrantResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) UserAdd(ctx context.Context, r *pb.AuthUserAddRequest) (*pb.AuthUserAddResponse, error) {
	resp, err := as.authenticator.UserAdd(ctx, r)
	if err != nil {
		return nil, togRPCError(err)
	}
	return resp, nil
}

func (as *AuthServer) UserDelete(ctx context.Context, r *pb.AuthUserDeleteRequest) (*pb.AuthUserDeleteResponse, error) {
	resp, err := as.authenticator.UserDelete(ctx, r)
	if err != nil {
		return nil, togRPCError(err)
	}
	return resp, nil
}

func (as *AuthServer) UserGet(ctx context.Context, r *pb.AuthUserGetRequest) (*pb.AuthUserGetResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) UserGrant(ctx context.Context, r *pb.AuthUserGrantRequest) (*pb.AuthUserGrantResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) UserRevoke(ctx context.Context, r *pb.AuthUserRevokeRequest) (*pb.AuthUserRevokeResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) UserChangePassword(ctx context.Context, r *pb.AuthUserChangePasswordRequest) (*pb.AuthUserChangePasswordResponse, error) {
	resp, err := as.authenticator.UserChangePassword(ctx, r)
	if err != nil {
		return nil, togRPCError(err)
	}
	return resp, nil
}
