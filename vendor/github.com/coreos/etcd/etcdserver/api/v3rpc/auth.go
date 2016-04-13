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
	return as.authenticator.AuthEnable(ctx, r)
}

func (as *AuthServer) AuthDisable(ctx context.Context, r *pb.AuthDisableRequest) (*pb.AuthDisableResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) Authenticate(ctx context.Context, r *pb.AuthenticateRequest) (*pb.AuthenticateResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) RoleAdd(ctx context.Context, r *pb.RoleAddRequest) (*pb.RoleAddResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) RoleDelete(ctx context.Context, r *pb.RoleDeleteRequest) (*pb.RoleDeleteResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) RoleGet(ctx context.Context, r *pb.RoleGetRequest) (*pb.RoleGetResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) RoleRevoke(ctx context.Context, r *pb.RoleRevokeRequest) (*pb.RoleRevokeResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) RoleGrant(ctx context.Context, r *pb.RoleGrantRequest) (*pb.RoleGrantResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) UserAdd(ctx context.Context, r *pb.UserAddRequest) (*pb.UserAddResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) UserDelete(ctx context.Context, r *pb.UserDeleteRequest) (*pb.UserDeleteResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) UserGet(ctx context.Context, r *pb.UserGetRequest) (*pb.UserGetResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) UserGrant(ctx context.Context, r *pb.UserGrantRequest) (*pb.UserGrantResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) UserRevoke(ctx context.Context, r *pb.UserRevokeRequest) (*pb.UserRevokeResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}

func (as *AuthServer) UserChangePassword(ctx context.Context, r *pb.UserChangePasswordRequest) (*pb.UserChangePasswordResponse, error) {
	plog.Info("not implemented yet")
	return nil, nil
}
