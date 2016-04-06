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
	"crypto/tls"

	"github.com/coreos/etcd/etcdserver"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

func Server(s *etcdserver.EtcdServer, tls *tls.Config) *grpc.Server {
	var opts []grpc.ServerOption
	if tls != nil {
		opts = append(opts, grpc.Creds(credentials.NewTLS(tls)))
	}

	grpcServer := grpc.NewServer(opts...)
	pb.RegisterKVServer(grpcServer, NewQuotaKVServer(s))
	pb.RegisterWatchServer(grpcServer, NewWatchServer(s))
	pb.RegisterLeaseServer(grpcServer, NewQuotaLeaseServer(s))
	pb.RegisterClusterServer(grpcServer, NewClusterServer(s))
	pb.RegisterAuthServer(grpcServer, NewAuthServer(s))
	pb.RegisterMaintenanceServer(grpcServer, NewMaintenanceServer(s))
	return grpcServer
}
