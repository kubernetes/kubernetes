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
	"github.com/coreos/etcd/auth"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/storage"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

func togRPCError(err error) error {
	switch err {
	case storage.ErrCompacted:
		return rpctypes.ErrCompacted
	case storage.ErrFutureRev:
		return rpctypes.ErrFutureRev
	case lease.ErrLeaseNotFound:
		return rpctypes.ErrLeaseNotFound
	// TODO: handle error from raft and timeout
	case etcdserver.ErrRequestTooLarge:
		return rpctypes.ErrRequestTooLarge
	case etcdserver.ErrNoSpace:
		return rpctypes.ErrNoSpace
	case auth.ErrUserAlreadyExist:
		return rpctypes.ErrUserAlreadyExist
	case auth.ErrUserNotFound:
		return rpctypes.ErrUserNotFound
	case auth.ErrRoleAlreadyExist:
		return rpctypes.ErrRoleAlreadyExist
	default:
		return grpc.Errorf(codes.Internal, err.Error())
	}
}
