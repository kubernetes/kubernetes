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
	"github.com/coreos/etcd/auth"
	"github.com/coreos/etcd/etcdserver"
	"github.com/coreos/etcd/etcdserver/api/v3rpc/rpctypes"
	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/mvcc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

func togRPCError(err error) error {
	switch err {
	case mvcc.ErrCompacted:
		return rpctypes.ErrGRPCCompacted
	case mvcc.ErrFutureRev:
		return rpctypes.ErrGRPCFutureRev
	case lease.ErrLeaseNotFound:
		return rpctypes.ErrGRPCLeaseNotFound
	// TODO: handle error from raft and timeout
	case etcdserver.ErrRequestTooLarge:
		return rpctypes.ErrGRPCRequestTooLarge
	case etcdserver.ErrNoSpace:
		return rpctypes.ErrGRPCNoSpace

	case etcdserver.ErrNoLeader:
		return rpctypes.ErrGRPCNoLeader
	case etcdserver.ErrStopped:
		return rpctypes.ErrGRPCStopped
	case etcdserver.ErrTimeout:
		return rpctypes.ErrGRPCTimeout
	case etcdserver.ErrTimeoutDueToLeaderFail:
		return rpctypes.ErrGRPCTimeoutDueToLeaderFail
	case etcdserver.ErrTimeoutDueToConnectionLost:
		return rpctypes.ErrGRPCTimeoutDueToConnectionLost

	case auth.ErrRootUserNotExist:
		return rpctypes.ErrGRPCRootUserNotExist
	case auth.ErrRootRoleNotExist:
		return rpctypes.ErrGRPCRootRoleNotExist
	case auth.ErrUserAlreadyExist:
		return rpctypes.ErrGRPCUserAlreadyExist
	case auth.ErrUserNotFound:
		return rpctypes.ErrGRPCUserNotFound
	case auth.ErrRoleAlreadyExist:
		return rpctypes.ErrGRPCRoleAlreadyExist
	case auth.ErrRoleNotFound:
		return rpctypes.ErrGRPCRoleNotFound
	case auth.ErrAuthFailed:
		return rpctypes.ErrGRPCAuthFailed
	case auth.ErrPermissionDenied:
		return rpctypes.ErrGRPCPermissionDenied
	case auth.ErrRoleNotGranted:
		return rpctypes.ErrGRPCRoleNotGranted
	case auth.ErrPermissionNotGranted:
		return rpctypes.ErrGRPCPermissionNotGranted
	default:
		return grpc.Errorf(codes.Internal, err.Error())
	}
}
