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
	"github.com/coreos/etcd/etcdserver/membership"
	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/mvcc"
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

func togRPCError(err error) error {
	switch err {
	case membership.ErrIDRemoved:
		return rpctypes.ErrGRPCMemberNotFound
	case membership.ErrIDNotFound:
		return rpctypes.ErrGRPCMemberNotFound
	case membership.ErrIDExists:
		return rpctypes.ErrGRPCMemberExist
	case membership.ErrPeerURLexists:
		return rpctypes.ErrGRPCPeerURLExist
	case etcdserver.ErrNotEnoughStartedMembers:
		return rpctypes.ErrMemberNotEnoughStarted

	case mvcc.ErrCompacted:
		return rpctypes.ErrGRPCCompacted
	case mvcc.ErrFutureRev:
		return rpctypes.ErrGRPCFutureRev
	case lease.ErrLeaseNotFound:
		return rpctypes.ErrGRPCLeaseNotFound
	case etcdserver.ErrRequestTooLarge:
		return rpctypes.ErrGRPCRequestTooLarge
	case etcdserver.ErrNoSpace:
		return rpctypes.ErrGRPCNoSpace
	case etcdserver.ErrTooManyRequests:
		return rpctypes.ErrTooManyRequests

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
	case etcdserver.ErrUnhealthy:
		return rpctypes.ErrGRPCUnhealthy

	case lease.ErrLeaseNotFound:
		return rpctypes.ErrGRPCLeaseNotFound
	case lease.ErrLeaseExists:
		return rpctypes.ErrGRPCLeaseExist

	case auth.ErrRootUserNotExist:
		return rpctypes.ErrGRPCRootUserNotExist
	case auth.ErrRootRoleNotExist:
		return rpctypes.ErrGRPCRootRoleNotExist
	case auth.ErrUserAlreadyExist:
		return rpctypes.ErrGRPCUserAlreadyExist
	case auth.ErrUserEmpty:
		return rpctypes.ErrGRPCUserEmpty
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
	case auth.ErrAuthNotEnabled:
		return rpctypes.ErrGRPCAuthNotEnabled
	case auth.ErrInvalidAuthToken:
		return rpctypes.ErrGRPCInvalidAuthToken
	default:
		return grpc.Errorf(codes.Unknown, err.Error())
	}
}
