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
	"context"
	"strings"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	"go.etcd.io/etcd/server/v3/auth"
	"go.etcd.io/etcd/server/v3/etcdserver"
	"go.etcd.io/etcd/server/v3/etcdserver/api/membership"
	"go.etcd.io/etcd/server/v3/lease"
	"go.etcd.io/etcd/server/v3/mvcc"

	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

var toGRPCErrorMap = map[error]error{
	membership.ErrIDRemoved:               rpctypes.ErrGRPCMemberNotFound,
	membership.ErrIDNotFound:              rpctypes.ErrGRPCMemberNotFound,
	membership.ErrIDExists:                rpctypes.ErrGRPCMemberExist,
	membership.ErrPeerURLexists:           rpctypes.ErrGRPCPeerURLExist,
	membership.ErrMemberNotLearner:        rpctypes.ErrGRPCMemberNotLearner,
	membership.ErrTooManyLearners:         rpctypes.ErrGRPCTooManyLearners,
	etcdserver.ErrNotEnoughStartedMembers: rpctypes.ErrMemberNotEnoughStarted,
	etcdserver.ErrLearnerNotReady:         rpctypes.ErrGRPCLearnerNotReady,

	mvcc.ErrCompacted:             rpctypes.ErrGRPCCompacted,
	mvcc.ErrFutureRev:             rpctypes.ErrGRPCFutureRev,
	etcdserver.ErrRequestTooLarge: rpctypes.ErrGRPCRequestTooLarge,
	etcdserver.ErrNoSpace:         rpctypes.ErrGRPCNoSpace,
	etcdserver.ErrTooManyRequests: rpctypes.ErrTooManyRequests,

	etcdserver.ErrNoLeader:                   rpctypes.ErrGRPCNoLeader,
	etcdserver.ErrNotLeader:                  rpctypes.ErrGRPCNotLeader,
	etcdserver.ErrLeaderChanged:              rpctypes.ErrGRPCLeaderChanged,
	etcdserver.ErrStopped:                    rpctypes.ErrGRPCStopped,
	etcdserver.ErrTimeout:                    rpctypes.ErrGRPCTimeout,
	etcdserver.ErrTimeoutDueToLeaderFail:     rpctypes.ErrGRPCTimeoutDueToLeaderFail,
	etcdserver.ErrTimeoutDueToConnectionLost: rpctypes.ErrGRPCTimeoutDueToConnectionLost,
	etcdserver.ErrUnhealthy:                  rpctypes.ErrGRPCUnhealthy,
	etcdserver.ErrKeyNotFound:                rpctypes.ErrGRPCKeyNotFound,
	etcdserver.ErrCorrupt:                    rpctypes.ErrGRPCCorrupt,
	etcdserver.ErrBadLeaderTransferee:        rpctypes.ErrGRPCBadLeaderTransferee,

	etcdserver.ErrClusterVersionUnavailable:     rpctypes.ErrGRPCClusterVersionUnavailable,
	etcdserver.ErrWrongDowngradeVersionFormat:   rpctypes.ErrGRPCWrongDowngradeVersionFormat,
	etcdserver.ErrInvalidDowngradeTargetVersion: rpctypes.ErrGRPCInvalidDowngradeTargetVersion,
	etcdserver.ErrDowngradeInProcess:            rpctypes.ErrGRPCDowngradeInProcess,
	etcdserver.ErrNoInflightDowngrade:           rpctypes.ErrGRPCNoInflightDowngrade,

	lease.ErrLeaseNotFound:    rpctypes.ErrGRPCLeaseNotFound,
	lease.ErrLeaseExists:      rpctypes.ErrGRPCLeaseExist,
	lease.ErrLeaseTTLTooLarge: rpctypes.ErrGRPCLeaseTTLTooLarge,

	auth.ErrRootUserNotExist:     rpctypes.ErrGRPCRootUserNotExist,
	auth.ErrRootRoleNotExist:     rpctypes.ErrGRPCRootRoleNotExist,
	auth.ErrUserAlreadyExist:     rpctypes.ErrGRPCUserAlreadyExist,
	auth.ErrUserEmpty:            rpctypes.ErrGRPCUserEmpty,
	auth.ErrUserNotFound:         rpctypes.ErrGRPCUserNotFound,
	auth.ErrRoleAlreadyExist:     rpctypes.ErrGRPCRoleAlreadyExist,
	auth.ErrRoleNotFound:         rpctypes.ErrGRPCRoleNotFound,
	auth.ErrRoleEmpty:            rpctypes.ErrGRPCRoleEmpty,
	auth.ErrAuthFailed:           rpctypes.ErrGRPCAuthFailed,
	auth.ErrPermissionDenied:     rpctypes.ErrGRPCPermissionDenied,
	auth.ErrRoleNotGranted:       rpctypes.ErrGRPCRoleNotGranted,
	auth.ErrPermissionNotGranted: rpctypes.ErrGRPCPermissionNotGranted,
	auth.ErrAuthNotEnabled:       rpctypes.ErrGRPCAuthNotEnabled,
	auth.ErrInvalidAuthToken:     rpctypes.ErrGRPCInvalidAuthToken,
	auth.ErrInvalidAuthMgmt:      rpctypes.ErrGRPCInvalidAuthMgmt,

	// In sync with status.FromContextError
	context.Canceled:         rpctypes.ErrGRPCCanceled,
	context.DeadlineExceeded: rpctypes.ErrGRPCDeadlineExceeded,
}

func togRPCError(err error) error {
	// let gRPC server convert to codes.Canceled, codes.DeadlineExceeded
	if err == context.Canceled || err == context.DeadlineExceeded {
		return err
	}
	grpcErr, ok := toGRPCErrorMap[err]
	if !ok {
		return status.Error(codes.Unknown, err.Error())
	}
	return grpcErr
}

func isClientCtxErr(ctxErr error, err error) bool {
	if ctxErr != nil {
		return true
	}

	ev, ok := status.FromError(err)
	if !ok {
		return false
	}

	switch ev.Code() {
	case codes.Canceled, codes.DeadlineExceeded:
		// client-side context cancel or deadline exceeded
		// "rpc error: code = Canceled desc = context canceled"
		// "rpc error: code = DeadlineExceeded desc = context deadline exceeded"
		return true
	case codes.Unavailable:
		msg := ev.Message()
		// client-side context cancel or deadline exceeded with TLS ("http2.errClientDisconnected")
		// "rpc error: code = Unavailable desc = client disconnected"
		if msg == "client disconnected" {
			return true
		}
		// "grpc/transport.ClientTransport.CloseStream" on canceled streams
		// "rpc error: code = Unavailable desc = stream error: stream ID 21; CANCEL")
		if strings.HasPrefix(msg, "stream error: ") && strings.HasSuffix(msg, "; CANCEL") {
			return true
		}
	}
	return false
}

// in v3.4, learner is allowed to serve serializable read and endpoint status
func isRPCSupportedForLearner(req interface{}) bool {
	switch r := req.(type) {
	case *pb.StatusRequest:
		return true
	case *pb.RangeRequest:
		return r.Serializable
	default:
		return false
	}
}
