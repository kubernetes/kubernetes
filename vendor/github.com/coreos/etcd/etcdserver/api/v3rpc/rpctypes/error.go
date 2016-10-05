// Copyright 2015 The etcd Authors
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

package rpctypes

import (
	"google.golang.org/grpc"
	"google.golang.org/grpc/codes"
)

var (
	// server-side error
	ErrGRPCEmptyKey     = grpc.Errorf(codes.InvalidArgument, "etcdserver: key is not provided")
	ErrGRPCTooManyOps   = grpc.Errorf(codes.InvalidArgument, "etcdserver: too many operations in txn request")
	ErrGRPCDuplicateKey = grpc.Errorf(codes.InvalidArgument, "etcdserver: duplicate key given in txn request")
	ErrGRPCCompacted    = grpc.Errorf(codes.OutOfRange, "etcdserver: mvcc: required revision has been compacted")
	ErrGRPCFutureRev    = grpc.Errorf(codes.OutOfRange, "etcdserver: mvcc: required revision is a future revision")
	ErrGRPCNoSpace      = grpc.Errorf(codes.ResourceExhausted, "etcdserver: mvcc: database space exceeded")

	ErrGRPCLeaseNotFound = grpc.Errorf(codes.NotFound, "etcdserver: requested lease not found")
	ErrGRPCLeaseExist    = grpc.Errorf(codes.FailedPrecondition, "etcdserver: lease already exists")

	ErrGRPCMemberExist    = grpc.Errorf(codes.FailedPrecondition, "etcdserver: member ID already exist")
	ErrGRPCPeerURLExist   = grpc.Errorf(codes.FailedPrecondition, "etcdserver: Peer URLs already exists")
	ErrGRPCMemberBadURLs  = grpc.Errorf(codes.InvalidArgument, "etcdserver: given member URLs are invalid")
	ErrGRPCMemberNotFound = grpc.Errorf(codes.NotFound, "etcdserver: member not found")

	ErrGRPCRequestTooLarge        = grpc.Errorf(codes.InvalidArgument, "etcdserver: request is too large")
	ErrGRPCRequestTooManyRequests = grpc.Errorf(codes.ResourceExhausted, "etcdserver: too many requests")

	ErrGRPCRootUserNotExist     = grpc.Errorf(codes.FailedPrecondition, "etcdserver: root user does not exist")
	ErrGRPCRootRoleNotExist     = grpc.Errorf(codes.FailedPrecondition, "etcdserver: root user does not have root role")
	ErrGRPCUserAlreadyExist     = grpc.Errorf(codes.FailedPrecondition, "etcdserver: user name already exists")
	ErrGRPCUserNotFound         = grpc.Errorf(codes.FailedPrecondition, "etcdserver: user name not found")
	ErrGRPCRoleAlreadyExist     = grpc.Errorf(codes.FailedPrecondition, "etcdserver: role name already exists")
	ErrGRPCRoleNotFound         = grpc.Errorf(codes.FailedPrecondition, "etcdserver: role name not found")
	ErrGRPCAuthFailed           = grpc.Errorf(codes.InvalidArgument, "etcdserver: authentication failed, invalid user ID or password")
	ErrGRPCPermissionDenied     = grpc.Errorf(codes.PermissionDenied, "etcdserver: permission denied")
	ErrGRPCRoleNotGranted       = grpc.Errorf(codes.FailedPrecondition, "etcdserver: role is not granted to the user")
	ErrGRPCPermissionNotGranted = grpc.Errorf(codes.FailedPrecondition, "etcdserver: permission is not granted to the role")
	ErrGRPCAuthNotEnabled       = grpc.Errorf(codes.FailedPrecondition, "etcdserver: authentication is not enabled")

	ErrGRPCNoLeader               = grpc.Errorf(codes.Unavailable, "etcdserver: no leader")
	ErrGRPCNotCapable             = grpc.Errorf(codes.Unavailable, "etcdserver: not capable")
	ErrGRPCStopped                = grpc.Errorf(codes.Unavailable, "etcdserver: server stopped")
	ErrGRPCTimeout                = grpc.Errorf(codes.Unavailable, "etcdserver: request timed out")
	ErrGRPCTimeoutDueToLeaderFail = grpc.Errorf(codes.Unavailable, "etcdserver: request timed out, possibly due to previous leader failure")
	ErrGRPCUnhealthy              = grpc.Errorf(codes.Unavailable, "etcdserver: unhealthy cluster")

	errStringToError = map[string]error{
		grpc.ErrorDesc(ErrGRPCEmptyKey):     ErrGRPCEmptyKey,
		grpc.ErrorDesc(ErrGRPCTooManyOps):   ErrGRPCTooManyOps,
		grpc.ErrorDesc(ErrGRPCDuplicateKey): ErrGRPCDuplicateKey,
		grpc.ErrorDesc(ErrGRPCCompacted):    ErrGRPCCompacted,
		grpc.ErrorDesc(ErrGRPCFutureRev):    ErrGRPCFutureRev,
		grpc.ErrorDesc(ErrGRPCNoSpace):      ErrGRPCNoSpace,

		grpc.ErrorDesc(ErrGRPCLeaseNotFound): ErrGRPCLeaseNotFound,
		grpc.ErrorDesc(ErrGRPCLeaseExist):    ErrGRPCLeaseExist,

		grpc.ErrorDesc(ErrGRPCMemberExist):    ErrGRPCMemberExist,
		grpc.ErrorDesc(ErrGRPCPeerURLExist):   ErrGRPCPeerURLExist,
		grpc.ErrorDesc(ErrGRPCMemberBadURLs):  ErrGRPCMemberBadURLs,
		grpc.ErrorDesc(ErrGRPCMemberNotFound): ErrGRPCMemberNotFound,

		grpc.ErrorDesc(ErrGRPCRequestTooLarge):        ErrGRPCRequestTooLarge,
		grpc.ErrorDesc(ErrGRPCRequestTooManyRequests): ErrGRPCRequestTooManyRequests,

		grpc.ErrorDesc(ErrGRPCRootUserNotExist):     ErrGRPCRootUserNotExist,
		grpc.ErrorDesc(ErrGRPCRootRoleNotExist):     ErrGRPCRootRoleNotExist,
		grpc.ErrorDesc(ErrGRPCUserAlreadyExist):     ErrGRPCUserAlreadyExist,
		grpc.ErrorDesc(ErrGRPCUserNotFound):         ErrGRPCUserNotFound,
		grpc.ErrorDesc(ErrGRPCRoleAlreadyExist):     ErrGRPCRoleAlreadyExist,
		grpc.ErrorDesc(ErrGRPCRoleNotFound):         ErrGRPCRoleNotFound,
		grpc.ErrorDesc(ErrGRPCAuthFailed):           ErrGRPCAuthFailed,
		grpc.ErrorDesc(ErrGRPCPermissionDenied):     ErrGRPCPermissionDenied,
		grpc.ErrorDesc(ErrGRPCRoleNotGranted):       ErrGRPCRoleNotGranted,
		grpc.ErrorDesc(ErrGRPCPermissionNotGranted): ErrGRPCPermissionNotGranted,
		grpc.ErrorDesc(ErrGRPCAuthNotEnabled):       ErrGRPCAuthNotEnabled,

		grpc.ErrorDesc(ErrGRPCNoLeader):               ErrGRPCNoLeader,
		grpc.ErrorDesc(ErrGRPCNotCapable):             ErrGRPCNotCapable,
		grpc.ErrorDesc(ErrGRPCStopped):                ErrGRPCStopped,
		grpc.ErrorDesc(ErrGRPCTimeout):                ErrGRPCTimeout,
		grpc.ErrorDesc(ErrGRPCTimeoutDueToLeaderFail): ErrGRPCTimeoutDueToLeaderFail,
		grpc.ErrorDesc(ErrGRPCUnhealthy):              ErrGRPCUnhealthy,
	}

	// client-side error
	ErrEmptyKey     = Error(ErrGRPCEmptyKey)
	ErrTooManyOps   = Error(ErrGRPCTooManyOps)
	ErrDuplicateKey = Error(ErrGRPCDuplicateKey)
	ErrCompacted    = Error(ErrGRPCCompacted)
	ErrFutureRev    = Error(ErrGRPCFutureRev)
	ErrNoSpace      = Error(ErrGRPCNoSpace)

	ErrLeaseNotFound = Error(ErrGRPCLeaseNotFound)
	ErrLeaseExist    = Error(ErrGRPCLeaseExist)

	ErrMemberExist    = Error(ErrGRPCMemberExist)
	ErrPeerURLExist   = Error(ErrGRPCPeerURLExist)
	ErrMemberBadURLs  = Error(ErrGRPCMemberBadURLs)
	ErrMemberNotFound = Error(ErrGRPCMemberNotFound)

	ErrRequestTooLarge = Error(ErrGRPCRequestTooLarge)
	ErrTooManyRequests = Error(ErrGRPCRequestTooManyRequests)

	ErrRootUserNotExist     = Error(ErrGRPCRootUserNotExist)
	ErrRootRoleNotExist     = Error(ErrGRPCRootRoleNotExist)
	ErrUserAlreadyExist     = Error(ErrGRPCUserAlreadyExist)
	ErrUserNotFound         = Error(ErrGRPCUserNotFound)
	ErrRoleAlreadyExist     = Error(ErrGRPCRoleAlreadyExist)
	ErrRoleNotFound         = Error(ErrGRPCRoleNotFound)
	ErrAuthFailed           = Error(ErrGRPCAuthFailed)
	ErrPermissionDenied     = Error(ErrGRPCPermissionDenied)
	ErrRoleNotGranted       = Error(ErrGRPCRoleNotGranted)
	ErrPermissionNotGranted = Error(ErrGRPCPermissionNotGranted)
	ErrAuthNotEnabled       = Error(ErrGRPCAuthNotEnabled)

	ErrNoLeader               = Error(ErrGRPCNoLeader)
	ErrNotCapable             = Error(ErrGRPCNotCapable)
	ErrStopped                = Error(ErrGRPCStopped)
	ErrTimeout                = Error(ErrGRPCTimeout)
	ErrTimeoutDueToLeaderFail = Error(ErrGRPCTimeoutDueToLeaderFail)
	ErrUnhealthy              = Error(ErrGRPCUnhealthy)
)

// EtcdError defines gRPC server errors.
// (https://github.com/grpc/grpc-go/blob/master/rpc_util.go#L319-L323)
type EtcdError struct {
	code codes.Code
	desc string
}

// Code returns grpc/codes.Code.
// TODO: define clientv3/codes.Code.
func (e EtcdError) Code() codes.Code {
	return e.code
}

func (e EtcdError) Error() string {
	return e.desc
}

func Error(err error) error {
	if err == nil {
		return nil
	}
	verr, ok := errStringToError[grpc.ErrorDesc(err)]
	if !ok { // not gRPC error
		return err
	}
	return EtcdError{code: grpc.Code(verr), desc: grpc.ErrorDesc(verr)}
}
