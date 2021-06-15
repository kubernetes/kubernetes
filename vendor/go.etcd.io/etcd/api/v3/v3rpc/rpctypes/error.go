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
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// server-side error
var (
	ErrGRPCEmptyKey      = status.New(codes.InvalidArgument, "etcdserver: key is not provided").Err()
	ErrGRPCKeyNotFound   = status.New(codes.InvalidArgument, "etcdserver: key not found").Err()
	ErrGRPCValueProvided = status.New(codes.InvalidArgument, "etcdserver: value is provided").Err()
	ErrGRPCLeaseProvided = status.New(codes.InvalidArgument, "etcdserver: lease is provided").Err()
	ErrGRPCTooManyOps    = status.New(codes.InvalidArgument, "etcdserver: too many operations in txn request").Err()
	ErrGRPCDuplicateKey  = status.New(codes.InvalidArgument, "etcdserver: duplicate key given in txn request").Err()
	ErrGRPCCompacted     = status.New(codes.OutOfRange, "etcdserver: mvcc: required revision has been compacted").Err()
	ErrGRPCFutureRev     = status.New(codes.OutOfRange, "etcdserver: mvcc: required revision is a future revision").Err()
	ErrGRPCNoSpace       = status.New(codes.ResourceExhausted, "etcdserver: mvcc: database space exceeded").Err()

	ErrGRPCLeaseNotFound    = status.New(codes.NotFound, "etcdserver: requested lease not found").Err()
	ErrGRPCLeaseExist       = status.New(codes.FailedPrecondition, "etcdserver: lease already exists").Err()
	ErrGRPCLeaseTTLTooLarge = status.New(codes.OutOfRange, "etcdserver: too large lease TTL").Err()

	ErrGRPCWatchCanceled = status.New(codes.Canceled, "etcdserver: watch canceled").Err()

	ErrGRPCMemberExist            = status.New(codes.FailedPrecondition, "etcdserver: member ID already exist").Err()
	ErrGRPCPeerURLExist           = status.New(codes.FailedPrecondition, "etcdserver: Peer URLs already exists").Err()
	ErrGRPCMemberNotEnoughStarted = status.New(codes.FailedPrecondition, "etcdserver: re-configuration failed due to not enough started members").Err()
	ErrGRPCMemberBadURLs          = status.New(codes.InvalidArgument, "etcdserver: given member URLs are invalid").Err()
	ErrGRPCMemberNotFound         = status.New(codes.NotFound, "etcdserver: member not found").Err()
	ErrGRPCMemberNotLearner       = status.New(codes.FailedPrecondition, "etcdserver: can only promote a learner member").Err()
	ErrGRPCLearnerNotReady        = status.New(codes.FailedPrecondition, "etcdserver: can only promote a learner member which is in sync with leader").Err()
	ErrGRPCTooManyLearners        = status.New(codes.FailedPrecondition, "etcdserver: too many learner members in cluster").Err()

	ErrGRPCRequestTooLarge        = status.New(codes.InvalidArgument, "etcdserver: request is too large").Err()
	ErrGRPCRequestTooManyRequests = status.New(codes.ResourceExhausted, "etcdserver: too many requests").Err()

	ErrGRPCRootUserNotExist     = status.New(codes.FailedPrecondition, "etcdserver: root user does not exist").Err()
	ErrGRPCRootRoleNotExist     = status.New(codes.FailedPrecondition, "etcdserver: root user does not have root role").Err()
	ErrGRPCUserAlreadyExist     = status.New(codes.FailedPrecondition, "etcdserver: user name already exists").Err()
	ErrGRPCUserEmpty            = status.New(codes.InvalidArgument, "etcdserver: user name is empty").Err()
	ErrGRPCUserNotFound         = status.New(codes.FailedPrecondition, "etcdserver: user name not found").Err()
	ErrGRPCRoleAlreadyExist     = status.New(codes.FailedPrecondition, "etcdserver: role name already exists").Err()
	ErrGRPCRoleNotFound         = status.New(codes.FailedPrecondition, "etcdserver: role name not found").Err()
	ErrGRPCRoleEmpty            = status.New(codes.InvalidArgument, "etcdserver: role name is empty").Err()
	ErrGRPCAuthFailed           = status.New(codes.InvalidArgument, "etcdserver: authentication failed, invalid user ID or password").Err()
	ErrGRPCPermissionDenied     = status.New(codes.PermissionDenied, "etcdserver: permission denied").Err()
	ErrGRPCRoleNotGranted       = status.New(codes.FailedPrecondition, "etcdserver: role is not granted to the user").Err()
	ErrGRPCPermissionNotGranted = status.New(codes.FailedPrecondition, "etcdserver: permission is not granted to the role").Err()
	ErrGRPCAuthNotEnabled       = status.New(codes.FailedPrecondition, "etcdserver: authentication is not enabled").Err()
	ErrGRPCInvalidAuthToken     = status.New(codes.Unauthenticated, "etcdserver: invalid auth token").Err()
	ErrGRPCInvalidAuthMgmt      = status.New(codes.InvalidArgument, "etcdserver: invalid auth management").Err()

	ErrGRPCNoLeader                   = status.New(codes.Unavailable, "etcdserver: no leader").Err()
	ErrGRPCNotLeader                  = status.New(codes.FailedPrecondition, "etcdserver: not leader").Err()
	ErrGRPCLeaderChanged              = status.New(codes.Unavailable, "etcdserver: leader changed").Err()
	ErrGRPCNotCapable                 = status.New(codes.Unavailable, "etcdserver: not capable").Err()
	ErrGRPCStopped                    = status.New(codes.Unavailable, "etcdserver: server stopped").Err()
	ErrGRPCTimeout                    = status.New(codes.Unavailable, "etcdserver: request timed out").Err()
	ErrGRPCTimeoutDueToLeaderFail     = status.New(codes.Unavailable, "etcdserver: request timed out, possibly due to previous leader failure").Err()
	ErrGRPCTimeoutDueToConnectionLost = status.New(codes.Unavailable, "etcdserver: request timed out, possibly due to connection lost").Err()
	ErrGRPCUnhealthy                  = status.New(codes.Unavailable, "etcdserver: unhealthy cluster").Err()
	ErrGRPCCorrupt                    = status.New(codes.DataLoss, "etcdserver: corrupt cluster").Err()
	ErrGPRCNotSupportedForLearner     = status.New(codes.Unavailable, "etcdserver: rpc not supported for learner").Err()
	ErrGRPCBadLeaderTransferee        = status.New(codes.FailedPrecondition, "etcdserver: bad leader transferee").Err()

	ErrGRPCClusterVersionUnavailable     = status.New(codes.Unavailable, "etcdserver: cluster version not found during downgrade").Err()
	ErrGRPCWrongDowngradeVersionFormat   = status.New(codes.InvalidArgument, "etcdserver: wrong downgrade target version format").Err()
	ErrGRPCInvalidDowngradeTargetVersion = status.New(codes.InvalidArgument, "etcdserver: invalid downgrade target version").Err()
	ErrGRPCDowngradeInProcess            = status.New(codes.FailedPrecondition, "etcdserver: cluster has a downgrade job in progress").Err()
	ErrGRPCNoInflightDowngrade           = status.New(codes.FailedPrecondition, "etcdserver: no inflight downgrade job").Err()

	ErrGRPCCanceled         = status.New(codes.Canceled, "etcdserver: request canceled").Err()
	ErrGRPCDeadlineExceeded = status.New(codes.DeadlineExceeded, "etcdserver: context deadline exceeded").Err()

	errStringToError = map[string]error{
		ErrorDesc(ErrGRPCEmptyKey):      ErrGRPCEmptyKey,
		ErrorDesc(ErrGRPCKeyNotFound):   ErrGRPCKeyNotFound,
		ErrorDesc(ErrGRPCValueProvided): ErrGRPCValueProvided,
		ErrorDesc(ErrGRPCLeaseProvided): ErrGRPCLeaseProvided,

		ErrorDesc(ErrGRPCTooManyOps):   ErrGRPCTooManyOps,
		ErrorDesc(ErrGRPCDuplicateKey): ErrGRPCDuplicateKey,
		ErrorDesc(ErrGRPCCompacted):    ErrGRPCCompacted,
		ErrorDesc(ErrGRPCFutureRev):    ErrGRPCFutureRev,
		ErrorDesc(ErrGRPCNoSpace):      ErrGRPCNoSpace,

		ErrorDesc(ErrGRPCLeaseNotFound):    ErrGRPCLeaseNotFound,
		ErrorDesc(ErrGRPCLeaseExist):       ErrGRPCLeaseExist,
		ErrorDesc(ErrGRPCLeaseTTLTooLarge): ErrGRPCLeaseTTLTooLarge,

		ErrorDesc(ErrGRPCMemberExist):            ErrGRPCMemberExist,
		ErrorDesc(ErrGRPCPeerURLExist):           ErrGRPCPeerURLExist,
		ErrorDesc(ErrGRPCMemberNotEnoughStarted): ErrGRPCMemberNotEnoughStarted,
		ErrorDesc(ErrGRPCMemberBadURLs):          ErrGRPCMemberBadURLs,
		ErrorDesc(ErrGRPCMemberNotFound):         ErrGRPCMemberNotFound,
		ErrorDesc(ErrGRPCMemberNotLearner):       ErrGRPCMemberNotLearner,
		ErrorDesc(ErrGRPCLearnerNotReady):        ErrGRPCLearnerNotReady,
		ErrorDesc(ErrGRPCTooManyLearners):        ErrGRPCTooManyLearners,

		ErrorDesc(ErrGRPCRequestTooLarge):        ErrGRPCRequestTooLarge,
		ErrorDesc(ErrGRPCRequestTooManyRequests): ErrGRPCRequestTooManyRequests,

		ErrorDesc(ErrGRPCRootUserNotExist):     ErrGRPCRootUserNotExist,
		ErrorDesc(ErrGRPCRootRoleNotExist):     ErrGRPCRootRoleNotExist,
		ErrorDesc(ErrGRPCUserAlreadyExist):     ErrGRPCUserAlreadyExist,
		ErrorDesc(ErrGRPCUserEmpty):            ErrGRPCUserEmpty,
		ErrorDesc(ErrGRPCUserNotFound):         ErrGRPCUserNotFound,
		ErrorDesc(ErrGRPCRoleAlreadyExist):     ErrGRPCRoleAlreadyExist,
		ErrorDesc(ErrGRPCRoleNotFound):         ErrGRPCRoleNotFound,
		ErrorDesc(ErrGRPCRoleEmpty):            ErrGRPCRoleEmpty,
		ErrorDesc(ErrGRPCAuthFailed):           ErrGRPCAuthFailed,
		ErrorDesc(ErrGRPCPermissionDenied):     ErrGRPCPermissionDenied,
		ErrorDesc(ErrGRPCRoleNotGranted):       ErrGRPCRoleNotGranted,
		ErrorDesc(ErrGRPCPermissionNotGranted): ErrGRPCPermissionNotGranted,
		ErrorDesc(ErrGRPCAuthNotEnabled):       ErrGRPCAuthNotEnabled,
		ErrorDesc(ErrGRPCInvalidAuthToken):     ErrGRPCInvalidAuthToken,
		ErrorDesc(ErrGRPCInvalidAuthMgmt):      ErrGRPCInvalidAuthMgmt,

		ErrorDesc(ErrGRPCNoLeader):                   ErrGRPCNoLeader,
		ErrorDesc(ErrGRPCNotLeader):                  ErrGRPCNotLeader,
		ErrorDesc(ErrGRPCLeaderChanged):              ErrGRPCLeaderChanged,
		ErrorDesc(ErrGRPCNotCapable):                 ErrGRPCNotCapable,
		ErrorDesc(ErrGRPCStopped):                    ErrGRPCStopped,
		ErrorDesc(ErrGRPCTimeout):                    ErrGRPCTimeout,
		ErrorDesc(ErrGRPCTimeoutDueToLeaderFail):     ErrGRPCTimeoutDueToLeaderFail,
		ErrorDesc(ErrGRPCTimeoutDueToConnectionLost): ErrGRPCTimeoutDueToConnectionLost,
		ErrorDesc(ErrGRPCUnhealthy):                  ErrGRPCUnhealthy,
		ErrorDesc(ErrGRPCCorrupt):                    ErrGRPCCorrupt,
		ErrorDesc(ErrGPRCNotSupportedForLearner):     ErrGPRCNotSupportedForLearner,
		ErrorDesc(ErrGRPCBadLeaderTransferee):        ErrGRPCBadLeaderTransferee,

		ErrorDesc(ErrGRPCClusterVersionUnavailable):     ErrGRPCClusterVersionUnavailable,
		ErrorDesc(ErrGRPCWrongDowngradeVersionFormat):   ErrGRPCWrongDowngradeVersionFormat,
		ErrorDesc(ErrGRPCInvalidDowngradeTargetVersion): ErrGRPCInvalidDowngradeTargetVersion,
		ErrorDesc(ErrGRPCDowngradeInProcess):            ErrGRPCDowngradeInProcess,
		ErrorDesc(ErrGRPCNoInflightDowngrade):           ErrGRPCNoInflightDowngrade,
	}
)

// client-side error
var (
	ErrEmptyKey      = Error(ErrGRPCEmptyKey)
	ErrKeyNotFound   = Error(ErrGRPCKeyNotFound)
	ErrValueProvided = Error(ErrGRPCValueProvided)
	ErrLeaseProvided = Error(ErrGRPCLeaseProvided)
	ErrTooManyOps    = Error(ErrGRPCTooManyOps)
	ErrDuplicateKey  = Error(ErrGRPCDuplicateKey)
	ErrCompacted     = Error(ErrGRPCCompacted)
	ErrFutureRev     = Error(ErrGRPCFutureRev)
	ErrNoSpace       = Error(ErrGRPCNoSpace)

	ErrLeaseNotFound    = Error(ErrGRPCLeaseNotFound)
	ErrLeaseExist       = Error(ErrGRPCLeaseExist)
	ErrLeaseTTLTooLarge = Error(ErrGRPCLeaseTTLTooLarge)

	ErrMemberExist            = Error(ErrGRPCMemberExist)
	ErrPeerURLExist           = Error(ErrGRPCPeerURLExist)
	ErrMemberNotEnoughStarted = Error(ErrGRPCMemberNotEnoughStarted)
	ErrMemberBadURLs          = Error(ErrGRPCMemberBadURLs)
	ErrMemberNotFound         = Error(ErrGRPCMemberNotFound)
	ErrMemberNotLearner       = Error(ErrGRPCMemberNotLearner)
	ErrMemberLearnerNotReady  = Error(ErrGRPCLearnerNotReady)
	ErrTooManyLearners        = Error(ErrGRPCTooManyLearners)

	ErrRequestTooLarge = Error(ErrGRPCRequestTooLarge)
	ErrTooManyRequests = Error(ErrGRPCRequestTooManyRequests)

	ErrRootUserNotExist     = Error(ErrGRPCRootUserNotExist)
	ErrRootRoleNotExist     = Error(ErrGRPCRootRoleNotExist)
	ErrUserAlreadyExist     = Error(ErrGRPCUserAlreadyExist)
	ErrUserEmpty            = Error(ErrGRPCUserEmpty)
	ErrUserNotFound         = Error(ErrGRPCUserNotFound)
	ErrRoleAlreadyExist     = Error(ErrGRPCRoleAlreadyExist)
	ErrRoleNotFound         = Error(ErrGRPCRoleNotFound)
	ErrRoleEmpty            = Error(ErrGRPCRoleEmpty)
	ErrAuthFailed           = Error(ErrGRPCAuthFailed)
	ErrPermissionDenied     = Error(ErrGRPCPermissionDenied)
	ErrRoleNotGranted       = Error(ErrGRPCRoleNotGranted)
	ErrPermissionNotGranted = Error(ErrGRPCPermissionNotGranted)
	ErrAuthNotEnabled       = Error(ErrGRPCAuthNotEnabled)
	ErrInvalidAuthToken     = Error(ErrGRPCInvalidAuthToken)
	ErrInvalidAuthMgmt      = Error(ErrGRPCInvalidAuthMgmt)

	ErrNoLeader                   = Error(ErrGRPCNoLeader)
	ErrNotLeader                  = Error(ErrGRPCNotLeader)
	ErrLeaderChanged              = Error(ErrGRPCLeaderChanged)
	ErrNotCapable                 = Error(ErrGRPCNotCapable)
	ErrStopped                    = Error(ErrGRPCStopped)
	ErrTimeout                    = Error(ErrGRPCTimeout)
	ErrTimeoutDueToLeaderFail     = Error(ErrGRPCTimeoutDueToLeaderFail)
	ErrTimeoutDueToConnectionLost = Error(ErrGRPCTimeoutDueToConnectionLost)
	ErrUnhealthy                  = Error(ErrGRPCUnhealthy)
	ErrCorrupt                    = Error(ErrGRPCCorrupt)
	ErrBadLeaderTransferee        = Error(ErrGRPCBadLeaderTransferee)

	ErrClusterVersionUnavailable     = Error(ErrGRPCClusterVersionUnavailable)
	ErrWrongDowngradeVersionFormat   = Error(ErrGRPCWrongDowngradeVersionFormat)
	ErrInvalidDowngradeTargetVersion = Error(ErrGRPCInvalidDowngradeTargetVersion)
	ErrDowngradeInProcess            = Error(ErrGRPCDowngradeInProcess)
	ErrNoInflightDowngrade           = Error(ErrGRPCNoInflightDowngrade)
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
	verr, ok := errStringToError[ErrorDesc(err)]
	if !ok { // not gRPC error
		return err
	}
	ev, ok := status.FromError(verr)
	var desc string
	if ok {
		desc = ev.Message()
	} else {
		desc = verr.Error()
	}
	return EtcdError{code: ev.Code(), desc: desc}
}

func ErrorDesc(err error) string {
	if s, ok := status.FromError(err); ok {
		return s.Message()
	}
	return err.Error()
}
