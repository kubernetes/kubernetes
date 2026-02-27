// Copyright 2022 The etcd Authors
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

package apply

import (
	"errors"
	"time"

	"go.uber.org/zap"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/server/v3/etcdserver/api/membership"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v3alarm"
	"go.etcd.io/etcd/server/v3/etcdserver/txn"
	"go.etcd.io/etcd/server/v3/storage/mvcc"
)

type UberApplier interface {
	Apply(r *pb.InternalRaftRequest, shouldApplyV3 membership.ShouldApplyV3) *Result
}

type uberApplier struct {
	lg *zap.Logger

	alarmStore           *v3alarm.AlarmStore
	warningApplyDuration time.Duration

	// This is the applier that is taking in consideration current alarms
	applyV3 applierV3

	// This is the applier used for wrapping when alarms change
	applyV3base applierV3
}

func NewUberApplier(opts ApplierOptions) UberApplier {
	applyV3base := newApplierV3(opts)

	ua := &uberApplier{
		lg:                   opts.Logger,
		alarmStore:           opts.AlarmStore,
		warningApplyDuration: opts.WarningApplyDuration,
		applyV3:              applyV3base,
		applyV3base:          applyV3base,
	}
	ua.restoreAlarms()
	return ua
}

func newApplierV3(opts ApplierOptions) applierV3 {
	applierBackend := newApplierV3Backend(opts)
	return newAuthApplierV3(
		opts.AuthStore,
		newQuotaApplierV3(opts.Logger, opts.QuotaBackendBytesCfg, opts.Backend, applierBackend),
		opts.Lessor,
	)
}

func (a *uberApplier) restoreAlarms() {
	noSpaceAlarms := len(a.alarmStore.Get(pb.AlarmType_NOSPACE)) > 0
	corruptAlarms := len(a.alarmStore.Get(pb.AlarmType_CORRUPT)) > 0
	a.applyV3 = a.applyV3base
	if noSpaceAlarms {
		a.applyV3 = newApplierV3Capped(a.applyV3)
	}
	if corruptAlarms {
		a.applyV3 = newApplierV3Corrupt(a.applyV3)
	}
}

func (a *uberApplier) Apply(r *pb.InternalRaftRequest, shouldApplyV3 membership.ShouldApplyV3) *Result {
	// We first execute chain of Apply() calls down the hierarchy:
	// (i.e. CorruptApplier -> CappedApplier -> Auth -> Quota -> Backend),
	// then dispatch() unpacks the request to a specific method (like Put),
	// that gets executed down the hierarchy again:
	// i.e. CorruptApplier.Put(CappedApplier.Put(...(BackendApplier.Put(...)))).
	return a.applyV3.Apply(r, shouldApplyV3, a.dispatch)
}

// dispatch translates the request (r) into appropriate call (like Put) on
// the underlying applyV3 object.
func (a *uberApplier) dispatch(r *pb.InternalRaftRequest, shouldApplyV3 membership.ShouldApplyV3) *Result {
	op := "unknown"
	ar := &Result{}
	defer func(start time.Time) {
		success := ar.Err == nil || errors.Is(ar.Err, mvcc.ErrCompacted)
		txn.ApplySecObserve("v3", op, success, time.Since(start))
		txn.WarnOfExpensiveRequest(a.lg, a.warningApplyDuration, start, &pb.InternalRaftStringer{Request: r}, ar.Resp, ar.Err)
		if !success {
			txn.WarnOfFailedRequest(a.lg, start, &pb.InternalRaftStringer{Request: r}, ar.Resp, ar.Err)
		}
	}(time.Now())

	switch {
	case r.ClusterVersionSet != nil:
		op = "ClusterVersionSet" // Implemented in 3.5.x
		a.applyV3.ClusterVersionSet(r.ClusterVersionSet, shouldApplyV3)
		return ar
	case r.ClusterMemberAttrSet != nil:
		op = "ClusterMemberAttrSet" // Implemented in 3.5.x
		a.applyV3.ClusterMemberAttrSet(r.ClusterMemberAttrSet, shouldApplyV3)
		return ar
	case r.DowngradeInfoSet != nil:
		op = "DowngradeInfoSet" // Implemented in 3.5.x
		a.applyV3.DowngradeInfoSet(r.DowngradeInfoSet, shouldApplyV3)
		return ar
	case r.DowngradeVersionTest != nil:
		op = "DowngradeVersionTest" // Implemented in 3.6 for test only
		// do nothing, we are just to ensure etcdserver don't panic in case
		// users(test cases) intentionally inject DowngradeVersionTestRequest
		// into the WAL files.
		return ar
	default:
	}
	if !shouldApplyV3 {
		return nil
	}

	switch {
	case r.Range != nil:
		op = "Range"
		ar.Resp, ar.Trace, ar.Err = a.applyV3.Range(r.Range)
	case r.Put != nil:
		op = "Put"
		ar.Resp, ar.Trace, ar.Err = a.applyV3.Put(r.Put)
	case r.DeleteRange != nil:
		op = "DeleteRange"
		ar.Resp, ar.Trace, ar.Err = a.applyV3.DeleteRange(r.DeleteRange)
	case r.Txn != nil:
		op = "Txn"
		ar.Resp, ar.Trace, ar.Err = a.applyV3.Txn(r.Txn)
	case r.Compaction != nil:
		op = "Compaction"
		ar.Resp, ar.Physc, ar.Trace, ar.Err = a.applyV3.Compaction(r.Compaction)
	case r.LeaseGrant != nil:
		op = "LeaseGrant"
		ar.Resp, ar.Err = a.applyV3.LeaseGrant(r.LeaseGrant)
	case r.LeaseRevoke != nil:
		op = "LeaseRevoke"
		ar.Resp, ar.Err = a.applyV3.LeaseRevoke(r.LeaseRevoke)
	case r.LeaseCheckpoint != nil:
		op = "LeaseCheckpoint"
		ar.Resp, ar.Err = a.applyV3.LeaseCheckpoint(r.LeaseCheckpoint)
	case r.Alarm != nil:
		op = "Alarm"
		ar.Resp, ar.Err = a.Alarm(r.Alarm)
	case r.Authenticate != nil:
		op = "Authenticate"
		ar.Resp, ar.Err = a.applyV3.Authenticate(r.Authenticate)
	case r.AuthEnable != nil:
		op = "AuthEnable"
		ar.Resp, ar.Err = a.applyV3.AuthEnable()
	case r.AuthDisable != nil:
		op = "AuthDisable"
		ar.Resp, ar.Err = a.applyV3.AuthDisable()
	case r.AuthStatus != nil:
		ar.Resp, ar.Err = a.applyV3.AuthStatus()
	case r.AuthUserAdd != nil:
		op = "AuthUserAdd"
		ar.Resp, ar.Err = a.applyV3.UserAdd(r.AuthUserAdd)
	case r.AuthUserDelete != nil:
		op = "AuthUserDelete"
		ar.Resp, ar.Err = a.applyV3.UserDelete(r.AuthUserDelete)
	case r.AuthUserChangePassword != nil:
		op = "AuthUserChangePassword"
		ar.Resp, ar.Err = a.applyV3.UserChangePassword(r.AuthUserChangePassword)
	case r.AuthUserGrantRole != nil:
		op = "AuthUserGrantRole"
		ar.Resp, ar.Err = a.applyV3.UserGrantRole(r.AuthUserGrantRole)
	case r.AuthUserGet != nil:
		op = "AuthUserGet"
		ar.Resp, ar.Err = a.applyV3.UserGet(r.AuthUserGet)
	case r.AuthUserRevokeRole != nil:
		op = "AuthUserRevokeRole"
		ar.Resp, ar.Err = a.applyV3.UserRevokeRole(r.AuthUserRevokeRole)
	case r.AuthRoleAdd != nil:
		op = "AuthRoleAdd"
		ar.Resp, ar.Err = a.applyV3.RoleAdd(r.AuthRoleAdd)
	case r.AuthRoleGrantPermission != nil:
		op = "AuthRoleGrantPermission"
		ar.Resp, ar.Err = a.applyV3.RoleGrantPermission(r.AuthRoleGrantPermission)
	case r.AuthRoleGet != nil:
		op = "AuthRoleGet"
		ar.Resp, ar.Err = a.applyV3.RoleGet(r.AuthRoleGet)
	case r.AuthRoleRevokePermission != nil:
		op = "AuthRoleRevokePermission"
		ar.Resp, ar.Err = a.applyV3.RoleRevokePermission(r.AuthRoleRevokePermission)
	case r.AuthRoleDelete != nil:
		op = "AuthRoleDelete"
		ar.Resp, ar.Err = a.applyV3.RoleDelete(r.AuthRoleDelete)
	case r.AuthUserList != nil:
		op = "AuthUserList"
		ar.Resp, ar.Err = a.applyV3.UserList(r.AuthUserList)
	case r.AuthRoleList != nil:
		op = "AuthRoleList"
		ar.Resp, ar.Err = a.applyV3.RoleList(r.AuthRoleList)
	default:
		a.lg.Panic("not implemented apply", zap.Stringer("raft-request", r))
	}
	return ar
}

func (a *uberApplier) Alarm(ar *pb.AlarmRequest) (*pb.AlarmResponse, error) {
	resp, err := a.applyV3.Alarm(ar)

	if ar.Action == pb.AlarmRequest_ACTIVATE ||
		ar.Action == pb.AlarmRequest_DEACTIVATE {
		a.restoreAlarms()
	}
	return resp, err
}
