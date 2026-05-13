// Copyright 2025 The etcd Authors
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
	"time"

	"github.com/gogo/protobuf/proto"
	"go.uber.org/zap"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/api/v3/membershippb"
	"go.etcd.io/etcd/client/pkg/v3/types"
	"go.etcd.io/etcd/pkg/v3/traceutil"
	"go.etcd.io/etcd/server/v3/auth"
	"go.etcd.io/etcd/server/v3/etcdserver/api/membership"
	"go.etcd.io/etcd/server/v3/etcdserver/api/v3alarm"
	"go.etcd.io/etcd/server/v3/etcdserver/cindex"
	"go.etcd.io/etcd/server/v3/lease"
	"go.etcd.io/etcd/server/v3/storage/backend"
	"go.etcd.io/etcd/server/v3/storage/mvcc"
)

// applierV3 is the interface for processing V3 raft messages
type applierV3 interface {
	// Apply executes the generic portion of application logic for the current applier, but
	// delegates the actual execution to the applyFunc method.
	Apply(r *pb.InternalRaftRequest, shouldApplyV3 membership.ShouldApplyV3, applyFunc applyFunc) *Result

	Put(p *pb.PutRequest) (*pb.PutResponse, *traceutil.Trace, error)
	Range(r *pb.RangeRequest) (*pb.RangeResponse, *traceutil.Trace, error)
	DeleteRange(dr *pb.DeleteRangeRequest) (*pb.DeleteRangeResponse, *traceutil.Trace, error)
	Txn(rt *pb.TxnRequest) (*pb.TxnResponse, *traceutil.Trace, error)
	Compaction(compaction *pb.CompactionRequest) (*pb.CompactionResponse, <-chan struct{}, *traceutil.Trace, error)

	LeaseGrant(lc *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error)
	LeaseRevoke(lc *pb.LeaseRevokeRequest) (*pb.LeaseRevokeResponse, error)

	LeaseCheckpoint(lc *pb.LeaseCheckpointRequest) (*pb.LeaseCheckpointResponse, error)

	Alarm(*pb.AlarmRequest) (*pb.AlarmResponse, error)

	Authenticate(r *pb.InternalAuthenticateRequest) (*pb.AuthenticateResponse, error)

	AuthEnable() (*pb.AuthEnableResponse, error)
	AuthDisable() (*pb.AuthDisableResponse, error)
	AuthStatus() (*pb.AuthStatusResponse, error)

	UserAdd(ua *pb.AuthUserAddRequest) (*pb.AuthUserAddResponse, error)
	UserDelete(ua *pb.AuthUserDeleteRequest) (*pb.AuthUserDeleteResponse, error)
	UserChangePassword(ua *pb.AuthUserChangePasswordRequest) (*pb.AuthUserChangePasswordResponse, error)
	UserGrantRole(ua *pb.AuthUserGrantRoleRequest) (*pb.AuthUserGrantRoleResponse, error)
	UserGet(ua *pb.AuthUserGetRequest) (*pb.AuthUserGetResponse, error)
	UserRevokeRole(ua *pb.AuthUserRevokeRoleRequest) (*pb.AuthUserRevokeRoleResponse, error)
	RoleAdd(ua *pb.AuthRoleAddRequest) (*pb.AuthRoleAddResponse, error)
	RoleGrantPermission(ua *pb.AuthRoleGrantPermissionRequest) (*pb.AuthRoleGrantPermissionResponse, error)
	RoleGet(ua *pb.AuthRoleGetRequest) (*pb.AuthRoleGetResponse, error)
	RoleRevokePermission(ua *pb.AuthRoleRevokePermissionRequest) (*pb.AuthRoleRevokePermissionResponse, error)
	RoleDelete(ua *pb.AuthRoleDeleteRequest) (*pb.AuthRoleDeleteResponse, error)
	UserList(ua *pb.AuthUserListRequest) (*pb.AuthUserListResponse, error)
	RoleList(ua *pb.AuthRoleListRequest) (*pb.AuthRoleListResponse, error)
	ClusterVersionSet(r *membershippb.ClusterVersionSetRequest, shouldApplyV3 membership.ShouldApplyV3)
	ClusterMemberAttrSet(r *membershippb.ClusterMemberAttrSetRequest, shouldApplyV3 membership.ShouldApplyV3)
	DowngradeInfoSet(r *membershippb.DowngradeInfoSetRequest, shouldApplyV3 membership.ShouldApplyV3)
}

type ApplierOptions struct {
	Logger                       *zap.Logger
	KV                           mvcc.KV
	AlarmStore                   *v3alarm.AlarmStore
	AuthStore                    auth.AuthStore
	Lessor                       lease.Lessor
	Cluster                      *membership.RaftCluster
	RaftStatus                   RaftStatusGetter
	SnapshotServer               SnapshotServer
	ConsistentIndex              cindex.ConsistentIndexer
	TxnModeWriteWithSharedBuffer bool
	Backend                      backend.Backend
	QuotaBackendBytesCfg         int64
	WarningApplyDuration         time.Duration
}

type SnapshotServer interface {
	ForceSnapshot()
}

// RaftStatusGetter represents etcd server and Raft progress.
type RaftStatusGetter interface {
	MemberID() types.ID
	Leader() types.ID
	CommittedIndex() uint64
	AppliedIndex() uint64
	Term() uint64
}

type Result struct {
	Resp proto.Message
	Err  error
	// Physc signals the physical effect of the request has completed in addition
	// to being logically reflected by the node. Currently, only used for
	// Compaction requests.
	Physc <-chan struct{}
	Trace *traceutil.Trace
}

type applyFunc func(*pb.InternalRaftRequest, membership.ShouldApplyV3) *Result
