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
	"go.uber.org/zap"
	"google.golang.org/protobuf/proto"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/pkg/v3/wait"
	"go.etcd.io/etcd/server/v3/etcdserver/api/membership"
	"go.etcd.io/etcd/server/v3/internal/legacyv2"
	"go.etcd.io/raft/v3/raftpb"
)

func Apply(lg *zap.Logger, e *raftpb.Entry, uberApply UberApplier, w wait.Wait, shouldApplyV3 membership.ShouldApplyV3) (ar *Result, id uint64) {
	var raftReq pb.InternalRaftRequest
	if err := proto.Unmarshal(e.Data, &raftReq); err != nil {
		var legacyReq legacyv2.Request
		if lerr := legacyReq.Unmarshal(e.Data); lerr != nil {
			lg.Panic("failed to unmarshal raft entry", zap.Error(err), zap.Error(lerr))
		}
		// Only the two metadata shapes written by etcd 3.5 and earlier are
		// expected; anything else is WAL corruption and must panic rather
		// than be dropped.
		if !legacyReq.IsMetadataRequest() {
			lg.Panic("unexpected non-metadata legacy v2 request",
				zap.Stringer("request", &legacyReq), zap.Error(err))
		}
		// Legacy v2 requests are dropped: the v2 store has been removed,
		// member attributes are re-published before the server starts
		// serving requests, and the cluster version is maintained by the
		// newer members.
		lg.Info("dropping legacy v2 WAL request", zap.Stringer("request", &legacyReq))
		return nil, legacyReq.ID
	}
	lg.Debug("Apply", zap.Stringer("raftReq", &raftReq))

	id = raftReq.ID
	if id == 0 {
		if raftReq.Header == nil {
			lg.Panic("Apply, could not find a header")
		}
		id = raftReq.Header.ID
	}

	needResult := w.IsRegistered(id)
	wrapper := &InternalRaftRequestWrapper{
		InternalRaftRequest: &raftReq,
		SkipRangeExecution:  !needResult && raftReq.Txn != nil,
	}
	if needResult || !noSideEffect(&raftReq) {
		return uberApply.Apply(wrapper, shouldApplyV3), id
	}
	return nil, id
}

func noSideEffect(r *pb.InternalRaftRequest) bool {
	return r.Range != nil || r.AuthUserGet != nil || r.AuthRoleGet != nil || r.AuthStatus != nil
}
