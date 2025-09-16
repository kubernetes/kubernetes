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

package etcdserver

import (
	"encoding/json"
	"net/http"
	"path"

	"go.uber.org/zap"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/api/v3/membershippb"
	"go.etcd.io/etcd/server/v3/etcdserver/api/membership"
)

func v2ToV3Request(lg *zap.Logger, r *RequestV2) pb.InternalRaftRequest {
	if r.Method != http.MethodPut || (!storeMemberAttributeRegexp.MatchString(r.Path) && r.Path != membership.StoreClusterVersionKey()) {
		lg.Panic("detected disallowed v2 WAL for stage --v2-deprecation=write-only", zap.String("method", r.Method))
	}
	if storeMemberAttributeRegexp.MatchString(r.Path) {
		id := membership.MustParseMemberIDFromKey(lg, path.Dir(r.Path))
		var attr membership.Attributes
		if err := json.Unmarshal([]byte(r.Val), &attr); err != nil {
			lg.Panic("failed to unmarshal", zap.String("value", r.Val), zap.Error(err))
		}
		return pb.InternalRaftRequest{
			Header: &pb.RequestHeader{
				ID: r.ID,
			},
			ClusterMemberAttrSet: &membershippb.ClusterMemberAttrSetRequest{
				Member_ID: uint64(id),
				MemberAttributes: &membershippb.Attributes{
					Name:       attr.Name,
					ClientUrls: attr.ClientURLs,
				},
			},
		}
	}
	if r.Path == membership.StoreClusterVersionKey() {
		return pb.InternalRaftRequest{
			Header: &pb.RequestHeader{
				ID: r.ID,
			},
			ClusterVersionSet: &membershippb.ClusterVersionSetRequest{
				Ver: r.Val,
			},
		}
	}
	lg.Panic("detected disallowed v2 WAL for stage --v2-deprecation=write-only", zap.String("method", r.Method))
	return pb.InternalRaftRequest{}
}
