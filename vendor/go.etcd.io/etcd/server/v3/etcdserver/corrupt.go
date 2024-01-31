// Copyright 2017 The etcd Authors
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
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"sort"
	"strings"
	"sync"
	"time"

	pb "go.etcd.io/etcd/api/v3/etcdserverpb"
	"go.etcd.io/etcd/api/v3/v3rpc/rpctypes"
	"go.etcd.io/etcd/client/pkg/v3/types"
	"go.etcd.io/etcd/pkg/v3/traceutil"
	"go.etcd.io/etcd/server/v3/etcdserver/api/rafthttp"
	"go.etcd.io/etcd/server/v3/mvcc"

	"go.uber.org/zap"
)

type CorruptionChecker interface {
	InitialCheck() error
	PeriodicCheck() error
	CompactHashCheck()
}

type corruptionChecker struct {
	lg *zap.Logger

	hasher Hasher

	mux                   sync.RWMutex
	latestRevisionChecked int64
}

type Hasher interface {
	mvcc.HashStorage
	ReqTimeout() time.Duration
	MemberId() types.ID
	PeerHashByRev(int64) []*peerHashKVResp
	LinearizableReadNotify(context.Context) error
	TriggerCorruptAlarm(types.ID)
}

func newCorruptionChecker(lg *zap.Logger, s *EtcdServer, storage mvcc.HashStorage) *corruptionChecker {
	return &corruptionChecker{
		lg:     lg,
		hasher: hasherAdapter{s, storage},
	}
}

type hasherAdapter struct {
	*EtcdServer
	mvcc.HashStorage
}

func (h hasherAdapter) MemberId() types.ID {
	return h.EtcdServer.ID()
}

func (h hasherAdapter) ReqTimeout() time.Duration {
	return h.EtcdServer.Cfg.ReqTimeout()
}

func (h hasherAdapter) PeerHashByRev(rev int64) []*peerHashKVResp {
	return h.EtcdServer.getPeerHashKVs(rev)
}

func (h hasherAdapter) TriggerCorruptAlarm(memberID types.ID) {
	h.EtcdServer.triggerCorruptAlarm(memberID)
}

// InitialCheck compares initial hash values with its peers
// before serving any peer/client traffic. Only mismatch when hashes
// are different at requested revision, with same compact revision.
func (cm *corruptionChecker) InitialCheck() error {

	cm.lg.Info(
		"starting initial corruption check",
		zap.String("local-member-id", cm.hasher.MemberId().String()),
		zap.Duration("timeout", cm.hasher.ReqTimeout()),
	)

	h, rev, err := cm.hasher.HashByRev(0)
	if err != nil {
		return fmt.Errorf("%s failed to fetch hash (%v)", cm.hasher.MemberId(), err)
	}
	peers := cm.hasher.PeerHashByRev(rev)
	mismatch := 0
	for _, p := range peers {
		if p.resp != nil {
			peerID := types.ID(p.resp.Header.MemberId)
			fields := []zap.Field{
				zap.String("local-member-id", cm.hasher.MemberId().String()),
				zap.Int64("local-member-revision", rev),
				zap.Int64("local-member-compact-revision", h.CompactRevision),
				zap.Uint32("local-member-hash", h.Hash),
				zap.String("remote-peer-id", peerID.String()),
				zap.Strings("remote-peer-endpoints", p.eps),
				zap.Int64("remote-peer-revision", p.resp.Header.Revision),
				zap.Int64("remote-peer-compact-revision", p.resp.CompactRevision),
				zap.Uint32("remote-peer-hash", p.resp.Hash),
			}

			if h.Hash != p.resp.Hash {
				if h.CompactRevision == p.resp.CompactRevision {
					cm.lg.Warn("found different hash values from remote peer", fields...)
					mismatch++
				} else {
					cm.lg.Warn("found different compact revision values from remote peer", fields...)
				}
			}

			continue
		}

		if p.err != nil {
			switch p.err {
			case rpctypes.ErrFutureRev:
				cm.lg.Warn(
					"cannot fetch hash from slow remote peer",
					zap.String("local-member-id", cm.hasher.MemberId().String()),
					zap.Int64("local-member-revision", rev),
					zap.Int64("local-member-compact-revision", h.CompactRevision),
					zap.Uint32("local-member-hash", h.Hash),
					zap.String("remote-peer-id", p.id.String()),
					zap.Strings("remote-peer-endpoints", p.eps),
					zap.Error(err),
				)
			case rpctypes.ErrCompacted:
				cm.lg.Warn(
					"cannot fetch hash from remote peer; local member is behind",
					zap.String("local-member-id", cm.hasher.MemberId().String()),
					zap.Int64("local-member-revision", rev),
					zap.Int64("local-member-compact-revision", h.CompactRevision),
					zap.Uint32("local-member-hash", h.Hash),
					zap.String("remote-peer-id", p.id.String()),
					zap.Strings("remote-peer-endpoints", p.eps),
					zap.Error(err),
				)
			case rpctypes.ErrClusterIdMismatch:
				cm.lg.Warn(
					"cluster ID mismatch",
					zap.String("local-member-id", cm.hasher.MemberId().String()),
					zap.Int64("local-member-revision", h.Revision),
					zap.Int64("local-member-compact-revision", h.CompactRevision),
					zap.Uint32("local-member-hash", h.Hash),
					zap.String("remote-peer-id", p.id.String()),
					zap.Strings("remote-peer-endpoints", p.eps),
					zap.Error(err),
				)
			}
		}
	}
	if mismatch > 0 {
		return fmt.Errorf("%s found data inconsistency with peers", cm.hasher.MemberId())
	}

	cm.lg.Info(
		"initial corruption checking passed; no corruption",
		zap.String("local-member-id", cm.hasher.MemberId().String()),
	)
	return nil
}

func (cm *corruptionChecker) PeriodicCheck() error {
	h, rev, err := cm.hasher.HashByRev(0)
	if err != nil {
		return err
	}
	peers := cm.hasher.PeerHashByRev(rev)

	ctx, cancel := context.WithTimeout(context.Background(), cm.hasher.ReqTimeout())
	err = cm.hasher.LinearizableReadNotify(ctx)
	cancel()
	if err != nil {
		return err
	}

	h2, rev2, err := cm.hasher.HashByRev(0)
	if err != nil {
		return err
	}

	alarmed := false
	mismatch := func(id types.ID) {
		if alarmed {
			return
		}
		alarmed = true
		// It isn't clear which member's data is corrupted, so we
		// intentionally set the memberID as 0. We will identify
		// the corrupted members using quorum in 3.6. Please see
		// discussion in https://github.com/etcd-io/etcd/pull/14828.
		cm.hasher.TriggerCorruptAlarm(types.ID(0))
	}

	if h2.Hash != h.Hash && rev2 == rev && h.CompactRevision == h2.CompactRevision {
		cm.lg.Warn(
			"found hash mismatch",
			zap.Int64("revision-1", rev),
			zap.Int64("compact-revision-1", h.CompactRevision),
			zap.Uint32("hash-1", h.Hash),
			zap.Int64("revision-2", rev2),
			zap.Int64("compact-revision-2", h2.CompactRevision),
			zap.Uint32("hash-2", h2.Hash),
		)
		mismatch(cm.hasher.MemberId())
	}

	checkedCount := 0
	for _, p := range peers {
		if p.resp == nil {
			continue
		}
		checkedCount++

		// leader expects follower's latest revision less than or equal to leader's
		if p.resp.Header.Revision > rev2 {
			cm.lg.Warn(
				"revision from follower must be less than or equal to leader's",
				zap.Int64("leader-revision", rev2),
				zap.Int64("follower-revision", p.resp.Header.Revision),
				zap.String("follower-peer-id", p.id.String()),
			)
			mismatch(p.id)
		}

		// leader expects follower's latest compact revision less than or equal to leader's
		if p.resp.CompactRevision > h2.CompactRevision {
			cm.lg.Warn(
				"compact revision from follower must be less than or equal to leader's",
				zap.Int64("leader-compact-revision", h2.CompactRevision),
				zap.Int64("follower-compact-revision", p.resp.CompactRevision),
				zap.String("follower-peer-id", p.id.String()),
			)
			mismatch(p.id)
		}

		// follower's compact revision is leader's old one, then hashes must match
		if p.resp.CompactRevision == h.CompactRevision && p.resp.Hash != h.Hash {
			cm.lg.Warn(
				"same compact revision then hashes must match",
				zap.Int64("leader-compact-revision", h2.CompactRevision),
				zap.Uint32("leader-hash", h.Hash),
				zap.Int64("follower-compact-revision", p.resp.CompactRevision),
				zap.Uint32("follower-hash", p.resp.Hash),
				zap.String("follower-peer-id", p.id.String()),
			)
			mismatch(p.id)
		}
	}
	cm.lg.Info("finished peer corruption check", zap.Int("number-of-peers-checked", checkedCount))
	return nil
}

func (cm *corruptionChecker) CompactHashCheck() {
	cm.lg.Info("starting compact hash check",
		zap.String("local-member-id", cm.hasher.MemberId().String()),
		zap.Duration("timeout", cm.hasher.ReqTimeout()),
	)
	hashes := cm.uncheckedRevisions()
	// Assume that revisions are ordered from largest to smallest
	for i, hash := range hashes {
		peers := cm.hasher.PeerHashByRev(hash.Revision)
		if len(peers) == 0 {
			continue
		}
		peersChecked := 0
		for _, p := range peers {
			if p.resp == nil || p.resp.CompactRevision != hash.CompactRevision {
				continue
			}

			// follower's compact revision is leader's old one, then hashes must match
			if p.resp.Hash != hash.Hash {
				// It isn't clear which member's data is corrupted, so we
				// intentionally set the memberID as 0. We will identify
				// the corrupted members using quorum in 3.6. Please see
				// discussion in https://github.com/etcd-io/etcd/pull/14828.
				cm.hasher.TriggerCorruptAlarm(types.ID(0))
				cm.lg.Error("failed compaction hash check",
					zap.Int64("revision", hash.Revision),
					zap.Int64("leader-compact-revision", hash.CompactRevision),
					zap.Uint32("leader-hash", hash.Hash),
					zap.Int64("follower-compact-revision", p.resp.CompactRevision),
					zap.Uint32("follower-hash", p.resp.Hash),
					zap.String("follower-peer-id", p.id.String()),
				)
				return
			}
			peersChecked++
			cm.lg.Info("successfully checked hash on follower",
				zap.Int64("revision", hash.Revision),
				zap.String("peer-id", p.id.String()),
			)
		}
		if len(peers) == peersChecked {
			cm.lg.Info("successfully checked hash on whole cluster",
				zap.Int("number-of-peers-checked", peersChecked),
				zap.Int64("revision", hash.Revision),
			)
			cm.mux.Lock()
			if hash.Revision > cm.latestRevisionChecked {
				cm.latestRevisionChecked = hash.Revision
			}
			cm.mux.Unlock()
			cm.lg.Info("finished compaction hash check", zap.Int("number-of-hashes-checked", i+1))
			return
		}
		cm.lg.Warn("skipped revision in compaction hash check; was not able to check all peers",
			zap.Int("number-of-peers-checked", peersChecked),
			zap.Int("number-of-peers", len(peers)),
			zap.Int64("revision", hash.Revision),
		)
	}
	cm.lg.Info("finished compaction hash check", zap.Int("number-of-hashes-checked", len(hashes)))
	return
}

func (cm *corruptionChecker) uncheckedRevisions() []mvcc.KeyValueHash {
	cm.mux.RLock()
	lastRevisionChecked := cm.latestRevisionChecked
	cm.mux.RUnlock()

	hashes := cm.hasher.Hashes()
	// Sort in descending order
	sort.Slice(hashes, func(i, j int) bool {
		return hashes[i].Revision > hashes[j].Revision
	})
	for i, hash := range hashes {
		if hash.Revision <= lastRevisionChecked {
			return hashes[:i]
		}
	}
	return hashes
}

func (s *EtcdServer) triggerCorruptAlarm(id types.ID) {
	a := &pb.AlarmRequest{
		MemberID: uint64(id),
		Action:   pb.AlarmRequest_ACTIVATE,
		Alarm:    pb.AlarmType_CORRUPT,
	}
	s.GoAttach(func() {
		s.raftRequest(s.ctx, pb.InternalRaftRequest{Alarm: a})
	})
}

type peerInfo struct {
	id  types.ID
	eps []string
}

type peerHashKVResp struct {
	peerInfo
	resp *pb.HashKVResponse
	err  error
}

func (s *EtcdServer) getPeerHashKVs(rev int64) []*peerHashKVResp {
	// TODO: handle the case when "s.cluster.Members" have not
	// been populated (e.g. no snapshot to load from disk)
	members := s.cluster.Members()
	peers := make([]peerInfo, 0, len(members))
	for _, m := range members {
		if m.ID == s.ID() {
			continue
		}
		peers = append(peers, peerInfo{id: m.ID, eps: m.PeerURLs})
	}

	lg := s.Logger()

	cc := &http.Client{
		Transport: s.peerRt,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return http.ErrUseLastResponse
		},
	}
	var resps []*peerHashKVResp
	for _, p := range peers {
		if len(p.eps) == 0 {
			continue
		}

		respsLen := len(resps)
		var lastErr error
		for _, ep := range p.eps {
			ctx, cancel := context.WithTimeout(context.Background(), s.Cfg.ReqTimeout())
			resp, lastErr := HashByRev(ctx, s.cluster.ID(), cc, ep, rev)
			cancel()
			if lastErr == nil {
				resps = append(resps, &peerHashKVResp{peerInfo: p, resp: resp, err: nil})
				break
			}
			lg.Warn(
				"failed hash kv request",
				zap.String("local-member-id", s.ID().String()),
				zap.Int64("requested-revision", rev),
				zap.String("remote-peer-endpoint", ep),
				zap.Error(lastErr),
			)
		}

		// failed to get hashKV from all endpoints of this peer
		if respsLen == len(resps) {
			resps = append(resps, &peerHashKVResp{peerInfo: p, resp: nil, err: lastErr})
		}
	}
	return resps
}

type applierV3Corrupt struct {
	applierV3
}

func newApplierV3Corrupt(a applierV3) *applierV3Corrupt { return &applierV3Corrupt{a} }

func (a *applierV3Corrupt) Put(ctx context.Context, txn mvcc.TxnWrite, p *pb.PutRequest) (*pb.PutResponse, *traceutil.Trace, error) {
	return nil, nil, ErrCorrupt
}

func (a *applierV3Corrupt) Range(ctx context.Context, txn mvcc.TxnRead, p *pb.RangeRequest) (*pb.RangeResponse, error) {
	return nil, ErrCorrupt
}

func (a *applierV3Corrupt) DeleteRange(txn mvcc.TxnWrite, p *pb.DeleteRangeRequest) (*pb.DeleteRangeResponse, error) {
	return nil, ErrCorrupt
}

func (a *applierV3Corrupt) Txn(ctx context.Context, rt *pb.TxnRequest) (*pb.TxnResponse, *traceutil.Trace, error) {
	return nil, nil, ErrCorrupt
}

func (a *applierV3Corrupt) Compaction(compaction *pb.CompactionRequest) (*pb.CompactionResponse, <-chan struct{}, *traceutil.Trace, error) {
	return nil, nil, nil, ErrCorrupt
}

func (a *applierV3Corrupt) LeaseGrant(lc *pb.LeaseGrantRequest) (*pb.LeaseGrantResponse, error) {
	return nil, ErrCorrupt
}

func (a *applierV3Corrupt) LeaseRevoke(lc *pb.LeaseRevokeRequest) (*pb.LeaseRevokeResponse, error) {
	return nil, ErrCorrupt
}

const PeerHashKVPath = "/members/hashkv"

type hashKVHandler struct {
	lg     *zap.Logger
	server *EtcdServer
}

func (s *EtcdServer) HashKVHandler() http.Handler {
	return &hashKVHandler{lg: s.Logger(), server: s}
}

func (h *hashKVHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	if r.Method != http.MethodGet {
		w.Header().Set("Allow", http.MethodGet)
		http.Error(w, "Method Not Allowed", http.StatusMethodNotAllowed)
		return
	}
	if r.URL.Path != PeerHashKVPath {
		http.Error(w, "bad path", http.StatusBadRequest)
		return
	}
	if gcid := r.Header.Get("X-Etcd-Cluster-ID"); gcid != "" && gcid != h.server.cluster.ID().String() {
		http.Error(w, rafthttp.ErrClusterIDMismatch.Error(), http.StatusPreconditionFailed)
		return
	}

	defer r.Body.Close()
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "error reading body", http.StatusBadRequest)
		return
	}

	req := &pb.HashKVRequest{}
	if err := json.Unmarshal(b, req); err != nil {
		h.lg.Warn("failed to unmarshal request", zap.Error(err))
		http.Error(w, "error unmarshalling request", http.StatusBadRequest)
		return
	}
	hash, rev, err := h.server.KV().HashStorage().HashByRev(req.Revision)
	if err != nil {
		h.lg.Warn(
			"failed to get hashKV",
			zap.Int64("requested-revision", req.Revision),
			zap.Error(err),
		)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	resp := &pb.HashKVResponse{Header: &pb.ResponseHeader{Revision: rev}, Hash: hash.Hash, CompactRevision: hash.CompactRevision}
	respBytes, err := json.Marshal(resp)
	if err != nil {
		h.lg.Warn("failed to marshal hashKV response", zap.Error(err))
		http.Error(w, err.Error(), http.StatusInternalServerError)
		return
	}

	w.Header().Set("X-Etcd-Cluster-ID", h.server.Cluster().ID().String())
	w.Header().Set("Content-Type", "application/json")
	w.Write(respBytes)
}

// HashByRev fetch hash of kv store at the given rev via http call to the given url
func HashByRev(ctx context.Context, cid types.ID, cc *http.Client, url string, rev int64) (*pb.HashKVResponse, error) {
	hashReq := &pb.HashKVRequest{Revision: rev}
	hashReqBytes, err := json.Marshal(hashReq)
	if err != nil {
		return nil, err
	}
	requestUrl := url + PeerHashKVPath
	req, err := http.NewRequest(http.MethodGet, requestUrl, bytes.NewReader(hashReqBytes))
	if err != nil {
		return nil, err
	}
	req = req.WithContext(ctx)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Etcd-Cluster-ID", cid.String())
	req.Cancel = ctx.Done()

	resp, err := cc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode == http.StatusBadRequest {
		if strings.Contains(string(b), mvcc.ErrCompacted.Error()) {
			return nil, rpctypes.ErrCompacted
		}
		if strings.Contains(string(b), mvcc.ErrFutureRev.Error()) {
			return nil, rpctypes.ErrFutureRev
		}
	} else if resp.StatusCode == http.StatusPreconditionFailed {
		if strings.Contains(string(b), rafthttp.ErrClusterIDMismatch.Error()) {
			return nil, rpctypes.ErrClusterIdMismatch
		}
	}
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("unknown error: %s", string(b))
	}

	hashResp := &pb.HashKVResponse{}
	if err := json.Unmarshal(b, hashResp); err != nil {
		return nil, err
	}
	return hashResp, nil
}
