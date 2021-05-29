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
	"strings"
	"time"

	"go.etcd.io/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "go.etcd.io/etcd/etcdserver/etcdserverpb"
	"go.etcd.io/etcd/mvcc"
	"go.etcd.io/etcd/pkg/traceutil"
	"go.etcd.io/etcd/pkg/types"

	"go.uber.org/zap"
)

// CheckInitialHashKV compares initial hash values with its peers
// before serving any peer/client traffic. Only mismatch when hashes
// are different at requested revision, with same compact revision.
func (s *EtcdServer) CheckInitialHashKV() error {
	if !s.Cfg.InitialCorruptCheck {
		return nil
	}

	lg := s.getLogger()

	if lg != nil {
		lg.Info(
			"starting initial corruption check",
			zap.String("local-member-id", s.ID().String()),
			zap.Duration("timeout", s.Cfg.ReqTimeout()),
		)
	} else {
		plog.Infof("%s starting initial corruption check with timeout %v...", s.ID(), s.Cfg.ReqTimeout())
	}

	h, rev, crev, err := s.kv.HashByRev(0)
	if err != nil {
		return fmt.Errorf("%s failed to fetch hash (%v)", s.ID(), err)
	}
	peers := s.getPeerHashKVs(rev)
	mismatch := 0
	for _, p := range peers {
		if p.resp != nil {
			peerID := types.ID(p.resp.Header.MemberId)
			fields := []zap.Field{
				zap.String("local-member-id", s.ID().String()),
				zap.Int64("local-member-revision", rev),
				zap.Int64("local-member-compact-revision", crev),
				zap.Uint32("local-member-hash", h),
				zap.String("remote-peer-id", peerID.String()),
				zap.Strings("remote-peer-endpoints", p.eps),
				zap.Int64("remote-peer-revision", p.resp.Header.Revision),
				zap.Int64("remote-peer-compact-revision", p.resp.CompactRevision),
				zap.Uint32("remote-peer-hash", p.resp.Hash),
			}

			if h != p.resp.Hash {
				if crev == p.resp.CompactRevision {
					if lg != nil {
						lg.Warn("found different hash values from remote peer", fields...)
					} else {
						plog.Errorf("%s's hash %d != %s's hash %d (revision %d, peer revision %d, compact revision %d)", s.ID(), h, peerID, p.resp.Hash, rev, p.resp.Header.Revision, crev)
					}
					mismatch++
				} else {
					if lg != nil {
						lg.Warn("found different compact revision values from remote peer", fields...)
					} else {
						plog.Warningf("%s cannot check hash of peer(%s): peer has a different compact revision %d (revision:%d)", s.ID(), peerID, p.resp.CompactRevision, rev)
					}
				}
			}

			continue
		}

		if p.err != nil {
			switch p.err {
			case rpctypes.ErrFutureRev:
				if lg != nil {
					lg.Warn(
						"cannot fetch hash from slow remote peer",
						zap.String("local-member-id", s.ID().String()),
						zap.Int64("local-member-revision", rev),
						zap.Int64("local-member-compact-revision", crev),
						zap.Uint32("local-member-hash", h),
						zap.String("remote-peer-id", p.id.String()),
						zap.Strings("remote-peer-endpoints", p.eps),
						zap.Error(err),
					)
				} else {
					plog.Warningf("%s cannot check the hash of peer(%q) at revision %d: peer is lagging behind(%q)", s.ID(), p.eps, rev, p.err.Error())
				}
			case rpctypes.ErrCompacted:
				if lg != nil {
					lg.Warn(
						"cannot fetch hash from remote peer; local member is behind",
						zap.String("local-member-id", s.ID().String()),
						zap.Int64("local-member-revision", rev),
						zap.Int64("local-member-compact-revision", crev),
						zap.Uint32("local-member-hash", h),
						zap.String("remote-peer-id", p.id.String()),
						zap.Strings("remote-peer-endpoints", p.eps),
						zap.Error(err),
					)
				} else {
					plog.Warningf("%s cannot check the hash of peer(%q) at revision %d: local node is lagging behind(%q)", s.ID(), p.eps, rev, p.err.Error())
				}
			}
		}
	}
	if mismatch > 0 {
		return fmt.Errorf("%s found data inconsistency with peers", s.ID())
	}

	if lg != nil {
		lg.Info(
			"initial corruption checking passed; no corruption",
			zap.String("local-member-id", s.ID().String()),
		)
	} else {
		plog.Infof("%s succeeded on initial corruption checking: no corruption", s.ID())
	}
	return nil
}

func (s *EtcdServer) monitorKVHash() {
	t := s.Cfg.CorruptCheckTime
	if t == 0 {
		return
	}

	lg := s.getLogger()
	if lg != nil {
		lg.Info(
			"enabled corruption checking",
			zap.String("local-member-id", s.ID().String()),
			zap.Duration("interval", t),
		)
	} else {
		plog.Infof("enabled corruption checking with %s interval", t)
	}

	for {
		select {
		case <-s.stopping:
			return
		case <-time.After(t):
		}
		if !s.isLeader() {
			continue
		}
		if err := s.checkHashKV(); err != nil {
			if lg != nil {
				lg.Warn("failed to check hash KV", zap.Error(err))
			} else {
				plog.Debugf("check hash kv failed %v", err)
			}
		}
	}
}

func (s *EtcdServer) checkHashKV() error {
	lg := s.getLogger()

	h, rev, crev, err := s.kv.HashByRev(0)
	if err != nil {
		return err
	}
	peers := s.getPeerHashKVs(rev)

	ctx, cancel := context.WithTimeout(context.Background(), s.Cfg.ReqTimeout())
	err = s.linearizableReadNotify(ctx)
	cancel()
	if err != nil {
		return err
	}

	h2, rev2, crev2, err := s.kv.HashByRev(0)
	if err != nil {
		return err
	}

	alarmed := false
	mismatch := func(id uint64) {
		if alarmed {
			return
		}
		alarmed = true
		a := &pb.AlarmRequest{
			MemberID: id,
			Action:   pb.AlarmRequest_ACTIVATE,
			Alarm:    pb.AlarmType_CORRUPT,
		}
		s.goAttach(func() {
			s.raftRequest(s.ctx, pb.InternalRaftRequest{Alarm: a})
		})
	}

	if h2 != h && rev2 == rev && crev == crev2 {
		if lg != nil {
			lg.Warn(
				"found hash mismatch",
				zap.Int64("revision-1", rev),
				zap.Int64("compact-revision-1", crev),
				zap.Uint32("hash-1", h),
				zap.Int64("revision-2", rev2),
				zap.Int64("compact-revision-2", crev2),
				zap.Uint32("hash-2", h2),
			)
		} else {
			plog.Warningf("mismatched hashes %d and %d for revision %d", h, h2, rev)
		}
		mismatch(uint64(s.ID()))
	}

	checkedCount := 0
	for _, p := range peers {
		if p.resp == nil {
			continue
		}
		checkedCount++
		id := p.resp.Header.MemberId

		// leader expects follower's latest revision less than or equal to leader's
		if p.resp.Header.Revision > rev2 {
			if lg != nil {
				lg.Warn(
					"revision from follower must be less than or equal to leader's",
					zap.Int64("leader-revision", rev2),
					zap.Int64("follower-revision", p.resp.Header.Revision),
					zap.String("follower-peer-id", types.ID(id).String()),
				)
			} else {
				plog.Warningf(
					"revision %d from member %v, expected at most %d",
					p.resp.Header.Revision,
					types.ID(id),
					rev2)
			}
			mismatch(id)
		}

		// leader expects follower's latest compact revision less than or equal to leader's
		if p.resp.CompactRevision > crev2 {
			if lg != nil {
				lg.Warn(
					"compact revision from follower must be less than or equal to leader's",
					zap.Int64("leader-compact-revision", crev2),
					zap.Int64("follower-compact-revision", p.resp.CompactRevision),
					zap.String("follower-peer-id", types.ID(id).String()),
				)
			} else {
				plog.Warningf(
					"compact revision %d from member %v, expected at most %d",
					p.resp.CompactRevision,
					types.ID(id),
					crev2,
				)
			}
			mismatch(id)
		}

		// follower's compact revision is leader's old one, then hashes must match
		if p.resp.CompactRevision == crev && p.resp.Hash != h {
			if lg != nil {
				lg.Warn(
					"same compact revision then hashes must match",
					zap.Int64("leader-compact-revision", crev2),
					zap.Uint32("leader-hash", h),
					zap.Int64("follower-compact-revision", p.resp.CompactRevision),
					zap.Uint32("follower-hash", p.resp.Hash),
					zap.String("follower-peer-id", types.ID(id).String()),
				)
			} else {
				plog.Warningf(
					"hash %d at revision %d from member %v, expected hash %d",
					p.resp.Hash,
					rev,
					types.ID(id),
					h,
				)
			}
			mismatch(id)
		}
	}
	if lg != nil {
		lg.Info("finished peer corruption check", zap.Int("number-of-peers-checked", checkedCount))
	} else {
		plog.Infof("finished peer corruption check")
	}

	return nil
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

	lg := s.getLogger()

	var resps []*peerHashKVResp
	for _, p := range peers {
		if len(p.eps) == 0 {
			continue
		}

		respsLen := len(resps)
		var lastErr error
		for _, ep := range p.eps {
			ctx, cancel := context.WithTimeout(context.Background(), s.Cfg.ReqTimeout())

			var resp *pb.HashKVResponse
			resp, lastErr = s.getPeerHashKVHTTP(ctx, ep, rev)
			cancel()
			if lastErr == nil {
				resps = append(resps, &peerHashKVResp{peerInfo: p, resp: resp, err: nil})
				break
			}
			if lg != nil {
				lg.Warn(
					"failed hash kv request",
					zap.String("local-member-id", s.ID().String()),
					zap.Int64("requested-revision", rev),
					zap.String("remote-peer-endpoint", ep),
					zap.Error(lastErr),
				)
			} else {
				plog.Warningf("%s hash-kv error %q on peer %q with revision %d", s.ID(), lastErr.Error(), ep, rev)
			}
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

func (a *applierV3Corrupt) Put(txn mvcc.TxnWrite, p *pb.PutRequest) (*pb.PutResponse, *traceutil.Trace, error) {
	return nil, nil, ErrCorrupt
}

func (a *applierV3Corrupt) Range(ctx context.Context, txn mvcc.TxnRead, p *pb.RangeRequest) (*pb.RangeResponse, error) {
	return nil, ErrCorrupt
}

func (a *applierV3Corrupt) DeleteRange(txn mvcc.TxnWrite, p *pb.DeleteRangeRequest) (*pb.DeleteRangeResponse, error) {
	return nil, ErrCorrupt
}

func (a *applierV3Corrupt) Txn(rt *pb.TxnRequest) (*pb.TxnResponse, error) {
	return nil, ErrCorrupt
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

type ServerPeerV2 interface {
	ServerPeer
	HashKVHandler() http.Handler
}

const PeerHashKVPath = "/members/hashkv"

type hashKVHandler struct {
	lg     *zap.Logger
	server *EtcdServer
}

func (s *EtcdServer) HashKVHandler() http.Handler {
	return &hashKVHandler{lg: s.getLogger(), server: s}
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

	defer r.Body.Close()
	b, err := ioutil.ReadAll(r.Body)
	if err != nil {
		http.Error(w, "error reading body", http.StatusBadRequest)
		return
	}

	req := &pb.HashKVRequest{}
	if err = json.Unmarshal(b, req); err != nil {
		h.lg.Warn("failed to unmarshal request", zap.Error(err))
		http.Error(w, "error unmarshalling request", http.StatusBadRequest)
		return
	}
	hash, rev, compactRev, err := h.server.KV().HashByRev(req.Revision)
	if err != nil {
		h.lg.Warn(
			"failed to get hashKV",
			zap.Int64("requested-revision", req.Revision),
			zap.Error(err),
		)
		http.Error(w, err.Error(), http.StatusBadRequest)
		return
	}
	resp := &pb.HashKVResponse{Header: &pb.ResponseHeader{Revision: rev}, Hash: hash, CompactRevision: compactRev}
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

// getPeerHashKVHTTP fetch hash of kv store at the given rev via http call to the given url
func (s *EtcdServer) getPeerHashKVHTTP(ctx context.Context, url string, rev int64) (*pb.HashKVResponse, error) {
	cc := &http.Client{Transport: s.peerRt}
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
