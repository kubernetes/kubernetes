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
	"crypto/sha256"
	"io"
	"time"

	"github.com/dustin/go-humanize"
	"go.etcd.io/etcd/auth"
	"go.etcd.io/etcd/etcdserver"
	"go.etcd.io/etcd/etcdserver/api/v3rpc/rpctypes"
	pb "go.etcd.io/etcd/etcdserver/etcdserverpb"
	"go.etcd.io/etcd/mvcc"
	"go.etcd.io/etcd/mvcc/backend"
	"go.etcd.io/etcd/raft"
	"go.etcd.io/etcd/version"

	"go.uber.org/zap"
)

type KVGetter interface {
	KV() mvcc.ConsistentWatchableKV
}

type BackendGetter interface {
	Backend() backend.Backend
}

type Alarmer interface {
	// Alarms is implemented in Server interface located in etcdserver/server.go
	// It returns a list of alarms present in the AlarmStore
	Alarms() []*pb.AlarmMember
	Alarm(ctx context.Context, ar *pb.AlarmRequest) (*pb.AlarmResponse, error)
}

type LeaderTransferrer interface {
	MoveLeader(ctx context.Context, lead, target uint64) error
}

type AuthGetter interface {
	AuthInfoFromCtx(ctx context.Context) (*auth.AuthInfo, error)
	AuthStore() auth.AuthStore
}

type ClusterStatusGetter interface {
	IsLearner() bool
}

type maintenanceServer struct {
	lg  *zap.Logger
	rg  etcdserver.RaftStatusGetter
	kg  KVGetter
	bg  BackendGetter
	a   Alarmer
	lt  LeaderTransferrer
	hdr header
	cs  ClusterStatusGetter
}

func NewMaintenanceServer(s *etcdserver.EtcdServer) pb.MaintenanceServer {
	srv := &maintenanceServer{lg: s.Cfg.Logger, rg: s, kg: s, bg: s, a: s, lt: s, hdr: newHeader(s), cs: s}
	return &authMaintenanceServer{srv, s}
}

func (ms *maintenanceServer) Defragment(ctx context.Context, sr *pb.DefragmentRequest) (*pb.DefragmentResponse, error) {
	if ms.lg != nil {
		ms.lg.Info("starting defragment")
	} else {
		plog.Noticef("starting to defragment the storage backend...")
	}
	err := ms.bg.Backend().Defrag()
	if err != nil {
		if ms.lg != nil {
			ms.lg.Warn("failed to defragment", zap.Error(err))
		} else {
			plog.Errorf("failed to defragment the storage backend (%v)", err)
		}
		return nil, err
	}
	if ms.lg != nil {
		ms.lg.Info("finished defragment")
	} else {
		plog.Noticef("finished defragmenting the storage backend")
	}
	return &pb.DefragmentResponse{}, nil
}

// big enough size to hold >1 OS pages in the buffer
const snapshotSendBufferSize = 32 * 1024

func (ms *maintenanceServer) Snapshot(sr *pb.SnapshotRequest, srv pb.Maintenance_SnapshotServer) error {
	snap := ms.bg.Backend().Snapshot()
	pr, pw := io.Pipe()

	defer pr.Close()

	go func() {
		snap.WriteTo(pw)
		if err := snap.Close(); err != nil {
			if ms.lg != nil {
				ms.lg.Warn("failed to close snapshot", zap.Error(err))
			} else {
				plog.Errorf("error closing snapshot (%v)", err)
			}
		}
		pw.Close()
	}()

	// record SHA digest of snapshot data
	// used for integrity checks during snapshot restore operation
	h := sha256.New()

	// buffer just holds read bytes from stream
	// response size is multiple of OS page size, fetched in boltdb
	// e.g. 4*1024
	buf := make([]byte, snapshotSendBufferSize)

	sent := int64(0)
	total := snap.Size()
	size := humanize.Bytes(uint64(total))

	start := time.Now()
	if ms.lg != nil {
		ms.lg.Info("sending database snapshot to client",
			zap.Int64("total-bytes", total),
			zap.String("size", size),
		)
	} else {
		plog.Infof("sending database snapshot to client %s [%d bytes]", size, total)
	}
	for total-sent > 0 {
		n, err := io.ReadFull(pr, buf)
		if err != nil && err != io.EOF && err != io.ErrUnexpectedEOF {
			return togRPCError(err)
		}
		sent += int64(n)

		// if total is x * snapshotSendBufferSize. it is possible that
		// resp.RemainingBytes == 0
		// resp.Blob == zero byte but not nil
		// does this make server response sent to client nil in proto
		// and client stops receiving from snapshot stream before
		// server sends snapshot SHA?
		// No, the client will still receive non-nil response
		// until server closes the stream with EOF

		resp := &pb.SnapshotResponse{
			RemainingBytes: uint64(total - sent),
			Blob:           buf[:n],
		}
		if err = srv.Send(resp); err != nil {
			return togRPCError(err)
		}
		h.Write(buf[:n])
	}

	// send SHA digest for integrity checks
	// during snapshot restore operation
	sha := h.Sum(nil)

	if ms.lg != nil {
		ms.lg.Info("sending database sha256 checksum to client",
			zap.Int64("total-bytes", total),
			zap.Int("checksum-size", len(sha)),
		)
	} else {
		plog.Infof("sending database sha256 checksum to client [%d bytes]", len(sha))
	}

	hresp := &pb.SnapshotResponse{RemainingBytes: 0, Blob: sha}
	if err := srv.Send(hresp); err != nil {
		return togRPCError(err)
	}

	if ms.lg != nil {
		ms.lg.Info("successfully sent database snapshot to client",
			zap.Int64("total-bytes", total),
			zap.String("size", size),
			zap.String("took", humanize.Time(start)),
		)
	} else {
		plog.Infof("successfully sent database snapshot to client %s [%d bytes]", size, total)
	}

	return nil
}

func (ms *maintenanceServer) Hash(ctx context.Context, r *pb.HashRequest) (*pb.HashResponse, error) {
	h, rev, err := ms.kg.KV().Hash()
	if err != nil {
		return nil, togRPCError(err)
	}
	resp := &pb.HashResponse{Header: &pb.ResponseHeader{Revision: rev}, Hash: h}
	ms.hdr.fill(resp.Header)
	return resp, nil
}

func (ms *maintenanceServer) HashKV(ctx context.Context, r *pb.HashKVRequest) (*pb.HashKVResponse, error) {
	h, rev, compactRev, err := ms.kg.KV().HashByRev(r.Revision)
	if err != nil {
		return nil, togRPCError(err)
	}

	resp := &pb.HashKVResponse{Header: &pb.ResponseHeader{Revision: rev}, Hash: h, CompactRevision: compactRev}
	ms.hdr.fill(resp.Header)
	return resp, nil
}

func (ms *maintenanceServer) Alarm(ctx context.Context, ar *pb.AlarmRequest) (*pb.AlarmResponse, error) {
	return ms.a.Alarm(ctx, ar)
}

func (ms *maintenanceServer) Status(ctx context.Context, ar *pb.StatusRequest) (*pb.StatusResponse, error) {
	hdr := &pb.ResponseHeader{}
	ms.hdr.fill(hdr)
	resp := &pb.StatusResponse{
		Header:           hdr,
		Version:          version.Version,
		Leader:           uint64(ms.rg.Leader()),
		RaftIndex:        ms.rg.CommittedIndex(),
		RaftAppliedIndex: ms.rg.AppliedIndex(),
		RaftTerm:         ms.rg.Term(),
		DbSize:           ms.bg.Backend().Size(),
		DbSizeInUse:      ms.bg.Backend().SizeInUse(),
		IsLearner:        ms.cs.IsLearner(),
	}
	if resp.Leader == raft.None {
		resp.Errors = append(resp.Errors, etcdserver.ErrNoLeader.Error())
	}
	for _, a := range ms.a.Alarms() {
		resp.Errors = append(resp.Errors, a.String())
	}
	return resp, nil
}

func (ms *maintenanceServer) MoveLeader(ctx context.Context, tr *pb.MoveLeaderRequest) (*pb.MoveLeaderResponse, error) {
	if ms.rg.ID() != ms.rg.Leader() {
		return nil, rpctypes.ErrGRPCNotLeader
	}

	if err := ms.lt.MoveLeader(ctx, uint64(ms.rg.Leader()), tr.TargetID); err != nil {
		return nil, togRPCError(err)
	}
	return &pb.MoveLeaderResponse{}, nil
}

type authMaintenanceServer struct {
	*maintenanceServer
	ag AuthGetter
}

func (ams *authMaintenanceServer) isAuthenticated(ctx context.Context) error {
	authInfo, err := ams.ag.AuthInfoFromCtx(ctx)
	if err != nil {
		return err
	}

	return ams.ag.AuthStore().IsAdminPermitted(authInfo)
}

func (ams *authMaintenanceServer) Defragment(ctx context.Context, sr *pb.DefragmentRequest) (*pb.DefragmentResponse, error) {
	if err := ams.isAuthenticated(ctx); err != nil {
		return nil, err
	}

	return ams.maintenanceServer.Defragment(ctx, sr)
}

func (ams *authMaintenanceServer) Snapshot(sr *pb.SnapshotRequest, srv pb.Maintenance_SnapshotServer) error {
	if err := ams.isAuthenticated(srv.Context()); err != nil {
		return err
	}

	return ams.maintenanceServer.Snapshot(sr, srv)
}

func (ams *authMaintenanceServer) Hash(ctx context.Context, r *pb.HashRequest) (*pb.HashResponse, error) {
	if err := ams.isAuthenticated(ctx); err != nil {
		return nil, err
	}

	return ams.maintenanceServer.Hash(ctx, r)
}

func (ams *authMaintenanceServer) HashKV(ctx context.Context, r *pb.HashKVRequest) (*pb.HashKVResponse, error) {
	if err := ams.isAuthenticated(ctx); err != nil {
		return nil, err
	}
	return ams.maintenanceServer.HashKV(ctx, r)
}

func (ams *authMaintenanceServer) Status(ctx context.Context, ar *pb.StatusRequest) (*pb.StatusResponse, error) {
	return ams.maintenanceServer.Status(ctx, ar)
}

func (ams *authMaintenanceServer) MoveLeader(ctx context.Context, tr *pb.MoveLeaderRequest) (*pb.MoveLeaderResponse, error) {
	return ams.maintenanceServer.MoveLeader(ctx, tr)
}
