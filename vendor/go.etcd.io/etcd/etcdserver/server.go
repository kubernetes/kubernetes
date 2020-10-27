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

package etcdserver

import (
	"context"
	"encoding/json"
	"expvar"
	"fmt"
	"math"
	"math/rand"
	"net/http"
	"os"
	"path"
	"regexp"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coreos/go-semver/semver"
	"github.com/coreos/pkg/capnslog"
	humanize "github.com/dustin/go-humanize"
	"github.com/prometheus/client_golang/prometheus"
	"go.etcd.io/etcd/auth"
	"go.etcd.io/etcd/etcdserver/api"
	"go.etcd.io/etcd/etcdserver/api/membership"
	"go.etcd.io/etcd/etcdserver/api/rafthttp"
	"go.etcd.io/etcd/etcdserver/api/snap"
	"go.etcd.io/etcd/etcdserver/api/v2discovery"
	"go.etcd.io/etcd/etcdserver/api/v2http/httptypes"
	stats "go.etcd.io/etcd/etcdserver/api/v2stats"
	"go.etcd.io/etcd/etcdserver/api/v2store"
	"go.etcd.io/etcd/etcdserver/api/v3alarm"
	"go.etcd.io/etcd/etcdserver/api/v3compactor"
	pb "go.etcd.io/etcd/etcdserver/etcdserverpb"
	"go.etcd.io/etcd/lease"
	"go.etcd.io/etcd/lease/leasehttp"
	"go.etcd.io/etcd/mvcc"
	"go.etcd.io/etcd/mvcc/backend"
	"go.etcd.io/etcd/pkg/fileutil"
	"go.etcd.io/etcd/pkg/idutil"
	"go.etcd.io/etcd/pkg/pbutil"
	"go.etcd.io/etcd/pkg/runtime"
	"go.etcd.io/etcd/pkg/schedule"
	"go.etcd.io/etcd/pkg/traceutil"
	"go.etcd.io/etcd/pkg/types"
	"go.etcd.io/etcd/pkg/wait"
	"go.etcd.io/etcd/raft"
	"go.etcd.io/etcd/raft/raftpb"
	"go.etcd.io/etcd/version"
	"go.etcd.io/etcd/wal"
	"go.uber.org/zap"
)

const (
	DefaultSnapshotCount = 100000

	// DefaultSnapshotCatchUpEntries is the number of entries for a slow follower
	// to catch-up after compacting the raft storage entries.
	// We expect the follower has a millisecond level latency with the leader.
	// The max throughput is around 10K. Keep a 5K entries is enough for helping
	// follower to catch up.
	DefaultSnapshotCatchUpEntries uint64 = 5000

	StoreClusterPrefix = "/0"
	StoreKeysPrefix    = "/1"

	// HealthInterval is the minimum time the cluster should be healthy
	// before accepting add member requests.
	HealthInterval = 5 * time.Second

	purgeFileInterval = 30 * time.Second
	// monitorVersionInterval should be smaller than the timeout
	// on the connection. Or we will not be able to reuse the connection
	// (since it will timeout).
	monitorVersionInterval = rafthttp.ConnWriteTimeout - time.Second

	// max number of in-flight snapshot messages etcdserver allows to have
	// This number is more than enough for most clusters with 5 machines.
	maxInFlightMsgSnap = 16

	releaseDelayAfterSnapshot = 30 * time.Second

	// maxPendingRevokes is the maximum number of outstanding expired lease revocations.
	maxPendingRevokes = 16

	recommendedMaxRequestBytes = 10 * 1024 * 1024

	readyPercent = 0.9
)

var (
	plog = capnslog.NewPackageLogger("go.etcd.io/etcd", "etcdserver")

	storeMemberAttributeRegexp = regexp.MustCompile(path.Join(membership.StoreMembersPrefix, "[[:xdigit:]]{1,16}", "attributes"))
)

func init() {
	rand.Seed(time.Now().UnixNano())

	expvar.Publish(
		"file_descriptor_limit",
		expvar.Func(
			func() interface{} {
				n, _ := runtime.FDLimit()
				return n
			},
		),
	)
}

type Response struct {
	Term    uint64
	Index   uint64
	Event   *v2store.Event
	Watcher v2store.Watcher
	Err     error
}

type ServerV2 interface {
	Server
	Leader() types.ID

	// Do takes a V2 request and attempts to fulfill it, returning a Response.
	Do(ctx context.Context, r pb.Request) (Response, error)
	stats.Stats
	ClientCertAuthEnabled() bool
}

type ServerV3 interface {
	Server
	RaftStatusGetter
}

func (s *EtcdServer) ClientCertAuthEnabled() bool { return s.Cfg.ClientCertAuthEnabled }

type Server interface {
	// AddMember attempts to add a member into the cluster. It will return
	// ErrIDRemoved if member ID is removed from the cluster, or return
	// ErrIDExists if member ID exists in the cluster.
	AddMember(ctx context.Context, memb membership.Member) ([]*membership.Member, error)
	// RemoveMember attempts to remove a member from the cluster. It will
	// return ErrIDRemoved if member ID is removed from the cluster, or return
	// ErrIDNotFound if member ID is not in the cluster.
	RemoveMember(ctx context.Context, id uint64) ([]*membership.Member, error)
	// UpdateMember attempts to update an existing member in the cluster. It will
	// return ErrIDNotFound if the member ID does not exist.
	UpdateMember(ctx context.Context, updateMemb membership.Member) ([]*membership.Member, error)
	// PromoteMember attempts to promote a non-voting node to a voting node. It will
	// return ErrIDNotFound if the member ID does not exist.
	// return ErrLearnerNotReady if the member are not ready.
	// return ErrMemberNotLearner if the member is not a learner.
	PromoteMember(ctx context.Context, id uint64) ([]*membership.Member, error)

	// ClusterVersion is the cluster-wide minimum major.minor version.
	// Cluster version is set to the min version that an etcd member is
	// compatible with when first bootstrap.
	//
	// ClusterVersion is nil until the cluster is bootstrapped (has a quorum).
	//
	// During a rolling upgrades, the ClusterVersion will be updated
	// automatically after a sync. (5 second by default)
	//
	// The API/raft component can utilize ClusterVersion to determine if
	// it can accept a client request or a raft RPC.
	// NOTE: ClusterVersion might be nil when etcd 2.1 works with etcd 2.0 and
	// the leader is etcd 2.0. etcd 2.0 leader will not update clusterVersion since
	// this feature is introduced post 2.0.
	ClusterVersion() *semver.Version
	Cluster() api.Cluster
	Alarms() []*pb.AlarmMember
}

// EtcdServer is the production implementation of the Server interface
type EtcdServer struct {
	// inflightSnapshots holds count the number of snapshots currently inflight.
	inflightSnapshots int64  // must use atomic operations to access; keep 64-bit aligned.
	appliedIndex      uint64 // must use atomic operations to access; keep 64-bit aligned.
	committedIndex    uint64 // must use atomic operations to access; keep 64-bit aligned.
	term              uint64 // must use atomic operations to access; keep 64-bit aligned.
	lead              uint64 // must use atomic operations to access; keep 64-bit aligned.

	// consistIndex used to hold the offset of current executing entry
	// It is initialized to 0 before executing any entry.
	consistIndex consistentIndex // must use atomic operations to access; keep 64-bit aligned.
	r            raftNode        // uses 64-bit atomics; keep 64-bit aligned.

	readych chan struct{}
	Cfg     ServerConfig

	lgMu *sync.RWMutex
	lg   *zap.Logger

	w wait.Wait

	readMu sync.RWMutex
	// read routine notifies etcd server that it waits for reading by sending an empty struct to
	// readwaitC
	readwaitc chan struct{}
	// readNotifier is used to notify the read routine that it can process the request
	// when there is no error
	readNotifier *notifier

	// stop signals the run goroutine should shutdown.
	stop chan struct{}
	// stopping is closed by run goroutine on shutdown.
	stopping chan struct{}
	// done is closed when all goroutines from start() complete.
	done chan struct{}
	// leaderChanged is used to notify the linearizable read loop to drop the old read requests.
	leaderChanged   chan struct{}
	leaderChangedMu sync.RWMutex

	errorc     chan error
	id         types.ID
	attributes membership.Attributes

	cluster *membership.RaftCluster

	v2store     v2store.Store
	snapshotter *snap.Snapshotter

	applyV2 ApplierV2

	// applyV3 is the applier with auth and quotas
	applyV3 applierV3
	// applyV3Base is the core applier without auth or quotas
	applyV3Base applierV3
	applyWait   wait.WaitTime

	kv         mvcc.ConsistentWatchableKV
	lessor     lease.Lessor
	bemu       sync.Mutex
	be         backend.Backend
	authStore  auth.AuthStore
	alarmStore *v3alarm.AlarmStore

	stats  *stats.ServerStats
	lstats *stats.LeaderStats

	SyncTicker *time.Ticker
	// compactor is used to auto-compact the KV.
	compactor v3compactor.Compactor

	// peerRt used to send requests (version, lease) to peers.
	peerRt   http.RoundTripper
	reqIDGen *idutil.Generator

	// forceVersionC is used to force the version monitor loop
	// to detect the cluster version immediately.
	forceVersionC chan struct{}

	// wgMu blocks concurrent waitgroup mutation while server stopping
	wgMu sync.RWMutex
	// wg is used to wait for the go routines that depends on the server state
	// to exit when stopping the server.
	wg sync.WaitGroup

	// ctx is used for etcd-initiated requests that may need to be canceled
	// on etcd server shutdown.
	ctx    context.Context
	cancel context.CancelFunc

	leadTimeMu      sync.RWMutex
	leadElectedTime time.Time

	*AccessController
}

// NewServer creates a new EtcdServer from the supplied configuration. The
// configuration is considered static for the lifetime of the EtcdServer.
func NewServer(cfg ServerConfig) (srv *EtcdServer, err error) {
	st := v2store.New(StoreClusterPrefix, StoreKeysPrefix)

	var (
		w  *wal.WAL
		n  raft.Node
		s  *raft.MemoryStorage
		id types.ID
		cl *membership.RaftCluster
	)

	if cfg.MaxRequestBytes > recommendedMaxRequestBytes {
		if cfg.Logger != nil {
			cfg.Logger.Warn(
				"exceeded recommended request limit",
				zap.Uint("max-request-bytes", cfg.MaxRequestBytes),
				zap.String("max-request-size", humanize.Bytes(uint64(cfg.MaxRequestBytes))),
				zap.Int("recommended-request-bytes", recommendedMaxRequestBytes),
				zap.String("recommended-request-size", humanize.Bytes(uint64(recommendedMaxRequestBytes))),
			)
		} else {
			plog.Warningf("MaxRequestBytes %v exceeds maximum recommended size %v", cfg.MaxRequestBytes, recommendedMaxRequestBytes)
		}
	}

	if terr := fileutil.TouchDirAll(cfg.DataDir); terr != nil {
		return nil, fmt.Errorf("cannot access data directory: %v", terr)
	}

	haveWAL := wal.Exist(cfg.WALDir())

	if err = fileutil.TouchDirAll(cfg.SnapDir()); err != nil {
		if cfg.Logger != nil {
			cfg.Logger.Fatal(
				"failed to create snapshot directory",
				zap.String("path", cfg.SnapDir()),
				zap.Error(err),
			)
		} else {
			plog.Fatalf("create snapshot directory error: %v", err)
		}
	}
	ss := snap.New(cfg.Logger, cfg.SnapDir())

	bepath := cfg.backendPath()
	beExist := fileutil.Exist(bepath)
	be := openBackend(cfg)

	defer func() {
		if err != nil {
			be.Close()
		}
	}()

	prt, err := rafthttp.NewRoundTripper(cfg.PeerTLSInfo, cfg.peerDialTimeout())
	if err != nil {
		return nil, err
	}
	var (
		remotes  []*membership.Member
		snapshot *raftpb.Snapshot
	)

	switch {
	case !haveWAL && !cfg.NewCluster:
		if err = cfg.VerifyJoinExisting(); err != nil {
			return nil, err
		}
		cl, err = membership.NewClusterFromURLsMap(cfg.Logger, cfg.InitialClusterToken, cfg.InitialPeerURLsMap)
		if err != nil {
			return nil, err
		}
		existingCluster, gerr := GetClusterFromRemotePeers(cfg.Logger, getRemotePeerURLs(cl, cfg.Name), prt)
		if gerr != nil {
			return nil, fmt.Errorf("cannot fetch cluster info from peer urls: %v", gerr)
		}
		if err = membership.ValidateClusterAndAssignIDs(cfg.Logger, cl, existingCluster); err != nil {
			return nil, fmt.Errorf("error validating peerURLs %s: %v", existingCluster, err)
		}
		if !isCompatibleWithCluster(cfg.Logger, cl, cl.MemberByName(cfg.Name).ID, prt) {
			return nil, fmt.Errorf("incompatible with current running cluster")
		}

		remotes = existingCluster.Members()
		cl.SetID(types.ID(0), existingCluster.ID())
		cl.SetStore(st)
		cl.SetBackend(be)
		id, n, s, w = startNode(cfg, cl, nil)
		cl.SetID(id, existingCluster.ID())

	case !haveWAL && cfg.NewCluster:
		if err = cfg.VerifyBootstrap(); err != nil {
			return nil, err
		}
		cl, err = membership.NewClusterFromURLsMap(cfg.Logger, cfg.InitialClusterToken, cfg.InitialPeerURLsMap)
		if err != nil {
			return nil, err
		}
		m := cl.MemberByName(cfg.Name)
		if isMemberBootstrapped(cfg.Logger, cl, cfg.Name, prt, cfg.bootstrapTimeout()) {
			return nil, fmt.Errorf("member %s has already been bootstrapped", m.ID)
		}
		if cfg.ShouldDiscover() {
			var str string
			str, err = v2discovery.JoinCluster(cfg.Logger, cfg.DiscoveryURL, cfg.DiscoveryProxy, m.ID, cfg.InitialPeerURLsMap.String())
			if err != nil {
				return nil, &DiscoveryError{Op: "join", Err: err}
			}
			var urlsmap types.URLsMap
			urlsmap, err = types.NewURLsMap(str)
			if err != nil {
				return nil, err
			}
			if checkDuplicateURL(urlsmap) {
				return nil, fmt.Errorf("discovery cluster %s has duplicate url", urlsmap)
			}
			if cl, err = membership.NewClusterFromURLsMap(cfg.Logger, cfg.InitialClusterToken, urlsmap); err != nil {
				return nil, err
			}
		}
		cl.SetStore(st)
		cl.SetBackend(be)
		id, n, s, w = startNode(cfg, cl, cl.MemberIDs())
		cl.SetID(id, cl.ID())

	case haveWAL:
		if err = fileutil.IsDirWriteable(cfg.MemberDir()); err != nil {
			return nil, fmt.Errorf("cannot write to member directory: %v", err)
		}

		if err = fileutil.IsDirWriteable(cfg.WALDir()); err != nil {
			return nil, fmt.Errorf("cannot write to WAL directory: %v", err)
		}

		if cfg.ShouldDiscover() {
			if cfg.Logger != nil {
				cfg.Logger.Warn(
					"discovery token is ignored since cluster already initialized; valid logs are found",
					zap.String("wal-dir", cfg.WALDir()),
				)
			} else {
				plog.Warningf("discovery token ignored since a cluster has already been initialized. Valid log found at %q", cfg.WALDir())
			}
		}

		// Find a snapshot to start/restart a raft node
		walSnaps, serr := wal.ValidSnapshotEntries(cfg.Logger, cfg.WALDir())
		if serr != nil {
			return nil, serr
		}
		// snapshot files can be orphaned if etcd crashes after writing them but before writing the corresponding
		// wal log entries
		snapshot, err = ss.LoadNewestAvailable(walSnaps)
		if err != nil && err != snap.ErrNoSnapshot {
			return nil, err
		}

		if snapshot != nil {
			if err = st.Recovery(snapshot.Data); err != nil {
				if cfg.Logger != nil {
					cfg.Logger.Panic("failed to recover from snapshot")
				} else {
					plog.Panicf("recovered store from snapshot error: %v", err)
				}
			}

			if cfg.Logger != nil {
				cfg.Logger.Info(
					"recovered v2 store from snapshot",
					zap.Uint64("snapshot-index", snapshot.Metadata.Index),
					zap.String("snapshot-size", humanize.Bytes(uint64(snapshot.Size()))),
				)
			} else {
				plog.Infof("recovered store from snapshot at index %d", snapshot.Metadata.Index)
			}

			if be, err = recoverSnapshotBackend(cfg, be, *snapshot); err != nil {
				if cfg.Logger != nil {
					cfg.Logger.Panic("failed to recover v3 backend from snapshot", zap.Error(err))
				} else {
					plog.Panicf("recovering backend from snapshot error: %v", err)
				}
			}
			if cfg.Logger != nil {
				s1, s2 := be.Size(), be.SizeInUse()
				cfg.Logger.Info(
					"recovered v3 backend from snapshot",
					zap.Int64("backend-size-bytes", s1),
					zap.String("backend-size", humanize.Bytes(uint64(s1))),
					zap.Int64("backend-size-in-use-bytes", s2),
					zap.String("backend-size-in-use", humanize.Bytes(uint64(s2))),
				)
			}
		}

		if !cfg.ForceNewCluster {
			id, cl, n, s, w = restartNode(cfg, snapshot)
		} else {
			id, cl, n, s, w = restartAsStandaloneNode(cfg, snapshot)
		}

		cl.SetStore(st)
		cl.SetBackend(be)
		cl.Recover(api.UpdateCapability)
		if cl.Version() != nil && !cl.Version().LessThan(semver.Version{Major: 3}) && !beExist {
			os.RemoveAll(bepath)
			return nil, fmt.Errorf("database file (%v) of the backend is missing", bepath)
		}

	default:
		return nil, fmt.Errorf("unsupported bootstrap config")
	}

	if terr := fileutil.TouchDirAll(cfg.MemberDir()); terr != nil {
		return nil, fmt.Errorf("cannot access member directory: %v", terr)
	}

	sstats := stats.NewServerStats(cfg.Name, id.String())
	lstats := stats.NewLeaderStats(id.String())

	heartbeat := time.Duration(cfg.TickMs) * time.Millisecond
	srv = &EtcdServer{
		readych:     make(chan struct{}),
		Cfg:         cfg,
		lgMu:        new(sync.RWMutex),
		lg:          cfg.Logger,
		errorc:      make(chan error, 1),
		v2store:     st,
		snapshotter: ss,
		r: *newRaftNode(
			raftNodeConfig{
				lg:          cfg.Logger,
				isIDRemoved: func(id uint64) bool { return cl.IsIDRemoved(types.ID(id)) },
				Node:        n,
				heartbeat:   heartbeat,
				raftStorage: s,
				storage:     NewStorage(w, ss),
			},
		),
		id:               id,
		attributes:       membership.Attributes{Name: cfg.Name, ClientURLs: cfg.ClientURLs.StringSlice()},
		cluster:          cl,
		stats:            sstats,
		lstats:           lstats,
		SyncTicker:       time.NewTicker(500 * time.Millisecond),
		peerRt:           prt,
		reqIDGen:         idutil.NewGenerator(uint16(id), time.Now()),
		forceVersionC:    make(chan struct{}),
		AccessController: &AccessController{CORS: cfg.CORS, HostWhitelist: cfg.HostWhitelist},
	}
	serverID.With(prometheus.Labels{"server_id": id.String()}).Set(1)

	srv.applyV2 = &applierV2store{store: srv.v2store, cluster: srv.cluster}

	srv.be = be
	minTTL := time.Duration((3*cfg.ElectionTicks)/2) * heartbeat

	// always recover lessor before kv. When we recover the mvcc.KV it will reattach keys to its leases.
	// If we recover mvcc.KV first, it will attach the keys to the wrong lessor before it recovers.
	srv.lessor = lease.NewLessor(
		srv.getLogger(),
		srv.be,
		lease.LessorConfig{
			MinLeaseTTL:                int64(math.Ceil(minTTL.Seconds())),
			CheckpointInterval:         cfg.LeaseCheckpointInterval,
			ExpiredLeasesRetryInterval: srv.Cfg.ReqTimeout(),
		})

	tp, err := auth.NewTokenProvider(cfg.Logger, cfg.AuthToken,
		func(index uint64) <-chan struct{} {
			return srv.applyWait.Wait(index)
		},
		time.Duration(cfg.TokenTTL)*time.Second,
	)
	if err != nil {
		if cfg.Logger != nil {
			cfg.Logger.Warn("failed to create token provider", zap.Error(err))
		} else {
			plog.Warningf("failed to create token provider,err is %v", err)
		}
		return nil, err
	}
	srv.authStore = auth.NewAuthStore(srv.getLogger(), srv.be, tp, int(cfg.BcryptCost))

	srv.kv = mvcc.New(srv.getLogger(), srv.be, srv.lessor, srv.authStore, &srv.consistIndex, mvcc.StoreConfig{CompactionBatchLimit: cfg.CompactionBatchLimit})
	if beExist {
		kvindex := srv.kv.ConsistentIndex()
		// TODO: remove kvindex != 0 checking when we do not expect users to upgrade
		// etcd from pre-3.0 release.
		if snapshot != nil && kvindex < snapshot.Metadata.Index {
			if kvindex != 0 {
				return nil, fmt.Errorf("database file (%v index %d) does not match with snapshot (index %d)", bepath, kvindex, snapshot.Metadata.Index)
			}
			if cfg.Logger != nil {
				cfg.Logger.Warn(
					"consistent index was never saved",
					zap.Uint64("snapshot-index", snapshot.Metadata.Index),
				)
			} else {
				plog.Warningf("consistent index never saved (snapshot index=%d)", snapshot.Metadata.Index)
			}
		}
	}
	newSrv := srv // since srv == nil in defer if srv is returned as nil
	defer func() {
		// closing backend without first closing kv can cause
		// resumed compactions to fail with closed tx errors
		if err != nil {
			newSrv.kv.Close()
		}
	}()

	srv.consistIndex.setConsistentIndex(srv.kv.ConsistentIndex())
	if num := cfg.AutoCompactionRetention; num != 0 {
		srv.compactor, err = v3compactor.New(cfg.Logger, cfg.AutoCompactionMode, num, srv.kv, srv)
		if err != nil {
			return nil, err
		}
		srv.compactor.Run()
	}

	srv.applyV3Base = srv.newApplierV3Backend()
	if err = srv.restoreAlarms(); err != nil {
		return nil, err
	}

	if srv.Cfg.EnableLeaseCheckpoint {
		// setting checkpointer enables lease checkpoint feature.
		srv.lessor.SetCheckpointer(func(ctx context.Context, cp *pb.LeaseCheckpointRequest) {
			srv.raftRequestOnce(ctx, pb.InternalRaftRequest{LeaseCheckpoint: cp})
		})
	}

	// TODO: move transport initialization near the definition of remote
	tr := &rafthttp.Transport{
		Logger:      cfg.Logger,
		TLSInfo:     cfg.PeerTLSInfo,
		DialTimeout: cfg.peerDialTimeout(),
		ID:          id,
		URLs:        cfg.PeerURLs,
		ClusterID:   cl.ID(),
		Raft:        srv,
		Snapshotter: ss,
		ServerStats: sstats,
		LeaderStats: lstats,
		ErrorC:      srv.errorc,
	}
	if err = tr.Start(); err != nil {
		return nil, err
	}
	// add all remotes into transport
	for _, m := range remotes {
		if m.ID != id {
			tr.AddRemote(m.ID, m.PeerURLs)
		}
	}
	for _, m := range cl.Members() {
		if m.ID != id {
			tr.AddPeer(m.ID, m.PeerURLs)
		}
	}
	srv.r.transport = tr

	return srv, nil
}

func (s *EtcdServer) getLogger() *zap.Logger {
	s.lgMu.RLock()
	l := s.lg
	s.lgMu.RUnlock()
	return l
}

func tickToDur(ticks int, tickMs uint) string {
	return fmt.Sprintf("%v", time.Duration(ticks)*time.Duration(tickMs)*time.Millisecond)
}

func (s *EtcdServer) adjustTicks() {
	lg := s.getLogger()
	clusterN := len(s.cluster.Members())

	// single-node fresh start, or single-node recovers from snapshot
	if clusterN == 1 {
		ticks := s.Cfg.ElectionTicks - 1
		if lg != nil {
			lg.Info(
				"started as single-node; fast-forwarding election ticks",
				zap.String("local-member-id", s.ID().String()),
				zap.Int("forward-ticks", ticks),
				zap.String("forward-duration", tickToDur(ticks, s.Cfg.TickMs)),
				zap.Int("election-ticks", s.Cfg.ElectionTicks),
				zap.String("election-timeout", tickToDur(s.Cfg.ElectionTicks, s.Cfg.TickMs)),
			)
		} else {
			plog.Infof("%s as single-node; fast-forwarding %d ticks (election ticks %d)", s.ID(), ticks, s.Cfg.ElectionTicks)
		}
		s.r.advanceTicks(ticks)
		return
	}

	if !s.Cfg.InitialElectionTickAdvance {
		if lg != nil {
			lg.Info("skipping initial election tick advance", zap.Int("election-ticks", s.Cfg.ElectionTicks))
		}
		return
	}
	if lg != nil {
		lg.Info("starting initial election tick advance", zap.Int("election-ticks", s.Cfg.ElectionTicks))
	}

	// retry up to "rafthttp.ConnReadTimeout", which is 5-sec
	// until peer connection reports; otherwise:
	// 1. all connections failed, or
	// 2. no active peers, or
	// 3. restarted single-node with no snapshot
	// then, do nothing, because advancing ticks would have no effect
	waitTime := rafthttp.ConnReadTimeout
	itv := 50 * time.Millisecond
	for i := int64(0); i < int64(waitTime/itv); i++ {
		select {
		case <-time.After(itv):
		case <-s.stopping:
			return
		}

		peerN := s.r.transport.ActivePeers()
		if peerN > 1 {
			// multi-node received peer connection reports
			// adjust ticks, in case slow leader message receive
			ticks := s.Cfg.ElectionTicks - 2

			if lg != nil {
				lg.Info(
					"initialized peer connections; fast-forwarding election ticks",
					zap.String("local-member-id", s.ID().String()),
					zap.Int("forward-ticks", ticks),
					zap.String("forward-duration", tickToDur(ticks, s.Cfg.TickMs)),
					zap.Int("election-ticks", s.Cfg.ElectionTicks),
					zap.String("election-timeout", tickToDur(s.Cfg.ElectionTicks, s.Cfg.TickMs)),
					zap.Int("active-remote-members", peerN),
				)
			} else {
				plog.Infof("%s initialized peer connection; fast-forwarding %d ticks (election ticks %d) with %d active peer(s)", s.ID(), ticks, s.Cfg.ElectionTicks, peerN)
			}

			s.r.advanceTicks(ticks)
			return
		}
	}
}

// Start performs any initialization of the Server necessary for it to
// begin serving requests. It must be called before Do or Process.
// Start must be non-blocking; any long-running server functionality
// should be implemented in goroutines.
func (s *EtcdServer) Start() {
	s.start()
	s.goAttach(func() { s.adjustTicks() })
	s.goAttach(func() { s.publish(s.Cfg.ReqTimeout()) })
	s.goAttach(s.purgeFile)
	s.goAttach(func() { monitorFileDescriptor(s.getLogger(), s.stopping) })
	s.goAttach(s.monitorVersions)
	s.goAttach(s.linearizableReadLoop)
	s.goAttach(s.monitorKVHash)
}

// start prepares and starts server in a new goroutine. It is no longer safe to
// modify a server's fields after it has been sent to Start.
// This function is just used for testing.
func (s *EtcdServer) start() {
	lg := s.getLogger()

	if s.Cfg.SnapshotCount == 0 {
		if lg != nil {
			lg.Info(
				"updating snapshot-count to default",
				zap.Uint64("given-snapshot-count", s.Cfg.SnapshotCount),
				zap.Uint64("updated-snapshot-count", DefaultSnapshotCount),
			)
		} else {
			plog.Infof("set snapshot count to default %d", DefaultSnapshotCount)
		}
		s.Cfg.SnapshotCount = DefaultSnapshotCount
	}
	if s.Cfg.SnapshotCatchUpEntries == 0 {
		if lg != nil {
			lg.Info(
				"updating snapshot catch-up entries to default",
				zap.Uint64("given-snapshot-catchup-entries", s.Cfg.SnapshotCatchUpEntries),
				zap.Uint64("updated-snapshot-catchup-entries", DefaultSnapshotCatchUpEntries),
			)
		}
		s.Cfg.SnapshotCatchUpEntries = DefaultSnapshotCatchUpEntries
	}

	s.w = wait.New()
	s.applyWait = wait.NewTimeList()
	s.done = make(chan struct{})
	s.stop = make(chan struct{})
	s.stopping = make(chan struct{})
	s.ctx, s.cancel = context.WithCancel(context.Background())
	s.readwaitc = make(chan struct{}, 1)
	s.readNotifier = newNotifier()
	s.leaderChanged = make(chan struct{})
	if s.ClusterVersion() != nil {
		if lg != nil {
			lg.Info(
				"starting etcd server",
				zap.String("local-member-id", s.ID().String()),
				zap.String("local-server-version", version.Version),
				zap.String("cluster-id", s.Cluster().ID().String()),
				zap.String("cluster-version", version.Cluster(s.ClusterVersion().String())),
			)
		} else {
			plog.Infof("starting server... [version: %v, cluster version: %v]", version.Version, version.Cluster(s.ClusterVersion().String()))
		}
		membership.ClusterVersionMetrics.With(prometheus.Labels{"cluster_version": version.Cluster(s.ClusterVersion().String())}).Set(1)
	} else {
		if lg != nil {
			lg.Info(
				"starting etcd server",
				zap.String("local-member-id", s.ID().String()),
				zap.String("local-server-version", version.Version),
				zap.String("cluster-version", "to_be_decided"),
			)
		} else {
			plog.Infof("starting server... [version: %v, cluster version: to_be_decided]", version.Version)
		}
	}

	// TODO: if this is an empty log, writes all peer infos
	// into the first entry
	go s.run()
}

func (s *EtcdServer) purgeFile() {
	var dberrc, serrc, werrc <-chan error
	var dbdonec, sdonec, wdonec <-chan struct{}
	if s.Cfg.MaxSnapFiles > 0 {
		dbdonec, dberrc = fileutil.PurgeFileWithDoneNotify(s.getLogger(), s.Cfg.SnapDir(), "snap.db", s.Cfg.MaxSnapFiles, purgeFileInterval, s.stopping)
		sdonec, serrc = fileutil.PurgeFileWithDoneNotify(s.getLogger(), s.Cfg.SnapDir(), "snap", s.Cfg.MaxSnapFiles, purgeFileInterval, s.stopping)
	}
	if s.Cfg.MaxWALFiles > 0 {
		wdonec, werrc = fileutil.PurgeFileWithDoneNotify(s.getLogger(), s.Cfg.WALDir(), "wal", s.Cfg.MaxWALFiles, purgeFileInterval, s.stopping)
	}

	lg := s.getLogger()
	select {
	case e := <-dberrc:
		if lg != nil {
			lg.Fatal("failed to purge snap db file", zap.Error(e))
		} else {
			plog.Fatalf("failed to purge snap db file %v", e)
		}
	case e := <-serrc:
		if lg != nil {
			lg.Fatal("failed to purge snap file", zap.Error(e))
		} else {
			plog.Fatalf("failed to purge snap file %v", e)
		}
	case e := <-werrc:
		if lg != nil {
			lg.Fatal("failed to purge wal file", zap.Error(e))
		} else {
			plog.Fatalf("failed to purge wal file %v", e)
		}
	case <-s.stopping:
		if dbdonec != nil {
			<-dbdonec
		}
		if sdonec != nil {
			<-sdonec
		}
		if wdonec != nil {
			<-wdonec
		}
		return
	}
}

func (s *EtcdServer) Cluster() api.Cluster { return s.cluster }

func (s *EtcdServer) ApplyWait() <-chan struct{} { return s.applyWait.Wait(s.getCommittedIndex()) }

type ServerPeer interface {
	ServerV2
	RaftHandler() http.Handler
	LeaseHandler() http.Handler
}

func (s *EtcdServer) LeaseHandler() http.Handler {
	if s.lessor == nil {
		return nil
	}
	return leasehttp.NewHandler(s.lessor, s.ApplyWait)
}

func (s *EtcdServer) RaftHandler() http.Handler { return s.r.transport.Handler() }

// Process takes a raft message and applies it to the server's raft state
// machine, respecting any timeout of the given context.
func (s *EtcdServer) Process(ctx context.Context, m raftpb.Message) error {
	if s.cluster.IsIDRemoved(types.ID(m.From)) {
		if lg := s.getLogger(); lg != nil {
			lg.Warn(
				"rejected Raft message from removed member",
				zap.String("local-member-id", s.ID().String()),
				zap.String("removed-member-id", types.ID(m.From).String()),
			)
		} else {
			plog.Warningf("reject message from removed member %s", types.ID(m.From).String())
		}
		return httptypes.NewHTTPError(http.StatusForbidden, "cannot process message from removed member")
	}
	if m.Type == raftpb.MsgApp {
		s.stats.RecvAppendReq(types.ID(m.From).String(), m.Size())
	}
	return s.r.Step(ctx, m)
}

func (s *EtcdServer) IsIDRemoved(id uint64) bool { return s.cluster.IsIDRemoved(types.ID(id)) }

func (s *EtcdServer) ReportUnreachable(id uint64) { s.r.ReportUnreachable(id) }

// ReportSnapshot reports snapshot sent status to the raft state machine,
// and clears the used snapshot from the snapshot store.
func (s *EtcdServer) ReportSnapshot(id uint64, status raft.SnapshotStatus) {
	s.r.ReportSnapshot(id, status)
}

type etcdProgress struct {
	confState raftpb.ConfState
	snapi     uint64
	appliedt  uint64
	appliedi  uint64
}

// raftReadyHandler contains a set of EtcdServer operations to be called by raftNode,
// and helps decouple state machine logic from Raft algorithms.
// TODO: add a state machine interface to apply the commit entries and do snapshot/recover
type raftReadyHandler struct {
	getLead              func() (lead uint64)
	updateLead           func(lead uint64)
	updateLeadership     func(newLeader bool)
	updateCommittedIndex func(uint64)
}

func (s *EtcdServer) run() {
	lg := s.getLogger()

	sn, err := s.r.raftStorage.Snapshot()
	if err != nil {
		if lg != nil {
			lg.Panic("failed to get snapshot from Raft storage", zap.Error(err))
		} else {
			plog.Panicf("get snapshot from raft storage error: %v", err)
		}
	}

	// asynchronously accept apply packets, dispatch progress in-order
	sched := schedule.NewFIFOScheduler()

	var (
		smu   sync.RWMutex
		syncC <-chan time.Time
	)
	setSyncC := func(ch <-chan time.Time) {
		smu.Lock()
		syncC = ch
		smu.Unlock()
	}
	getSyncC := func() (ch <-chan time.Time) {
		smu.RLock()
		ch = syncC
		smu.RUnlock()
		return
	}
	rh := &raftReadyHandler{
		getLead:    func() (lead uint64) { return s.getLead() },
		updateLead: func(lead uint64) { s.setLead(lead) },
		updateLeadership: func(newLeader bool) {
			if !s.isLeader() {
				if s.lessor != nil {
					s.lessor.Demote()
				}
				if s.compactor != nil {
					s.compactor.Pause()
				}
				setSyncC(nil)
			} else {
				if newLeader {
					t := time.Now()
					s.leadTimeMu.Lock()
					s.leadElectedTime = t
					s.leadTimeMu.Unlock()
				}
				setSyncC(s.SyncTicker.C)
				if s.compactor != nil {
					s.compactor.Resume()
				}
			}
			if newLeader {
				s.leaderChangedMu.Lock()
				lc := s.leaderChanged
				s.leaderChanged = make(chan struct{})
				close(lc)
				s.leaderChangedMu.Unlock()
			}
			// TODO: remove the nil checking
			// current test utility does not provide the stats
			if s.stats != nil {
				s.stats.BecomeLeader()
			}
		},
		updateCommittedIndex: func(ci uint64) {
			cci := s.getCommittedIndex()
			if ci > cci {
				s.setCommittedIndex(ci)
			}
		},
	}
	s.r.start(rh)

	ep := etcdProgress{
		confState: sn.Metadata.ConfState,
		snapi:     sn.Metadata.Index,
		appliedt:  sn.Metadata.Term,
		appliedi:  sn.Metadata.Index,
	}

	defer func() {
		s.wgMu.Lock() // block concurrent waitgroup adds in goAttach while stopping
		close(s.stopping)
		s.wgMu.Unlock()
		s.cancel()

		sched.Stop()

		// wait for gouroutines before closing raft so wal stays open
		s.wg.Wait()

		s.SyncTicker.Stop()

		// must stop raft after scheduler-- etcdserver can leak rafthttp pipelines
		// by adding a peer after raft stops the transport
		s.r.stop()

		// kv, lessor and backend can be nil if running without v3 enabled
		// or running unit tests.
		if s.lessor != nil {
			s.lessor.Stop()
		}
		if s.kv != nil {
			s.kv.Close()
		}
		if s.authStore != nil {
			s.authStore.Close()
		}
		if s.be != nil {
			s.be.Close()
		}
		if s.compactor != nil {
			s.compactor.Stop()
		}
		close(s.done)
	}()

	var expiredLeaseC <-chan []*lease.Lease
	if s.lessor != nil {
		expiredLeaseC = s.lessor.ExpiredLeasesC()
	}

	for {
		select {
		case ap := <-s.r.apply():
			f := func(context.Context) { s.applyAll(&ep, &ap) }
			sched.Schedule(f)
		case leases := <-expiredLeaseC:
			s.goAttach(func() {
				// Increases throughput of expired leases deletion process through parallelization
				c := make(chan struct{}, maxPendingRevokes)
				for _, lease := range leases {
					select {
					case c <- struct{}{}:
					case <-s.stopping:
						return
					}
					lid := lease.ID
					s.goAttach(func() {
						ctx := s.authStore.WithRoot(s.ctx)
						_, lerr := s.LeaseRevoke(ctx, &pb.LeaseRevokeRequest{ID: int64(lid)})
						if lerr == nil {
							leaseExpired.Inc()
						} else {
							if lg != nil {
								lg.Warn(
									"failed to revoke lease",
									zap.String("lease-id", fmt.Sprintf("%016x", lid)),
									zap.Error(lerr),
								)
							} else {
								plog.Warningf("failed to revoke %016x (%q)", lid, lerr.Error())
							}
						}

						<-c
					})
				}
			})
		case err := <-s.errorc:
			if lg != nil {
				lg.Warn("server error", zap.Error(err))
				lg.Warn("data-dir used by this member must be removed")
			} else {
				plog.Errorf("%s", err)
				plog.Infof("the data-dir used by this member must be removed.")
			}
			return
		case <-getSyncC():
			if s.v2store.HasTTLKeys() {
				s.sync(s.Cfg.ReqTimeout())
			}
		case <-s.stop:
			return
		}
	}
}

func (s *EtcdServer) applyAll(ep *etcdProgress, apply *apply) {
	s.applySnapshot(ep, apply)
	s.applyEntries(ep, apply)

	proposalsApplied.Set(float64(ep.appliedi))
	s.applyWait.Trigger(ep.appliedi)

	// wait for the raft routine to finish the disk writes before triggering a
	// snapshot. or applied index might be greater than the last index in raft
	// storage, since the raft routine might be slower than apply routine.
	<-apply.notifyc

	s.triggerSnapshot(ep)
	select {
	// snapshot requested via send()
	case m := <-s.r.msgSnapC:
		merged := s.createMergedSnapshotMessage(m, ep.appliedt, ep.appliedi, ep.confState)
		s.sendMergedSnap(merged)
	default:
	}
}

func (s *EtcdServer) applySnapshot(ep *etcdProgress, apply *apply) {
	if raft.IsEmptySnap(apply.snapshot) {
		return
	}
	applySnapshotInProgress.Inc()

	lg := s.getLogger()
	if lg != nil {
		lg.Info(
			"applying snapshot",
			zap.Uint64("current-snapshot-index", ep.snapi),
			zap.Uint64("current-applied-index", ep.appliedi),
			zap.Uint64("incoming-leader-snapshot-index", apply.snapshot.Metadata.Index),
			zap.Uint64("incoming-leader-snapshot-term", apply.snapshot.Metadata.Term),
		)
	} else {
		plog.Infof("applying snapshot at index %d...", ep.snapi)
	}
	defer func() {
		if lg != nil {
			lg.Info(
				"applied snapshot",
				zap.Uint64("current-snapshot-index", ep.snapi),
				zap.Uint64("current-applied-index", ep.appliedi),
				zap.Uint64("incoming-leader-snapshot-index", apply.snapshot.Metadata.Index),
				zap.Uint64("incoming-leader-snapshot-term", apply.snapshot.Metadata.Term),
			)
		} else {
			plog.Infof("finished applying incoming snapshot at index %d", ep.snapi)
		}
		applySnapshotInProgress.Dec()
	}()

	if apply.snapshot.Metadata.Index <= ep.appliedi {
		if lg != nil {
			lg.Panic(
				"unexpected leader snapshot from outdated index",
				zap.Uint64("current-snapshot-index", ep.snapi),
				zap.Uint64("current-applied-index", ep.appliedi),
				zap.Uint64("incoming-leader-snapshot-index", apply.snapshot.Metadata.Index),
				zap.Uint64("incoming-leader-snapshot-term", apply.snapshot.Metadata.Term),
			)
		} else {
			plog.Panicf("snapshot index [%d] should > appliedi[%d] + 1",
				apply.snapshot.Metadata.Index, ep.appliedi)
		}
	}

	// wait for raftNode to persist snapshot onto the disk
	<-apply.notifyc

	newbe, err := openSnapshotBackend(s.Cfg, s.snapshotter, apply.snapshot)
	if err != nil {
		if lg != nil {
			lg.Panic("failed to open snapshot backend", zap.Error(err))
		} else {
			plog.Panic(err)
		}
	}

	// always recover lessor before kv. When we recover the mvcc.KV it will reattach keys to its leases.
	// If we recover mvcc.KV first, it will attach the keys to the wrong lessor before it recovers.
	if s.lessor != nil {
		if lg != nil {
			lg.Info("restoring lease store")
		} else {
			plog.Info("recovering lessor...")
		}

		s.lessor.Recover(newbe, func() lease.TxnDelete { return s.kv.Write(traceutil.TODO()) })

		if lg != nil {
			lg.Info("restored lease store")
		} else {
			plog.Info("finished recovering lessor")
		}
	}

	if lg != nil {
		lg.Info("restoring mvcc store")
	} else {
		plog.Info("restoring mvcc store...")
	}

	if err := s.kv.Restore(newbe); err != nil {
		if lg != nil {
			lg.Panic("failed to restore mvcc store", zap.Error(err))
		} else {
			plog.Panicf("restore KV error: %v", err)
		}
	}

	s.consistIndex.setConsistentIndex(s.kv.ConsistentIndex())
	if lg != nil {
		lg.Info("restored mvcc store")
	} else {
		plog.Info("finished restoring mvcc store")
	}

	// Closing old backend might block until all the txns
	// on the backend are finished.
	// We do not want to wait on closing the old backend.
	s.bemu.Lock()
	oldbe := s.be
	go func() {
		if lg != nil {
			lg.Info("closing old backend file")
		} else {
			plog.Info("closing old backend...")
		}
		defer func() {
			if lg != nil {
				lg.Info("closed old backend file")
			} else {
				plog.Info("finished closing old backend")
			}
		}()
		if err := oldbe.Close(); err != nil {
			if lg != nil {
				lg.Panic("failed to close old backend", zap.Error(err))
			} else {
				plog.Panicf("close backend error: %v", err)
			}
		}
	}()

	s.be = newbe
	s.bemu.Unlock()

	if lg != nil {
		lg.Info("restoring alarm store")
	} else {
		plog.Info("recovering alarms...")
	}

	if err := s.restoreAlarms(); err != nil {
		if lg != nil {
			lg.Panic("failed to restore alarm store", zap.Error(err))
		} else {
			plog.Panicf("restore alarms error: %v", err)
		}
	}

	if lg != nil {
		lg.Info("restored alarm store")
	} else {
		plog.Info("finished recovering alarms")
	}

	if s.authStore != nil {
		if lg != nil {
			lg.Info("restoring auth store")
		} else {
			plog.Info("recovering auth store...")
		}

		s.authStore.Recover(newbe)

		if lg != nil {
			lg.Info("restored auth store")
		} else {
			plog.Info("finished recovering auth store")
		}
	}

	if lg != nil {
		lg.Info("restoring v2 store")
	} else {
		plog.Info("recovering store v2...")
	}
	if err := s.v2store.Recovery(apply.snapshot.Data); err != nil {
		if lg != nil {
			lg.Panic("failed to restore v2 store", zap.Error(err))
		} else {
			plog.Panicf("recovery store error: %v", err)
		}
	}

	if lg != nil {
		lg.Info("restored v2 store")
	} else {
		plog.Info("finished recovering store v2")
	}

	s.cluster.SetBackend(newbe)

	if lg != nil {
		lg.Info("restoring cluster configuration")
	} else {
		plog.Info("recovering cluster configuration...")
	}

	s.cluster.Recover(api.UpdateCapability)

	if lg != nil {
		lg.Info("restored cluster configuration")
		lg.Info("removing old peers from network")
	} else {
		plog.Info("finished recovering cluster configuration")
		plog.Info("removing old peers from network...")
	}

	// recover raft transport
	s.r.transport.RemoveAllPeers()

	if lg != nil {
		lg.Info("removed old peers from network")
		lg.Info("adding peers from new cluster configuration")
	} else {
		plog.Info("finished removing old peers from network")
		plog.Info("adding peers from new cluster configuration into network...")
	}

	for _, m := range s.cluster.Members() {
		if m.ID == s.ID() {
			continue
		}
		s.r.transport.AddPeer(m.ID, m.PeerURLs)
	}

	if lg != nil {
		lg.Info("added peers from new cluster configuration")
	} else {
		plog.Info("finished adding peers from new cluster configuration into network...")
	}

	ep.appliedt = apply.snapshot.Metadata.Term
	ep.appliedi = apply.snapshot.Metadata.Index
	ep.snapi = ep.appliedi
	ep.confState = apply.snapshot.Metadata.ConfState
}

func (s *EtcdServer) applyEntries(ep *etcdProgress, apply *apply) {
	if len(apply.entries) == 0 {
		return
	}
	firsti := apply.entries[0].Index
	if firsti > ep.appliedi+1 {
		if lg := s.getLogger(); lg != nil {
			lg.Panic(
				"unexpected committed entry index",
				zap.Uint64("current-applied-index", ep.appliedi),
				zap.Uint64("first-committed-entry-index", firsti),
			)
		} else {
			plog.Panicf("first index of committed entry[%d] should <= appliedi[%d] + 1", firsti, ep.appliedi)
		}
	}
	var ents []raftpb.Entry
	if ep.appliedi+1-firsti < uint64(len(apply.entries)) {
		ents = apply.entries[ep.appliedi+1-firsti:]
	}
	if len(ents) == 0 {
		return
	}
	var shouldstop bool
	if ep.appliedt, ep.appliedi, shouldstop = s.apply(ents, &ep.confState); shouldstop {
		go s.stopWithDelay(10*100*time.Millisecond, fmt.Errorf("the member has been permanently removed from the cluster"))
	}
}

func (s *EtcdServer) triggerSnapshot(ep *etcdProgress) {
	if ep.appliedi-ep.snapi <= s.Cfg.SnapshotCount {
		return
	}

	if lg := s.getLogger(); lg != nil {
		lg.Info(
			"triggering snapshot",
			zap.String("local-member-id", s.ID().String()),
			zap.Uint64("local-member-applied-index", ep.appliedi),
			zap.Uint64("local-member-snapshot-index", ep.snapi),
			zap.Uint64("local-member-snapshot-count", s.Cfg.SnapshotCount),
		)
	} else {
		plog.Infof("start to snapshot (applied: %d, lastsnap: %d)", ep.appliedi, ep.snapi)
	}

	s.snapshot(ep.appliedi, ep.confState)
	ep.snapi = ep.appliedi
}

func (s *EtcdServer) hasMultipleVotingMembers() bool {
	return s.cluster != nil && len(s.cluster.VotingMemberIDs()) > 1
}

func (s *EtcdServer) isLeader() bool {
	return uint64(s.ID()) == s.Lead()
}

// MoveLeader transfers the leader to the given transferee.
func (s *EtcdServer) MoveLeader(ctx context.Context, lead, transferee uint64) error {
	if !s.cluster.IsMemberExist(types.ID(transferee)) || s.cluster.Member(types.ID(transferee)).IsLearner {
		return ErrBadLeaderTransferee
	}

	now := time.Now()
	interval := time.Duration(s.Cfg.TickMs) * time.Millisecond

	if lg := s.getLogger(); lg != nil {
		lg.Info(
			"leadership transfer starting",
			zap.String("local-member-id", s.ID().String()),
			zap.String("current-leader-member-id", types.ID(lead).String()),
			zap.String("transferee-member-id", types.ID(transferee).String()),
		)
	} else {
		plog.Infof("%s starts leadership transfer from %s to %s", s.ID(), types.ID(lead), types.ID(transferee))
	}

	s.r.TransferLeadership(ctx, lead, transferee)
	for s.Lead() != transferee {
		select {
		case <-ctx.Done(): // time out
			return ErrTimeoutLeaderTransfer
		case <-time.After(interval):
		}
	}

	// TODO: drain all requests, or drop all messages to the old leader
	if lg := s.getLogger(); lg != nil {
		lg.Info(
			"leadership transfer finished",
			zap.String("local-member-id", s.ID().String()),
			zap.String("old-leader-member-id", types.ID(lead).String()),
			zap.String("new-leader-member-id", types.ID(transferee).String()),
			zap.Duration("took", time.Since(now)),
		)
	} else {
		plog.Infof("%s finished leadership transfer from %s to %s (took %v)", s.ID(), types.ID(lead), types.ID(transferee), time.Since(now))
	}
	return nil
}

// TransferLeadership transfers the leader to the chosen transferee.
func (s *EtcdServer) TransferLeadership() error {
	if !s.isLeader() {
		if lg := s.getLogger(); lg != nil {
			lg.Info(
				"skipped leadership transfer; local server is not leader",
				zap.String("local-member-id", s.ID().String()),
				zap.String("current-leader-member-id", types.ID(s.Lead()).String()),
			)
		} else {
			plog.Printf("skipped leadership transfer for stopping non-leader member")
		}
		return nil
	}

	if !s.hasMultipleVotingMembers() {
		if lg := s.getLogger(); lg != nil {
			lg.Info(
				"skipped leadership transfer for single voting member cluster",
				zap.String("local-member-id", s.ID().String()),
				zap.String("current-leader-member-id", types.ID(s.Lead()).String()),
			)
		} else {
			plog.Printf("skipped leadership transfer for single voting member cluster")
		}
		return nil
	}

	transferee, ok := longestConnected(s.r.transport, s.cluster.VotingMemberIDs())
	if !ok {
		return ErrUnhealthy
	}

	tm := s.Cfg.ReqTimeout()
	ctx, cancel := context.WithTimeout(s.ctx, tm)
	err := s.MoveLeader(ctx, s.Lead(), uint64(transferee))
	cancel()
	return err
}

// HardStop stops the server without coordination with other members in the cluster.
func (s *EtcdServer) HardStop() {
	select {
	case s.stop <- struct{}{}:
	case <-s.done:
		return
	}
	<-s.done
}

// Stop stops the server gracefully, and shuts down the running goroutine.
// Stop should be called after a Start(s), otherwise it will block forever.
// When stopping leader, Stop transfers its leadership to one of its peers
// before stopping the server.
// Stop terminates the Server and performs any necessary finalization.
// Do and Process cannot be called after Stop has been invoked.
func (s *EtcdServer) Stop() {
	if err := s.TransferLeadership(); err != nil {
		if lg := s.getLogger(); lg != nil {
			lg.Warn("leadership transfer failed", zap.String("local-member-id", s.ID().String()), zap.Error(err))
		} else {
			plog.Warningf("%s failed to transfer leadership (%v)", s.ID(), err)
		}
	}
	s.HardStop()
}

// ReadyNotify returns a channel that will be closed when the server
// is ready to serve client requests
func (s *EtcdServer) ReadyNotify() <-chan struct{} { return s.readych }

func (s *EtcdServer) stopWithDelay(d time.Duration, err error) {
	select {
	case <-time.After(d):
	case <-s.done:
	}
	select {
	case s.errorc <- err:
	default:
	}
}

// StopNotify returns a channel that receives a empty struct
// when the server is stopped.
func (s *EtcdServer) StopNotify() <-chan struct{} { return s.done }

func (s *EtcdServer) SelfStats() []byte { return s.stats.JSON() }

func (s *EtcdServer) LeaderStats() []byte {
	lead := s.getLead()
	if lead != uint64(s.id) {
		return nil
	}
	return s.lstats.JSON()
}

func (s *EtcdServer) StoreStats() []byte { return s.v2store.JsonStats() }

func (s *EtcdServer) checkMembershipOperationPermission(ctx context.Context) error {
	if s.authStore == nil {
		// In the context of ordinary etcd process, s.authStore will never be nil.
		// This branch is for handling cases in server_test.go
		return nil
	}

	// Note that this permission check is done in the API layer,
	// so TOCTOU problem can be caused potentially in a schedule like this:
	// update membership with user A -> revoke root role of A -> apply membership change
	// in the state machine layer
	// However, both of membership change and role management requires the root privilege.
	// So careful operation by admins can prevent the problem.
	authInfo, err := s.AuthInfoFromCtx(ctx)
	if err != nil {
		return err
	}

	return s.AuthStore().IsAdminPermitted(authInfo)
}

func (s *EtcdServer) AddMember(ctx context.Context, memb membership.Member) ([]*membership.Member, error) {
	if err := s.checkMembershipOperationPermission(ctx); err != nil {
		return nil, err
	}

	// TODO: move Member to protobuf type
	b, err := json.Marshal(memb)
	if err != nil {
		return nil, err
	}

	// by default StrictReconfigCheck is enabled; reject new members if unhealthy.
	if err := s.mayAddMember(memb); err != nil {
		return nil, err
	}

	cc := raftpb.ConfChange{
		Type:    raftpb.ConfChangeAddNode,
		NodeID:  uint64(memb.ID),
		Context: b,
	}

	if memb.IsLearner {
		cc.Type = raftpb.ConfChangeAddLearnerNode
	}

	return s.configure(ctx, cc)
}

func (s *EtcdServer) mayAddMember(memb membership.Member) error {
	if !s.Cfg.StrictReconfigCheck {
		return nil
	}

	// protect quorum when adding voting member
	if !memb.IsLearner && !s.cluster.IsReadyToAddVotingMember() {
		if lg := s.getLogger(); lg != nil {
			lg.Warn(
				"rejecting member add request; not enough healthy members",
				zap.String("local-member-id", s.ID().String()),
				zap.String("requested-member-add", fmt.Sprintf("%+v", memb)),
				zap.Error(ErrNotEnoughStartedMembers),
			)
		} else {
			plog.Warningf("not enough started members, rejecting member add %+v", memb)
		}
		return ErrNotEnoughStartedMembers
	}

	if !isConnectedFullySince(s.r.transport, time.Now().Add(-HealthInterval), s.ID(), s.cluster.VotingMembers()) {
		if lg := s.getLogger(); lg != nil {
			lg.Warn(
				"rejecting member add request; local member has not been connected to all peers, reconfigure breaks active quorum",
				zap.String("local-member-id", s.ID().String()),
				zap.String("requested-member-add", fmt.Sprintf("%+v", memb)),
				zap.Error(ErrUnhealthy),
			)
		} else {
			plog.Warningf("not healthy for reconfigure, rejecting member add %+v", memb)
		}
		return ErrUnhealthy
	}

	return nil
}

func (s *EtcdServer) RemoveMember(ctx context.Context, id uint64) ([]*membership.Member, error) {
	if err := s.checkMembershipOperationPermission(ctx); err != nil {
		return nil, err
	}

	// by default StrictReconfigCheck is enabled; reject removal if leads to quorum loss
	if err := s.mayRemoveMember(types.ID(id)); err != nil {
		return nil, err
	}

	cc := raftpb.ConfChange{
		Type:   raftpb.ConfChangeRemoveNode,
		NodeID: id,
	}
	return s.configure(ctx, cc)
}

// PromoteMember promotes a learner node to a voting node.
func (s *EtcdServer) PromoteMember(ctx context.Context, id uint64) ([]*membership.Member, error) {
	// only raft leader has information on whether the to-be-promoted learner node is ready. If promoteMember call
	// fails with ErrNotLeader, forward the request to leader node via HTTP. If promoteMember call fails with error
	// other than ErrNotLeader, return the error.
	resp, err := s.promoteMember(ctx, id)
	if err == nil {
		learnerPromoteSucceed.Inc()
		return resp, nil
	}
	if err != ErrNotLeader {
		learnerPromoteFailed.WithLabelValues(err.Error()).Inc()
		return resp, err
	}

	cctx, cancel := context.WithTimeout(ctx, s.Cfg.ReqTimeout())
	defer cancel()
	// forward to leader
	for cctx.Err() == nil {
		leader, err := s.waitLeader(cctx)
		if err != nil {
			return nil, err
		}
		for _, url := range leader.PeerURLs {
			resp, err := promoteMemberHTTP(cctx, url, id, s.peerRt)
			if err == nil {
				return resp, nil
			}
			// If member promotion failed, return early. Otherwise keep retry.
			if err == ErrLearnerNotReady || err == membership.ErrIDNotFound || err == membership.ErrMemberNotLearner {
				return nil, err
			}
		}
	}

	if cctx.Err() == context.DeadlineExceeded {
		return nil, ErrTimeout
	}
	return nil, ErrCanceled
}

// promoteMember checks whether the to-be-promoted learner node is ready before sending the promote
// request to raft.
// The function returns ErrNotLeader if the local node is not raft leader (therefore does not have
// enough information to determine if the learner node is ready), returns ErrLearnerNotReady if the
// local node is leader (therefore has enough information) but decided the learner node is not ready
// to be promoted.
func (s *EtcdServer) promoteMember(ctx context.Context, id uint64) ([]*membership.Member, error) {
	if err := s.checkMembershipOperationPermission(ctx); err != nil {
		return nil, err
	}

	// check if we can promote this learner.
	if err := s.mayPromoteMember(types.ID(id)); err != nil {
		return nil, err
	}

	// build the context for the promote confChange. mark IsLearner to false and IsPromote to true.
	promoteChangeContext := membership.ConfigChangeContext{
		Member: membership.Member{
			ID: types.ID(id),
		},
		IsPromote: true,
	}

	b, err := json.Marshal(promoteChangeContext)
	if err != nil {
		return nil, err
	}

	cc := raftpb.ConfChange{
		Type:    raftpb.ConfChangeAddNode,
		NodeID:  id,
		Context: b,
	}

	return s.configure(ctx, cc)
}

func (s *EtcdServer) mayPromoteMember(id types.ID) error {
	err := s.isLearnerReady(uint64(id))
	if err != nil {
		return err
	}

	if !s.Cfg.StrictReconfigCheck {
		return nil
	}
	if !s.cluster.IsReadyToPromoteMember(uint64(id)) {
		if lg := s.getLogger(); lg != nil {
			lg.Warn(
				"rejecting member promote request; not enough healthy members",
				zap.String("local-member-id", s.ID().String()),
				zap.String("requested-member-remove-id", id.String()),
				zap.Error(ErrNotEnoughStartedMembers),
			)
		} else {
			plog.Warningf("not enough started members, rejecting promote member %s", id)
		}
		return ErrNotEnoughStartedMembers
	}

	return nil
}

// check whether the learner catches up with leader or not.
// Note: it will return nil if member is not found in cluster or if member is not learner.
// These two conditions will be checked before apply phase later.
func (s *EtcdServer) isLearnerReady(id uint64) error {
	rs := s.raftStatus()

	// leader's raftStatus.Progress is not nil
	if rs.Progress == nil {
		return ErrNotLeader
	}

	var learnerMatch uint64
	isFound := false
	leaderID := rs.ID
	for memberID, progress := range rs.Progress {
		if id == memberID {
			// check its status
			learnerMatch = progress.Match
			isFound = true
			break
		}
	}

	if isFound {
		leaderMatch := rs.Progress[leaderID].Match
		// the learner's Match not caught up with leader yet
		if float64(learnerMatch) < float64(leaderMatch)*readyPercent {
			return ErrLearnerNotReady
		}
	}

	return nil
}

func (s *EtcdServer) mayRemoveMember(id types.ID) error {
	if !s.Cfg.StrictReconfigCheck {
		return nil
	}

	isLearner := s.cluster.IsMemberExist(id) && s.cluster.Member(id).IsLearner
	// no need to check quorum when removing non-voting member
	if isLearner {
		return nil
	}

	if !s.cluster.IsReadyToRemoveVotingMember(uint64(id)) {
		if lg := s.getLogger(); lg != nil {
			lg.Warn(
				"rejecting member remove request; not enough healthy members",
				zap.String("local-member-id", s.ID().String()),
				zap.String("requested-member-remove-id", id.String()),
				zap.Error(ErrNotEnoughStartedMembers),
			)
		} else {
			plog.Warningf("not enough started members, rejecting remove member %s", id)
		}
		return ErrNotEnoughStartedMembers
	}

	// downed member is safe to remove since it's not part of the active quorum
	if t := s.r.transport.ActiveSince(id); id != s.ID() && t.IsZero() {
		return nil
	}

	// protect quorum if some members are down
	m := s.cluster.VotingMembers()
	active := numConnectedSince(s.r.transport, time.Now().Add(-HealthInterval), s.ID(), m)
	if (active - 1) < 1+((len(m)-1)/2) {
		if lg := s.getLogger(); lg != nil {
			lg.Warn(
				"rejecting member remove request; local member has not been connected to all peers, reconfigure breaks active quorum",
				zap.String("local-member-id", s.ID().String()),
				zap.String("requested-member-remove", id.String()),
				zap.Int("active-peers", active),
				zap.Error(ErrUnhealthy),
			)
		} else {
			plog.Warningf("reconfigure breaks active quorum, rejecting remove member %s", id)
		}
		return ErrUnhealthy
	}

	return nil
}

func (s *EtcdServer) UpdateMember(ctx context.Context, memb membership.Member) ([]*membership.Member, error) {
	b, merr := json.Marshal(memb)
	if merr != nil {
		return nil, merr
	}

	if err := s.checkMembershipOperationPermission(ctx); err != nil {
		return nil, err
	}
	cc := raftpb.ConfChange{
		Type:    raftpb.ConfChangeUpdateNode,
		NodeID:  uint64(memb.ID),
		Context: b,
	}
	return s.configure(ctx, cc)
}

func (s *EtcdServer) setCommittedIndex(v uint64) {
	atomic.StoreUint64(&s.committedIndex, v)
}

func (s *EtcdServer) getCommittedIndex() uint64 {
	return atomic.LoadUint64(&s.committedIndex)
}

func (s *EtcdServer) setAppliedIndex(v uint64) {
	atomic.StoreUint64(&s.appliedIndex, v)
}

func (s *EtcdServer) getAppliedIndex() uint64 {
	return atomic.LoadUint64(&s.appliedIndex)
}

func (s *EtcdServer) setTerm(v uint64) {
	atomic.StoreUint64(&s.term, v)
}

func (s *EtcdServer) getTerm() uint64 {
	return atomic.LoadUint64(&s.term)
}

func (s *EtcdServer) setLead(v uint64) {
	atomic.StoreUint64(&s.lead, v)
}

func (s *EtcdServer) getLead() uint64 {
	return atomic.LoadUint64(&s.lead)
}

func (s *EtcdServer) leaderChangedNotify() <-chan struct{} {
	s.leaderChangedMu.RLock()
	defer s.leaderChangedMu.RUnlock()
	return s.leaderChanged
}

// RaftStatusGetter represents etcd server and Raft progress.
type RaftStatusGetter interface {
	ID() types.ID
	Leader() types.ID
	CommittedIndex() uint64
	AppliedIndex() uint64
	Term() uint64
}

func (s *EtcdServer) ID() types.ID { return s.id }

func (s *EtcdServer) Leader() types.ID { return types.ID(s.getLead()) }

func (s *EtcdServer) Lead() uint64 { return s.getLead() }

func (s *EtcdServer) CommittedIndex() uint64 { return s.getCommittedIndex() }

func (s *EtcdServer) AppliedIndex() uint64 { return s.getAppliedIndex() }

func (s *EtcdServer) Term() uint64 { return s.getTerm() }

type confChangeResponse struct {
	membs []*membership.Member
	err   error
}

// configure sends a configuration change through consensus and
// then waits for it to be applied to the server. It
// will block until the change is performed or there is an error.
func (s *EtcdServer) configure(ctx context.Context, cc raftpb.ConfChange) ([]*membership.Member, error) {
	cc.ID = s.reqIDGen.Next()
	ch := s.w.Register(cc.ID)

	start := time.Now()
	if err := s.r.ProposeConfChange(ctx, cc); err != nil {
		s.w.Trigger(cc.ID, nil)
		return nil, err
	}

	select {
	case x := <-ch:
		if x == nil {
			if lg := s.getLogger(); lg != nil {
				lg.Panic("failed to configure")
			} else {
				plog.Panicf("configure trigger value should never be nil")
			}
		}
		resp := x.(*confChangeResponse)
		if lg := s.getLogger(); lg != nil {
			lg.Info(
				"applied a configuration change through raft",
				zap.String("local-member-id", s.ID().String()),
				zap.String("raft-conf-change", cc.Type.String()),
				zap.String("raft-conf-change-node-id", types.ID(cc.NodeID).String()),
			)
		}
		return resp.membs, resp.err

	case <-ctx.Done():
		s.w.Trigger(cc.ID, nil) // GC wait
		return nil, s.parseProposeCtxErr(ctx.Err(), start)

	case <-s.stopping:
		return nil, ErrStopped
	}
}

// sync proposes a SYNC request and is non-blocking.
// This makes no guarantee that the request will be proposed or performed.
// The request will be canceled after the given timeout.
func (s *EtcdServer) sync(timeout time.Duration) {
	req := pb.Request{
		Method: "SYNC",
		ID:     s.reqIDGen.Next(),
		Time:   time.Now().UnixNano(),
	}
	data := pbutil.MustMarshal(&req)
	// There is no promise that node has leader when do SYNC request,
	// so it uses goroutine to propose.
	ctx, cancel := context.WithTimeout(s.ctx, timeout)
	s.goAttach(func() {
		s.r.Propose(ctx, data)
		cancel()
	})
}

// publish registers server information into the cluster. The information
// is the JSON representation of this server's member struct, updated with the
// static clientURLs of the server.
// The function keeps attempting to register until it succeeds,
// or its server is stopped.
//
// Use v2 store to encode member attributes, and apply through Raft
// but does not go through v2 API endpoint, which means even with v2
// client handler disabled (e.g. --enable-v2=false), cluster can still
// process publish requests through rafthttp
// TODO: Deprecate v2 store
func (s *EtcdServer) publish(timeout time.Duration) {
	b, err := json.Marshal(s.attributes)
	if err != nil {
		if lg := s.getLogger(); lg != nil {
			lg.Panic("failed to marshal JSON", zap.Error(err))
		} else {
			plog.Panicf("json marshal error: %v", err)
		}
		return
	}
	req := pb.Request{
		Method: "PUT",
		Path:   membership.MemberAttributesStorePath(s.id),
		Val:    string(b),
	}

	for {
		ctx, cancel := context.WithTimeout(s.ctx, timeout)
		_, err := s.Do(ctx, req)
		cancel()
		switch err {
		case nil:
			close(s.readych)
			if lg := s.getLogger(); lg != nil {
				lg.Info(
					"published local member to cluster through raft",
					zap.String("local-member-id", s.ID().String()),
					zap.String("local-member-attributes", fmt.Sprintf("%+v", s.attributes)),
					zap.String("request-path", req.Path),
					zap.String("cluster-id", s.cluster.ID().String()),
					zap.Duration("publish-timeout", timeout),
				)
			} else {
				plog.Infof("published %+v to cluster %s", s.attributes, s.cluster.ID())
			}
			return

		case ErrStopped:
			if lg := s.getLogger(); lg != nil {
				lg.Warn(
					"stopped publish because server is stopped",
					zap.String("local-member-id", s.ID().String()),
					zap.String("local-member-attributes", fmt.Sprintf("%+v", s.attributes)),
					zap.Duration("publish-timeout", timeout),
					zap.Error(err),
				)
			} else {
				plog.Infof("aborting publish because server is stopped")
			}
			return

		default:
			if lg := s.getLogger(); lg != nil {
				lg.Warn(
					"failed to publish local member to cluster through raft",
					zap.String("local-member-id", s.ID().String()),
					zap.String("local-member-attributes", fmt.Sprintf("%+v", s.attributes)),
					zap.String("request-path", req.Path),
					zap.Duration("publish-timeout", timeout),
					zap.Error(err),
				)
			} else {
				plog.Errorf("publish error: %v", err)
			}
		}
	}
}

func (s *EtcdServer) sendMergedSnap(merged snap.Message) {
	atomic.AddInt64(&s.inflightSnapshots, 1)

	lg := s.getLogger()
	fields := []zap.Field{
		zap.String("from", s.ID().String()),
		zap.String("to", types.ID(merged.To).String()),
		zap.Int64("bytes", merged.TotalSize),
		zap.String("size", humanize.Bytes(uint64(merged.TotalSize))),
	}

	now := time.Now()
	s.r.transport.SendSnapshot(merged)
	if lg != nil {
		lg.Info("sending merged snapshot", fields...)
	}

	s.goAttach(func() {
		select {
		case ok := <-merged.CloseNotify():
			// delay releasing inflight snapshot for another 30 seconds to
			// block log compaction.
			// If the follower still fails to catch up, it is probably just too slow
			// to catch up. We cannot avoid the snapshot cycle anyway.
			if ok {
				select {
				case <-time.After(releaseDelayAfterSnapshot):
				case <-s.stopping:
				}
			}

			atomic.AddInt64(&s.inflightSnapshots, -1)

			if lg != nil {
				lg.Info("sent merged snapshot", append(fields, zap.Duration("took", time.Since(now)))...)
			}

		case <-s.stopping:
			if lg != nil {
				lg.Warn("canceled sending merged snapshot; server stopping", fields...)
			}
			return
		}
	})
}

// apply takes entries received from Raft (after it has been committed) and
// applies them to the current state of the EtcdServer.
// The given entries should not be empty.
func (s *EtcdServer) apply(
	es []raftpb.Entry,
	confState *raftpb.ConfState,
) (appliedt uint64, appliedi uint64, shouldStop bool) {
	for i := range es {
		e := es[i]
		switch e.Type {
		case raftpb.EntryNormal:
			s.applyEntryNormal(&e)
			s.setAppliedIndex(e.Index)
			s.setTerm(e.Term)

		case raftpb.EntryConfChange:
			// set the consistent index of current executing entry
			if e.Index > s.consistIndex.ConsistentIndex() {
				s.consistIndex.setConsistentIndex(e.Index)
			}
			var cc raftpb.ConfChange
			pbutil.MustUnmarshal(&cc, e.Data)
			removedSelf, err := s.applyConfChange(cc, confState)
			s.setAppliedIndex(e.Index)
			s.setTerm(e.Term)
			shouldStop = shouldStop || removedSelf
			s.w.Trigger(cc.ID, &confChangeResponse{s.cluster.Members(), err})

		default:
			if lg := s.getLogger(); lg != nil {
				lg.Panic(
					"unknown entry type; must be either EntryNormal or EntryConfChange",
					zap.String("type", e.Type.String()),
				)
			} else {
				plog.Panicf("entry type should be either EntryNormal or EntryConfChange")
			}
		}
		appliedi, appliedt = e.Index, e.Term
	}
	return appliedt, appliedi, shouldStop
}

// applyEntryNormal apples an EntryNormal type raftpb request to the EtcdServer
func (s *EtcdServer) applyEntryNormal(e *raftpb.Entry) {
	shouldApplyV3 := false
	if e.Index > s.consistIndex.ConsistentIndex() {
		// set the consistent index of current executing entry
		s.consistIndex.setConsistentIndex(e.Index)
		shouldApplyV3 = true
	}

	// raft state machine may generate noop entry when leader confirmation.
	// skip it in advance to avoid some potential bug in the future
	if len(e.Data) == 0 {
		select {
		case s.forceVersionC <- struct{}{}:
		default:
		}
		// promote lessor when the local member is leader and finished
		// applying all entries from the last term.
		if s.isLeader() {
			s.lessor.Promote(s.Cfg.electionTimeout())
		}
		return
	}

	var raftReq pb.InternalRaftRequest
	if !pbutil.MaybeUnmarshal(&raftReq, e.Data) { // backward compatible
		var r pb.Request
		rp := &r
		pbutil.MustUnmarshal(rp, e.Data)
		s.w.Trigger(r.ID, s.applyV2Request((*RequestV2)(rp)))
		return
	}
	if raftReq.V2 != nil {
		req := (*RequestV2)(raftReq.V2)
		s.w.Trigger(req.ID, s.applyV2Request(req))
		return
	}

	// do not re-apply applied entries.
	if !shouldApplyV3 {
		return
	}

	id := raftReq.ID
	if id == 0 {
		id = raftReq.Header.ID
	}

	var ar *applyResult
	needResult := s.w.IsRegistered(id)
	if needResult || !noSideEffect(&raftReq) {
		if !needResult && raftReq.Txn != nil {
			removeNeedlessRangeReqs(raftReq.Txn)
		}
		ar = s.applyV3.Apply(&raftReq)
	}

	if ar == nil {
		return
	}

	if ar.err != ErrNoSpace || len(s.alarmStore.Get(pb.AlarmType_NOSPACE)) > 0 {
		s.w.Trigger(id, ar)
		return
	}

	if lg := s.getLogger(); lg != nil {
		lg.Warn(
			"message exceeded backend quota; raising alarm",
			zap.Int64("quota-size-bytes", s.Cfg.QuotaBackendBytes),
			zap.String("quota-size", humanize.Bytes(uint64(s.Cfg.QuotaBackendBytes))),
			zap.Error(ar.err),
		)
	} else {
		plog.Errorf("applying raft message exceeded backend quota")
	}

	s.goAttach(func() {
		a := &pb.AlarmRequest{
			MemberID: uint64(s.ID()),
			Action:   pb.AlarmRequest_ACTIVATE,
			Alarm:    pb.AlarmType_NOSPACE,
		}
		s.raftRequest(s.ctx, pb.InternalRaftRequest{Alarm: a})
		s.w.Trigger(id, ar)
	})
}

// applyConfChange applies a ConfChange to the server. It is only
// invoked with a ConfChange that has already passed through Raft
func (s *EtcdServer) applyConfChange(cc raftpb.ConfChange, confState *raftpb.ConfState) (bool, error) {
	if err := s.cluster.ValidateConfigurationChange(cc); err != nil {
		cc.NodeID = raft.None
		s.r.ApplyConfChange(cc)
		return false, err
	}

	lg := s.getLogger()
	*confState = *s.r.ApplyConfChange(cc)
	switch cc.Type {
	case raftpb.ConfChangeAddNode, raftpb.ConfChangeAddLearnerNode:
		confChangeContext := new(membership.ConfigChangeContext)
		if err := json.Unmarshal(cc.Context, confChangeContext); err != nil {
			if lg != nil {
				lg.Panic("failed to unmarshal member", zap.Error(err))
			} else {
				plog.Panicf("unmarshal member should never fail: %v", err)
			}
		}
		if cc.NodeID != uint64(confChangeContext.Member.ID) {
			if lg != nil {
				lg.Panic(
					"got different member ID",
					zap.String("member-id-from-config-change-entry", types.ID(cc.NodeID).String()),
					zap.String("member-id-from-message", confChangeContext.Member.ID.String()),
				)
			} else {
				plog.Panicf("nodeID should always be equal to member ID")
			}
		}
		if confChangeContext.IsPromote {
			s.cluster.PromoteMember(confChangeContext.Member.ID)
		} else {
			s.cluster.AddMember(&confChangeContext.Member)

			if confChangeContext.Member.ID != s.id {
				s.r.transport.AddPeer(confChangeContext.Member.ID, confChangeContext.PeerURLs)
			}
		}

		// update the isLearner metric when this server id is equal to the id in raft member confChange
		if confChangeContext.Member.ID == s.id {
			if cc.Type == raftpb.ConfChangeAddLearnerNode {
				isLearner.Set(1)
			} else {
				isLearner.Set(0)
			}
		}

	case raftpb.ConfChangeRemoveNode:
		id := types.ID(cc.NodeID)
		s.cluster.RemoveMember(id)
		if id == s.id {
			return true, nil
		}
		s.r.transport.RemovePeer(id)

	case raftpb.ConfChangeUpdateNode:
		m := new(membership.Member)
		if err := json.Unmarshal(cc.Context, m); err != nil {
			if lg != nil {
				lg.Panic("failed to unmarshal member", zap.Error(err))
			} else {
				plog.Panicf("unmarshal member should never fail: %v", err)
			}
		}
		if cc.NodeID != uint64(m.ID) {
			if lg != nil {
				lg.Panic(
					"got different member ID",
					zap.String("member-id-from-config-change-entry", types.ID(cc.NodeID).String()),
					zap.String("member-id-from-message", m.ID.String()),
				)
			} else {
				plog.Panicf("nodeID should always be equal to member ID")
			}
		}
		s.cluster.UpdateRaftAttributes(m.ID, m.RaftAttributes)
		if m.ID != s.id {
			s.r.transport.UpdatePeer(m.ID, m.PeerURLs)
		}
	}
	return false, nil
}

// TODO: non-blocking snapshot
func (s *EtcdServer) snapshot(snapi uint64, confState raftpb.ConfState) {
	clone := s.v2store.Clone()
	// commit kv to write metadata (for example: consistent index) to disk.
	// KV().commit() updates the consistent index in backend.
	// All operations that update consistent index must be called sequentially
	// from applyAll function.
	// So KV().Commit() cannot run in parallel with apply. It has to be called outside
	// the go routine created below.
	s.KV().Commit()

	s.goAttach(func() {
		lg := s.getLogger()

		d, err := clone.SaveNoCopy()
		// TODO: current store will never fail to do a snapshot
		// what should we do if the store might fail?
		if err != nil {
			if lg != nil {
				lg.Panic("failed to save v2 store", zap.Error(err))
			} else {
				plog.Panicf("store save should never fail: %v", err)
			}
		}
		snap, err := s.r.raftStorage.CreateSnapshot(snapi, &confState, d)
		if err != nil {
			// the snapshot was done asynchronously with the progress of raft.
			// raft might have already got a newer snapshot.
			if err == raft.ErrSnapOutOfDate {
				return
			}
			if lg != nil {
				lg.Panic("failed to create snapshot", zap.Error(err))
			} else {
				plog.Panicf("unexpected create snapshot error %v", err)
			}
		}
		// SaveSnap saves the snapshot to file and appends the corresponding WAL entry.
		if err = s.r.storage.SaveSnap(snap); err != nil {
			if lg != nil {
				lg.Panic("failed to save snapshot", zap.Error(err))
			} else {
				plog.Fatalf("save snapshot error: %v", err)
			}
		}
		if lg != nil {
			lg.Info(
				"saved snapshot",
				zap.Uint64("snapshot-index", snap.Metadata.Index),
			)
		} else {
			plog.Infof("saved snapshot at index %d", snap.Metadata.Index)
		}
		if err = s.r.storage.Release(snap); err != nil {
			if lg != nil {
				lg.Panic("failed to release wal", zap.Error(err))
			} else {
				plog.Panicf("failed to release wal %v", err)
			}
		}

		// When sending a snapshot, etcd will pause compaction.
		// After receives a snapshot, the slow follower needs to get all the entries right after
		// the snapshot sent to catch up. If we do not pause compaction, the log entries right after
		// the snapshot sent might already be compacted. It happens when the snapshot takes long time
		// to send and save. Pausing compaction avoids triggering a snapshot sending cycle.
		if atomic.LoadInt64(&s.inflightSnapshots) != 0 {
			if lg != nil {
				lg.Info("skip compaction since there is an inflight snapshot")
			} else {
				plog.Infof("skip compaction since there is an inflight snapshot")
			}
			return
		}

		// keep some in memory log entries for slow followers.
		compacti := uint64(1)
		if snapi > s.Cfg.SnapshotCatchUpEntries {
			compacti = snapi - s.Cfg.SnapshotCatchUpEntries
		}

		err = s.r.raftStorage.Compact(compacti)
		if err != nil {
			// the compaction was done asynchronously with the progress of raft.
			// raft log might already been compact.
			if err == raft.ErrCompacted {
				return
			}
			if lg != nil {
				lg.Panic("failed to compact", zap.Error(err))
			} else {
				plog.Panicf("unexpected compaction error %v", err)
			}
		}
		if lg != nil {
			lg.Info(
				"compacted Raft logs",
				zap.Uint64("compact-index", compacti),
			)
		} else {
			plog.Infof("compacted raft log at %d", compacti)
		}
	})
}

// CutPeer drops messages to the specified peer.
func (s *EtcdServer) CutPeer(id types.ID) {
	tr, ok := s.r.transport.(*rafthttp.Transport)
	if ok {
		tr.CutPeer(id)
	}
}

// MendPeer recovers the message dropping behavior of the given peer.
func (s *EtcdServer) MendPeer(id types.ID) {
	tr, ok := s.r.transport.(*rafthttp.Transport)
	if ok {
		tr.MendPeer(id)
	}
}

func (s *EtcdServer) PauseSending() { s.r.pauseSending() }

func (s *EtcdServer) ResumeSending() { s.r.resumeSending() }

func (s *EtcdServer) ClusterVersion() *semver.Version {
	if s.cluster == nil {
		return nil
	}
	return s.cluster.Version()
}

// monitorVersions checks the member's version every monitorVersionInterval.
// It updates the cluster version if all members agrees on a higher one.
// It prints out log if there is a member with a higher version than the
// local version.
func (s *EtcdServer) monitorVersions() {
	for {
		select {
		case <-s.forceVersionC:
		case <-time.After(monitorVersionInterval):
		case <-s.stopping:
			return
		}

		if s.Leader() != s.ID() {
			continue
		}

		v := decideClusterVersion(s.getLogger(), getVersions(s.getLogger(), s.cluster, s.id, s.peerRt))
		if v != nil {
			// only keep major.minor version for comparison
			v = &semver.Version{
				Major: v.Major,
				Minor: v.Minor,
			}
		}

		// if the current version is nil:
		// 1. use the decided version if possible
		// 2. or use the min cluster version
		if s.cluster.Version() == nil {
			verStr := version.MinClusterVersion
			if v != nil {
				verStr = v.String()
			}
			s.goAttach(func() { s.updateClusterVersion(verStr) })
			continue
		}

		// update cluster version only if the decided version is greater than
		// the current cluster version
		if v != nil && s.cluster.Version().LessThan(*v) {
			s.goAttach(func() { s.updateClusterVersion(v.String()) })
		}
	}
}

func (s *EtcdServer) updateClusterVersion(ver string) {
	lg := s.getLogger()

	if s.cluster.Version() == nil {
		if lg != nil {
			lg.Info(
				"setting up initial cluster version",
				zap.String("cluster-version", version.Cluster(ver)),
			)
		} else {
			plog.Infof("setting up the initial cluster version to %s", version.Cluster(ver))
		}
	} else {
		if lg != nil {
			lg.Info(
				"updating cluster version",
				zap.String("from", version.Cluster(s.cluster.Version().String())),
				zap.String("to", version.Cluster(ver)),
			)
		} else {
			plog.Infof("updating the cluster version from %s to %s", version.Cluster(s.cluster.Version().String()), version.Cluster(ver))
		}
	}

	req := pb.Request{
		Method: "PUT",
		Path:   membership.StoreClusterVersionKey(),
		Val:    ver,
	}

	ctx, cancel := context.WithTimeout(s.ctx, s.Cfg.ReqTimeout())
	_, err := s.Do(ctx, req)
	cancel()

	switch err {
	case nil:
		if lg != nil {
			lg.Info("cluster version is updated", zap.String("cluster-version", version.Cluster(ver)))
		}
		return

	case ErrStopped:
		if lg != nil {
			lg.Warn("aborting cluster version update; server is stopped", zap.Error(err))
		} else {
			plog.Infof("aborting update cluster version because server is stopped")
		}
		return

	default:
		if lg != nil {
			lg.Warn("failed to update cluster version", zap.Error(err))
		} else {
			plog.Errorf("error updating cluster version (%v)", err)
		}
	}
}

func (s *EtcdServer) parseProposeCtxErr(err error, start time.Time) error {
	switch err {
	case context.Canceled:
		return ErrCanceled

	case context.DeadlineExceeded:
		s.leadTimeMu.RLock()
		curLeadElected := s.leadElectedTime
		s.leadTimeMu.RUnlock()
		prevLeadLost := curLeadElected.Add(-2 * time.Duration(s.Cfg.ElectionTicks) * time.Duration(s.Cfg.TickMs) * time.Millisecond)
		if start.After(prevLeadLost) && start.Before(curLeadElected) {
			return ErrTimeoutDueToLeaderFail
		}
		lead := types.ID(s.getLead())
		switch lead {
		case types.ID(raft.None):
			// TODO: return error to specify it happens because the cluster does not have leader now
		case s.ID():
			if !isConnectedToQuorumSince(s.r.transport, start, s.ID(), s.cluster.Members()) {
				return ErrTimeoutDueToConnectionLost
			}
		default:
			if !isConnectedSince(s.r.transport, start, lead) {
				return ErrTimeoutDueToConnectionLost
			}
		}
		return ErrTimeout

	default:
		return err
	}
}

func (s *EtcdServer) KV() mvcc.ConsistentWatchableKV { return s.kv }
func (s *EtcdServer) Backend() backend.Backend {
	s.bemu.Lock()
	defer s.bemu.Unlock()
	return s.be
}

func (s *EtcdServer) AuthStore() auth.AuthStore { return s.authStore }

func (s *EtcdServer) restoreAlarms() error {
	s.applyV3 = s.newApplierV3()
	as, err := v3alarm.NewAlarmStore(s)
	if err != nil {
		return err
	}
	s.alarmStore = as
	if len(as.Get(pb.AlarmType_NOSPACE)) > 0 {
		s.applyV3 = newApplierV3Capped(s.applyV3)
	}
	if len(as.Get(pb.AlarmType_CORRUPT)) > 0 {
		s.applyV3 = newApplierV3Corrupt(s.applyV3)
	}
	return nil
}

// goAttach creates a goroutine on a given function and tracks it using
// the etcdserver waitgroup.
func (s *EtcdServer) goAttach(f func()) {
	s.wgMu.RLock() // this blocks with ongoing close(s.stopping)
	defer s.wgMu.RUnlock()
	select {
	case <-s.stopping:
		if lg := s.getLogger(); lg != nil {
			lg.Warn("server has stopped; skipping goAttach")
		} else {
			plog.Warning("server has stopped (skipping goAttach)")
		}
		return
	default:
	}

	// now safe to add since waitgroup wait has not started yet
	s.wg.Add(1)
	go func() {
		defer s.wg.Done()
		f()
	}()
}

func (s *EtcdServer) Alarms() []*pb.AlarmMember {
	return s.alarmStore.Get(pb.AlarmType_NONE)
}

func (s *EtcdServer) Logger() *zap.Logger {
	return s.lg
}

// IsLearner returns if the local member is raft learner
func (s *EtcdServer) IsLearner() bool {
	return s.cluster.IsLocalMemberLearner()
}

// IsMemberExist returns if the member with the given id exists in cluster.
func (s *EtcdServer) IsMemberExist(id types.ID) bool {
	return s.cluster.IsMemberExist(id)
}

// raftStatus returns the raft status of this etcd node.
func (s *EtcdServer) raftStatus() raft.Status {
	return s.r.Node.Status()
}
