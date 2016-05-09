// Copyright 2015 CoreOS, Inc.
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
	"expvar"
	"fmt"
	"math/rand"
	"net/http"
	"os"
	"path"
	"regexp"
	"sync"
	"sync/atomic"
	"time"

	"github.com/coreos/etcd/alarm"
	"github.com/coreos/etcd/auth"
	"github.com/coreos/etcd/compactor"
	"github.com/coreos/etcd/discovery"
	"github.com/coreos/etcd/etcdserver/api/v2http/httptypes"
	pb "github.com/coreos/etcd/etcdserver/etcdserverpb"
	"github.com/coreos/etcd/etcdserver/membership"
	"github.com/coreos/etcd/etcdserver/stats"
	"github.com/coreos/etcd/lease"
	"github.com/coreos/etcd/pkg/fileutil"
	"github.com/coreos/etcd/pkg/idutil"
	"github.com/coreos/etcd/pkg/pbutil"
	"github.com/coreos/etcd/pkg/runtime"
	"github.com/coreos/etcd/pkg/schedule"
	"github.com/coreos/etcd/pkg/types"
	"github.com/coreos/etcd/pkg/wait"
	"github.com/coreos/etcd/raft"
	"github.com/coreos/etcd/raft/raftpb"
	"github.com/coreos/etcd/rafthttp"
	"github.com/coreos/etcd/snap"
	dstorage "github.com/coreos/etcd/storage"
	"github.com/coreos/etcd/storage/backend"
	"github.com/coreos/etcd/store"
	"github.com/coreos/etcd/version"
	"github.com/coreos/etcd/wal"
	"github.com/coreos/go-semver/semver"
	"github.com/coreos/pkg/capnslog"
	"golang.org/x/net/context"
)

const (
	// owner can make/remove files inside the directory
	privateDirMode = 0700

	DefaultSnapCount = 10000

	StoreClusterPrefix = "/0"
	StoreKeysPrefix    = "/1"

	purgeFileInterval = 30 * time.Second
	// monitorVersionInterval should be smaller than the timeout
	// on the connection. Or we will not be able to reuse the connection
	// (since it will timeout).
	monitorVersionInterval = rafthttp.ConnWriteTimeout - time.Second

	databaseFilename = "db"
	// max number of in-flight snapshot messages etcdserver allows to have
	// This number is more than enough for most clusters with 5 machines.
	maxInFlightMsgSnap = 16

	releaseDelayAfterSnapshot = 30 * time.Second
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd", "etcdserver")

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
	Event   *store.Event
	Watcher store.Watcher
	err     error
}

type Server interface {
	// Start performs any initialization of the Server necessary for it to
	// begin serving requests. It must be called before Do or Process.
	// Start must be non-blocking; any long-running server functionality
	// should be implemented in goroutines.
	Start()
	// Stop terminates the Server and performs any necessary finalization.
	// Do and Process cannot be called after Stop has been invoked.
	Stop()
	// ID returns the ID of the Server.
	ID() types.ID
	// Leader returns the ID of the leader Server.
	Leader() types.ID
	// Do takes a request and attempts to fulfill it, returning a Response.
	Do(ctx context.Context, r pb.Request) (Response, error)
	// Process takes a raft message and applies it to the server's raft state
	// machine, respecting any timeout of the given context.
	Process(ctx context.Context, m raftpb.Message) error
	// AddMember attempts to add a member into the cluster. It will return
	// ErrIDRemoved if member ID is removed from the cluster, or return
	// ErrIDExists if member ID exists in the cluster.
	AddMember(ctx context.Context, memb membership.Member) error
	// RemoveMember attempts to remove a member from the cluster. It will
	// return ErrIDRemoved if member ID is removed from the cluster, or return
	// ErrIDNotFound if member ID is not in the cluster.
	RemoveMember(ctx context.Context, id uint64) error

	// UpdateMember attempts to update an existing member in the cluster. It will
	// return ErrIDNotFound if the member ID does not exist.
	UpdateMember(ctx context.Context, updateMemb membership.Member) error

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
}

// EtcdServer is the production implementation of the Server interface
type EtcdServer struct {
	// r must be the first element to keep 64-bit alignment for atomic
	// access to fields
	r raftNode

	cfg       *ServerConfig
	snapCount uint64

	w          wait.Wait
	stop       chan struct{}
	done       chan struct{}
	errorc     chan error
	id         types.ID
	attributes membership.Attributes

	cluster *membership.RaftCluster

	store store.Store

	applyV3    applierV3
	kv         dstorage.ConsistentWatchableKV
	lessor     lease.Lessor
	bemu       sync.Mutex
	be         backend.Backend
	authStore  auth.AuthStore
	alarmStore *alarm.AlarmStore

	stats  *stats.ServerStats
	lstats *stats.LeaderStats

	SyncTicker <-chan time.Time
	// compactor is used to auto-compact the KV.
	compactor *compactor.Periodic

	// consistent index used to hold the offset of current executing entry
	// It is initialized to 0 before executing any entry.
	consistIndex consistentIndex

	// peerRt used to send requests (version, lease) to peers.
	peerRt   http.RoundTripper
	reqIDGen *idutil.Generator

	// forceVersionC is used to force the version monitor loop
	// to detect the cluster version immediately.
	forceVersionC chan struct{}

	msgSnapC chan raftpb.Message

	// count the number of inflight snapshots.
	// MUST use atomic operation to access this field.
	inflightSnapshots int64
}

// NewServer creates a new EtcdServer from the supplied configuration. The
// configuration is considered static for the lifetime of the EtcdServer.
func NewServer(cfg *ServerConfig) (*EtcdServer, error) {
	st := store.New(StoreClusterPrefix, StoreKeysPrefix)
	var (
		w  *wal.WAL
		n  raft.Node
		s  *raft.MemoryStorage
		id types.ID
		cl *membership.RaftCluster
	)

	if terr := fileutil.TouchDirAll(cfg.DataDir); terr != nil {
		return nil, fmt.Errorf("cannot access data directory: %v", terr)
	}

	// Run the migrations.
	dataVer, err := version.DetectDataDir(cfg.DataDir)
	if err != nil {
		return nil, err
	}
	if err := upgradeDataDir(cfg.DataDir, cfg.Name, dataVer); err != nil {
		return nil, err
	}

	haveWAL := wal.Exist(cfg.WALDir())
	ss := snap.New(cfg.SnapDir())

	prt, err := rafthttp.NewRoundTripper(cfg.PeerTLSInfo, cfg.peerDialTimeout())
	if err != nil {
		return nil, err
	}
	var remotes []*membership.Member
	switch {
	case !haveWAL && !cfg.NewCluster:
		if err := cfg.VerifyJoinExisting(); err != nil {
			return nil, err
		}
		cl, err = membership.NewClusterFromURLsMap(cfg.InitialClusterToken, cfg.InitialPeerURLsMap)
		if err != nil {
			return nil, err
		}
		existingCluster, err := GetClusterFromRemotePeers(getRemotePeerURLs(cl, cfg.Name), prt)
		if err != nil {
			return nil, fmt.Errorf("cannot fetch cluster info from peer urls: %v", err)
		}
		if err := membership.ValidateClusterAndAssignIDs(cl, existingCluster); err != nil {
			return nil, fmt.Errorf("error validating peerURLs %s: %v", existingCluster, err)
		}
		if !isCompatibleWithCluster(cl, cl.MemberByName(cfg.Name).ID, prt) {
			return nil, fmt.Errorf("incomptible with current running cluster")
		}

		remotes = existingCluster.Members()
		cl.SetID(existingCluster.ID())
		cl.SetStore(st)
		cfg.Print()
		id, n, s, w = startNode(cfg, cl, nil)
	case !haveWAL && cfg.NewCluster:
		if err := cfg.VerifyBootstrap(); err != nil {
			return nil, err
		}
		cl, err = membership.NewClusterFromURLsMap(cfg.InitialClusterToken, cfg.InitialPeerURLsMap)
		if err != nil {
			return nil, err
		}
		m := cl.MemberByName(cfg.Name)
		if isMemberBootstrapped(cl, cfg.Name, prt, cfg.bootstrapTimeout()) {
			return nil, fmt.Errorf("member %s has already been bootstrapped", m.ID)
		}
		if cfg.ShouldDiscover() {
			var str string
			var err error
			str, err = discovery.JoinCluster(cfg.DiscoveryURL, cfg.DiscoveryProxy, m.ID, cfg.InitialPeerURLsMap.String())
			if err != nil {
				return nil, &DiscoveryError{Op: "join", Err: err}
			}
			urlsmap, err := types.NewURLsMap(str)
			if err != nil {
				return nil, err
			}
			if checkDuplicateURL(urlsmap) {
				return nil, fmt.Errorf("discovery cluster %s has duplicate url", urlsmap)
			}
			if cl, err = membership.NewClusterFromURLsMap(cfg.InitialClusterToken, urlsmap); err != nil {
				return nil, err
			}
		}
		cl.SetStore(st)
		cfg.PrintWithInitial()
		id, n, s, w = startNode(cfg, cl, cl.MemberIDs())
	case haveWAL:
		if err := fileutil.IsDirWriteable(cfg.MemberDir()); err != nil {
			return nil, fmt.Errorf("cannot write to member directory: %v", err)
		}

		if err := fileutil.IsDirWriteable(cfg.WALDir()); err != nil {
			return nil, fmt.Errorf("cannot write to WAL directory: %v", err)
		}

		if cfg.ShouldDiscover() {
			plog.Warningf("discovery token ignored since a cluster has already been initialized. Valid log found at %q", cfg.WALDir())
		}
		var snapshot *raftpb.Snapshot
		var err error
		snapshot, err = ss.Load()
		if err != nil && err != snap.ErrNoSnapshot {
			return nil, err
		}
		if snapshot != nil {
			if err := st.Recovery(snapshot.Data); err != nil {
				plog.Panicf("recovered store from snapshot error: %v", err)
			}
			plog.Infof("recovered store from snapshot at index %d", snapshot.Metadata.Index)
		}
		cfg.Print()
		if !cfg.ForceNewCluster {
			id, cl, n, s, w = restartNode(cfg, snapshot)
		} else {
			id, cl, n, s, w = restartAsStandaloneNode(cfg, snapshot)
		}
		cl.SetStore(st)
		cl.Recover()
	default:
		return nil, fmt.Errorf("unsupported bootstrap config")
	}

	if terr := fileutil.TouchDirAll(cfg.MemberDir()); terr != nil {
		return nil, fmt.Errorf("cannot access member directory: %v", terr)
	}

	sstats := &stats.ServerStats{
		Name: cfg.Name,
		ID:   id.String(),
	}
	sstats.Initialize()
	lstats := stats.NewLeaderStats(id.String())

	srv := &EtcdServer{
		cfg:       cfg,
		snapCount: cfg.SnapCount,
		errorc:    make(chan error, 1),
		store:     st,
		r: raftNode{
			Node:        n,
			ticker:      time.Tick(time.Duration(cfg.TickMs) * time.Millisecond),
			raftStorage: s,
			storage:     NewStorage(w, ss),
		},
		id:            id,
		attributes:    membership.Attributes{Name: cfg.Name, ClientURLs: cfg.ClientURLs.StringSlice()},
		cluster:       cl,
		stats:         sstats,
		lstats:        lstats,
		SyncTicker:    time.Tick(500 * time.Millisecond),
		peerRt:        prt,
		reqIDGen:      idutil.NewGenerator(uint16(id), time.Now()),
		forceVersionC: make(chan struct{}),
		msgSnapC:      make(chan raftpb.Message, maxInFlightMsgSnap),
	}

	srv.be = backend.NewDefaultBackend(path.Join(cfg.SnapDir(), databaseFilename))
	srv.lessor = lease.NewLessor(srv.be)
	srv.kv = dstorage.New(srv.be, srv.lessor, &srv.consistIndex)
	srv.consistIndex.setConsistentIndex(srv.kv.ConsistentIndex())
	srv.authStore = auth.NewAuthStore(srv.be)
	if h := cfg.AutoCompactionRetention; h != 0 {
		srv.compactor = compactor.NewPeriodic(h, srv.kv, srv)
		srv.compactor.Run()
	}

	if err := srv.restoreAlarms(); err != nil {
		return nil, err
	}

	// TODO: move transport initialization near the definition of remote
	tr := &rafthttp.Transport{
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
	if err := tr.Start(); err != nil {
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

// Start prepares and starts server in a new goroutine. It is no longer safe to
// modify a server's fields after it has been sent to Start.
// It also starts a goroutine to publish its server information.
func (s *EtcdServer) Start() {
	s.start()
	go s.publish(s.cfg.ReqTimeout())
	go s.purgeFile()
	go monitorFileDescriptor(s.done)
	go s.monitorVersions()
}

// start prepares and starts server in a new goroutine. It is no longer safe to
// modify a server's fields after it has been sent to Start.
// This function is just used for testing.
func (s *EtcdServer) start() {
	if s.snapCount == 0 {
		plog.Infof("set snapshot count to default %d", DefaultSnapCount)
		s.snapCount = DefaultSnapCount
	}
	s.w = wait.New()
	s.done = make(chan struct{})
	s.stop = make(chan struct{})
	if s.ClusterVersion() != nil {
		plog.Infof("starting server... [version: %v, cluster version: %v]", version.Version, version.Cluster(s.ClusterVersion().String()))
	} else {
		plog.Infof("starting server... [version: %v, cluster version: to_be_decided]", version.Version)
	}
	// TODO: if this is an empty log, writes all peer infos
	// into the first entry
	go s.run()
}

func (s *EtcdServer) purgeFile() {
	var serrc, werrc <-chan error
	if s.cfg.MaxSnapFiles > 0 {
		serrc = fileutil.PurgeFile(s.cfg.SnapDir(), "snap", s.cfg.MaxSnapFiles, purgeFileInterval, s.done)
	}
	if s.cfg.MaxWALFiles > 0 {
		werrc = fileutil.PurgeFile(s.cfg.WALDir(), "wal", s.cfg.MaxWALFiles, purgeFileInterval, s.done)
	}
	select {
	case e := <-werrc:
		plog.Fatalf("failed to purge wal file %v", e)
	case e := <-serrc:
		plog.Fatalf("failed to purge snap file %v", e)
	case <-s.done:
		return
	}
}

func (s *EtcdServer) ID() types.ID { return s.id }

func (s *EtcdServer) Cluster() *membership.RaftCluster { return s.cluster }

func (s *EtcdServer) RaftHandler() http.Handler { return s.r.transport.Handler() }

func (s *EtcdServer) Lessor() lease.Lessor { return s.lessor }

func (s *EtcdServer) Process(ctx context.Context, m raftpb.Message) error {
	if s.cluster.IsIDRemoved(types.ID(m.From)) {
		plog.Warningf("reject message from removed member %s", types.ID(m.From).String())
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
	appliedi  uint64
}

func (s *EtcdServer) run() {
	snap, err := s.r.raftStorage.Snapshot()
	if err != nil {
		plog.Panicf("get snapshot from raft storage error: %v", err)
	}
	s.r.start(s)

	// asynchronously accept apply packets, dispatch progress in-order
	sched := schedule.NewFIFOScheduler()
	ep := etcdProgress{
		confState: snap.Metadata.ConfState,
		snapi:     snap.Metadata.Index,
		appliedi:  snap.Metadata.Index,
	}

	defer func() {
		s.r.stop()
		sched.Stop()

		// kv, lessor and backend can be nil if running without v3 enabled
		// or running unit tests.
		if s.lessor != nil {
			s.lessor.Stop()
		}
		if s.kv != nil {
			s.kv.Close()
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
			go func() {
				for _, l := range leases {
					s.LeaseRevoke(context.TODO(), &pb.LeaseRevokeRequest{ID: int64(l.ID)})
				}
			}()
		case err := <-s.errorc:
			plog.Errorf("%s", err)
			plog.Infof("the data-dir used by this member must be removed.")
			return
		case <-s.stop:
			return
		}
	}
}

func (s *EtcdServer) applyAll(ep *etcdProgress, apply *apply) {
	s.applySnapshot(ep, apply)
	s.applyEntries(ep, apply)
	// wait for the raft routine to finish the disk writes before triggering a
	// snapshot. or applied index might be greater than the last index in raft
	// storage, since the raft routine might be slower than apply routine.
	<-apply.raftDone
	s.triggerSnapshot(ep)
	select {
	// snapshot requested via send()
	case m := <-s.msgSnapC:
		merged := s.createMergedSnapshotMessage(m, ep.appliedi, ep.confState)
		s.sendMergedSnap(merged)
	default:
	}
}

func (s *EtcdServer) applySnapshot(ep *etcdProgress, apply *apply) {
	if raft.IsEmptySnap(apply.snapshot) {
		return
	}

	if apply.snapshot.Metadata.Index <= ep.appliedi {
		plog.Panicf("snapshot index [%d] should > appliedi[%d] + 1",
			apply.snapshot.Metadata.Index, ep.appliedi)
	}

	snapfn, err := s.r.storage.DBFilePath(apply.snapshot.Metadata.Index)
	if err != nil {
		plog.Panicf("get database snapshot file path error: %v", err)
	}

	fn := path.Join(s.cfg.SnapDir(), databaseFilename)
	if err := os.Rename(snapfn, fn); err != nil {
		plog.Panicf("rename snapshot file error: %v", err)
	}

	newbe := backend.NewDefaultBackend(fn)
	if err := s.kv.Restore(newbe); err != nil {
		plog.Panicf("restore KV error: %v", err)
	}
	s.consistIndex.setConsistentIndex(s.kv.ConsistentIndex())

	// Closing old backend might block until all the txns
	// on the backend are finished.
	// We do not want to wait on closing the old backend.
	s.bemu.Lock()
	oldbe := s.be
	go func() {
		if err := oldbe.Close(); err != nil {
			plog.Panicf("close backend error: %v", err)
		}
	}()

	s.be = newbe
	s.bemu.Unlock()

	if s.lessor != nil {
		s.lessor.Recover(newbe, s.kv)
	}

	if err := s.restoreAlarms(); err != nil {
		plog.Panicf("restore alarms error: %v", err)
	}

	if s.authStore != nil {
		s.authStore.Recover(newbe)
	}

	if err := s.store.Recovery(apply.snapshot.Data); err != nil {
		plog.Panicf("recovery store error: %v", err)
	}
	s.cluster.Recover()

	// recover raft transport
	s.r.transport.RemoveAllPeers()
	for _, m := range s.cluster.Members() {
		if m.ID == s.ID() {
			continue
		}
		s.r.transport.AddPeer(m.ID, m.PeerURLs)
	}

	ep.appliedi = apply.snapshot.Metadata.Index
	ep.snapi = ep.appliedi
	ep.confState = apply.snapshot.Metadata.ConfState
	plog.Infof("recovered from incoming snapshot at index %d", ep.snapi)
}

func (s *EtcdServer) applyEntries(ep *etcdProgress, apply *apply) {
	if len(apply.entries) == 0 {
		return
	}
	firsti := apply.entries[0].Index
	if firsti > ep.appliedi+1 {
		plog.Panicf("first index of committed entry[%d] should <= appliedi[%d] + 1", firsti, ep.appliedi)
	}
	var ents []raftpb.Entry
	if ep.appliedi+1-firsti < uint64(len(apply.entries)) {
		ents = apply.entries[ep.appliedi+1-firsti:]
	}
	if len(ents) == 0 {
		return
	}
	var shouldstop bool
	if ep.appliedi, shouldstop = s.apply(ents, &ep.confState); shouldstop {
		go s.stopWithDelay(10*100*time.Millisecond, fmt.Errorf("the member has been permanently removed from the cluster"))
	}
}

func (s *EtcdServer) triggerSnapshot(ep *etcdProgress) {
	if ep.appliedi-ep.snapi <= s.snapCount {
		return
	}

	// When sending a snapshot, etcd will pause compaction.
	// After receives a snapshot, the slow follower needs to get all the entries right after
	// the snapshot sent to catch up. If we do not pause compaction, the log entries right after
	// the snapshot sent might already be compacted. It happens when the snapshot takes long time
	// to send and save. Pausing compaction avoids triggering a snapshot sending cycle.
	if atomic.LoadInt64(&s.inflightSnapshots) != 0 {
		return
	}

	plog.Infof("start to snapshot (applied: %d, lastsnap: %d)", ep.appliedi, ep.snapi)
	s.snapshot(ep.appliedi, ep.confState)
	ep.snapi = ep.appliedi
}

// Stop stops the server gracefully, and shuts down the running goroutine.
// Stop should be called after a Start(s), otherwise it will block forever.
func (s *EtcdServer) Stop() {
	select {
	case s.stop <- struct{}{}:
	case <-s.done:
		return
	}
	<-s.done
}

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

// Do interprets r and performs an operation on s.store according to r.Method
// and other fields. If r.Method is "POST", "PUT", "DELETE", or a "GET" with
// Quorum == true, r will be sent through consensus before performing its
// respective operation. Do will block until an action is performed or there is
// an error.
func (s *EtcdServer) Do(ctx context.Context, r pb.Request) (Response, error) {
	r.ID = s.reqIDGen.Next()
	if r.Method == "GET" && r.Quorum {
		r.Method = "QGET"
	}
	switch r.Method {
	case "POST", "PUT", "DELETE", "QGET":
		data, err := r.Marshal()
		if err != nil {
			return Response{}, err
		}
		ch := s.w.Register(r.ID)

		// TODO: benchmark the cost of time.Now()
		// might be sampling?
		start := time.Now()
		s.r.Propose(ctx, data)

		proposePending.Inc()
		defer proposePending.Dec()

		select {
		case x := <-ch:
			proposeDurations.Observe(float64(time.Since(start)) / float64(time.Second))
			resp := x.(Response)
			return resp, resp.err
		case <-ctx.Done():
			proposeFailed.Inc()
			s.w.Trigger(r.ID, nil) // GC wait
			return Response{}, s.parseProposeCtxErr(ctx.Err(), start)
		case <-s.done:
			return Response{}, ErrStopped
		}
	case "GET":
		switch {
		case r.Wait:
			wc, err := s.store.Watch(r.Path, r.Recursive, r.Stream, r.Since)
			if err != nil {
				return Response{}, err
			}
			return Response{Watcher: wc}, nil
		default:
			ev, err := s.store.Get(r.Path, r.Recursive, r.Sorted)
			if err != nil {
				return Response{}, err
			}
			return Response{Event: ev}, nil
		}
	case "HEAD":
		ev, err := s.store.Get(r.Path, r.Recursive, r.Sorted)
		if err != nil {
			return Response{}, err
		}
		return Response{Event: ev}, nil
	default:
		return Response{}, ErrUnknownMethod
	}
}

func (s *EtcdServer) SelfStats() []byte { return s.stats.JSON() }

func (s *EtcdServer) LeaderStats() []byte {
	lead := atomic.LoadUint64(&s.r.lead)
	if lead != uint64(s.id) {
		return nil
	}
	return s.lstats.JSON()
}

func (s *EtcdServer) StoreStats() []byte { return s.store.JsonStats() }

func (s *EtcdServer) AddMember(ctx context.Context, memb membership.Member) error {
	if s.cfg.StrictReconfigCheck && !s.cluster.IsReadyToAddNewMember() {
		// If s.cfg.StrictReconfigCheck is false, it means the option --strict-reconfig-check isn't passed to etcd.
		// In such a case adding a new member is allowed unconditionally
		return ErrNotEnoughStartedMembers
	}

	// TODO: move Member to protobuf type
	b, err := json.Marshal(memb)
	if err != nil {
		return err
	}
	cc := raftpb.ConfChange{
		Type:    raftpb.ConfChangeAddNode,
		NodeID:  uint64(memb.ID),
		Context: b,
	}
	return s.configure(ctx, cc)
}

func (s *EtcdServer) RemoveMember(ctx context.Context, id uint64) error {
	if s.cfg.StrictReconfigCheck && !s.cluster.IsReadyToRemoveMember(id) {
		// If s.cfg.StrictReconfigCheck is false, it means the option --strict-reconfig-check isn't passed to etcd.
		// In such a case removing a member is allowed unconditionally
		return ErrNotEnoughStartedMembers
	}

	cc := raftpb.ConfChange{
		Type:   raftpb.ConfChangeRemoveNode,
		NodeID: id,
	}
	return s.configure(ctx, cc)
}

func (s *EtcdServer) UpdateMember(ctx context.Context, memb membership.Member) error {
	b, err := json.Marshal(memb)
	if err != nil {
		return err
	}
	cc := raftpb.ConfChange{
		Type:    raftpb.ConfChangeUpdateNode,
		NodeID:  uint64(memb.ID),
		Context: b,
	}
	return s.configure(ctx, cc)
}

// Implement the RaftTimer interface

func (s *EtcdServer) Index() uint64 { return atomic.LoadUint64(&s.r.index) }

func (s *EtcdServer) Term() uint64 { return atomic.LoadUint64(&s.r.term) }

// Lead is only for testing purposes.
// TODO: add Raft server interface to expose raft related info:
// Index, Term, Lead, Committed, Applied, LastIndex, etc.
func (s *EtcdServer) Lead() uint64 { return atomic.LoadUint64(&s.r.lead) }

func (s *EtcdServer) Leader() types.ID { return types.ID(s.Lead()) }

func (s *EtcdServer) IsPprofEnabled() bool { return s.cfg.EnablePprof }

// configure sends a configuration change through consensus and
// then waits for it to be applied to the server. It
// will block until the change is performed or there is an error.
func (s *EtcdServer) configure(ctx context.Context, cc raftpb.ConfChange) error {
	cc.ID = s.reqIDGen.Next()
	ch := s.w.Register(cc.ID)
	start := time.Now()
	if err := s.r.ProposeConfChange(ctx, cc); err != nil {
		s.w.Trigger(cc.ID, nil)
		return err
	}
	select {
	case x := <-ch:
		if err, ok := x.(error); ok {
			return err
		}
		if x != nil {
			plog.Panicf("return type should always be error")
		}
		return nil
	case <-ctx.Done():
		s.w.Trigger(cc.ID, nil) // GC wait
		return s.parseProposeCtxErr(ctx.Err(), start)
	case <-s.done:
		return ErrStopped
	}
}

// sync proposes a SYNC request and is non-blocking.
// This makes no guarantee that the request will be proposed or performed.
// The request will be canceled after the given timeout.
func (s *EtcdServer) sync(timeout time.Duration) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	req := pb.Request{
		Method: "SYNC",
		ID:     s.reqIDGen.Next(),
		Time:   time.Now().UnixNano(),
	}
	data := pbutil.MustMarshal(&req)
	// There is no promise that node has leader when do SYNC request,
	// so it uses goroutine to propose.
	go func() {
		s.r.Propose(ctx, data)
		cancel()
	}()
}

// publish registers server information into the cluster. The information
// is the JSON representation of this server's member struct, updated with the
// static clientURLs of the server.
// The function keeps attempting to register until it succeeds,
// or its server is stopped.
func (s *EtcdServer) publish(timeout time.Duration) {
	b, err := json.Marshal(s.attributes)
	if err != nil {
		plog.Panicf("json marshal error: %v", err)
		return
	}
	req := pb.Request{
		Method: "PUT",
		Path:   membership.MemberAttributesStorePath(s.id),
		Val:    string(b),
	}

	for {
		ctx, cancel := context.WithTimeout(context.Background(), timeout)
		_, err := s.Do(ctx, req)
		cancel()
		switch err {
		case nil:
			plog.Infof("published %+v to cluster %s", s.attributes, s.cluster.ID())
			return
		case ErrStopped:
			plog.Infof("aborting publish because server is stopped")
			return
		default:
			plog.Errorf("publish error: %v", err)
		}
	}
}

// TODO: move this function into raft.go
func (s *EtcdServer) send(ms []raftpb.Message) {
	for i := range ms {
		if s.cluster.IsIDRemoved(types.ID(ms[i].To)) {
			ms[i].To = 0
		}

		if ms[i].Type == raftpb.MsgSnap {
			// There are two separate data store: the store for v2, and the KV for v3.
			// The msgSnap only contains the most recent snapshot of store without KV.
			// So we need to redirect the msgSnap to etcd server main loop for merging in the
			// current store snapshot and KV snapshot.
			select {
			case s.msgSnapC <- ms[i]:
			default:
				// drop msgSnap if the inflight chan if full.
			}
			ms[i].To = 0
		}
		if ms[i].Type == raftpb.MsgHeartbeat {
			ok, exceed := s.r.td.Observe(ms[i].To)
			if !ok {
				// TODO: limit request rate.
				plog.Warningf("failed to send out heartbeat on time (deadline exceeded for %v)", exceed)
				plog.Warningf("server is likely overloaded")
			}
		}
	}

	s.r.transport.Send(ms)
}

func (s *EtcdServer) sendMergedSnap(merged snap.Message) {
	atomic.AddInt64(&s.inflightSnapshots, 1)

	s.r.transport.SendSnapshot(merged)
	go func() {
		select {
		case ok := <-merged.CloseNotify():
			// delay releasing inflight snapshot for another 30 seconds to
			// block log compaction.
			// If the follower still fails to catch up, it is probably just too slow
			// to catch up. We cannot avoid the snapshot cycle anyway.
			if ok {
				select {
				case <-time.After(releaseDelayAfterSnapshot):
				case <-s.done:
				}
			}
			atomic.AddInt64(&s.inflightSnapshots, -1)
		case <-s.done:
			return
		}
	}()
}

// apply takes entries received from Raft (after it has been committed) and
// applies them to the current state of the EtcdServer.
// The given entries should not be empty.
func (s *EtcdServer) apply(es []raftpb.Entry, confState *raftpb.ConfState) (uint64, bool) {
	var applied uint64
	var shouldstop bool
	for i := range es {
		e := es[i]
		switch e.Type {
		case raftpb.EntryNormal:
			// raft state machine may generate noop entry when leader confirmation.
			// skip it in advance to avoid some potential bug in the future
			if len(e.Data) == 0 {
				select {
				case s.forceVersionC <- struct{}{}:
				default:
				}
				break
			}

			var raftReq pb.InternalRaftRequest
			if !pbutil.MaybeUnmarshal(&raftReq, e.Data) { // backward compatible
				var r pb.Request
				pbutil.MustUnmarshal(&r, e.Data)
				s.w.Trigger(r.ID, s.applyRequest(r))
			} else if raftReq.V2 != nil {
				req := raftReq.V2
				s.w.Trigger(req.ID, s.applyRequest(*req))
			} else {
				// do not re-apply applied entries.
				if e.Index <= s.consistIndex.ConsistentIndex() {
					break
				}
				// set the consistent index of current executing entry
				s.consistIndex.setConsistentIndex(e.Index)
				ar := s.applyV3Request(&raftReq)
				if ar.err != ErrNoSpace || len(s.alarmStore.Get(pb.AlarmType_NOSPACE)) > 0 {
					s.w.Trigger(raftReq.ID, ar)
					break
				}
				plog.Errorf("applying raft message exceeded backend quota")
				go func() {
					a := &pb.AlarmRequest{
						MemberID: uint64(s.ID()),
						Action:   pb.AlarmRequest_ACTIVATE,
						Alarm:    pb.AlarmType_NOSPACE,
					}
					r := pb.InternalRaftRequest{Alarm: a}
					s.processInternalRaftRequest(context.TODO(), r)
					s.w.Trigger(raftReq.ID, ar)
				}()
			}
		case raftpb.EntryConfChange:
			var cc raftpb.ConfChange
			pbutil.MustUnmarshal(&cc, e.Data)
			removedSelf, err := s.applyConfChange(cc, confState)
			shouldstop = shouldstop || removedSelf
			s.w.Trigger(cc.ID, err)
		default:
			plog.Panicf("entry type should be either EntryNormal or EntryConfChange")
		}
		atomic.StoreUint64(&s.r.index, e.Index)
		atomic.StoreUint64(&s.r.term, e.Term)
		applied = e.Index
	}
	return applied, shouldstop
}

// applyRequest interprets r as a call to store.X and returns a Response interpreted
// from store.Event
func (s *EtcdServer) applyRequest(r pb.Request) Response {
	f := func(ev *store.Event, err error) Response {
		return Response{Event: ev, err: err}
	}

	refresh, _ := pbutil.GetBool(r.Refresh)
	ttlOptions := store.TTLOptionSet{Refresh: refresh}
	if r.Expiration != 0 {
		ttlOptions.ExpireTime = time.Unix(0, r.Expiration)
	}

	switch r.Method {
	case "POST":
		return f(s.store.Create(r.Path, r.Dir, r.Val, true, ttlOptions))
	case "PUT":
		exists, existsSet := pbutil.GetBool(r.PrevExist)
		switch {
		case existsSet:
			if exists {
				if r.PrevIndex == 0 && r.PrevValue == "" {
					return f(s.store.Update(r.Path, r.Val, ttlOptions))
				} else {
					return f(s.store.CompareAndSwap(r.Path, r.PrevValue, r.PrevIndex, r.Val, ttlOptions))
				}
			}
			return f(s.store.Create(r.Path, r.Dir, r.Val, false, ttlOptions))
		case r.PrevIndex > 0 || r.PrevValue != "":
			return f(s.store.CompareAndSwap(r.Path, r.PrevValue, r.PrevIndex, r.Val, ttlOptions))
		default:
			// TODO (yicheng): cluster should be the owner of cluster prefix store
			// we should not modify cluster store here.
			if storeMemberAttributeRegexp.MatchString(r.Path) {
				id := membership.MustParseMemberIDFromKey(path.Dir(r.Path))
				var attr membership.Attributes
				if err := json.Unmarshal([]byte(r.Val), &attr); err != nil {
					plog.Panicf("unmarshal %s should never fail: %v", r.Val, err)
				}
				ok := s.cluster.UpdateAttributes(id, attr)
				if !ok {
					return Response{}
				}
			}
			if r.Path == path.Join(StoreClusterPrefix, "version") {
				s.cluster.SetVersion(semver.Must(semver.NewVersion(r.Val)))
			}
			return f(s.store.Set(r.Path, r.Dir, r.Val, ttlOptions))
		}
	case "DELETE":
		switch {
		case r.PrevIndex > 0 || r.PrevValue != "":
			return f(s.store.CompareAndDelete(r.Path, r.PrevValue, r.PrevIndex))
		default:
			return f(s.store.Delete(r.Path, r.Dir, r.Recursive))
		}
	case "QGET":
		return f(s.store.Get(r.Path, r.Recursive, r.Sorted))
	case "SYNC":
		s.store.DeleteExpiredKeys(time.Unix(0, r.Time))
		return Response{}
	default:
		// This should never be reached, but just in case:
		return Response{err: ErrUnknownMethod}
	}
}

// applyConfChange applies a ConfChange to the server. It is only
// invoked with a ConfChange that has already passed through Raft
func (s *EtcdServer) applyConfChange(cc raftpb.ConfChange, confState *raftpb.ConfState) (bool, error) {
	if err := s.cluster.ValidateConfigurationChange(cc); err != nil {
		cc.NodeID = raft.None
		s.r.ApplyConfChange(cc)
		return false, err
	}
	*confState = *s.r.ApplyConfChange(cc)
	switch cc.Type {
	case raftpb.ConfChangeAddNode:
		m := new(membership.Member)
		if err := json.Unmarshal(cc.Context, m); err != nil {
			plog.Panicf("unmarshal member should never fail: %v", err)
		}
		if cc.NodeID != uint64(m.ID) {
			plog.Panicf("nodeID should always be equal to member ID")
		}
		s.cluster.AddMember(m)
		if m.ID == s.id {
			plog.Noticef("added local member %s %v to cluster %s", m.ID, m.PeerURLs, s.cluster.ID())
		} else {
			s.r.transport.AddPeer(m.ID, m.PeerURLs)
			plog.Noticef("added member %s %v to cluster %s", m.ID, m.PeerURLs, s.cluster.ID())
		}
	case raftpb.ConfChangeRemoveNode:
		id := types.ID(cc.NodeID)
		s.cluster.RemoveMember(id)
		if id == s.id {
			return true, nil
		} else {
			s.r.transport.RemovePeer(id)
			plog.Noticef("removed member %s from cluster %s", id, s.cluster.ID())
		}
	case raftpb.ConfChangeUpdateNode:
		m := new(membership.Member)
		if err := json.Unmarshal(cc.Context, m); err != nil {
			plog.Panicf("unmarshal member should never fail: %v", err)
		}
		if cc.NodeID != uint64(m.ID) {
			plog.Panicf("nodeID should always be equal to member ID")
		}
		s.cluster.UpdateRaftAttributes(m.ID, m.RaftAttributes)
		if m.ID == s.id {
			plog.Noticef("update local member %s %v in cluster %s", m.ID, m.PeerURLs, s.cluster.ID())
		} else {
			s.r.transport.UpdatePeer(m.ID, m.PeerURLs)
			plog.Noticef("update member %s %v in cluster %s", m.ID, m.PeerURLs, s.cluster.ID())
		}
	}
	return false, nil
}

// TODO: non-blocking snapshot
func (s *EtcdServer) snapshot(snapi uint64, confState raftpb.ConfState) {
	clone := s.store.Clone()

	go func() {
		d, err := clone.SaveNoCopy()
		// TODO: current store will never fail to do a snapshot
		// what should we do if the store might fail?
		if err != nil {
			plog.Panicf("store save should never fail: %v", err)
		}
		snap, err := s.r.raftStorage.CreateSnapshot(snapi, &confState, d)
		if err != nil {
			// the snapshot was done asynchronously with the progress of raft.
			// raft might have already got a newer snapshot.
			if err == raft.ErrSnapOutOfDate {
				return
			}
			plog.Panicf("unexpected create snapshot error %v", err)
		}
		// commit v3 storage because WAL file before snapshot index
		// could be removed after SaveSnap.
		s.KV().Commit()
		// SaveSnap saves the snapshot and releases the locked wal files
		// to the snapshot index.
		if err = s.r.storage.SaveSnap(snap); err != nil {
			plog.Fatalf("save snapshot error: %v", err)
		}
		plog.Infof("saved snapshot at index %d", snap.Metadata.Index)

		// keep some in memory log entries for slow followers.
		compacti := uint64(1)
		if snapi > numberOfCatchUpEntries {
			compacti = snapi - numberOfCatchUpEntries
		}
		err = s.r.raftStorage.Compact(compacti)
		if err != nil {
			// the compaction was done asynchronously with the progress of raft.
			// raft log might already been compact.
			if err == raft.ErrCompacted {
				return
			}
			plog.Panicf("unexpected compaction error %v", err)
		}
		plog.Infof("compacted raft log at %d", compacti)
	}()
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
		case <-s.done:
			return
		}

		if s.Leader() != s.ID() {
			continue
		}

		v := decideClusterVersion(getVersions(s.cluster, s.id, s.peerRt))
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
			if v != nil {
				go s.updateClusterVersion(v.String())
			} else {
				go s.updateClusterVersion(version.MinClusterVersion)
			}
			continue
		}

		// update cluster version only if the decided version is greater than
		// the current cluster version
		if v != nil && s.cluster.Version().LessThan(*v) {
			go s.updateClusterVersion(v.String())
		}
	}
}

func (s *EtcdServer) updateClusterVersion(ver string) {
	if s.cluster.Version() == nil {
		plog.Infof("setting up the initial cluster version to %s", version.Cluster(ver))
	} else {
		plog.Infof("updating the cluster version from %s to %s", version.Cluster(s.cluster.Version().String()), version.Cluster(ver))
	}
	req := pb.Request{
		Method: "PUT",
		Path:   path.Join(StoreClusterPrefix, "version"),
		Val:    ver,
	}
	ctx, cancel := context.WithTimeout(context.Background(), s.cfg.ReqTimeout())
	_, err := s.Do(ctx, req)
	cancel()
	switch err {
	case nil:
		return
	case ErrStopped:
		plog.Infof("aborting update cluster version because server is stopped")
		return
	default:
		plog.Errorf("error updating cluster version (%v)", err)
	}
}

func (s *EtcdServer) parseProposeCtxErr(err error, start time.Time) error {
	switch err {
	case context.Canceled:
		return ErrCanceled
	case context.DeadlineExceeded:
		curLeadElected := s.r.leadElectedTime()
		prevLeadLost := curLeadElected.Add(-2 * time.Duration(s.cfg.ElectionTicks) * time.Duration(s.cfg.TickMs) * time.Millisecond)
		if start.After(prevLeadLost) && start.Before(curLeadElected) {
			return ErrTimeoutDueToLeaderFail
		}

		lead := types.ID(atomic.LoadUint64(&s.r.lead))
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

func (s *EtcdServer) KV() dstorage.ConsistentWatchableKV { return s.kv }
func (s *EtcdServer) Backend() backend.Backend {
	s.bemu.Lock()
	defer s.bemu.Unlock()
	return s.be
}

func (s *EtcdServer) AuthStore() auth.AuthStore { return s.authStore }

func (s *EtcdServer) restoreAlarms() error {
	s.applyV3 = newQuotaApplierV3(s, &applierV3backend{s})

	as, err := alarm.NewAlarmStore(s)
	if err != nil {
		return err
	}
	s.alarmStore = as
	if len(as.Get(pb.AlarmType_NOSPACE)) > 0 {
		s.applyV3 = newApplierV3Capped(s.applyV3)
	}
	return nil
}
