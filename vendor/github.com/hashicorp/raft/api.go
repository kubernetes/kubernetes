package raft

import (
	"errors"
	"fmt"
	"log"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/armon/go-metrics"
)

var (
	// ErrLeader is returned when an operation can't be completed on a
	// leader node.
	ErrLeader = errors.New("node is the leader")

	// ErrNotLeader is returned when an operation can't be completed on a
	// follower or candidate node.
	ErrNotLeader = errors.New("node is not the leader")

	// ErrLeadershipLost is returned when a leader fails to commit a log entry
	// because it's been deposed in the process.
	ErrLeadershipLost = errors.New("leadership lost while committing log")

	// ErrRaftShutdown is returned when operations are requested against an
	// inactive Raft.
	ErrRaftShutdown = errors.New("raft is already shutdown")

	// ErrEnqueueTimeout is returned when a command fails due to a timeout.
	ErrEnqueueTimeout = errors.New("timed out enqueuing operation")

	// ErrNothingNewToSnapshot is returned when trying to create a snapshot
	// but there's nothing new commited to the FSM since we started.
	ErrNothingNewToSnapshot = errors.New("nothing new to snapshot")

	// ErrUnsupportedProtocol is returned when an operation is attempted
	// that's not supported by the current protocol version.
	ErrUnsupportedProtocol = errors.New("operation not supported with current protocol version")

	// ErrCantBootstrap is returned when attempt is made to bootstrap a
	// cluster that already has state present.
	ErrCantBootstrap = errors.New("bootstrap only works on new clusters")
)

// Raft implements a Raft node.
type Raft struct {
	raftState

	// protocolVersion is used to inter-operate with Raft servers running
	// different versions of the library. See comments in config.go for more
	// details.
	protocolVersion ProtocolVersion

	// applyCh is used to async send logs to the main thread to
	// be committed and applied to the FSM.
	applyCh chan *logFuture

	// Configuration provided at Raft initialization
	conf Config

	// FSM is the client state machine to apply commands to
	fsm FSM

	// fsmCommitCh is used to trigger async application of logs to the fsm
	fsmCommitCh chan commitTuple

	// fsmRestoreCh is used to trigger a restore from snapshot
	fsmRestoreCh chan *restoreFuture

	// fsmSnapshotCh is used to trigger a new snapshot being taken
	fsmSnapshotCh chan *reqSnapshotFuture

	// lastContact is the last time we had contact from the
	// leader node. This can be used to gauge staleness.
	lastContact     time.Time
	lastContactLock sync.RWMutex

	// Leader is the current cluster leader
	leader     ServerAddress
	leaderLock sync.RWMutex

	// leaderCh is used to notify of leadership changes
	leaderCh chan bool

	// leaderState used only while state is leader
	leaderState leaderState

	// Stores our local server ID, used to avoid sending RPCs to ourself
	localID ServerID

	// Stores our local addr
	localAddr ServerAddress

	// Used for our logging
	logger *log.Logger

	// LogStore provides durable storage for logs
	logs LogStore

	// Used to request the leader to make configuration changes.
	configurationChangeCh chan *configurationChangeFuture

	// Tracks the latest configuration and latest committed configuration from
	// the log/snapshot.
	configurations configurations

	// RPC chan comes from the transport layer
	rpcCh <-chan RPC

	// Shutdown channel to exit, protected to prevent concurrent exits
	shutdown     bool
	shutdownCh   chan struct{}
	shutdownLock sync.Mutex

	// snapshots is used to store and retrieve snapshots
	snapshots SnapshotStore

	// snapshotCh is used for user triggered snapshots
	snapshotCh chan *snapshotFuture

	// stable is a StableStore implementation for durable state
	// It provides stable storage for many fields in raftState
	stable StableStore

	// The transport layer we use
	trans Transport

	// verifyCh is used to async send verify futures to the main thread
	// to verify we are still the leader
	verifyCh chan *verifyFuture

	// configurationsCh is used to get the configuration data safely from
	// outside of the main thread.
	configurationsCh chan *configurationsFuture

	// bootstrapCh is used to attempt an initial bootstrap from outside of
	// the main thread.
	bootstrapCh chan *bootstrapFuture

	// List of observers and the mutex that protects them. The observers list
	// is indexed by an artificial ID which is used for deregistration.
	observersLock sync.RWMutex
	observers     map[uint64]*Observer
}

// BootstrapCluster initializes a server's storage with the given cluster
// configuration. This should only be called at the beginning of time for the
// cluster, and you absolutely must make sure that you call it with the same
// configuration on all the Voter servers. There is no need to bootstrap
// Nonvoter and Staging servers.
//
// One sane approach is to boostrap a single server with a configuration
// listing just itself as a Voter, then invoke AddVoter() on it to add other
// servers to the cluster.
func BootstrapCluster(conf *Config, logs LogStore, stable StableStore,
	snaps SnapshotStore, trans Transport, configuration Configuration) error {
	// Validate the Raft server config.
	if err := ValidateConfig(conf); err != nil {
		return err
	}

	// Sanity check the Raft peer configuration.
	if err := checkConfiguration(configuration); err != nil {
		return err
	}

	// Make sure the cluster is in a clean state.
	hasState, err := HasExistingState(logs, stable, snaps)
	if err != nil {
		return fmt.Errorf("failed to check for existing state: %v", err)
	}
	if hasState {
		return ErrCantBootstrap
	}

	// Set current term to 1.
	if err := stable.SetUint64(keyCurrentTerm, 1); err != nil {
		return fmt.Errorf("failed to save current term: %v", err)
	}

	// Append configuration entry to log.
	entry := &Log{
		Index: 1,
		Term:  1,
	}
	if conf.ProtocolVersion < 3 {
		entry.Type = LogRemovePeerDeprecated
		entry.Data = encodePeers(configuration, trans)
	} else {
		entry.Type = LogConfiguration
		entry.Data = encodeConfiguration(configuration)
	}
	if err := logs.StoreLog(entry); err != nil {
		return fmt.Errorf("failed to append configuration entry to log: %v", err)
	}

	return nil
}

// RecoverCluster is used to manually force a new configuration in order to
// recover from a loss of quorum where the current configuration cannot be
// restored, such as when several servers die at the same time. This works by
// reading all the current state for this server, creating a snapshot with the
// supplied configuration, and then truncating the Raft log. This is the only
// safe way to force a given configuration without actually altering the log to
// insert any new entries, which could cause conflicts with other servers with
// different state.
//
// WARNING! This operation implicitly commits all entries in the Raft log, so
// in general this is an extremely unsafe operation. If you've lost your other
// servers and are performing a manual recovery, then you've also lost the
// commit information, so this is likely the best you can do, but you should be
// aware that calling this can cause Raft log entries that were in the process
// of being replicated but not yet be committed to be committed.
//
// Note the FSM passed here is used for the snapshot operations and will be
// left in a state that should not be used by the application. Be sure to
// discard this FSM and any associated state and provide a fresh one when
// calling NewRaft later.
//
// A typical way to recover the cluster is to shut down all servers and then
// run RecoverCluster on every server using an identical configuration. When
// the cluster is then restarted, and election should occur and then Raft will
// resume normal operation. If it's desired to make a particular server the
// leader, this can be used to inject a new configuration with that server as
// the sole voter, and then join up other new clean-state peer servers using
// the usual APIs in order to bring the cluster back into a known state.
func RecoverCluster(conf *Config, fsm FSM, logs LogStore, stable StableStore,
	snaps SnapshotStore, trans Transport, configuration Configuration) error {
	// Validate the Raft server config.
	if err := ValidateConfig(conf); err != nil {
		return err
	}

	// Sanity check the Raft peer configuration.
	if err := checkConfiguration(configuration); err != nil {
		return err
	}

	// Refuse to recover if there's no existing state. This would be safe to
	// do, but it is likely an indication of an operator error where they
	// expect data to be there and it's not. By refusing, we force them
	// to show intent to start a cluster fresh by explicitly doing a
	// bootstrap, rather than quietly fire up a fresh cluster here.
	hasState, err := HasExistingState(logs, stable, snaps)
	if err != nil {
		return fmt.Errorf("failed to check for existing state: %v", err)
	}
	if !hasState {
		return fmt.Errorf("refused to recover cluster with no initial state, this is probably an operator error")
	}

	// Attempt to restore any snapshots we find, newest to oldest.
	var snapshotIndex uint64
	var snapshotTerm uint64
	snapshots, err := snaps.List()
	if err != nil {
		return fmt.Errorf("failed to list snapshots: %v", err)
	}
	for _, snapshot := range snapshots {
		_, source, err := snaps.Open(snapshot.ID)
		if err != nil {
			// Skip this one and try the next. We will detect if we
			// couldn't open any snapshots.
			continue
		}
		defer source.Close()

		if err := fsm.Restore(source); err != nil {
			// Same here, skip and try the next one.
			continue
		}

		snapshotIndex = snapshot.Index
		snapshotTerm = snapshot.Term
		break
	}
	if len(snapshots) > 0 && (snapshotIndex == 0 || snapshotTerm == 0) {
		return fmt.Errorf("failed to restore any of the available snapshots")
	}

	// The snapshot information is the best known end point for the data
	// until we play back the Raft log entries.
	lastIndex := snapshotIndex
	lastTerm := snapshotTerm

	// Apply any Raft log entries past the snapshot.
	lastLogIndex, err := logs.LastIndex()
	if err != nil {
		return fmt.Errorf("failed to find last log: %v", err)
	}
	for index := snapshotIndex + 1; index <= lastLogIndex; index++ {
		var entry Log
		if err := logs.GetLog(index, &entry); err != nil {
			return fmt.Errorf("failed to get log at index %d: %v", index, err)
		}
		if entry.Type == LogCommand {
			_ = fsm.Apply(&entry)
		}
		lastIndex = entry.Index
		lastTerm = entry.Term
	}

	// Create a new snapshot, placing the configuration in as if it was
	// committed at index 1.
	snapshot, err := fsm.Snapshot()
	if err != nil {
		return fmt.Errorf("failed to snapshot FSM: %v", err)
	}
	version := getSnapshotVersion(conf.ProtocolVersion)
	sink, err := snaps.Create(version, lastIndex, lastTerm, configuration, 1, trans)
	if err != nil {
		return fmt.Errorf("failed to create snapshot: %v", err)
	}
	if err := snapshot.Persist(sink); err != nil {
		return fmt.Errorf("failed to persist snapshot: %v", err)
	}
	if err := sink.Close(); err != nil {
		return fmt.Errorf("failed to finalize snapshot: %v", err)
	}

	// Compact the log so that we don't get bad interference from any
	// configuration change log entries that might be there.
	firstLogIndex, err := logs.FirstIndex()
	if err != nil {
		return fmt.Errorf("failed to get first log index: %v", err)
	}
	if err := logs.DeleteRange(firstLogIndex, lastLogIndex); err != nil {
		return fmt.Errorf("log compaction failed: %v", err)
	}

	return nil
}

// HasExistingState returns true if the server has any existing state (logs,
// knowledge of a current term, or any snapshots).
func HasExistingState(logs LogStore, stable StableStore, snaps SnapshotStore) (bool, error) {
	// Make sure we don't have a current term.
	currentTerm, err := stable.GetUint64(keyCurrentTerm)
	if err == nil {
		if currentTerm > 0 {
			return true, nil
		}
	} else {
		if err.Error() != "not found" {
			return false, fmt.Errorf("failed to read current term: %v", err)
		}
	}

	// Make sure we have an empty log.
	lastIndex, err := logs.LastIndex()
	if err != nil {
		return false, fmt.Errorf("failed to get last log index: %v", err)
	}
	if lastIndex > 0 {
		return true, nil
	}

	// Make sure we have no snapshots
	snapshots, err := snaps.List()
	if err != nil {
		return false, fmt.Errorf("failed to list snapshots: %v", err)
	}
	if len(snapshots) > 0 {
		return true, nil
	}

	return false, nil
}

// NewRaft is used to construct a new Raft node. It takes a configuration, as well
// as implementations of various interfaces that are required. If we have any
// old state, such as snapshots, logs, peers, etc, all those will be restored
// when creating the Raft node.
func NewRaft(conf *Config, fsm FSM, logs LogStore, stable StableStore, snaps SnapshotStore, trans Transport) (*Raft, error) {
	// Validate the configuration.
	if err := ValidateConfig(conf); err != nil {
		return nil, err
	}

	// Ensure we have a LogOutput.
	var logger *log.Logger
	if conf.Logger != nil {
		logger = conf.Logger
	} else {
		if conf.LogOutput == nil {
			conf.LogOutput = os.Stderr
		}
		logger = log.New(conf.LogOutput, "", log.LstdFlags)
	}

	// Try to restore the current term.
	currentTerm, err := stable.GetUint64(keyCurrentTerm)
	if err != nil && err.Error() != "not found" {
		return nil, fmt.Errorf("failed to load current term: %v", err)
	}

	// Read the index of the last log entry.
	lastIndex, err := logs.LastIndex()
	if err != nil {
		return nil, fmt.Errorf("failed to find last log: %v", err)
	}

	// Get the last log entry.
	var lastLog Log
	if lastIndex > 0 {
		if err = logs.GetLog(lastIndex, &lastLog); err != nil {
			return nil, fmt.Errorf("failed to get last log at index %d: %v", lastIndex, err)
		}
	}

	// Make sure we have a valid server address and ID.
	protocolVersion := conf.ProtocolVersion
	localAddr := ServerAddress(trans.LocalAddr())
	localID := conf.LocalID

	// TODO (slackpad) - When we deprecate protocol version 2, remove this
	// along with the AddPeer() and RemovePeer() APIs.
	if protocolVersion < 3 && string(localID) != string(localAddr) {
		return nil, fmt.Errorf("when running with ProtocolVersion < 3, LocalID must be set to the network address")
	}

	// Create Raft struct.
	r := &Raft{
		protocolVersion: protocolVersion,
		applyCh:         make(chan *logFuture),
		conf:            *conf,
		fsm:             fsm,
		fsmCommitCh:     make(chan commitTuple, 128),
		fsmRestoreCh:    make(chan *restoreFuture),
		fsmSnapshotCh:   make(chan *reqSnapshotFuture),
		leaderCh:        make(chan bool),
		localID:         localID,
		localAddr:       localAddr,
		logger:          logger,
		logs:            logs,
		configurationChangeCh: make(chan *configurationChangeFuture),
		configurations:        configurations{},
		rpcCh:                 trans.Consumer(),
		snapshots:             snaps,
		snapshotCh:            make(chan *snapshotFuture),
		shutdownCh:            make(chan struct{}),
		stable:                stable,
		trans:                 trans,
		verifyCh:              make(chan *verifyFuture, 64),
		configurationsCh:      make(chan *configurationsFuture, 8),
		bootstrapCh:           make(chan *bootstrapFuture),
		observers:             make(map[uint64]*Observer),
	}

	// Initialize as a follower.
	r.setState(Follower)

	// Start as leader if specified. This should only be used
	// for testing purposes.
	if conf.StartAsLeader {
		r.setState(Leader)
		r.setLeader(r.localAddr)
	}

	// Restore the current term and the last log.
	r.setCurrentTerm(currentTerm)
	r.setLastLog(lastLog.Index, lastLog.Term)

	// Attempt to restore a snapshot if there are any.
	if err := r.restoreSnapshot(); err != nil {
		return nil, err
	}

	// Scan through the log for any configuration change entries.
	snapshotIndex, _ := r.getLastSnapshot()
	for index := snapshotIndex + 1; index <= lastLog.Index; index++ {
		var entry Log
		if err := r.logs.GetLog(index, &entry); err != nil {
			r.logger.Printf("[ERR] raft: Failed to get log at %d: %v", index, err)
			panic(err)
		}
		r.processConfigurationLogEntry(&entry)
	}
	r.logger.Printf("[INFO] raft: Initial configuration (index=%d): %+v",
		r.configurations.latestIndex, r.configurations.latest.Servers)

	// Setup a heartbeat fast-path to avoid head-of-line
	// blocking where possible. It MUST be safe for this
	// to be called concurrently with a blocking RPC.
	trans.SetHeartbeatHandler(r.processHeartbeat)

	// Start the background work.
	r.goFunc(r.run)
	r.goFunc(r.runFSM)
	r.goFunc(r.runSnapshots)
	return r, nil
}

// restoreSnapshot attempts to restore the latest snapshots, and fails if none
// of them can be restored. This is called at initialization time, and is
// completely unsafe to call at any other time.
func (r *Raft) restoreSnapshot() error {
	snapshots, err := r.snapshots.List()
	if err != nil {
		r.logger.Printf("[ERR] raft: Failed to list snapshots: %v", err)
		return err
	}

	// Try to load in order of newest to oldest
	for _, snapshot := range snapshots {
		_, source, err := r.snapshots.Open(snapshot.ID)
		if err != nil {
			r.logger.Printf("[ERR] raft: Failed to open snapshot %v: %v", snapshot.ID, err)
			continue
		}
		defer source.Close()

		if err := r.fsm.Restore(source); err != nil {
			r.logger.Printf("[ERR] raft: Failed to restore snapshot %v: %v", snapshot.ID, err)
			continue
		}

		// Log success
		r.logger.Printf("[INFO] raft: Restored from snapshot %v", snapshot.ID)

		// Update the lastApplied so we don't replay old logs
		r.setLastApplied(snapshot.Index)

		// Update the last stable snapshot info
		r.setLastSnapshot(snapshot.Index, snapshot.Term)

		// Update the configuration
		if snapshot.Version > 0 {
			r.configurations.committed = snapshot.Configuration
			r.configurations.committedIndex = snapshot.ConfigurationIndex
			r.configurations.latest = snapshot.Configuration
			r.configurations.latestIndex = snapshot.ConfigurationIndex
		} else {
			configuration := decodePeers(snapshot.Peers, r.trans)
			r.configurations.committed = configuration
			r.configurations.committedIndex = snapshot.Index
			r.configurations.latest = configuration
			r.configurations.latestIndex = snapshot.Index
		}

		// Success!
		return nil
	}

	// If we had snapshots and failed to load them, its an error
	if len(snapshots) > 0 {
		return fmt.Errorf("failed to load any existing snapshots")
	}
	return nil
}

// BootstrapCluster is equivalent to non-member BootstrapCluster but can be
// called on an un-bootstrapped Raft instance after it has been created. This
// should only be called at the beginning of time for the cluster, and you
// absolutely must make sure that you call it with the same configuration on all
// the Voter servers. There is no need to bootstrap Nonvoter and Staging
// servers.
func (r *Raft) BootstrapCluster(configuration Configuration) Future {
	bootstrapReq := &bootstrapFuture{}
	bootstrapReq.init()
	bootstrapReq.configuration = configuration
	select {
	case <-r.shutdownCh:
		return errorFuture{ErrRaftShutdown}
	case r.bootstrapCh <- bootstrapReq:
		return bootstrapReq
	}
}

// Leader is used to return the current leader of the cluster.
// It may return empty string if there is no current leader
// or the leader is unknown.
func (r *Raft) Leader() ServerAddress {
	r.leaderLock.RLock()
	leader := r.leader
	r.leaderLock.RUnlock()
	return leader
}

// Apply is used to apply a command to the FSM in a highly consistent
// manner. This returns a future that can be used to wait on the application.
// An optional timeout can be provided to limit the amount of time we wait
// for the command to be started. This must be run on the leader or it
// will fail.
func (r *Raft) Apply(cmd []byte, timeout time.Duration) ApplyFuture {
	metrics.IncrCounter([]string{"raft", "apply"}, 1)
	var timer <-chan time.Time
	if timeout > 0 {
		timer = time.After(timeout)
	}

	// Create a log future, no index or term yet
	logFuture := &logFuture{
		log: Log{
			Type: LogCommand,
			Data: cmd,
		},
	}
	logFuture.init()

	select {
	case <-timer:
		return errorFuture{ErrEnqueueTimeout}
	case <-r.shutdownCh:
		return errorFuture{ErrRaftShutdown}
	case r.applyCh <- logFuture:
		return logFuture
	}
}

// Barrier is used to issue a command that blocks until all preceeding
// operations have been applied to the FSM. It can be used to ensure the
// FSM reflects all queued writes. An optional timeout can be provided to
// limit the amount of time we wait for the command to be started. This
// must be run on the leader or it will fail.
func (r *Raft) Barrier(timeout time.Duration) Future {
	metrics.IncrCounter([]string{"raft", "barrier"}, 1)
	var timer <-chan time.Time
	if timeout > 0 {
		timer = time.After(timeout)
	}

	// Create a log future, no index or term yet
	logFuture := &logFuture{
		log: Log{
			Type: LogBarrier,
		},
	}
	logFuture.init()

	select {
	case <-timer:
		return errorFuture{ErrEnqueueTimeout}
	case <-r.shutdownCh:
		return errorFuture{ErrRaftShutdown}
	case r.applyCh <- logFuture:
		return logFuture
	}
}

// VerifyLeader is used to ensure the current node is still
// the leader. This can be done to prevent stale reads when a
// new leader has potentially been elected.
func (r *Raft) VerifyLeader() Future {
	metrics.IncrCounter([]string{"raft", "verify_leader"}, 1)
	verifyFuture := &verifyFuture{}
	verifyFuture.init()
	select {
	case <-r.shutdownCh:
		return errorFuture{ErrRaftShutdown}
	case r.verifyCh <- verifyFuture:
		return verifyFuture
	}
}

// GetConfiguration returns the latest configuration and its associated index
// currently in use. This may not yet be committed. This must not be called on
// the main thread (which can access the information directly).
func (r *Raft) GetConfiguration() ConfigurationFuture {
	configReq := &configurationsFuture{}
	configReq.init()
	select {
	case <-r.shutdownCh:
		configReq.respond(ErrRaftShutdown)
		return configReq
	case r.configurationsCh <- configReq:
		return configReq
	}
}

// AddPeer (deprecated) is used to add a new peer into the cluster. This must be
// run on the leader or it will fail. Use AddVoter/AddNonvoter instead.
func (r *Raft) AddPeer(peer ServerAddress) Future {
	if r.protocolVersion > 2 {
		return errorFuture{ErrUnsupportedProtocol}
	}

	return r.requestConfigChange(configurationChangeRequest{
		command:       AddStaging,
		serverID:      ServerID(peer),
		serverAddress: peer,
		prevIndex:     0,
	}, 0)
}

// RemovePeer (deprecated) is used to remove a peer from the cluster. If the
// current leader is being removed, it will cause a new election
// to occur. This must be run on the leader or it will fail.
// Use RemoveServer instead.
func (r *Raft) RemovePeer(peer ServerAddress) Future {
	if r.protocolVersion > 2 {
		return errorFuture{ErrUnsupportedProtocol}
	}

	return r.requestConfigChange(configurationChangeRequest{
		command:   RemoveServer,
		serverID:  ServerID(peer),
		prevIndex: 0,
	}, 0)
}

// AddVoter will add the given server to the cluster as a staging server. If the
// server is already in the cluster as a voter, this does nothing. This must be
// run on the leader or it will fail. The leader will promote the staging server
// to a voter once that server is ready. If nonzero, prevIndex is the index of
// the only configuration upon which this change may be applied; if another
// configuration entry has been added in the meantime, this request will fail.
// If nonzero, timeout is how long this server should wait before the
// configuration change log entry is appended.
func (r *Raft) AddVoter(id ServerID, address ServerAddress, prevIndex uint64, timeout time.Duration) IndexFuture {
	if r.protocolVersion < 2 {
		return errorFuture{ErrUnsupportedProtocol}
	}

	return r.requestConfigChange(configurationChangeRequest{
		command:       AddStaging,
		serverID:      id,
		serverAddress: address,
		prevIndex:     prevIndex,
	}, timeout)
}

// AddNonvoter will add the given server to the cluster but won't assign it a
// vote. The server will receive log entries, but it won't participate in
// elections or log entry commitment. If the server is already in the cluster as
// a staging server or voter, this does nothing. This must be run on the leader
// or it will fail. For prevIndex and timeout, see AddVoter.
func (r *Raft) AddNonvoter(id ServerID, address ServerAddress, prevIndex uint64, timeout time.Duration) IndexFuture {
	if r.protocolVersion < 3 {
		return errorFuture{ErrUnsupportedProtocol}
	}

	return r.requestConfigChange(configurationChangeRequest{
		command:       AddNonvoter,
		serverID:      id,
		serverAddress: address,
		prevIndex:     prevIndex,
	}, timeout)
}

// RemoveServer will remove the given server from the cluster. If the current
// leader is being removed, it will cause a new election to occur. This must be
// run on the leader or it will fail. For prevIndex and timeout, see AddVoter.
func (r *Raft) RemoveServer(id ServerID, prevIndex uint64, timeout time.Duration) IndexFuture {
	if r.protocolVersion < 2 {
		return errorFuture{ErrUnsupportedProtocol}
	}

	return r.requestConfigChange(configurationChangeRequest{
		command:   RemoveServer,
		serverID:  id,
		prevIndex: prevIndex,
	}, timeout)
}

// DemoteVoter will take away a server's vote, if it has one. If present, the
// server will continue to receive log entries, but it won't participate in
// elections or log entry commitment. If the server is not in the cluster, this
// does nothing. This must be run on the leader or it will fail. For prevIndex
// and timeout, see AddVoter.
func (r *Raft) DemoteVoter(id ServerID, prevIndex uint64, timeout time.Duration) IndexFuture {
	if r.protocolVersion < 3 {
		return errorFuture{ErrUnsupportedProtocol}
	}

	return r.requestConfigChange(configurationChangeRequest{
		command:   DemoteVoter,
		serverID:  id,
		prevIndex: prevIndex,
	}, timeout)
}

// Shutdown is used to stop the Raft background routines.
// This is not a graceful operation. Provides a future that
// can be used to block until all background routines have exited.
func (r *Raft) Shutdown() Future {
	r.shutdownLock.Lock()
	defer r.shutdownLock.Unlock()

	if !r.shutdown {
		close(r.shutdownCh)
		r.shutdown = true
		r.setState(Shutdown)
		return &shutdownFuture{r}
	}

	// avoid closing transport twice
	return &shutdownFuture{nil}
}

// Snapshot is used to manually force Raft to take a snapshot.
// Returns a future that can be used to block until complete.
func (r *Raft) Snapshot() Future {
	snapFuture := &snapshotFuture{}
	snapFuture.init()
	select {
	case r.snapshotCh <- snapFuture:
		return snapFuture
	case <-r.shutdownCh:
		return errorFuture{ErrRaftShutdown}
	}

}

// State is used to return the current raft state.
func (r *Raft) State() RaftState {
	return r.getState()
}

// LeaderCh is used to get a channel which delivers signals on
// acquiring or losing leadership. It sends true if we become
// the leader, and false if we lose it. The channel is not buffered,
// and does not block on writes.
func (r *Raft) LeaderCh() <-chan bool {
	return r.leaderCh
}

// String returns a string representation of this Raft node.
func (r *Raft) String() string {
	return fmt.Sprintf("Node at %s [%v]", r.localAddr, r.getState())
}

// LastContact returns the time of last contact by a leader.
// This only makes sense if we are currently a follower.
func (r *Raft) LastContact() time.Time {
	r.lastContactLock.RLock()
	last := r.lastContact
	r.lastContactLock.RUnlock()
	return last
}

// Stats is used to return a map of various internal stats. This
// should only be used for informative purposes or debugging.
//
// Keys are: "state", "term", "last_log_index", "last_log_term",
// "commit_index", "applied_index", "fsm_pending",
// "last_snapshot_index", "last_snapshot_term",
// "latest_configuration", "last_contact", and "num_peers".
//
// The value of "state" is a numerical value representing a
// RaftState const.
//
// The value of "latest_configuration" is a string which contains
// the id of each server, its suffrage status, and its address.
//
// The value of "last_contact" is either "never" if there
// has been no contact with a leader, "0" if the node is in the
// leader state, or the time since last contact with a leader
// formatted as a string.
//
// The value of "num_peers" is the number of other voting servers in the
// cluster, not including this node. If this node isn't part of the
// configuration then this will be "0".
//
// All other values are uint64s, formatted as strings.
func (r *Raft) Stats() map[string]string {
	toString := func(v uint64) string {
		return strconv.FormatUint(v, 10)
	}
	lastLogIndex, lastLogTerm := r.getLastLog()
	lastSnapIndex, lastSnapTerm := r.getLastSnapshot()
	s := map[string]string{
		"state":                r.getState().String(),
		"term":                 toString(r.getCurrentTerm()),
		"last_log_index":       toString(lastLogIndex),
		"last_log_term":        toString(lastLogTerm),
		"commit_index":         toString(r.getCommitIndex()),
		"applied_index":        toString(r.getLastApplied()),
		"fsm_pending":          toString(uint64(len(r.fsmCommitCh))),
		"last_snapshot_index":  toString(lastSnapIndex),
		"last_snapshot_term":   toString(lastSnapTerm),
		"protocol_version":     toString(uint64(r.protocolVersion)),
		"protocol_version_min": toString(uint64(ProtocolVersionMin)),
		"protocol_version_max": toString(uint64(ProtocolVersionMax)),
		"snapshot_version_min": toString(uint64(SnapshotVersionMin)),
		"snapshot_version_max": toString(uint64(SnapshotVersionMax)),
	}

	future := r.GetConfiguration()
	if err := future.Error(); err != nil {
		r.logger.Printf("[WARN] raft: could not get configuration for Stats: %v", err)
	} else {
		configuration := future.Configuration()
		s["latest_configuration_index"] = toString(future.Index())
		s["latest_configuration"] = fmt.Sprintf("%+v", configuration.Servers)

		// This is a legacy metric that we've seen people use in the wild.
		hasUs := false
		numPeers := 0
		for _, server := range configuration.Servers {
			if server.Suffrage == Voter {
				if server.ID == r.localID {
					hasUs = true
				} else {
					numPeers++
				}
			}
		}
		if !hasUs {
			numPeers = 0
		}
		s["num_peers"] = toString(uint64(numPeers))
	}

	last := r.LastContact()
	if last.IsZero() {
		s["last_contact"] = "never"
	} else if r.getState() == Leader {
		s["last_contact"] = "0"
	} else {
		s["last_contact"] = fmt.Sprintf("%v", time.Now().Sub(last))
	}
	return s
}

// LastIndex returns the last index in stable storage,
// either from the last log or from the last snapshot.
func (r *Raft) LastIndex() uint64 {
	return r.getLastIndex()
}

// AppliedIndex returns the last index applied to the FSM. This is generally
// lagging behind the last index, especially for indexes that are persisted but
// have not yet been considered committed by the leader. NOTE - this reflects
// the last index that was sent to the application's FSM over the apply channel
// but DOES NOT mean that the application's FSM has yet consumed it and applied
// it to its internal state. Thus, the application's state may lag behind this
// index.
func (r *Raft) AppliedIndex() uint64 {
	return r.getLastApplied()
}
