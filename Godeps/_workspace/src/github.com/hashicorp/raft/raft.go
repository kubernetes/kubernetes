package raft

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"sync"
	"time"

	"github.com/armon/go-metrics"
)

const (
	minCheckInterval = 10 * time.Millisecond
)

var (
	keyCurrentTerm  = []byte("CurrentTerm")
	keyLastVoteTerm = []byte("LastVoteTerm")
	keyLastVoteCand = []byte("LastVoteCand")

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

	// ErrKnownPeer is returned when trying to add a peer to the configuration
	// that already exists.
	ErrKnownPeer = errors.New("peer already known")

	// ErrUnknownPeer is returned when trying to remove a peer from the
	// configuration that doesn't exist.
	ErrUnknownPeer = errors.New("peer is unknown")

	// ErrNothingNewToSnapshot is returned when trying to create a snapshot
	// but there's nothing new commited to the FSM since we started.
	ErrNothingNewToSnapshot = errors.New("Nothing new to snapshot")
)

// commitTuple is used to send an index that was committed,
// with an optional associated future that should be invoked.
type commitTuple struct {
	log    *Log
	future *logFuture
}

// leaderState is state that is used while we are a leader.
type leaderState struct {
	commitCh  chan struct{}
	inflight  *inflight
	replState map[string]*followerReplication
	notify    map[*verifyFuture]struct{}
	stepDown  chan struct{}
}

// Raft implements a Raft node.
type Raft struct {
	raftState

	// applyCh is used to async send logs to the main thread to
	// be committed and applied to the FSM.
	applyCh chan *logFuture

	// Configuration provided at Raft initialization
	conf *Config

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
	leader     string
	leaderLock sync.RWMutex

	// leaderCh is used to notify of leadership changes
	leaderCh chan bool

	// leaderState used only while state is leader
	leaderState leaderState

	// Stores our local addr
	localAddr string

	// Used for our logging
	logger *log.Logger

	// LogStore provides durable storage for logs
	logs LogStore

	// Track our known peers
	peerCh    chan *peerFuture
	peers     []string
	peerStore PeerStore

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
}

// NewRaft is used to construct a new Raft node. It takes a configuration, as well
// as implementations of various interfaces that are required. If we have any old state,
// such as snapshots, logs, peers, etc, all those will be restored when creating the
// Raft node.
func NewRaft(conf *Config, fsm FSM, logs LogStore, stable StableStore, snaps SnapshotStore,
	peerStore PeerStore, trans Transport) (*Raft, error) {
	// Validate the configuration
	if err := ValidateConfig(conf); err != nil {
		return nil, err
	}

	// Ensure we have a LogOutput
	var logger *log.Logger
	if conf.Logger != nil {
		logger = conf.Logger
	} else {
		if conf.LogOutput == nil {
			conf.LogOutput = os.Stderr
		}
		logger = log.New(conf.LogOutput, "", log.LstdFlags)
	}

	// Try to restore the current term
	currentTerm, err := stable.GetUint64(keyCurrentTerm)
	if err != nil && err.Error() != "not found" {
		return nil, fmt.Errorf("failed to load current term: %v", err)
	}

	// Read the last log value
	lastIdx, err := logs.LastIndex()
	if err != nil {
		return nil, fmt.Errorf("failed to find last log: %v", err)
	}

	// Get the log
	var lastLog Log
	if lastIdx > 0 {
		if err := logs.GetLog(lastIdx, &lastLog); err != nil {
			return nil, fmt.Errorf("failed to get last log: %v", err)
		}
	}

	// Construct the list of peers that excludes us
	localAddr := trans.LocalAddr()
	peers, err := peerStore.Peers()
	if err != nil {
		return nil, fmt.Errorf("failed to get list of peers: %v", err)
	}
	peers = ExcludePeer(peers, localAddr)

	// Create Raft struct
	r := &Raft{
		applyCh:       make(chan *logFuture),
		conf:          conf,
		fsm:           fsm,
		fsmCommitCh:   make(chan commitTuple, 128),
		fsmRestoreCh:  make(chan *restoreFuture),
		fsmSnapshotCh: make(chan *reqSnapshotFuture),
		leaderCh:      make(chan bool),
		localAddr:     localAddr,
		logger:        logger,
		logs:          logs,
		peerCh:        make(chan *peerFuture),
		peers:         peers,
		peerStore:     peerStore,
		rpcCh:         trans.Consumer(),
		snapshots:     snaps,
		snapshotCh:    make(chan *snapshotFuture),
		shutdownCh:    make(chan struct{}),
		stable:        stable,
		trans:         trans,
		verifyCh:      make(chan *verifyFuture, 64),
	}

	// Initialize as a follower
	r.setState(Follower)

	// Start as leader if specified. This should only be used
	// for testing purposes.
	if conf.StartAsLeader {
		r.setState(Leader)
		r.setLeader(r.localAddr)
	}

	// Restore the current term and the last log
	r.setCurrentTerm(currentTerm)
	r.setLastLogIndex(lastLog.Index)
	r.setLastLogTerm(lastLog.Term)

	// Attempt to restore a snapshot if there are any
	if err := r.restoreSnapshot(); err != nil {
		return nil, err
	}

	// Setup a heartbeat fast-path to avoid head-of-line
	// blocking where possible. It MUST be safe for this
	// to be called concurrently with a blocking RPC.
	trans.SetHeartbeatHandler(r.processHeartbeat)

	// Start the background work
	r.goFunc(r.run)
	r.goFunc(r.runFSM)
	r.goFunc(r.runSnapshots)
	return r, nil
}

// Leader is used to return the current leader of the cluster.
// It may return empty string if there is no current leader
// or the leader is unknown.
func (r *Raft) Leader() string {
	r.leaderLock.RLock()
	leader := r.leader
	r.leaderLock.RUnlock()
	return leader
}

// setLeader is used to modify the current leader of the cluster
func (r *Raft) setLeader(leader string) {
	r.leaderLock.Lock()
	r.leader = leader
	r.leaderLock.Unlock()
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

// AddPeer is used to add a new peer into the cluster. This must be
// run on the leader or it will fail.
func (r *Raft) AddPeer(peer string) Future {
	logFuture := &logFuture{
		log: Log{
			Type: LogAddPeer,
			peer: peer,
		},
	}
	logFuture.init()
	select {
	case r.applyCh <- logFuture:
		return logFuture
	case <-r.shutdownCh:
		return errorFuture{ErrRaftShutdown}
	}
}

// RemovePeer is used to remove a peer from the cluster. If the
// current leader is being removed, it will cause a new election
// to occur. This must be run on the leader or it will fail.
func (r *Raft) RemovePeer(peer string) Future {
	logFuture := &logFuture{
		log: Log{
			Type: LogRemovePeer,
			peer: peer,
		},
	}
	logFuture.init()
	select {
	case r.applyCh <- logFuture:
		return logFuture
	case <-r.shutdownCh:
		return errorFuture{ErrRaftShutdown}
	}
}

// SetPeers is used to forcibly replace the set of internal peers and
// the peerstore with the ones specified. This can be considered unsafe.
func (r *Raft) SetPeers(p []string) Future {
	peerFuture := &peerFuture{
		peers: p,
	}
	peerFuture.init()

	select {
	case r.peerCh <- peerFuture:
		return peerFuture
	case <-r.shutdownCh:
		return errorFuture{ErrRaftShutdown}
	}
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
	}

	return &shutdownFuture{r}
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

// Stats is used to return a map of various internal stats. This should only
// be used for informative purposes or debugging.
func (r *Raft) Stats() map[string]string {
	toString := func(v uint64) string {
		return strconv.FormatUint(v, 10)
	}
	s := map[string]string{
		"state":               r.getState().String(),
		"term":                toString(r.getCurrentTerm()),
		"last_log_index":      toString(r.getLastLogIndex()),
		"last_log_term":       toString(r.getLastLogTerm()),
		"commit_index":        toString(r.getCommitIndex()),
		"applied_index":       toString(r.getLastApplied()),
		"fsm_pending":         toString(uint64(len(r.fsmCommitCh))),
		"last_snapshot_index": toString(r.getLastSnapshotIndex()),
		"last_snapshot_term":  toString(r.getLastSnapshotTerm()),
		"num_peers":           toString(uint64(len(r.peers))),
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

// AppliedIndex returns the last index applied to the FSM.
// This is generally lagging behind the last index, especially
// for indexes that are persisted but have not yet been considered
// committed by the leader.
func (r *Raft) AppliedIndex() uint64 {
	return r.getLastApplied()
}

// runFSM is a long running goroutine responsible for applying logs
// to the FSM. This is done async of other logs since we don't want
// the FSM to block our internal operations.
func (r *Raft) runFSM() {
	var lastIndex, lastTerm uint64
	for {
		select {
		case req := <-r.fsmRestoreCh:
			// Open the snapshot
			meta, source, err := r.snapshots.Open(req.ID)
			if err != nil {
				req.respond(fmt.Errorf("failed to open snapshot %v: %v", req.ID, err))
				continue
			}

			// Attempt to restore
			start := time.Now()
			if err := r.fsm.Restore(source); err != nil {
				req.respond(fmt.Errorf("failed to restore snapshot %v: %v", req.ID, err))
				source.Close()
				continue
			}
			source.Close()
			metrics.MeasureSince([]string{"raft", "fsm", "restore"}, start)

			// Update the last index and term
			lastIndex = meta.Index
			lastTerm = meta.Term
			req.respond(nil)

		case req := <-r.fsmSnapshotCh:
			// Is there something to snapshot?
			if lastIndex == 0 {
				req.respond(ErrNothingNewToSnapshot)
				continue
			}

			// Get our peers
			peers, err := r.peerStore.Peers()
			if err != nil {
				req.respond(err)
				continue
			}

			// Start a snapshot
			start := time.Now()
			snap, err := r.fsm.Snapshot()
			metrics.MeasureSince([]string{"raft", "fsm", "snapshot"}, start)

			// Respond to the request
			req.index = lastIndex
			req.term = lastTerm
			req.peers = peers
			req.snapshot = snap
			req.respond(err)

		case commitTuple := <-r.fsmCommitCh:
			// Apply the log if a command
			var resp interface{}
			if commitTuple.log.Type == LogCommand {
				start := time.Now()
				resp = r.fsm.Apply(commitTuple.log)
				metrics.MeasureSince([]string{"raft", "fsm", "apply"}, start)
			}

			// Update the indexes
			lastIndex = commitTuple.log.Index
			lastTerm = commitTuple.log.Term

			// Invoke the future if given
			if commitTuple.future != nil {
				commitTuple.future.response = resp
				commitTuple.future.respond(nil)
			}
		case <-r.shutdownCh:
			return
		}
	}
}

// run is a long running goroutine that runs the Raft FSM.
func (r *Raft) run() {
	for {
		// Check if we are doing a shutdown
		select {
		case <-r.shutdownCh:
			// Clear the leader to prevent forwarding
			r.setLeader("")
			return
		default:
		}

		// Enter into a sub-FSM
		switch r.getState() {
		case Follower:
			r.runFollower()
		case Candidate:
			r.runCandidate()
		case Leader:
			r.runLeader()
		}
	}
}

// runFollower runs the FSM for a follower.
func (r *Raft) runFollower() {
	didWarn := false
	r.logger.Printf("[INFO] raft: %v entering Follower state", r)
	metrics.IncrCounter([]string{"raft", "state", "follower"}, 1)
	heartbeatTimer := randomTimeout(r.conf.HeartbeatTimeout)
	for {
		select {
		case rpc := <-r.rpcCh:
			r.processRPC(rpc)

		case a := <-r.applyCh:
			// Reject any operations since we are not the leader
			a.respond(ErrNotLeader)

		case v := <-r.verifyCh:
			// Reject any operations since we are not the leader
			v.respond(ErrNotLeader)

		case p := <-r.peerCh:
			// Set the peers
			r.peers = ExcludePeer(p.peers, r.localAddr)
			p.respond(r.peerStore.SetPeers(p.peers))

		case <-heartbeatTimer:
			// Restart the heartbeat timer
			heartbeatTimer = randomTimeout(r.conf.HeartbeatTimeout)

			// Check if we have had a successful contact
			lastContact := r.LastContact()
			if time.Now().Sub(lastContact) < r.conf.HeartbeatTimeout {
				continue
			}

			// Heartbeat failed! Transition to the candidate state
			r.setLeader("")
			if len(r.peers) == 0 && !r.conf.EnableSingleNode {
				if !didWarn {
					r.logger.Printf("[WARN] raft: EnableSingleNode disabled, and no known peers. Aborting election.")
					didWarn = true
				}
			} else {
				r.logger.Printf("[WARN] raft: Heartbeat timeout reached, starting election")

				metrics.IncrCounter([]string{"raft", "transition", "heartbeat_timout"}, 1)
				r.setState(Candidate)
				return
			}

		case <-r.shutdownCh:
			return
		}
	}
}

// runCandidate runs the FSM for a candidate.
func (r *Raft) runCandidate() {
	r.logger.Printf("[INFO] raft: %v entering Candidate state", r)
	metrics.IncrCounter([]string{"raft", "state", "candidate"}, 1)

	// Start vote for us, and set a timeout
	voteCh := r.electSelf()
	electionTimer := randomTimeout(r.conf.ElectionTimeout)

	// Tally the votes, need a simple majority
	grantedVotes := 0
	votesNeeded := r.quorumSize()
	r.logger.Printf("[DEBUG] raft: Votes needed: %d", votesNeeded)

	for r.getState() == Candidate {
		select {
		case rpc := <-r.rpcCh:
			r.processRPC(rpc)

		case vote := <-voteCh:
			// Check if the term is greater than ours, bail
			if vote.Term > r.getCurrentTerm() {
				r.logger.Printf("[DEBUG] raft: Newer term discovered, fallback to follower")
				r.setState(Follower)
				r.setCurrentTerm(vote.Term)
				return
			}

			// Check if the vote is granted
			if vote.Granted {
				grantedVotes++
				r.logger.Printf("[DEBUG] raft: Vote granted from %s. Tally: %d", vote.voter, grantedVotes)
			}

			// Check if we've become the leader
			if grantedVotes >= votesNeeded {
				r.logger.Printf("[INFO] raft: Election won. Tally: %d", grantedVotes)
				r.setState(Leader)
				r.setLeader(r.localAddr)
				return
			}

		case a := <-r.applyCh:
			// Reject any operations since we are not the leader
			a.respond(ErrNotLeader)

		case v := <-r.verifyCh:
			// Reject any operations since we are not the leader
			v.respond(ErrNotLeader)

		case p := <-r.peerCh:
			// Set the peers
			r.peers = ExcludePeer(p.peers, r.localAddr)
			p.respond(r.peerStore.SetPeers(p.peers))
			// Become a follower again
			r.setState(Follower)
			return

		case <-electionTimer:
			// Election failed! Restart the election. We simply return,
			// which will kick us back into runCandidate
			r.logger.Printf("[WARN] raft: Election timeout reached, restarting election")
			return

		case <-r.shutdownCh:
			return
		}
	}
}

// runLeader runs the FSM for a leader. Do the setup here and drop into
// the leaderLoop for the hot loop.
func (r *Raft) runLeader() {
	r.logger.Printf("[INFO] raft: %v entering Leader state", r)
	metrics.IncrCounter([]string{"raft", "state", "leader"}, 1)

	// Notify that we are the leader
	asyncNotifyBool(r.leaderCh, true)

	// Push to the notify channel if given
	if notify := r.conf.NotifyCh; notify != nil {
		select {
		case notify <- true:
		case <-r.shutdownCh:
		}
	}

	// Setup leader state
	r.leaderState.commitCh = make(chan struct{}, 1)
	r.leaderState.inflight = newInflight(r.leaderState.commitCh)
	r.leaderState.replState = make(map[string]*followerReplication)
	r.leaderState.notify = make(map[*verifyFuture]struct{})
	r.leaderState.stepDown = make(chan struct{}, 1)

	// Cleanup state on step down
	defer func() {
		// Since we were the leader previously, we update our
		// last contact time when we step down, so that we are not
		// reporting a last contact time from before we were the
		// leader. Otherwise, to a client it would seem our data
		// is extremely stale.
		r.setLastContact()

		// Stop replication
		for _, p := range r.leaderState.replState {
			close(p.stopCh)
		}

		// Cancel inflight requests
		r.leaderState.inflight.Cancel(ErrLeadershipLost)

		// Respond to any pending verify requests
		for future := range r.leaderState.notify {
			future.respond(ErrLeadershipLost)
		}

		// Clear all the state
		r.leaderState.commitCh = nil
		r.leaderState.inflight = nil
		r.leaderState.replState = nil
		r.leaderState.notify = nil
		r.leaderState.stepDown = nil

		// If we are stepping down for some reason, no known leader.
		// We may have stepped down due to an RPC call, which would
		// provide the leader, so we cannot always blank this out.
		r.leaderLock.Lock()
		if r.leader == r.localAddr {
			r.leader = ""
		}
		r.leaderLock.Unlock()

		// Notify that we are not the leader
		asyncNotifyBool(r.leaderCh, false)

		// Push to the notify channel if given
		if notify := r.conf.NotifyCh; notify != nil {
			select {
			case notify <- false:
			case <-r.shutdownCh:
				// On shutdown, make a best effort but do not block
				select {
				case notify <- false:
				default:
				}
			}
		}
	}()

	// Start a replication routine for each peer
	for _, peer := range r.peers {
		r.startReplication(peer)
	}

	// Dispatch a no-op log first. Instead of LogNoop,
	// we use a LogAddPeer with our peerset. This acts like
	// a no-op as well, but when doing an initial bootstrap, ensures
	// that all nodes share a common peerset.
	peerSet := append([]string{r.localAddr}, r.peers...)
	noop := &logFuture{
		log: Log{
			Type: LogAddPeer,
			Data: encodePeers(peerSet, r.trans),
		},
	}
	r.dispatchLogs([]*logFuture{noop})

	// Disable EnableSingleNode after we've been elected leader.
	// This is to prevent a split brain in the future, if we are removed
	// from the cluster and then elect ourself as leader.
	if r.conf.DisableBootstrapAfterElect && r.conf.EnableSingleNode {
		r.logger.Printf("[INFO] raft: Disabling EnableSingleNode (bootstrap)")
		r.conf.EnableSingleNode = false
	}

	// Sit in the leader loop until we step down
	r.leaderLoop()
}

// startReplication is a helper to setup state and start async replication to a peer.
func (r *Raft) startReplication(peer string) {
	lastIdx := r.getLastIndex()
	s := &followerReplication{
		peer:        peer,
		inflight:    r.leaderState.inflight,
		stopCh:      make(chan uint64, 1),
		triggerCh:   make(chan struct{}, 1),
		currentTerm: r.getCurrentTerm(),
		matchIndex:  0,
		nextIndex:   lastIdx + 1,
		lastContact: time.Now(),
		notifyCh:    make(chan struct{}, 1),
		stepDown:    r.leaderState.stepDown,
	}
	r.leaderState.replState[peer] = s
	r.goFunc(func() { r.replicate(s) })
	asyncNotifyCh(s.triggerCh)
}

// leaderLoop is the hot loop for a leader. It is invoked
// after all the various leader setup is done.
func (r *Raft) leaderLoop() {
	// stepDown is used to track if there is an inflight log that
	// would cause us to lose leadership (specifically a RemovePeer of
	// ourselves). If this is the case, we must not allow any logs to
	// be processed in parallel, otherwise we are basing commit on
	// only a single peer (ourself) and replicating to an undefined set
	// of peers.
	stepDown := false

	lease := time.After(r.conf.LeaderLeaseTimeout)
	for r.getState() == Leader {
		select {
		case rpc := <-r.rpcCh:
			r.processRPC(rpc)

		case <-r.leaderState.stepDown:
			r.setState(Follower)

		case <-r.leaderState.commitCh:
			// Get the committed messages
			committed := r.leaderState.inflight.Committed()
			for e := committed.Front(); e != nil; e = e.Next() {
				// Measure the commit time
				commitLog := e.Value.(*logFuture)
				metrics.MeasureSince([]string{"raft", "commitTime"}, commitLog.dispatch)

				// Increment the commit index
				idx := commitLog.log.Index
				r.setCommitIndex(idx)
				r.processLogs(idx, commitLog)
			}

		case v := <-r.verifyCh:
			if v.quorumSize == 0 {
				// Just dispatched, start the verification
				r.verifyLeader(v)

			} else if v.votes < v.quorumSize {
				// Early return, means there must be a new leader
				r.logger.Printf("[WARN] raft: New leader elected, stepping down")
				r.setState(Follower)
				delete(r.leaderState.notify, v)
				v.respond(ErrNotLeader)

			} else {
				// Quorum of members agree, we are still leader
				delete(r.leaderState.notify, v)
				v.respond(nil)
			}

		case p := <-r.peerCh:
			p.respond(ErrLeader)

		case newLog := <-r.applyCh:
			// Group commit, gather all the ready commits
			ready := []*logFuture{newLog}
			for i := 0; i < r.conf.MaxAppendEntries; i++ {
				select {
				case newLog := <-r.applyCh:
					ready = append(ready, newLog)
				default:
					break
				}
			}

			// Handle any peer set changes
			n := len(ready)
			for i := 0; i < n; i++ {
				// Fail all future transactions once stepDown is on
				if stepDown {
					ready[i].respond(ErrNotLeader)
					ready[i], ready[n-1] = ready[n-1], nil
					n--
					i--
					continue
				}

				// Special case AddPeer and RemovePeer
				log := ready[i]
				if log.log.Type != LogAddPeer && log.log.Type != LogRemovePeer {
					continue
				}

				// Check if this log should be ignored. The logs can be
				// reordered here since we have not yet assigned an index
				// and are not violating any promises.
				if !r.preparePeerChange(log) {
					ready[i], ready[n-1] = ready[n-1], nil
					n--
					i--
					continue
				}

				// Apply peer set changes early and check if we will step
				// down after the commit of this log. If so, we must not
				// allow any future entries to make progress to avoid undefined
				// behavior.
				if ok := r.processLog(&log.log, nil, true); ok {
					stepDown = true
				}
			}

			// Nothing to do if all logs are invalid
			if n == 0 {
				continue
			}

			// Dispatch the logs
			ready = ready[:n]
			r.dispatchLogs(ready)

		case <-lease:
			// Check if we've exceeded the lease, potentially stepping down
			maxDiff := r.checkLeaderLease()

			// Next check interval should adjust for the last node we've
			// contacted, without going negative
			checkInterval := r.conf.LeaderLeaseTimeout - maxDiff
			if checkInterval < minCheckInterval {
				checkInterval = minCheckInterval
			}

			// Renew the lease timer
			lease = time.After(checkInterval)

		case <-r.shutdownCh:
			return
		}
	}
}

// verifyLeader must be called from the main thread for safety.
// Causes the followers to attempt an immediate heartbeat.
func (r *Raft) verifyLeader(v *verifyFuture) {
	// Current leader always votes for self
	v.votes = 1

	// Set the quorum size, hot-path for single node
	v.quorumSize = r.quorumSize()
	if v.quorumSize == 1 {
		v.respond(nil)
		return
	}

	// Track this request
	v.notifyCh = r.verifyCh
	r.leaderState.notify[v] = struct{}{}

	// Trigger immediate heartbeats
	for _, repl := range r.leaderState.replState {
		repl.notifyLock.Lock()
		repl.notify = append(repl.notify, v)
		repl.notifyLock.Unlock()
		asyncNotifyCh(repl.notifyCh)
	}
}

// checkLeaderLease is used to check if we can contact a quorum of nodes
// within the last leader lease interval. If not, we need to step down,
// as we may have lost connectivity. Returns the maximum duration without
// contact.
func (r *Raft) checkLeaderLease() time.Duration {
	// Track contacted nodes, we can always contact ourself
	contacted := 1

	// Check each follower
	var maxDiff time.Duration
	now := time.Now()
	for peer, f := range r.leaderState.replState {
		diff := now.Sub(f.LastContact())
		if diff <= r.conf.LeaderLeaseTimeout {
			contacted++
			if diff > maxDiff {
				maxDiff = diff
			}
		} else {
			// Log at least once at high value, then debug. Otherwise it gets very verbose.
			if diff <= 3*r.conf.LeaderLeaseTimeout {
				r.logger.Printf("[WARN] raft: Failed to contact %v in %v", peer, diff)
			} else {
				r.logger.Printf("[DEBUG] raft: Failed to contact %v in %v", peer, diff)
			}
		}
		metrics.AddSample([]string{"raft", "leader", "lastContact"}, float32(diff/time.Millisecond))
	}

	// Verify we can contact a quorum
	quorum := r.quorumSize()
	if contacted < quorum {
		r.logger.Printf("[WARN] raft: Failed to contact quorum of nodes, stepping down")
		r.setState(Follower)
		metrics.IncrCounter([]string{"raft", "transition", "leader_lease_timeout"}, 1)
	}
	return maxDiff
}

// quorumSize is used to return the quorum size
func (r *Raft) quorumSize() int {
	return ((len(r.peers) + 1) / 2) + 1
}

// preparePeerChange checks if a LogAddPeer or LogRemovePeer should be performed,
// and properly formats the data field on the log before dispatching it.
func (r *Raft) preparePeerChange(l *logFuture) bool {
	// Check if this is a known peer
	p := l.log.peer
	knownPeer := PeerContained(r.peers, p) || r.localAddr == p

	// Ignore known peers on add
	if l.log.Type == LogAddPeer && knownPeer {
		l.respond(ErrKnownPeer)
		return false
	}

	// Ignore unknown peers on remove
	if l.log.Type == LogRemovePeer && !knownPeer {
		l.respond(ErrUnknownPeer)
		return false
	}

	// Construct the peer set
	var peerSet []string
	if l.log.Type == LogAddPeer {
		peerSet = append([]string{p, r.localAddr}, r.peers...)
	} else {
		peerSet = ExcludePeer(append([]string{r.localAddr}, r.peers...), p)
	}

	// Setup the log
	l.log.Data = encodePeers(peerSet, r.trans)
	return true
}

// dispatchLog is called to push a log to disk, mark it
// as inflight and begin replication of it.
func (r *Raft) dispatchLogs(applyLogs []*logFuture) {
	now := time.Now()
	defer metrics.MeasureSince([]string{"raft", "leader", "dispatchLog"}, now)

	term := r.getCurrentTerm()
	lastIndex := r.getLastIndex()
	logs := make([]*Log, len(applyLogs))

	for idx, applyLog := range applyLogs {
		applyLog.dispatch = now
		applyLog.log.Index = lastIndex + uint64(idx) + 1
		applyLog.log.Term = term
		applyLog.policy = newMajorityQuorum(len(r.peers) + 1)
		logs[idx] = &applyLog.log
	}

	// Write the log entry locally
	if err := r.logs.StoreLogs(logs); err != nil {
		r.logger.Printf("[ERR] raft: Failed to commit logs: %v", err)
		for _, applyLog := range applyLogs {
			applyLog.respond(err)
		}
		r.setState(Follower)
		return
	}

	// Add this to the inflight logs, commit
	r.leaderState.inflight.StartAll(applyLogs)

	// Update the last log since it's on disk now
	r.setLastLogIndex(lastIndex + uint64(len(applyLogs)))
	r.setLastLogTerm(term)

	// Notify the replicators of the new log
	for _, f := range r.leaderState.replState {
		asyncNotifyCh(f.triggerCh)
	}
}

// processLogs is used to process all the logs from the lastApplied
// up to the given index.
func (r *Raft) processLogs(index uint64, future *logFuture) {
	// Reject logs we've applied already
	lastApplied := r.getLastApplied()
	if index <= lastApplied {
		r.logger.Printf("[WARN] raft: Skipping application of old log: %d", index)
		return
	}

	// Apply all the preceding logs
	for idx := r.getLastApplied() + 1; idx <= index; idx++ {
		// Get the log, either from the future or from our log store
		if future != nil && future.log.Index == idx {
			r.processLog(&future.log, future, false)

		} else {
			l := new(Log)
			if err := r.logs.GetLog(idx, l); err != nil {
				r.logger.Printf("[ERR] raft: Failed to get log at %d: %v", idx, err)
				panic(err)
			}
			r.processLog(l, nil, false)
		}

		// Update the lastApplied index and term
		r.setLastApplied(idx)
	}
}

// processLog is invoked to process the application of a single committed log.
// Returns if this log entry would cause us to stepDown after it commits.
func (r *Raft) processLog(l *Log, future *logFuture, precommit bool) (stepDown bool) {
	switch l.Type {
	case LogBarrier:
		// Barrier is handled by the FSM
		fallthrough

	case LogCommand:
		// Forward to the fsm handler
		select {
		case r.fsmCommitCh <- commitTuple{l, future}:
		case <-r.shutdownCh:
			if future != nil {
				future.respond(ErrRaftShutdown)
			}
		}

		// Return so that the future is only responded to
		// by the FSM handler when the application is done
		return

	case LogAddPeer:
		fallthrough
	case LogRemovePeer:
		peers := decodePeers(l.Data, r.trans)
		r.logger.Printf("[DEBUG] raft: Node %v updated peer set (%v): %v", r.localAddr, l.Type, peers)

		// If the peer set does not include us, remove all other peers
		removeSelf := !PeerContained(peers, r.localAddr) && l.Type == LogRemovePeer
		if removeSelf {
			// Mark that this operation will cause us to step down as
			// leader. This prevents the future logs from being Applied
			// from this leader.
			stepDown = true

			// We only modify the peers after the commit, otherwise we
			// would be using a quorum size of 1 for the RemovePeer operation.
			// This is used with the stepDown guard to prevent any other logs.
			if !precommit {
				r.peers = nil
				r.peerStore.SetPeers([]string{r.localAddr})
			}
		} else {
			r.peers = ExcludePeer(peers, r.localAddr)
			r.peerStore.SetPeers(peers)
		}

		// Handle replication if we are the leader
		if r.getState() == Leader {
			for _, p := range r.peers {
				if _, ok := r.leaderState.replState[p]; !ok {
					r.logger.Printf("[INFO] raft: Added peer %v, starting replication", p)
					r.startReplication(p)
				}
			}
		}

		// Stop replication for old nodes
		if r.getState() == Leader && !precommit {
			var toDelete []string
			for _, repl := range r.leaderState.replState {
				if !PeerContained(r.peers, repl.peer) {
					r.logger.Printf("[INFO] raft: Removed peer %v, stopping replication (Index: %d)", repl.peer, l.Index)

					// Replicate up to this index and stop
					repl.stopCh <- l.Index
					close(repl.stopCh)
					toDelete = append(toDelete, repl.peer)
				}
			}
			for _, name := range toDelete {
				delete(r.leaderState.replState, name)
			}
		}

		// Handle removing ourself
		if removeSelf && !precommit {
			if r.conf.ShutdownOnRemove {
				r.logger.Printf("[INFO] raft: Removed ourself, shutting down")
				r.Shutdown()
			} else {
				r.logger.Printf("[INFO] raft: Removed ourself, transitioning to follower")
				r.setState(Follower)
			}
		}

	case LogNoop:
		// Ignore the no-op
	default:
		r.logger.Printf("[ERR] raft: Got unrecognized log type: %#v", l)
	}

	// Invoke the future if given
	if future != nil && !precommit {
		future.respond(nil)
	}
	return
}

// processRPC is called to handle an incoming RPC request.
func (r *Raft) processRPC(rpc RPC) {
	switch cmd := rpc.Command.(type) {
	case *AppendEntriesRequest:
		r.appendEntries(rpc, cmd)
	case *RequestVoteRequest:
		r.requestVote(rpc, cmd)
	case *InstallSnapshotRequest:
		r.installSnapshot(rpc, cmd)
	default:
		r.logger.Printf("[ERR] raft: Got unexpected command: %#v", rpc.Command)
		rpc.Respond(nil, fmt.Errorf("unexpected command"))
	}
}

// processHeartbeat is a special handler used just for heartbeat requests
// so that they can be fast-pathed if a transport supports it.
func (r *Raft) processHeartbeat(rpc RPC) {
	defer metrics.MeasureSince([]string{"raft", "rpc", "processHeartbeat"}, time.Now())

	// Check if we are shutdown, just ignore the RPC
	select {
	case <-r.shutdownCh:
		return
	default:
	}

	// Ensure we are only handling a heartbeat
	switch cmd := rpc.Command.(type) {
	case *AppendEntriesRequest:
		r.appendEntries(rpc, cmd)
	default:
		r.logger.Printf("[ERR] raft: Expected heartbeat, got command: %#v", rpc.Command)
		rpc.Respond(nil, fmt.Errorf("unexpected command"))
	}
}

// appendEntries is invoked when we get an append entries RPC call.
func (r *Raft) appendEntries(rpc RPC, a *AppendEntriesRequest) {
	defer metrics.MeasureSince([]string{"raft", "rpc", "appendEntries"}, time.Now())
	// Setup a response
	resp := &AppendEntriesResponse{
		Term:           r.getCurrentTerm(),
		LastLog:        r.getLastIndex(),
		Success:        false,
		NoRetryBackoff: false,
	}
	var rpcErr error
	defer func() {
		rpc.Respond(resp, rpcErr)
	}()

	// Ignore an older term
	if a.Term < r.getCurrentTerm() {
		return
	}

	// Increase the term if we see a newer one, also transition to follower
	// if we ever get an appendEntries call
	if a.Term > r.getCurrentTerm() || r.getState() != Follower {
		// Ensure transition to follower
		r.setState(Follower)
		r.setCurrentTerm(a.Term)
		resp.Term = a.Term
	}

	// Save the current leader
	r.setLeader(r.trans.DecodePeer(a.Leader))

	// Verify the last log entry
	if a.PrevLogEntry > 0 {
		lastIdx, lastTerm := r.getLastEntry()

		var prevLogTerm uint64
		if a.PrevLogEntry == lastIdx {
			prevLogTerm = lastTerm

		} else {
			var prevLog Log
			if err := r.logs.GetLog(a.PrevLogEntry, &prevLog); err != nil {
				r.logger.Printf("[WARN] raft: Failed to get previous log: %d %v (last: %d)",
					a.PrevLogEntry, err, lastIdx)
				resp.NoRetryBackoff = true
				return
			}
			prevLogTerm = prevLog.Term
		}

		if a.PrevLogTerm != prevLogTerm {
			r.logger.Printf("[WARN] raft: Previous log term mis-match: ours: %d remote: %d",
				prevLogTerm, a.PrevLogTerm)
			resp.NoRetryBackoff = true
			return
		}
	}

	// Process any new entries
	if n := len(a.Entries); n > 0 {
		start := time.Now()
		first := a.Entries[0]
		last := a.Entries[n-1]

		// Delete any conflicting entries
		lastLogIdx := r.getLastLogIndex()
		if first.Index <= lastLogIdx {
			r.logger.Printf("[WARN] raft: Clearing log suffix from %d to %d", first.Index, lastLogIdx)
			if err := r.logs.DeleteRange(first.Index, lastLogIdx); err != nil {
				r.logger.Printf("[ERR] raft: Failed to clear log suffix: %v", err)
				return
			}
		}

		// Append the entry
		if err := r.logs.StoreLogs(a.Entries); err != nil {
			r.logger.Printf("[ERR] raft: Failed to append to logs: %v", err)
			return
		}

		// Update the lastLog
		r.setLastLogIndex(last.Index)
		r.setLastLogTerm(last.Term)
		metrics.MeasureSince([]string{"raft", "rpc", "appendEntries", "storeLogs"}, start)
	}

	// Update the commit index
	if a.LeaderCommitIndex > 0 && a.LeaderCommitIndex > r.getCommitIndex() {
		start := time.Now()
		idx := min(a.LeaderCommitIndex, r.getLastIndex())
		r.setCommitIndex(idx)
		r.processLogs(idx, nil)
		metrics.MeasureSince([]string{"raft", "rpc", "appendEntries", "processLogs"}, start)
	}

	// Everything went well, set success
	resp.Success = true
	r.setLastContact()
	return
}

// requestVote is invoked when we get an request vote RPC call.
func (r *Raft) requestVote(rpc RPC, req *RequestVoteRequest) {
	defer metrics.MeasureSince([]string{"raft", "rpc", "requestVote"}, time.Now())
	// Setup a response
	resp := &RequestVoteResponse{
		Term:    r.getCurrentTerm(),
		Peers:   encodePeers(r.peers, r.trans),
		Granted: false,
	}
	var rpcErr error
	defer func() {
		rpc.Respond(resp, rpcErr)
	}()

	// Check if we have an existing leader [who's not the candidate]
	candidate := r.trans.DecodePeer(req.Candidate)
	if leader := r.Leader(); leader != "" && leader != candidate {
		r.logger.Printf("[WARN] raft: Rejecting vote request from %v since we have a leader: %v",
			candidate, leader)
		return
	}

	// Ignore an older term
	if req.Term < r.getCurrentTerm() {
		return
	}

	// Increase the term if we see a newer one
	if req.Term > r.getCurrentTerm() {
		// Ensure transition to follower
		r.setState(Follower)
		r.setCurrentTerm(req.Term)
		resp.Term = req.Term
	}

	// Check if we have voted yet
	lastVoteTerm, err := r.stable.GetUint64(keyLastVoteTerm)
	if err != nil && err.Error() != "not found" {
		r.logger.Printf("[ERR] raft: Failed to get last vote term: %v", err)
		return
	}
	lastVoteCandBytes, err := r.stable.Get(keyLastVoteCand)
	if err != nil && err.Error() != "not found" {
		r.logger.Printf("[ERR] raft: Failed to get last vote candidate: %v", err)
		return
	}

	// Check if we've voted in this election before
	if lastVoteTerm == req.Term && lastVoteCandBytes != nil {
		r.logger.Printf("[INFO] raft: Duplicate RequestVote for same term: %d", req.Term)
		if bytes.Compare(lastVoteCandBytes, req.Candidate) == 0 {
			r.logger.Printf("[WARN] raft: Duplicate RequestVote from candidate: %s", req.Candidate)
			resp.Granted = true
		}
		return
	}

	// Reject if their term is older
	lastIdx, lastTerm := r.getLastEntry()
	if lastTerm > req.LastLogTerm {
		r.logger.Printf("[WARN] raft: Rejecting vote request from %v since our last term is greater (%d, %d)",
			candidate, lastTerm, req.LastLogTerm)
		return
	}

	if lastIdx > req.LastLogIndex {
		r.logger.Printf("[WARN] raft: Rejecting vote request from %v since our last index is greater (%d, %d)",
			candidate, lastIdx, req.LastLogIndex)
		return
	}

	// Persist a vote for safety
	if err := r.persistVote(req.Term, req.Candidate); err != nil {
		r.logger.Printf("[ERR] raft: Failed to persist vote: %v", err)
		return
	}

	resp.Granted = true
	return
}

// installSnapshot is invoked when we get a InstallSnapshot RPC call.
// We must be in the follower state for this, since it means we are
// too far behind a leader for log replay.
func (r *Raft) installSnapshot(rpc RPC, req *InstallSnapshotRequest) {
	defer metrics.MeasureSince([]string{"raft", "rpc", "installSnapshot"}, time.Now())
	// Setup a response
	resp := &InstallSnapshotResponse{
		Term:    r.getCurrentTerm(),
		Success: false,
	}
	var rpcErr error
	defer func() {
		rpc.Respond(resp, rpcErr)
	}()

	// Ignore an older term
	if req.Term < r.getCurrentTerm() {
		return
	}

	// Increase the term if we see a newer one
	if req.Term > r.getCurrentTerm() {
		// Ensure transition to follower
		r.setState(Follower)
		r.setCurrentTerm(req.Term)
		resp.Term = req.Term
	}

	// Save the current leader
	r.setLeader(r.trans.DecodePeer(req.Leader))

	// Create a new snapshot
	sink, err := r.snapshots.Create(req.LastLogIndex, req.LastLogTerm, req.Peers)
	if err != nil {
		r.logger.Printf("[ERR] raft: Failed to create snapshot to install: %v", err)
		rpcErr = fmt.Errorf("failed to create snapshot: %v", err)
		return
	}

	// Spill the remote snapshot to disk
	n, err := io.Copy(sink, rpc.Reader)
	if err != nil {
		sink.Cancel()
		r.logger.Printf("[ERR] raft: Failed to copy snapshot: %v", err)
		rpcErr = err
		return
	}

	// Check that we received it all
	if n != req.Size {
		sink.Cancel()
		r.logger.Printf("[ERR] raft: Failed to receive whole snapshot: %d / %d", n, req.Size)
		rpcErr = fmt.Errorf("short read")
		return
	}

	// Finalize the snapshot
	if err := sink.Close(); err != nil {
		r.logger.Printf("[ERR] raft: Failed to finalize snapshot: %v", err)
		rpcErr = err
		return
	}
	r.logger.Printf("[INFO] raft: Copied %d bytes to local snapshot", n)

	// Restore snapshot
	future := &restoreFuture{ID: sink.ID()}
	future.init()
	select {
	case r.fsmRestoreCh <- future:
	case <-r.shutdownCh:
		future.respond(ErrRaftShutdown)
		return
	}

	// Wait for the restore to happen
	if err := future.Error(); err != nil {
		r.logger.Printf("[ERR] raft: Failed to restore snapshot: %v", err)
		rpcErr = err
		return
	}

	// Update the lastApplied so we don't replay old logs
	r.setLastApplied(req.LastLogIndex)

	// Update the last stable snapshot info
	r.setLastSnapshotIndex(req.LastLogIndex)
	r.setLastSnapshotTerm(req.LastLogTerm)

	// Restore the peer set
	peers := decodePeers(req.Peers, r.trans)
	r.peers = ExcludePeer(peers, r.localAddr)
	r.peerStore.SetPeers(peers)

	// Compact logs, continue even if this fails
	if err := r.compactLogs(req.LastLogIndex); err != nil {
		r.logger.Printf("[ERR] raft: Failed to compact logs: %v", err)
	}

	r.logger.Printf("[INFO] raft: Installed remote snapshot")
	resp.Success = true
	r.setLastContact()
	return
}

// setLastContact is used to set the last contact time to now
func (r *Raft) setLastContact() {
	r.lastContactLock.Lock()
	r.lastContact = time.Now()
	r.lastContactLock.Unlock()
}

type voteResult struct {
	RequestVoteResponse
	voter string
}

// electSelf is used to send a RequestVote RPC to all peers,
// and vote for ourself. This has the side affecting of incrementing
// the current term. The response channel returned is used to wait
// for all the responses (including a vote for ourself).
func (r *Raft) electSelf() <-chan *voteResult {
	// Create a response channel
	respCh := make(chan *voteResult, len(r.peers)+1)

	// Increment the term
	r.setCurrentTerm(r.getCurrentTerm() + 1)

	// Construct the request
	lastIdx, lastTerm := r.getLastEntry()
	req := &RequestVoteRequest{
		Term:         r.getCurrentTerm(),
		Candidate:    r.trans.EncodePeer(r.localAddr),
		LastLogIndex: lastIdx,
		LastLogTerm:  lastTerm,
	}

	// Construct a function to ask for a vote
	askPeer := func(peer string) {
		r.goFunc(func() {
			defer metrics.MeasureSince([]string{"raft", "candidate", "electSelf"}, time.Now())
			resp := &voteResult{voter: peer}
			err := r.trans.RequestVote(peer, req, &resp.RequestVoteResponse)
			if err != nil {
				r.logger.Printf("[ERR] raft: Failed to make RequestVote RPC to %v: %v", peer, err)
				resp.Term = req.Term
				resp.Granted = false
			}

			// If we are not a peer, we could have been removed but failed
			// to receive the log message. OR it could mean an improperly configured
			// cluster. Either way, we should warn
			if err == nil {
				peerSet := decodePeers(resp.Peers, r.trans)
				if !PeerContained(peerSet, r.localAddr) {
					r.logger.Printf("[WARN] raft: Remote peer %v does not have local node %v as a peer",
						peer, r.localAddr)
				}
			}

			respCh <- resp
		})
	}

	// For each peer, request a vote
	for _, peer := range r.peers {
		askPeer(peer)
	}

	// Persist a vote for ourselves
	if err := r.persistVote(req.Term, req.Candidate); err != nil {
		r.logger.Printf("[ERR] raft: Failed to persist vote : %v", err)
		return nil
	}

	// Include our own vote
	respCh <- &voteResult{
		RequestVoteResponse: RequestVoteResponse{
			Term:    req.Term,
			Granted: true,
		},
		voter: r.localAddr,
	}
	return respCh
}

// persistVote is used to persist our vote for safety.
func (r *Raft) persistVote(term uint64, candidate []byte) error {
	if err := r.stable.SetUint64(keyLastVoteTerm, term); err != nil {
		return err
	}
	if err := r.stable.Set(keyLastVoteCand, candidate); err != nil {
		return err
	}
	return nil
}

// setCurrentTerm is used to set the current term in a durable manner.
func (r *Raft) setCurrentTerm(t uint64) {
	// Persist to disk first
	if err := r.stable.SetUint64(keyCurrentTerm, t); err != nil {
		panic(fmt.Errorf("failed to save current term: %v", err))
	}
	r.raftState.setCurrentTerm(t)
}

// setState is used to update the current state. Any state
// transition causes the known leader to be cleared. This means
// that leader should be set only after updating the state.
func (r *Raft) setState(state RaftState) {
	r.setLeader("")
	r.raftState.setState(state)
}

// runSnapshots is a long running goroutine used to manage taking
// new snapshots of the FSM. It runs in parallel to the FSM and
// main goroutines, so that snapshots do not block normal operation.
func (r *Raft) runSnapshots() {
	for {
		select {
		case <-randomTimeout(r.conf.SnapshotInterval):
			// Check if we should snapshot
			if !r.shouldSnapshot() {
				continue
			}

			// Trigger a snapshot
			if err := r.takeSnapshot(); err != nil {
				r.logger.Printf("[ERR] raft: Failed to take snapshot: %v", err)
			}

		case future := <-r.snapshotCh:
			// User-triggered, run immediately
			err := r.takeSnapshot()
			if err != nil {
				r.logger.Printf("[ERR] raft: Failed to take snapshot: %v", err)
			}
			future.respond(err)

		case <-r.shutdownCh:
			return
		}
	}
}

// shouldSnapshot checks if we meet the conditions to take
// a new snapshot.
func (r *Raft) shouldSnapshot() bool {
	// Check the last snapshot index
	lastSnap := r.getLastSnapshotIndex()

	// Check the last log index
	lastIdx, err := r.logs.LastIndex()
	if err != nil {
		r.logger.Printf("[ERR] raft: Failed to get last log index: %v", err)
		return false
	}

	// Compare the delta to the threshold
	delta := lastIdx - lastSnap
	return delta >= r.conf.SnapshotThreshold
}

// takeSnapshot is used to take a new snapshot.
func (r *Raft) takeSnapshot() error {
	defer metrics.MeasureSince([]string{"raft", "snapshot", "takeSnapshot"}, time.Now())
	// Create a snapshot request
	req := &reqSnapshotFuture{}
	req.init()

	// Wait for dispatch or shutdown
	select {
	case r.fsmSnapshotCh <- req:
	case <-r.shutdownCh:
		return ErrRaftShutdown
	}

	// Wait until we get a response
	if err := req.Error(); err != nil {
		if err != ErrNothingNewToSnapshot {
			err = fmt.Errorf("failed to start snapshot: %v", err)
		}
		return err
	}
	defer req.snapshot.Release()

	// Log that we are starting the snapshot
	r.logger.Printf("[INFO] raft: Starting snapshot up to %d", req.index)

	// Encode the peerset
	peerSet := encodePeers(req.peers, r.trans)

	// Create a new snapshot
	start := time.Now()
	sink, err := r.snapshots.Create(req.index, req.term, peerSet)
	if err != nil {
		return fmt.Errorf("failed to create snapshot: %v", err)
	}
	metrics.MeasureSince([]string{"raft", "snapshot", "create"}, start)

	// Try to persist the snapshot
	start = time.Now()
	if err := req.snapshot.Persist(sink); err != nil {
		sink.Cancel()
		return fmt.Errorf("failed to persist snapshot: %v", err)
	}
	metrics.MeasureSince([]string{"raft", "snapshot", "persist"}, start)

	// Close and check for error
	if err := sink.Close(); err != nil {
		return fmt.Errorf("failed to close snapshot: %v", err)
	}

	// Update the last stable snapshot info
	r.setLastSnapshotIndex(req.index)
	r.setLastSnapshotTerm(req.term)

	// Compact the logs
	if err := r.compactLogs(req.index); err != nil {
		return err
	}

	// Log completion
	r.logger.Printf("[INFO] raft: Snapshot to %d complete", req.index)
	return nil
}

// compactLogs takes the last inclusive index of a snapshot
// and trims the logs that are no longer needed.
func (r *Raft) compactLogs(snapIdx uint64) error {
	defer metrics.MeasureSince([]string{"raft", "compactLogs"}, time.Now())
	// Determine log ranges to compact
	minLog, err := r.logs.FirstIndex()
	if err != nil {
		return fmt.Errorf("failed to get first log index: %v", err)
	}

	// Check if we have enough logs to truncate
	if r.getLastLogIndex() <= r.conf.TrailingLogs {
		return nil
	}

	// Truncate up to the end of the snapshot, or `TrailingLogs`
	// back from the head, which ever is further back. This ensures
	// at least `TrailingLogs` entries, but does not allow logs
	// after the snapshot to be removed.
	maxLog := min(snapIdx, r.getLastLogIndex()-r.conf.TrailingLogs)

	// Log this
	r.logger.Printf("[INFO] raft: Compacting logs from %d to %d", minLog, maxLog)

	// Compact the logs
	if err := r.logs.DeleteRange(minLog, maxLog); err != nil {
		return fmt.Errorf("log compaction failed: %v", err)
	}
	return nil
}

// restoreSnapshot attempts to restore the latest snapshots, and fails
// if none of them can be restored. This is called at initialization time,
// and is completely unsafe to call at any other time.
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
		r.setLastSnapshotIndex(snapshot.Index)
		r.setLastSnapshotTerm(snapshot.Term)

		// Success!
		return nil
	}

	// If we had snapshots and failed to load them, its an error
	if len(snapshots) > 0 {
		return fmt.Errorf("failed to load any existing snapshots")
	}
	return nil
}
