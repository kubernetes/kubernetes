package raft

import (
	"bytes"
	"container/list"
	"fmt"
	"io"
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
)

// getRPCHeader returns an initialized RPCHeader struct for the given
// Raft instance. This structure is sent along with RPC requests and
// responses.
func (r *Raft) getRPCHeader() RPCHeader {
	return RPCHeader{
		ProtocolVersion: r.conf.ProtocolVersion,
	}
}

// checkRPCHeader houses logic about whether this instance of Raft can process
// the given RPC message.
func (r *Raft) checkRPCHeader(rpc RPC) error {
	// Get the header off the RPC message.
	wh, ok := rpc.Command.(WithRPCHeader)
	if !ok {
		return fmt.Errorf("RPC does not have a header")
	}
	header := wh.GetRPCHeader()

	// First check is to just make sure the code can understand the
	// protocol at all.
	if header.ProtocolVersion < ProtocolVersionMin ||
		header.ProtocolVersion > ProtocolVersionMax {
		return ErrUnsupportedProtocol
	}

	// Second check is whether we should support this message, given the
	// current protocol we are configured to run. This will drop support
	// for protocol version 0 starting at protocol version 2, which is
	// currently what we want, and in general support one version back. We
	// may need to revisit this policy depending on how future protocol
	// changes evolve.
	if header.ProtocolVersion < r.conf.ProtocolVersion-1 {
		return ErrUnsupportedProtocol
	}

	return nil
}

// getSnapshotVersion returns the snapshot version that should be used when
// creating snapshots, given the protocol version in use.
func getSnapshotVersion(protocolVersion ProtocolVersion) SnapshotVersion {
	// Right now we only have two versions and they are backwards compatible
	// so we don't need to look at the protocol version.
	return 1
}

// commitTuple is used to send an index that was committed,
// with an optional associated future that should be invoked.
type commitTuple struct {
	log    *Log
	future *logFuture
}

// leaderState is state that is used while we are a leader.
type leaderState struct {
	commitCh   chan struct{}
	commitment *commitment
	inflight   *list.List // list of logFuture in log index order
	replState  map[ServerID]*followerReplication
	notify     map[*verifyFuture]struct{}
	stepDown   chan struct{}
}

// setLeader is used to modify the current leader of the cluster
func (r *Raft) setLeader(leader ServerAddress) {
	r.leaderLock.Lock()
	r.leader = leader
	r.leaderLock.Unlock()
}

// requestConfigChange is a helper for the above functions that make
// configuration change requests. 'req' describes the change. For timeout,
// see AddVoter.
func (r *Raft) requestConfigChange(req configurationChangeRequest, timeout time.Duration) IndexFuture {
	var timer <-chan time.Time
	if timeout > 0 {
		timer = time.After(timeout)
	}
	future := &configurationChangeFuture{
		req: req,
	}
	future.init()
	select {
	case <-timer:
		return errorFuture{ErrEnqueueTimeout}
	case r.configurationChangeCh <- future:
		return future
	case <-r.shutdownCh:
		return errorFuture{ErrRaftShutdown}
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
	r.logger.Printf("[INFO] raft: %v entering Follower state (Leader: %q)", r, r.Leader())
	metrics.IncrCounter([]string{"raft", "state", "follower"}, 1)
	heartbeatTimer := randomTimeout(r.conf.HeartbeatTimeout)
	for {
		select {
		case rpc := <-r.rpcCh:
			r.processRPC(rpc)

		case c := <-r.configurationChangeCh:
			// Reject any operations since we are not the leader
			c.respond(ErrNotLeader)

		case a := <-r.applyCh:
			// Reject any operations since we are not the leader
			a.respond(ErrNotLeader)

		case v := <-r.verifyCh:
			// Reject any operations since we are not the leader
			v.respond(ErrNotLeader)

		case c := <-r.configurationsCh:
			c.configurations = r.configurations.Clone()
			c.respond(nil)

		case b := <-r.bootstrapCh:
			b.respond(r.liveBootstrap(b.configuration))

		case <-heartbeatTimer:
			// Restart the heartbeat timer
			heartbeatTimer = randomTimeout(r.conf.HeartbeatTimeout)

			// Check if we have had a successful contact
			lastContact := r.LastContact()
			if time.Now().Sub(lastContact) < r.conf.HeartbeatTimeout {
				continue
			}

			// Heartbeat failed! Transition to the candidate state
			lastLeader := r.Leader()
			r.setLeader("")

			if r.configurations.latestIndex == 0 {
				if !didWarn {
					r.logger.Printf("[WARN] raft: no known peers, aborting election")
					didWarn = true
				}
			} else if r.configurations.latestIndex == r.configurations.committedIndex &&
				!hasVote(r.configurations.latest, r.localID) {
				if !didWarn {
					r.logger.Printf("[WARN] raft: not part of stable configuration, aborting election")
					didWarn = true
				}
			} else {
				r.logger.Printf(`[WARN] raft: Heartbeat timeout from %q reached, starting election`, lastLeader)
				metrics.IncrCounter([]string{"raft", "transition", "heartbeat_timeout"}, 1)
				r.setState(Candidate)
				return
			}

		case <-r.shutdownCh:
			return
		}
	}
}

// liveBootstrap attempts to seed an initial configuration for the cluster. See
// the Raft object's member BootstrapCluster for more details. This must only be
// called on the main thread, and only makes sense in the follower state.
func (r *Raft) liveBootstrap(configuration Configuration) error {
	// Use the pre-init API to make the static updates.
	err := BootstrapCluster(&r.conf, r.logs, r.stable, r.snapshots,
		r.trans, configuration)
	if err != nil {
		return err
	}

	// Make the configuration live.
	var entry Log
	if err := r.logs.GetLog(1, &entry); err != nil {
		panic(err)
	}
	r.setCurrentTerm(1)
	r.setLastLog(entry.Index, entry.Term)
	r.processConfigurationLogEntry(&entry)
	return nil
}

// runCandidate runs the FSM for a candidate.
func (r *Raft) runCandidate() {
	r.logger.Printf("[INFO] raft: %v entering Candidate state in term %v",
		r, r.getCurrentTerm()+1)
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
				r.logger.Printf("[DEBUG] raft: Vote granted from %s in term %v. Tally: %d",
					vote.voterID, vote.Term, grantedVotes)
			}

			// Check if we've become the leader
			if grantedVotes >= votesNeeded {
				r.logger.Printf("[INFO] raft: Election won. Tally: %d", grantedVotes)
				r.setState(Leader)
				r.setLeader(r.localAddr)
				return
			}

		case c := <-r.configurationChangeCh:
			// Reject any operations since we are not the leader
			c.respond(ErrNotLeader)

		case a := <-r.applyCh:
			// Reject any operations since we are not the leader
			a.respond(ErrNotLeader)

		case v := <-r.verifyCh:
			// Reject any operations since we are not the leader
			v.respond(ErrNotLeader)

		case c := <-r.configurationsCh:
			c.configurations = r.configurations.Clone()
			c.respond(nil)

		case b := <-r.bootstrapCh:
			b.respond(ErrCantBootstrap)

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
	r.leaderState.commitment = newCommitment(r.leaderState.commitCh,
		r.configurations.latest,
		r.getLastIndex()+1 /* first index that may be committed in this term */)
	r.leaderState.inflight = list.New()
	r.leaderState.replState = make(map[ServerID]*followerReplication)
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

		// Respond to all inflight operations
		for e := r.leaderState.inflight.Front(); e != nil; e = e.Next() {
			e.Value.(*logFuture).respond(ErrLeadershipLost)
		}

		// Respond to any pending verify requests
		for future := range r.leaderState.notify {
			future.respond(ErrLeadershipLost)
		}

		// Clear all the state
		r.leaderState.commitCh = nil
		r.leaderState.commitment = nil
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
	r.startStopReplication()

	// Dispatch a no-op log entry first. This gets this leader up to the latest
	// possible commit index, even in the absence of client commands. This used
	// to append a configuration entry instead of a noop. However, that permits
	// an unbounded number of uncommitted configurations in the log. We now
	// maintain that there exists at most one uncommitted configuration entry in
	// any log, so we have to do proper no-ops here.
	noop := &logFuture{
		log: Log{
			Type: LogNoop,
		},
	}
	r.dispatchLogs([]*logFuture{noop})

	// Sit in the leader loop until we step down
	r.leaderLoop()
}

// startStopReplication will set up state and start asynchronous replication to
// new peers, and stop replication to removed peers. Before removing a peer,
// it'll instruct the replication routines to try to replicate to the current
// index. This must only be called from the main thread.
func (r *Raft) startStopReplication() {
	inConfig := make(map[ServerID]bool, len(r.configurations.latest.Servers))
	lastIdx := r.getLastIndex()

	// Start replication goroutines that need starting
	for _, server := range r.configurations.latest.Servers {
		if server.ID == r.localID {
			continue
		}
		inConfig[server.ID] = true
		if _, ok := r.leaderState.replState[server.ID]; !ok {
			r.logger.Printf("[INFO] raft: Added peer %v, starting replication", server.ID)
			s := &followerReplication{
				peer:        server,
				commitment:  r.leaderState.commitment,
				stopCh:      make(chan uint64, 1),
				triggerCh:   make(chan struct{}, 1),
				currentTerm: r.getCurrentTerm(),
				nextIndex:   lastIdx + 1,
				lastContact: time.Now(),
				notifyCh:    make(chan struct{}, 1),
				stepDown:    r.leaderState.stepDown,
			}
			r.leaderState.replState[server.ID] = s
			r.goFunc(func() { r.replicate(s) })
			asyncNotifyCh(s.triggerCh)
		}
	}

	// Stop replication goroutines that need stopping
	for serverID, repl := range r.leaderState.replState {
		if inConfig[serverID] {
			continue
		}
		// Replicate up to lastIdx and stop
		r.logger.Printf("[INFO] raft: Removed peer %v, stopping replication after %v", serverID, lastIdx)
		repl.stopCh <- lastIdx
		close(repl.stopCh)
		delete(r.leaderState.replState, serverID)
	}
}

// configurationChangeChIfStable returns r.configurationChangeCh if it's safe
// to process requests from it, or nil otherwise. This must only be called
// from the main thread.
//
// Note that if the conditions here were to change outside of leaderLoop to take
// this from nil to non-nil, we would need leaderLoop to be kicked.
func (r *Raft) configurationChangeChIfStable() chan *configurationChangeFuture {
	// Have to wait until:
	// 1. The latest configuration is committed, and
	// 2. This leader has committed some entry (the noop) in this term
	//    https://groups.google.com/forum/#!msg/raft-dev/t4xj6dJTP6E/d2D9LrWRza8J
	if r.configurations.latestIndex == r.configurations.committedIndex &&
		r.getCommitIndex() >= r.leaderState.commitment.startIndex {
		return r.configurationChangeCh
	}
	return nil
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
			// Process the newly committed entries
			oldCommitIndex := r.getCommitIndex()
			commitIndex := r.leaderState.commitment.getCommitIndex()
			r.setCommitIndex(commitIndex)

			if r.configurations.latestIndex > oldCommitIndex &&
				r.configurations.latestIndex <= commitIndex {
				r.configurations.committed = r.configurations.latest
				r.configurations.committedIndex = r.configurations.latestIndex
				if !hasVote(r.configurations.committed, r.localID) {
					stepDown = true
				}
			}

			for {
				e := r.leaderState.inflight.Front()
				if e == nil {
					break
				}
				commitLog := e.Value.(*logFuture)
				idx := commitLog.log.Index
				if idx > commitIndex {
					break
				}
				// Measure the commit time
				metrics.MeasureSince([]string{"raft", "commitTime"}, commitLog.dispatch)
				r.processLogs(idx, commitLog)
				r.leaderState.inflight.Remove(e)
			}

			if stepDown {
				if r.conf.ShutdownOnRemove {
					r.logger.Printf("[INFO] raft: Removed ourself, shutting down")
					r.Shutdown()
				} else {
					r.logger.Printf("[INFO] raft: Removed ourself, transitioning to follower")
					r.setState(Follower)
				}
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

		case c := <-r.configurationsCh:
			c.configurations = r.configurations.Clone()
			c.respond(nil)

		case future := <-r.configurationChangeChIfStable():
			r.appendConfigurationEntry(future)

		case b := <-r.bootstrapCh:
			b.respond(ErrCantBootstrap)

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

			// Dispatch the logs
			if stepDown {
				// we're in the process of stepping down as leader, don't process anything new
				for i := range ready {
					ready[i].respond(ErrNotLeader)
				}
			} else {
				r.dispatchLogs(ready)
			}

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
// contact. This must only be called from the main thread.
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

// quorumSize is used to return the quorum size. This must only be called on
// the main thread.
// TODO: revisit usage
func (r *Raft) quorumSize() int {
	voters := 0
	for _, server := range r.configurations.latest.Servers {
		if server.Suffrage == Voter {
			voters++
		}
	}
	return voters/2 + 1
}

// appendConfigurationEntry changes the configuration and adds a new
// configuration entry to the log. This must only be called from the
// main thread.
func (r *Raft) appendConfigurationEntry(future *configurationChangeFuture) {
	configuration, err := nextConfiguration(r.configurations.latest, r.configurations.latestIndex, future.req)
	if err != nil {
		future.respond(err)
		return
	}

	r.logger.Printf("[INFO] raft: Updating configuration with %s (%v, %v) to %+v",
		future.req.command, future.req.serverID, future.req.serverAddress, configuration.Servers)

	// In pre-ID compatibility mode we translate all configuration changes
	// in to an old remove peer message, which can handle all supported
	// cases for peer changes in the pre-ID world (adding and removing
	// voters). Both add peer and remove peer log entries are handled
	// similarly on old Raft servers, but remove peer does extra checks to
	// see if a leader needs to step down. Since they both assert the full
	// configuration, then we can safely call remove peer for everything.
	if r.protocolVersion < 2 {
		future.log = Log{
			Type: LogRemovePeerDeprecated,
			Data: encodePeers(configuration, r.trans),
		}
	} else {
		future.log = Log{
			Type: LogConfiguration,
			Data: encodeConfiguration(configuration),
		}
	}

	r.dispatchLogs([]*logFuture{&future.logFuture})
	index := future.Index()
	r.configurations.latest = configuration
	r.configurations.latestIndex = index
	r.leaderState.commitment.setConfiguration(configuration)
	r.startStopReplication()
}

// dispatchLog is called on the leader to push a log to disk, mark it
// as inflight and begin replication of it.
func (r *Raft) dispatchLogs(applyLogs []*logFuture) {
	now := time.Now()
	defer metrics.MeasureSince([]string{"raft", "leader", "dispatchLog"}, now)

	term := r.getCurrentTerm()
	lastIndex := r.getLastIndex()
	logs := make([]*Log, len(applyLogs))

	for idx, applyLog := range applyLogs {
		applyLog.dispatch = now
		lastIndex++
		applyLog.log.Index = lastIndex
		applyLog.log.Term = term
		logs[idx] = &applyLog.log
		r.leaderState.inflight.PushBack(applyLog)
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
	r.leaderState.commitment.match(r.localID, lastIndex)

	// Update the last log since it's on disk now
	r.setLastLog(lastIndex, term)

	// Notify the replicators of the new log
	for _, f := range r.leaderState.replState {
		asyncNotifyCh(f.triggerCh)
	}
}

// processLogs is used to apply all the committed entires that haven't been
// applied up to the given index limit.
// This can be called from both leaders and followers.
// Followers call this from AppendEntires, for n entires at a time, and always
// pass future=nil.
// Leaders call this once per inflight when entries are committed. They pass
// the future from inflights.
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
			r.processLog(&future.log, future)

		} else {
			l := new(Log)
			if err := r.logs.GetLog(idx, l); err != nil {
				r.logger.Printf("[ERR] raft: Failed to get log at %d: %v", idx, err)
				panic(err)
			}
			r.processLog(l, nil)
		}

		// Update the lastApplied index and term
		r.setLastApplied(idx)
	}
}

// processLog is invoked to process the application of a single committed log entry.
func (r *Raft) processLog(l *Log, future *logFuture) {
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

	case LogConfiguration:
	case LogAddPeerDeprecated:
	case LogRemovePeerDeprecated:
	case LogNoop:
		// Ignore the no-op

	default:
		panic(fmt.Errorf("unrecognized log type: %#v", l))
	}

	// Invoke the future if given
	if future != nil {
		future.respond(nil)
	}
}

// processRPC is called to handle an incoming RPC request. This must only be
// called from the main thread.
func (r *Raft) processRPC(rpc RPC) {
	if err := r.checkRPCHeader(rpc); err != nil {
		rpc.Respond(nil, err)
		return
	}

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
// so that they can be fast-pathed if a transport supports it. This must only
// be called from the main thread.
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

// appendEntries is invoked when we get an append entries RPC call. This must
// only be called from the main thread.
func (r *Raft) appendEntries(rpc RPC, a *AppendEntriesRequest) {
	defer metrics.MeasureSince([]string{"raft", "rpc", "appendEntries"}, time.Now())
	// Setup a response
	resp := &AppendEntriesResponse{
		RPCHeader:      r.getRPCHeader(),
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
	r.setLeader(ServerAddress(r.trans.DecodePeer(a.Leader)))

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
	if len(a.Entries) > 0 {
		start := time.Now()

		// Delete any conflicting entries, skip any duplicates
		lastLogIdx, _ := r.getLastLog()
		var newEntries []*Log
		for i, entry := range a.Entries {
			if entry.Index > lastLogIdx {
				newEntries = a.Entries[i:]
				break
			}
			var storeEntry Log
			if err := r.logs.GetLog(entry.Index, &storeEntry); err != nil {
				r.logger.Printf("[WARN] raft: Failed to get log entry %d: %v",
					entry.Index, err)
				return
			}
			if entry.Term != storeEntry.Term {
				r.logger.Printf("[WARN] raft: Clearing log suffix from %d to %d", entry.Index, lastLogIdx)
				if err := r.logs.DeleteRange(entry.Index, lastLogIdx); err != nil {
					r.logger.Printf("[ERR] raft: Failed to clear log suffix: %v", err)
					return
				}
				if entry.Index <= r.configurations.latestIndex {
					r.configurations.latest = r.configurations.committed
					r.configurations.latestIndex = r.configurations.committedIndex
				}
				newEntries = a.Entries[i:]
				break
			}
		}

		if n := len(newEntries); n > 0 {
			// Append the new entries
			if err := r.logs.StoreLogs(newEntries); err != nil {
				r.logger.Printf("[ERR] raft: Failed to append to logs: %v", err)
				// TODO: leaving r.getLastLog() in the wrong
				// state if there was a truncation above
				return
			}

			// Handle any new configuration changes
			for _, newEntry := range newEntries {
				r.processConfigurationLogEntry(newEntry)
			}

			// Update the lastLog
			last := newEntries[n-1]
			r.setLastLog(last.Index, last.Term)
		}

		metrics.MeasureSince([]string{"raft", "rpc", "appendEntries", "storeLogs"}, start)
	}

	// Update the commit index
	if a.LeaderCommitIndex > 0 && a.LeaderCommitIndex > r.getCommitIndex() {
		start := time.Now()
		idx := min(a.LeaderCommitIndex, r.getLastIndex())
		r.setCommitIndex(idx)
		if r.configurations.latestIndex <= idx {
			r.configurations.committed = r.configurations.latest
			r.configurations.committedIndex = r.configurations.latestIndex
		}
		r.processLogs(idx, nil)
		metrics.MeasureSince([]string{"raft", "rpc", "appendEntries", "processLogs"}, start)
	}

	// Everything went well, set success
	resp.Success = true
	r.setLastContact()
	return
}

// processConfigurationLogEntry takes a log entry and updates the latest
// configuration if the entry results in a new configuration. This must only be
// called from the main thread, or from NewRaft() before any threads have begun.
func (r *Raft) processConfigurationLogEntry(entry *Log) {
	if entry.Type == LogConfiguration {
		r.configurations.committed = r.configurations.latest
		r.configurations.committedIndex = r.configurations.latestIndex
		r.configurations.latest = decodeConfiguration(entry.Data)
		r.configurations.latestIndex = entry.Index
	} else if entry.Type == LogAddPeerDeprecated || entry.Type == LogRemovePeerDeprecated {
		r.configurations.committed = r.configurations.latest
		r.configurations.committedIndex = r.configurations.latestIndex
		r.configurations.latest = decodePeers(entry.Data, r.trans)
		r.configurations.latestIndex = entry.Index
	}
}

// requestVote is invoked when we get an request vote RPC call.
func (r *Raft) requestVote(rpc RPC, req *RequestVoteRequest) {
	defer metrics.MeasureSince([]string{"raft", "rpc", "requestVote"}, time.Now())
	r.observe(*req)

	// Setup a response
	resp := &RequestVoteResponse{
		RPCHeader: r.getRPCHeader(),
		Term:      r.getCurrentTerm(),
		Granted:   false,
	}
	var rpcErr error
	defer func() {
		rpc.Respond(resp, rpcErr)
	}()

	// Version 0 servers will panic unless the peers is present. It's only
	// used on them to produce a warning message.
	if r.protocolVersion < 2 {
		resp.Peers = encodePeers(r.configurations.latest, r.trans)
	}

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

	if lastTerm == req.LastLogTerm && lastIdx > req.LastLogIndex {
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
	r.setLastContact()
	return
}

// installSnapshot is invoked when we get a InstallSnapshot RPC call.
// We must be in the follower state for this, since it means we are
// too far behind a leader for log replay. This must only be called
// from the main thread.
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

	// Sanity check the version
	if req.SnapshotVersion < SnapshotVersionMin ||
		req.SnapshotVersion > SnapshotVersionMax {
		rpcErr = fmt.Errorf("unsupported snapshot version %d", req.SnapshotVersion)
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

	// Save the current leader
	r.setLeader(ServerAddress(r.trans.DecodePeer(req.Leader)))

	// Create a new snapshot
	var reqConfiguration Configuration
	var reqConfigurationIndex uint64
	if req.SnapshotVersion > 0 {
		reqConfiguration = decodeConfiguration(req.Configuration)
		reqConfigurationIndex = req.ConfigurationIndex
	} else {
		reqConfiguration = decodePeers(req.Peers, r.trans)
		reqConfigurationIndex = req.LastLogIndex
	}
	version := getSnapshotVersion(r.protocolVersion)
	sink, err := r.snapshots.Create(version, req.LastLogIndex, req.LastLogTerm,
		reqConfiguration, reqConfigurationIndex, r.trans)
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
	r.setLastSnapshot(req.LastLogIndex, req.LastLogTerm)

	// Restore the peer set
	r.configurations.latest = reqConfiguration
	r.configurations.latestIndex = reqConfigurationIndex
	r.configurations.committed = reqConfiguration
	r.configurations.committedIndex = reqConfigurationIndex

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
	voterID ServerID
}

// electSelf is used to send a RequestVote RPC to all peers, and vote for
// ourself. This has the side affecting of incrementing the current term. The
// response channel returned is used to wait for all the responses (including a
// vote for ourself). This must only be called from the main thread.
func (r *Raft) electSelf() <-chan *voteResult {
	// Create a response channel
	respCh := make(chan *voteResult, len(r.configurations.latest.Servers))

	// Increment the term
	r.setCurrentTerm(r.getCurrentTerm() + 1)

	// Construct the request
	lastIdx, lastTerm := r.getLastEntry()
	req := &RequestVoteRequest{
		RPCHeader:    r.getRPCHeader(),
		Term:         r.getCurrentTerm(),
		Candidate:    r.trans.EncodePeer(r.localAddr),
		LastLogIndex: lastIdx,
		LastLogTerm:  lastTerm,
	}

	// Construct a function to ask for a vote
	askPeer := func(peer Server) {
		r.goFunc(func() {
			defer metrics.MeasureSince([]string{"raft", "candidate", "electSelf"}, time.Now())
			resp := &voteResult{voterID: peer.ID}
			err := r.trans.RequestVote(peer.Address, req, &resp.RequestVoteResponse)
			if err != nil {
				r.logger.Printf("[ERR] raft: Failed to make RequestVote RPC to %v: %v", peer, err)
				resp.Term = req.Term
				resp.Granted = false
			}
			respCh <- resp
		})
	}

	// For each peer, request a vote
	for _, server := range r.configurations.latest.Servers {
		if server.Suffrage == Voter {
			if server.ID == r.localID {
				// Persist a vote for ourselves
				if err := r.persistVote(req.Term, req.Candidate); err != nil {
					r.logger.Printf("[ERR] raft: Failed to persist vote : %v", err)
					return nil
				}
				// Include our own vote
				respCh <- &voteResult{
					RequestVoteResponse: RequestVoteResponse{
						RPCHeader: r.getRPCHeader(),
						Term:      req.Term,
						Granted:   true,
					},
					voterID: r.localID,
				}
			} else {
				askPeer(server)
			}
		}
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
	oldState := r.raftState.getState()
	r.raftState.setState(state)
	if oldState != state {
		r.observe(state)
	}
}
