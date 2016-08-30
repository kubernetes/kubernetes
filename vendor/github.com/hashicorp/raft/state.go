package raft

import (
	"sync/atomic"
)

// RaftState captures the state of a Raft node: Follower, Candidate, Leader,
// or Shutdown.
type RaftState uint32

const (
	// Follower is the initial state of a Raft node.
	Follower RaftState = iota

	// Candidate is one of the valid states of a Raft node.
	Candidate

	// Leader is one of the valid states of a Raft node.
	Leader

	// Shutdown is the terminal state of a Raft node.
	Shutdown
)

func (s RaftState) String() string {
	switch s {
	case Follower:
		return "Follower"
	case Candidate:
		return "Candidate"
	case Leader:
		return "Leader"
	case Shutdown:
		return "Shutdown"
	default:
		return "Unknown"
	}
}

// raftState is used to maintain various state variables
// and provides an interface to set/get the variables in a
// thread safe manner.
type raftState struct {
	// The current term, cache of StableStore
	currentTerm uint64

	// Cache the latest log from LogStore
	LastLogIndex uint64
	LastLogTerm  uint64

	// Highest committed log entry
	commitIndex uint64

	// Last applied log to the FSM
	lastApplied uint64

	// Cache the latest snapshot index/term
	lastSnapshotIndex uint64
	lastSnapshotTerm  uint64

	// Tracks the number of live routines
	runningRoutines int32

	// The current state
	state RaftState
}

func (r *raftState) getState() RaftState {
	stateAddr := (*uint32)(&r.state)
	return RaftState(atomic.LoadUint32(stateAddr))
}

func (r *raftState) setState(s RaftState) {
	stateAddr := (*uint32)(&r.state)
	atomic.StoreUint32(stateAddr, uint32(s))
}

func (r *raftState) getCurrentTerm() uint64 {
	return atomic.LoadUint64(&r.currentTerm)
}

func (r *raftState) setCurrentTerm(term uint64) {
	atomic.StoreUint64(&r.currentTerm, term)
}

func (r *raftState) getLastLogIndex() uint64 {
	return atomic.LoadUint64(&r.LastLogIndex)
}

func (r *raftState) setLastLogIndex(term uint64) {
	atomic.StoreUint64(&r.LastLogIndex, term)
}

func (r *raftState) getLastLogTerm() uint64 {
	return atomic.LoadUint64(&r.LastLogTerm)
}

func (r *raftState) setLastLogTerm(term uint64) {
	atomic.StoreUint64(&r.LastLogTerm, term)
}

func (r *raftState) getCommitIndex() uint64 {
	return atomic.LoadUint64(&r.commitIndex)
}

func (r *raftState) setCommitIndex(term uint64) {
	atomic.StoreUint64(&r.commitIndex, term)
}

func (r *raftState) getLastApplied() uint64 {
	return atomic.LoadUint64(&r.lastApplied)
}

func (r *raftState) setLastApplied(term uint64) {
	atomic.StoreUint64(&r.lastApplied, term)
}

func (r *raftState) getLastSnapshotIndex() uint64 {
	return atomic.LoadUint64(&r.lastSnapshotIndex)
}

func (r *raftState) setLastSnapshotIndex(term uint64) {
	atomic.StoreUint64(&r.lastSnapshotIndex, term)
}

func (r *raftState) getLastSnapshotTerm() uint64 {
	return atomic.LoadUint64(&r.lastSnapshotTerm)
}

func (r *raftState) setLastSnapshotTerm(term uint64) {
	atomic.StoreUint64(&r.lastSnapshotTerm, term)
}

func (r *raftState) incrRoutines() {
	atomic.AddInt32(&r.runningRoutines, 1)
}

func (r *raftState) decrRoutines() {
	atomic.AddInt32(&r.runningRoutines, -1)
}

func (r *raftState) getRoutines() int32 {
	return atomic.LoadInt32(&r.runningRoutines)
}

// Start a goroutine and properly handle the race between a routine
// starting and incrementing, and exiting and decrementing.
func (r *raftState) goFunc(f func()) {
	r.incrRoutines()
	go func() {
		defer r.decrRoutines()
		f()
	}()
}

// getLastIndex returns the last index in stable storage.
// Either from the last log or from the last snapshot.
func (r *raftState) getLastIndex() uint64 {
	return max(r.getLastLogIndex(), r.getLastSnapshotIndex())
}

// getLastEntry returns the last index and term in stable storage.
// Either from the last log or from the last snapshot.
func (r *raftState) getLastEntry() (uint64, uint64) {
	if r.getLastLogIndex() >= r.getLastSnapshotIndex() {
		return r.getLastLogIndex(), r.getLastLogTerm()
	}
	return r.getLastSnapshotIndex(), r.getLastSnapshotTerm()
}
