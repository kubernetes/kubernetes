package raft

import (
	"sync"
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

	// Highest committed log entry
	commitIndex uint64

	// Last applied log to the FSM
	lastApplied uint64

	// protects 4 next fields
	lastLock sync.Mutex

	// Cache the latest snapshot index/term
	lastSnapshotIndex uint64
	lastSnapshotTerm  uint64

	// Cache the latest log from LogStore
	lastLogIndex uint64
	lastLogTerm  uint64

	// Tracks running goroutines
	routinesGroup sync.WaitGroup

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

func (r *raftState) getLastLog() (index, term uint64) {
	r.lastLock.Lock()
	index = r.lastLogIndex
	term = r.lastLogTerm
	r.lastLock.Unlock()
	return
}

func (r *raftState) setLastLog(index, term uint64) {
	r.lastLock.Lock()
	r.lastLogIndex = index
	r.lastLogTerm = term
	r.lastLock.Unlock()
}

func (r *raftState) getLastSnapshot() (index, term uint64) {
	r.lastLock.Lock()
	index = r.lastSnapshotIndex
	term = r.lastSnapshotTerm
	r.lastLock.Unlock()
	return
}

func (r *raftState) setLastSnapshot(index, term uint64) {
	r.lastLock.Lock()
	r.lastSnapshotIndex = index
	r.lastSnapshotTerm = term
	r.lastLock.Unlock()
}

func (r *raftState) getCommitIndex() uint64 {
	return atomic.LoadUint64(&r.commitIndex)
}

func (r *raftState) setCommitIndex(index uint64) {
	atomic.StoreUint64(&r.commitIndex, index)
}

func (r *raftState) getLastApplied() uint64 {
	return atomic.LoadUint64(&r.lastApplied)
}

func (r *raftState) setLastApplied(index uint64) {
	atomic.StoreUint64(&r.lastApplied, index)
}

// Start a goroutine and properly handle the race between a routine
// starting and incrementing, and exiting and decrementing.
func (r *raftState) goFunc(f func()) {
	r.routinesGroup.Add(1)
	go func() {
		defer r.routinesGroup.Done()
		f()
	}()
}

func (r *raftState) waitShutdown() {
	r.routinesGroup.Wait()
}

// getLastIndex returns the last index in stable storage.
// Either from the last log or from the last snapshot.
func (r *raftState) getLastIndex() uint64 {
	r.lastLock.Lock()
	defer r.lastLock.Unlock()
	return max(r.lastLogIndex, r.lastSnapshotIndex)
}

// getLastEntry returns the last index and term in stable storage.
// Either from the last log or from the last snapshot.
func (r *raftState) getLastEntry() (uint64, uint64) {
	r.lastLock.Lock()
	defer r.lastLock.Unlock()
	if r.lastLogIndex >= r.lastSnapshotIndex {
		return r.lastLogIndex, r.lastLogTerm
	}
	return r.lastSnapshotIndex, r.lastSnapshotTerm
}
