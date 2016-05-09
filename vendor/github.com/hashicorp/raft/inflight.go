package raft

import (
	"container/list"
	"sync"
)

// QuorumPolicy allows individual logFutures to have different
// commitment rules while still using the inflight mechanism.
type quorumPolicy interface {
	// Checks if a commit from a given peer is enough to
	// satisfy the commitment rules
	Commit() bool

	// Checks if a commit is committed
	IsCommitted() bool
}

// MajorityQuorum is used by Apply transactions and requires
// a simple majority of nodes.
type majorityQuorum struct {
	count       int
	votesNeeded int
}

func newMajorityQuorum(clusterSize int) *majorityQuorum {
	votesNeeded := (clusterSize / 2) + 1
	return &majorityQuorum{count: 0, votesNeeded: votesNeeded}
}

func (m *majorityQuorum) Commit() bool {
	m.count++
	return m.count >= m.votesNeeded
}

func (m *majorityQuorum) IsCommitted() bool {
	return m.count >= m.votesNeeded
}

// Inflight is used to track operations that are still in-flight.
type inflight struct {
	sync.Mutex
	committed  *list.List
	commitCh   chan struct{}
	minCommit  uint64
	maxCommit  uint64
	operations map[uint64]*logFuture
	stopCh     chan struct{}
}

// NewInflight returns an inflight struct that notifies
// the provided channel when logs are finished committing.
func newInflight(commitCh chan struct{}) *inflight {
	return &inflight{
		committed:  list.New(),
		commitCh:   commitCh,
		minCommit:  0,
		maxCommit:  0,
		operations: make(map[uint64]*logFuture),
		stopCh:     make(chan struct{}),
	}
}

// Start is used to mark a logFuture as being inflight. It
// also commits the entry, as it is assumed the leader is
// starting.
func (i *inflight) Start(l *logFuture) {
	i.Lock()
	defer i.Unlock()
	i.start(l)
}

// StartAll is used to mark a list of logFuture's as being
// inflight. It also commits each entry as the leader is
// assumed to be starting.
func (i *inflight) StartAll(logs []*logFuture) {
	i.Lock()
	defer i.Unlock()
	for _, l := range logs {
		i.start(l)
	}
}

// start is used to mark a single entry as inflight,
// must be invoked with the lock held.
func (i *inflight) start(l *logFuture) {
	idx := l.log.Index
	i.operations[idx] = l

	if idx > i.maxCommit {
		i.maxCommit = idx
	}
	if i.minCommit == 0 {
		i.minCommit = idx
	}
	i.commit(idx)
}

// Cancel is used to cancel all in-flight operations.
// This is done when the leader steps down, and all futures
// are sent the given error.
func (i *inflight) Cancel(err error) {
	// Close the channel first to unblock any pending commits
	close(i.stopCh)

	// Lock after close to avoid deadlock
	i.Lock()
	defer i.Unlock()

	// Respond to all inflight operations
	for _, op := range i.operations {
		op.respond(err)
	}

	// Clear all the committed but not processed
	for e := i.committed.Front(); e != nil; e = e.Next() {
		e.Value.(*logFuture).respond(err)
	}

	// Clear the map
	i.operations = make(map[uint64]*logFuture)

	// Clear the list of committed
	i.committed = list.New()

	// Close the commmitCh
	close(i.commitCh)

	// Reset indexes
	i.minCommit = 0
	i.maxCommit = 0
}

// Committed returns all the committed operations in order.
func (i *inflight) Committed() (l *list.List) {
	i.Lock()
	l, i.committed = i.committed, list.New()
	i.Unlock()
	return l
}

// Commit is used by leader replication routines to indicate that
// a follower was finished committing a log to disk.
func (i *inflight) Commit(index uint64) {
	i.Lock()
	defer i.Unlock()
	i.commit(index)
}

// CommitRange is used to commit a range of indexes inclusively.
// It is optimized to avoid commits for indexes that are not tracked.
func (i *inflight) CommitRange(minIndex, maxIndex uint64) {
	i.Lock()
	defer i.Unlock()

	// Update the minimum index
	minIndex = max(i.minCommit, minIndex)

	// Commit each index
	for idx := minIndex; idx <= maxIndex; idx++ {
		i.commit(idx)
	}
}

// commit is used to commit a single index. Must be called with the lock held.
func (i *inflight) commit(index uint64) {
	op, ok := i.operations[index]
	if !ok {
		// Ignore if not in the map, as it may be committed already
		return
	}

	// Check if we've satisfied the commit
	if !op.policy.Commit() {
		return
	}

	// Cannot commit if this is not the minimum inflight. This can happen
	// if the quorum size changes, meaning a previous commit requires a larger
	// quorum that this commit. We MUST block until the previous log is committed,
	// otherwise logs will be applied out of order.
	if index != i.minCommit {
		return
	}

NOTIFY:
	// Add the operation to the committed list
	i.committed.PushBack(op)

	// Stop tracking since it is committed
	delete(i.operations, index)

	// Update the indexes
	if index == i.maxCommit {
		i.minCommit = 0
		i.maxCommit = 0

	} else {
		i.minCommit++
	}

	// Check if the next in-flight operation is ready
	if i.minCommit != 0 {
		op = i.operations[i.minCommit]
		if op.policy.IsCommitted() {
			index = i.minCommit
			goto NOTIFY
		}
	}

	// Async notify of ready operations
	asyncNotifyCh(i.commitCh)
}
