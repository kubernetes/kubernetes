package raft

import (
	"fmt"
	"io"
	"time"

	"github.com/armon/go-metrics"
)

// FSM provides an interface that can be implemented by
// clients to make use of the replicated log.
type FSM interface {
	// Apply log is invoked once a log entry is committed.
	// It returns a value which will be made available in the
	// ApplyFuture returned by Raft.Apply method if that
	// method was called on the same Raft node as the FSM.
	Apply(*Log) interface{}

	// Snapshot is used to support log compaction. This call should
	// return an FSMSnapshot which can be used to save a point-in-time
	// snapshot of the FSM. Apply and Snapshot are not called in multiple
	// threads, but Apply will be called concurrently with Persist. This means
	// the FSM should be implemented in a fashion that allows for concurrent
	// updates while a snapshot is happening.
	Snapshot() (FSMSnapshot, error)

	// Restore is used to restore an FSM from a snapshot. It is not called
	// concurrently with any other command. The FSM must discard all previous
	// state.
	Restore(io.ReadCloser) error
}

// FSMSnapshot is returned by an FSM in response to a Snapshot
// It must be safe to invoke FSMSnapshot methods with concurrent
// calls to Apply.
type FSMSnapshot interface {
	// Persist should dump all necessary state to the WriteCloser 'sink',
	// and call sink.Close() when finished or call sink.Cancel() on error.
	Persist(sink SnapshotSink) error

	// Release is invoked when we are finished with the snapshot.
	Release()
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

			// Start a snapshot
			start := time.Now()
			snap, err := r.fsm.Snapshot()
			metrics.MeasureSince([]string{"raft", "fsm", "snapshot"}, start)

			// Respond to the request
			req.index = lastIndex
			req.term = lastTerm
			req.snapshot = snap
			req.respond(err)

		case commitEntry := <-r.fsmCommitCh:
			// Apply the log if a command
			var resp interface{}
			if commitEntry.log.Type == LogCommand {
				start := time.Now()
				resp = r.fsm.Apply(commitEntry.log)
				metrics.MeasureSince([]string{"raft", "fsm", "apply"}, start)
			}

			// Update the indexes
			lastIndex = commitEntry.log.Index
			lastTerm = commitEntry.log.Term

			// Invoke the future if given
			if commitEntry.future != nil {
				commitEntry.future.response = resp
				commitEntry.future.respond(nil)
			}
		case <-r.shutdownCh:
			return
		}
	}
}
