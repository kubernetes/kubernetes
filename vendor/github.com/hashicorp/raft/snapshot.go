package raft

import (
	"fmt"
	"io"
	"time"

	"github.com/armon/go-metrics"
)

// SnapshotMeta is for metadata of a snapshot.
type SnapshotMeta struct {
	// Version is the version number of the snapshot metadata. This does not cover
	// the application's data in the snapshot, that should be versioned
	// separately.
	Version SnapshotVersion

	// ID is opaque to the store, and is used for opening.
	ID string

	// Index and Term store when the snapshot was taken.
	Index uint64
	Term  uint64

	// Peers is deprecated and used to support version 0 snapshots, but will
	// be populated in version 1 snapshots as well to help with upgrades.
	Peers []byte

	// Configuration and ConfigurationIndex are present in version 1
	// snapshots and later.
	Configuration      Configuration
	ConfigurationIndex uint64

	// Size is the size of the snapshot in bytes.
	Size int64
}

// SnapshotStore interface is used to allow for flexible implementations
// of snapshot storage and retrieval. For example, a client could implement
// a shared state store such as S3, allowing new nodes to restore snapshots
// without streaming from the leader.
type SnapshotStore interface {
	// Create is used to begin a snapshot at a given index and term, and with
	// the given committed configuration. The version parameter controls
	// which snapshot version to create.
	Create(version SnapshotVersion, index, term uint64, configuration Configuration,
		configurationIndex uint64, trans Transport) (SnapshotSink, error)

	// List is used to list the available snapshots in the store.
	// It should return then in descending order, with the highest index first.
	List() ([]*SnapshotMeta, error)

	// Open takes a snapshot ID and provides a ReadCloser. Once close is
	// called it is assumed the snapshot is no longer needed.
	Open(id string) (*SnapshotMeta, io.ReadCloser, error)
}

// SnapshotSink is returned by StartSnapshot. The FSM will Write state
// to the sink and call Close on completion. On error, Cancel will be invoked.
type SnapshotSink interface {
	io.WriteCloser
	ID() string
	Cancel() error
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
	lastSnap, _ := r.getLastSnapshot()

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

// takeSnapshot is used to take a new snapshot. This must only be called from
// the snapshot thread, never the main thread.
func (r *Raft) takeSnapshot() error {
	defer metrics.MeasureSince([]string{"raft", "snapshot", "takeSnapshot"}, time.Now())

	// Create a request for the FSM to perform a snapshot.
	snapReq := &reqSnapshotFuture{}
	snapReq.init()

	// Wait for dispatch or shutdown.
	select {
	case r.fsmSnapshotCh <- snapReq:
	case <-r.shutdownCh:
		return ErrRaftShutdown
	}

	// Wait until we get a response
	if err := snapReq.Error(); err != nil {
		if err != ErrNothingNewToSnapshot {
			err = fmt.Errorf("failed to start snapshot: %v", err)
		}
		return err
	}
	defer snapReq.snapshot.Release()

	// Make a request for the configurations and extract the committed info.
	// We have to use the future here to safely get this information since
	// it is owned by the main thread.
	configReq := &configurationsFuture{}
	configReq.init()
	select {
	case r.configurationsCh <- configReq:
	case <-r.shutdownCh:
		return ErrRaftShutdown
	}
	if err := configReq.Error(); err != nil {
		return err
	}
	committed := configReq.configurations.committed
	committedIndex := configReq.configurations.committedIndex

	// We don't support snapshots while there's a config change outstanding
	// since the snapshot doesn't have a means to represent this state. This
	// is a little weird because we need the FSM to apply an index that's
	// past the configuration change, even though the FSM itself doesn't see
	// the configuration changes. It should be ok in practice with normal
	// application traffic flowing through the FSM. If there's none of that
	// then it's not crucial that we snapshot, since there's not much going
	// on Raft-wise.
	if snapReq.index < committedIndex {
		return fmt.Errorf("cannot take snapshot now, wait until the configuration entry at %v has been applied (have applied %v)",
			committedIndex, snapReq.index)
	}

	// Create a new snapshot.
	r.logger.Printf("[INFO] raft: Starting snapshot up to %d", snapReq.index)
	start := time.Now()
	version := getSnapshotVersion(r.protocolVersion)
	sink, err := r.snapshots.Create(version, snapReq.index, snapReq.term, committed, committedIndex, r.trans)
	if err != nil {
		return fmt.Errorf("failed to create snapshot: %v", err)
	}
	metrics.MeasureSince([]string{"raft", "snapshot", "create"}, start)

	// Try to persist the snapshot.
	start = time.Now()
	if err := snapReq.snapshot.Persist(sink); err != nil {
		sink.Cancel()
		return fmt.Errorf("failed to persist snapshot: %v", err)
	}
	metrics.MeasureSince([]string{"raft", "snapshot", "persist"}, start)

	// Close and check for error.
	if err := sink.Close(); err != nil {
		return fmt.Errorf("failed to close snapshot: %v", err)
	}

	// Update the last stable snapshot info.
	r.setLastSnapshot(snapReq.index, snapReq.term)

	// Compact the logs.
	if err := r.compactLogs(snapReq.index); err != nil {
		return err
	}

	r.logger.Printf("[INFO] raft: Snapshot to %d complete", snapReq.index)
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
	lastLogIdx, _ := r.getLastLog()
	if lastLogIdx <= r.conf.TrailingLogs {
		return nil
	}

	// Truncate up to the end of the snapshot, or `TrailingLogs`
	// back from the head, which ever is further back. This ensures
	// at least `TrailingLogs` entries, but does not allow logs
	// after the snapshot to be removed.
	maxLog := min(snapIdx, lastLogIdx-r.conf.TrailingLogs)

	// Log this
	r.logger.Printf("[INFO] raft: Compacting logs from %d to %d", minLog, maxLog)

	// Compact the logs
	if err := r.logs.DeleteRange(minLog, maxLog); err != nil {
		return fmt.Errorf("log compaction failed: %v", err)
	}
	return nil
}
