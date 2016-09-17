package consul

import (
	"fmt"
	"sort"
	"time"

	"github.com/armon/go-metrics"
	"github.com/hashicorp/consul/consul/structs"
	"github.com/hashicorp/consul/lib"
)

// aclIterator simplifies the algorithm below by providing a basic iterator that
// moves through a list of ACLs and returns nil when it's exhausted. It also has
// methods for pre-sorting the ACLs being iterated over by ID, which should
// already be true, but since this is crucial for correctness and we are taking
// input from other servers, we sort to make sure.
type aclIterator struct {
	acls structs.ACLs

	// index is the current position of the iterator.
	index int
}

// newACLIterator returns a new ACL iterator.
func newACLIterator(acls structs.ACLs) *aclIterator {
	return &aclIterator{acls: acls}
}

// See sort.Interface.
func (a *aclIterator) Len() int {
	return len(a.acls)
}

// See sort.Interface.
func (a *aclIterator) Swap(i, j int) {
	a.acls[i], a.acls[j] = a.acls[j], a.acls[i]
}

// See sort.Interface.
func (a *aclIterator) Less(i, j int) bool {
	return a.acls[i].ID < a.acls[j].ID
}

// Front returns the item at index position, or nil if the list is exhausted.
func (a *aclIterator) Front() *structs.ACL {
	if a.index < len(a.acls) {
		return a.acls[a.index]
	} else {
		return nil
	}
}

// Next advances the iterator to the next index.
func (a *aclIterator) Next() {
	a.index++
}

// reconcileACLs takes the local and remote ACL state, and produces a list of
// changes required in order to bring the local ACLs into sync with the remote
// ACLs. You can supply lastRemoteIndex as a hint that replication has succeeded
// up to that remote index and it will make this process more efficient by only
// comparing ACL entries modified after that index. Setting this to 0 will force
// a full compare of all existing ACLs.
func reconcileACLs(local, remote structs.ACLs, lastRemoteIndex uint64) structs.ACLRequests {
	// Since sorting the lists is crucial for correctness, we are depending
	// on data coming from other servers potentially running a different,
	// version of Consul, and sorted-ness is kind of a subtle property of
	// the state store indexing, it's prudent to make sure things are sorted
	// before we begin.
	localIter, remoteIter := newACLIterator(local), newACLIterator(remote)
	sort.Sort(localIter)
	sort.Sort(remoteIter)

	// Run through both lists and reconcile them.
	var changes structs.ACLRequests
	for localIter.Front() != nil || remoteIter.Front() != nil {
		// If the local list is exhausted, then process this as a remote
		// add. We know from the loop condition that there's something
		// in the remote list.
		if localIter.Front() == nil {
			changes = append(changes, &structs.ACLRequest{
				Op:  structs.ACLSet,
				ACL: *(remoteIter.Front()),
			})
			remoteIter.Next()
			continue
		}

		// If the remote list is exhausted, then process this as a local
		// delete. We know from the loop condition that there's something
		// in the local list.
		if remoteIter.Front() == nil {
			changes = append(changes, &structs.ACLRequest{
				Op:  structs.ACLDelete,
				ACL: *(localIter.Front()),
			})
			localIter.Next()
			continue
		}

		// At this point we know there's something at the front of each
		// list we need to resolve.

		// If the remote list has something local doesn't, we add it.
		if localIter.Front().ID > remoteIter.Front().ID {
			changes = append(changes, &structs.ACLRequest{
				Op:  structs.ACLSet,
				ACL: *(remoteIter.Front()),
			})
			remoteIter.Next()
			continue
		}

		// If local has something remote doesn't, we delete it.
		if localIter.Front().ID < remoteIter.Front().ID {
			changes = append(changes, &structs.ACLRequest{
				Op:  structs.ACLDelete,
				ACL: *(localIter.Front()),
			})
			localIter.Next()
			continue
		}

		// Local and remote have an ACL with the same ID, so we might
		// need to compare them.
		l, r := localIter.Front(), remoteIter.Front()
		if r.RaftIndex.ModifyIndex > lastRemoteIndex && !r.IsSame(l) {
			changes = append(changes, &structs.ACLRequest{
				Op:  structs.ACLSet,
				ACL: *r,
			})
		}
		localIter.Next()
		remoteIter.Next()
	}
	return changes
}

// FetchLocalACLs returns the ACLs in the local state store.
func (s *Server) fetchLocalACLs() (structs.ACLs, error) {
	_, local, err := s.fsm.State().ACLList()
	if err != nil {
		return nil, err
	}
	return local, nil
}

// FetchRemoteACLs is used to get the remote set of ACLs from the ACL
// datacenter. The lastIndex parameter is a hint about which remote index we
// have replicated to, so this is expected to block until something changes.
func (s *Server) fetchRemoteACLs(lastRemoteIndex uint64) (*structs.IndexedACLs, error) {
	defer metrics.MeasureSince([]string{"consul", "leader", "fetchRemoteACLs"}, time.Now())

	args := structs.DCSpecificRequest{
		Datacenter: s.config.ACLDatacenter,
		QueryOptions: structs.QueryOptions{
			Token:         s.config.ACLReplicationToken,
			MinQueryIndex: lastRemoteIndex,
			AllowStale:    true,
		},
	}
	var remote structs.IndexedACLs
	if err := s.RPC("ACL.List", &args, &remote); err != nil {
		return nil, err
	}
	return &remote, nil
}

// UpdateLocalACLs is given a list of changes to apply in order to bring the
// local ACLs in-line with the remote ACLs from the ACL datacenter.
func (s *Server) updateLocalACLs(changes structs.ACLRequests) error {
	defer metrics.MeasureSince([]string{"consul", "leader", "updateLocalACLs"}, time.Now())

	minTimePerOp := time.Second / time.Duration(s.config.ACLReplicationApplyLimit)
	for _, change := range changes {
		// Note that we are using the single ACL interface here and not
		// performing all this inside a single transaction. This is OK
		// for two reasons. First, there's nothing else other than this
		// replication routine that alters the local ACLs, so there's
		// nothing to contend with locally. Second, if an apply fails
		// in the middle (most likely due to losing leadership), the
		// next replication pass will clean up and check everything
		// again.
		var reply string
		start := time.Now()
		if err := aclApplyInternal(s, change, &reply); err != nil {
			return err
		}

		// Do a smooth rate limit to wait out the min time allowed for
		// each op. If this op took longer than the min, then the sleep
		// time will be negative and we will just move on.
		elapsed := time.Now().Sub(start)
		time.Sleep(minTimePerOp - elapsed)
	}
	return nil
}

// replicateACLs is a runs one pass of the algorithm for replicating ACLs from
// a remote ACL datacenter to local state. If there's any error, this will return
// 0 for the lastRemoteIndex, which will cause us to immediately do a full sync
// next time.
func (s *Server) replicateACLs(lastRemoteIndex uint64) (uint64, error) {
	remote, err := s.fetchRemoteACLs(lastRemoteIndex)
	if err != nil {
		return 0, fmt.Errorf("failed to retrieve remote ACLs: %v", err)
	}

	// This will be pretty common because we will be blocking for a long time
	// and may have lost leadership, so lets control the message here instead
	// of returning deeper error messages from from Raft.
	if !s.IsLeader() {
		return 0, fmt.Errorf("no longer cluster leader")
	}

	// Measure everything after the remote query, which can block for long
	// periods of time. This metric is a good measure of how expensive the
	// replication process is.
	defer metrics.MeasureSince([]string{"consul", "leader", "replicateACLs"}, time.Now())

	local, err := s.fetchLocalACLs()
	if err != nil {
		return 0, fmt.Errorf("failed to retrieve local ACLs: %v", err)
	}

	// If the remote index ever goes backwards, it's a good indication that
	// the remote side was rebuilt and we should do a full sync since we
	// can't make any assumptions about what's going on.
	if remote.QueryMeta.Index < lastRemoteIndex {
		s.logger.Printf("[WARN] consul: ACL replication remote index moved backwards (%d to %d), forcing a full ACL sync", lastRemoteIndex, remote.QueryMeta.Index)
		lastRemoteIndex = 0
	}

	// Calculate the changes required to bring the state into sync and then
	// apply them.
	changes := reconcileACLs(local, remote.ACLs, lastRemoteIndex)
	if err := s.updateLocalACLs(changes); err != nil {
		return 0, fmt.Errorf("failed to sync ACL changes: %v", err)
	}

	// Return the index we got back from the remote side, since we've synced
	// up with the remote state as of that index.
	return remote.QueryMeta.Index, nil
}

// IsACLReplicationEnabled returns true if ACL replication is enabled.
func (s *Server) IsACLReplicationEnabled() bool {
	authDC := s.config.ACLDatacenter
	return len(authDC) > 0 && (authDC != s.config.Datacenter) &&
		len(s.config.ACLReplicationToken) > 0
}

// updateACLReplicationStatus safely updates the ACL replication status.
func (s *Server) updateACLReplicationStatus(status structs.ACLReplicationStatus) {
	// Fixup the times to shed some useless precision to ease formattting,
	// and always report UTC.
	status.LastError = status.LastError.Round(time.Second).UTC()
	status.LastSuccess = status.LastSuccess.Round(time.Second).UTC()

	// Set the shared state.
	s.aclReplicationStatusLock.Lock()
	s.aclReplicationStatus = status
	s.aclReplicationStatusLock.Unlock()
}

// runACLReplication is a long-running goroutine that will attempt to replicate
// ACLs while the server is the leader, until the shutdown channel closes.
func (s *Server) runACLReplication() {
	var status structs.ACLReplicationStatus
	status.Enabled = true
	status.SourceDatacenter = s.config.ACLDatacenter
	s.updateACLReplicationStatus(status)

	// Show that it's not running on the way out.
	defer func() {
		status.Running = false
		s.updateACLReplicationStatus(status)
	}()

	// Give each server's replicator a random initial phase for good
	// measure.
	select {
	case <-s.shutdownCh:
		return

	case <-time.After(lib.RandomStagger(s.config.ACLReplicationInterval)):
	}

	// We are fairly conservative with the lastRemoteIndex so that after a
	// leadership change or an error we re-sync everything (we also don't
	// want to block the first time after one of these events so we can
	// show a successful sync in the status endpoint).
	var lastRemoteIndex uint64
	replicate := func() {
		if !status.Running {
			lastRemoteIndex = 0 // Re-sync everything.
			status.Running = true
			s.updateACLReplicationStatus(status)
			s.logger.Printf("[INFO] consul: ACL replication started")
		}

		index, err := s.replicateACLs(lastRemoteIndex)
		if err != nil {
			lastRemoteIndex = 0 // Re-sync everything.
			status.LastError = time.Now()
			s.updateACLReplicationStatus(status)
			s.logger.Printf("[WARN] consul: ACL replication error (will retry if still leader): %v", err)
		} else {
			lastRemoteIndex = index
			status.ReplicatedIndex = index
			status.LastSuccess = time.Now()
			s.updateACLReplicationStatus(status)
			s.logger.Printf("[DEBUG] consul: ACL replication completed through remote index %d", index)
		}
	}
	pause := func() {
		if status.Running {
			lastRemoteIndex = 0 // Re-sync everything.
			status.Running = false
			s.updateACLReplicationStatus(status)
			s.logger.Printf("[INFO] consul: ACL replication stopped (no longer leader)")
		}
	}

	// This will slowly poll to see if replication should be active. Once it
	// is and we've caught up, the replicate() call will begin to block and
	// only wake up when the query timer expires or there are new ACLs to
	// replicate. We've chosen this design so that the ACLReplicationInterval
	// is the lower bound for how quickly we will replicate, no matter how
	// much ACL churn is happening on the remote side.
	//
	// The blocking query inside replicate() respects the shutdown channel,
	// so we won't get stuck in here as things are torn down.
	for {
		select {
		case <-s.shutdownCh:
			return

		case <-time.After(s.config.ACLReplicationInterval):
			if s.IsLeader() {
				replicate()
			} else {
				pause()
			}
		}
	}
}
