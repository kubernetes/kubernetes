// Copyright 2019 The etcd Authors
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

package tracker

import (
	"fmt"
	"sort"
	"strings"
)

// Progress represents a follower’s progress in the view of the leader. Leader
// maintains progresses of all followers, and sends entries to the follower
// based on its progress.
//
// NB(tbg): Progress is basically a state machine whose transitions are mostly
// strewn around `*raft.raft`. Additionally, some fields are only used when in a
// certain State. All of this isn't ideal.
type Progress struct {
	Match, Next uint64
	// State defines how the leader should interact with the follower.
	//
	// When in StateProbe, leader sends at most one replication message
	// per heartbeat interval. It also probes actual progress of the follower.
	//
	// When in StateReplicate, leader optimistically increases next
	// to the latest entry sent after sending replication message. This is
	// an optimized state for fast replicating log entries to the follower.
	//
	// When in StateSnapshot, leader should have sent out snapshot
	// before and stops sending any replication message.
	State StateType

	// PendingSnapshot is used in StateSnapshot.
	// If there is a pending snapshot, the pendingSnapshot will be set to the
	// index of the snapshot. If pendingSnapshot is set, the replication process of
	// this Progress will be paused. raft will not resend snapshot until the pending one
	// is reported to be failed.
	PendingSnapshot uint64

	// RecentActive is true if the progress is recently active. Receiving any messages
	// from the corresponding follower indicates the progress is active.
	// RecentActive can be reset to false after an election timeout.
	//
	// TODO(tbg): the leader should always have this set to true.
	RecentActive bool

	// ProbeSent is used while this follower is in StateProbe. When ProbeSent is
	// true, raft should pause sending replication message to this peer until
	// ProbeSent is reset. See ProbeAcked() and IsPaused().
	ProbeSent bool

	// Inflights is a sliding window for the inflight messages.
	// Each inflight message contains one or more log entries.
	// The max number of entries per message is defined in raft config as MaxSizePerMsg.
	// Thus inflight effectively limits both the number of inflight messages
	// and the bandwidth each Progress can use.
	// When inflights is Full, no more message should be sent.
	// When a leader sends out a message, the index of the last
	// entry should be added to inflights. The index MUST be added
	// into inflights in order.
	// When a leader receives a reply, the previous inflights should
	// be freed by calling inflights.FreeLE with the index of the last
	// received entry.
	Inflights *Inflights

	// IsLearner is true if this progress is tracked for a learner.
	IsLearner bool
}

// ResetState moves the Progress into the specified State, resetting ProbeSent,
// PendingSnapshot, and Inflights.
func (pr *Progress) ResetState(state StateType) {
	pr.ProbeSent = false
	pr.PendingSnapshot = 0
	pr.State = state
	pr.Inflights.reset()
}

func max(a, b uint64) uint64 {
	if a > b {
		return a
	}
	return b
}

func min(a, b uint64) uint64 {
	if a > b {
		return b
	}
	return a
}

// ProbeAcked is called when this peer has accepted an append. It resets
// ProbeSent to signal that additional append messages should be sent without
// further delay.
func (pr *Progress) ProbeAcked() {
	pr.ProbeSent = false
}

// BecomeProbe transitions into StateProbe. Next is reset to Match+1 or,
// optionally and if larger, the index of the pending snapshot.
func (pr *Progress) BecomeProbe() {
	// If the original state is StateSnapshot, progress knows that
	// the pending snapshot has been sent to this peer successfully, then
	// probes from pendingSnapshot + 1.
	if pr.State == StateSnapshot {
		pendingSnapshot := pr.PendingSnapshot
		pr.ResetState(StateProbe)
		pr.Next = max(pr.Match+1, pendingSnapshot+1)
	} else {
		pr.ResetState(StateProbe)
		pr.Next = pr.Match + 1
	}
}

// BecomeReplicate transitions into StateReplicate, resetting Next to Match+1.
func (pr *Progress) BecomeReplicate() {
	pr.ResetState(StateReplicate)
	pr.Next = pr.Match + 1
}

// BecomeSnapshot moves the Progress to StateSnapshot with the specified pending
// snapshot index.
func (pr *Progress) BecomeSnapshot(snapshoti uint64) {
	pr.ResetState(StateSnapshot)
	pr.PendingSnapshot = snapshoti
}

// MaybeUpdate is called when an MsgAppResp arrives from the follower, with the
// index acked by it. The method returns false if the given n index comes from
// an outdated message. Otherwise it updates the progress and returns true.
func (pr *Progress) MaybeUpdate(n uint64) bool {
	var updated bool
	if pr.Match < n {
		pr.Match = n
		updated = true
		pr.ProbeAcked()
	}
	if pr.Next < n+1 {
		pr.Next = n + 1
	}
	return updated
}

// OptimisticUpdate signals that appends all the way up to and including index n
// are in-flight. As a result, Next is increased to n+1.
func (pr *Progress) OptimisticUpdate(n uint64) { pr.Next = n + 1 }

// MaybeDecrTo adjusts the Progress to the receipt of a MsgApp rejection. The
// arguments are the index the follower rejected to append to its log, and its
// last index.
//
// Rejections can happen spuriously as messages are sent out of order or
// duplicated. In such cases, the rejection pertains to an index that the
// Progress already knows were previously acknowledged, and false is returned
// without changing the Progress.
//
// If the rejection is genuine, Next is lowered sensibly, and the Progress is
// cleared for sending log entries.
func (pr *Progress) MaybeDecrTo(rejected, last uint64) bool {
	if pr.State == StateReplicate {
		// The rejection must be stale if the progress has matched and "rejected"
		// is smaller than "match".
		if rejected <= pr.Match {
			return false
		}
		// Directly decrease next to match + 1.
		//
		// TODO(tbg): why not use last if it's larger?
		pr.Next = pr.Match + 1
		return true
	}

	// The rejection must be stale if "rejected" does not match next - 1. This
	// is because non-replicating followers are probed one entry at a time.
	if pr.Next-1 != rejected {
		return false
	}

	if pr.Next = min(rejected, last+1); pr.Next < 1 {
		pr.Next = 1
	}
	pr.ProbeSent = false
	return true
}

// IsPaused returns whether sending log entries to this node has been throttled.
// This is done when a node has rejected recent MsgApps, is currently waiting
// for a snapshot, or has reached the MaxInflightMsgs limit. In normal
// operation, this is false. A throttled node will be contacted less frequently
// until it has reached a state in which it's able to accept a steady stream of
// log entries again.
func (pr *Progress) IsPaused() bool {
	switch pr.State {
	case StateProbe:
		return pr.ProbeSent
	case StateReplicate:
		return pr.Inflights.Full()
	case StateSnapshot:
		return true
	default:
		panic("unexpected state")
	}
}

func (pr *Progress) String() string {
	var buf strings.Builder
	fmt.Fprintf(&buf, "%s match=%d next=%d", pr.State, pr.Match, pr.Next)
	if pr.IsLearner {
		fmt.Fprint(&buf, " learner")
	}
	if pr.IsPaused() {
		fmt.Fprint(&buf, " paused")
	}
	if pr.PendingSnapshot > 0 {
		fmt.Fprintf(&buf, " pendingSnap=%d", pr.PendingSnapshot)
	}
	if !pr.RecentActive {
		fmt.Fprintf(&buf, " inactive")
	}
	if n := pr.Inflights.Count(); n > 0 {
		fmt.Fprintf(&buf, " inflight=%d", n)
		if pr.Inflights.Full() {
			fmt.Fprint(&buf, "[full]")
		}
	}
	return buf.String()
}

// ProgressMap is a map of *Progress.
type ProgressMap map[uint64]*Progress

// String prints the ProgressMap in sorted key order, one Progress per line.
func (m ProgressMap) String() string {
	ids := make([]uint64, 0, len(m))
	for k := range m {
		ids = append(ids, k)
	}
	sort.Slice(ids, func(i, j int) bool {
		return ids[i] < ids[j]
	})
	var buf strings.Builder
	for _, id := range ids {
		fmt.Fprintf(&buf, "%d: %s\n", id, m[id])
	}
	return buf.String()
}
