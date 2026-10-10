/*
Copyright The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package nodememo

import (
	"context"

	fwk "k8s.io/kube-scheduler/framework"
)

// NodeMemo memoizes a per node contribution across scheduling cycles and validates it with
// framework.NodeInfo.GetGeneration instead of chasing informer events.
//
// Why this shape. The obvious way to make a per cycle cluster walk cheaper is a cache that a worker
// goroutine maintains from informer events, and that design has to guess when it is behind - which it
// cannot. A pod deletion reaches such a worker some hundreds of microseconds after the scheduler
// removed the pod from its own cache and woke the pods that deletion unblocked, so the retry runs on
// counts that still include the deleted pod. Nothing in the cache can say "this number is stale", and
// because the queue only wakes a pod for events its own rejectors registered, a pod that fails on a
// stale count then waits for whichever event happens to interest it next: a race worth microseconds
// becomes a stall worth minutes.
//
// Deriving the numbers from the scheduling cycle snapshot instead removes the question. The snapshot
// is what Filter is allowed to read, Cache.UpdateSnapshot refreshes it at the top of every scheduling
// cycle (schedule_one.go, and once per pod group cycle in schedule_one_podgroup.go), and every
// mutation that can change a node's contribution bumps that node's generation - NodeInfo.update for
// pods (framework/types.go), SetNode and RemoveNode for the node itself, including the label changes
// that move it between topology domains.
//
// Generations come from one process wide monotonic counter (nextGeneration in framework/types.go,
// deliberately collision free so that a node deleted and recreated under the same name cannot reuse
// a number), and one generation belongs to exactly one state of one node. Everything the memo
// decides follows from that single fact, in two rules:
//
//   - A stored contribution is reused only when the node still has the generation it was computed
//     from. Comparing generations instead of asking "is it newer than the watermark" also survives a
//     snapshot that goes back in time, which is what preemption's mutation session does:
//     StartMutations hands out clones of every NodeInfo and EndMutations puts the originals back,
//     generations included (backend/cache/snapshot.go).
//   - A node with no stored contribution and a generation no newer than the watermark contributed
//     nothing when a previous pass looked at that very same state, so it contributes nothing now.
//     Empty contributions are not stored, which is what keeps the memo sparse on a large cluster.
//
// The second rule needs one more thing: the node must have been looked at. A caller may hand in a
// filtered list - InterPodAffinity's existing pods counts walk the snapshot's
// HavePodsWithRequiredAntiAffinityList - and a node can drop out of such a list and come back later
// carrying the generation it left with, so the memo remembers the names of the nodes that left while
// contributing and recomputes them when they reappear instead of reading them as empty.
//
// The one thing a generation cannot express is a mutation that changes a node without bumping it.
// Snapshot.AssumePod does exactly that, on purpose: it puts the old number back so that the snapshot
// stays consistent with the cache, which is what lets UpdateSnapshot's generation comparison keep
// working (backend/cache/snapshot.go). A single pod cycle assumes only after its scheduling decisions
// are made, so nothing reads the memo afterwards, but a pod group cycle assumes every pod of the gang
// into one snapshot between two Cache.UpdateSnapshot calls (AssumeAndReserveInSnapshot in
// schedule_one_podgroup.go) and the later pods of the gang have to see the earlier ones. Callers
// therefore stay away from the memo when CycleState.IsPodGroupSchedulingCycle, because a generation
// that does not move while the node does would silently under-count.
//
// This is the same argument Cache.UpdateSnapshot itself makes when it walks the cache's generation
// ordered node list and stops at the first node that is not newer than the snapshot
// (backend/cache/cache.go).
//
// Cost per pass is one generation read and one map lookup per node; only nodes that changed are
// recomputed, and only their delta is folded into the aggregate. A pass runs in three parts, and the
// split is what lets the expensive part be parallel without the memo ever needing a lock:
//
//  1. serial, in the scheduling goroutine: walk the nodes, compare generations, and collect the ones
//     that have to be recomputed. This is the only part that reads or writes the memo.
//  2. parallel: compute the contribution of each collected node, one result slot per node, no
//     sharing between them. This is the part that matches every pod of a node against the terms, so
//     it is the part worth spreading - a cold pass, where a key is seen for the first time and every
//     node has to be computed, would otherwise run serially in the scheduling goroutine and cost
//     more than the full computation it replaces.
//  3. serial again: fold the results into the caller's aggregate and store them.
//
// The memo's own state is therefore only ever touched by one goroutine and can never be observed
// half applied. compute has to be safe to call concurrently, which it is for both InterPodAffinity
// counts: they are the same per node closures the full computation already runs through the
// framework parallelizer.
// Parallel says how a memo may spread the per node computations of one pass over several
// goroutines. The zero value keeps the memo serial.
type Parallel struct {
	// Until runs fn for every index in [0, count), possibly concurrently. It is normally the
	// framework parallelizer bound to a plugin name.
	//
	// Until must not return before every one of its fn calls has completed, however it gets there -
	// the framework's ParallelizeUntil waits on a WaitGroup, and a canceled context is no exception.
	// Refresh relies on it: once Until is back it writes to the same result slots again to finish a
	// pass the context abandoned, and an fn still running would be writing its own slot at the same
	// time.
	Until func(ctx context.Context, count int, fn func(index int))
	// MinNodes is how many nodes a pass has to recompute before Until is used at all: waking workers
	// costs more than a handful of nodes does.
	MinNodes int
}

// DefaultParallelMinNodes is the MinNodes used by callers that have a parallelizer but no opinion of
// their own. A pass that recomputes fewer nodes than this is dominated by the wake up cost, not by
// the matching.
const DefaultParallelMinNodes = 64

type NodeMemo[C any] struct {
	// how this memo may parallelize the compute part of a pass
	parallel Parallel

	// contributions of the nodes that have one, keyed by node name. Nodes that contribute nothing
	// are absent and covered by the watermark.
	entries map[string]*memoEntry[C]
	// generations of the nodes that left the caller's list while contributing, keyed by node name.
	// They may come back carrying that same generation, and then they must be recomputed rather than
	// read as "contributes nothing". Bounded by leftCap.
	//
	// With the callers that exist today this set stays empty, and it is worth writing down why, since
	// "never used" is not the same as "useless". Every list they hand in is either the whole snapshot
	// or one of the snapshot's filtered lists, and a filtered list's membership is itself a function
	// of node state: UpdateSnapshot moves a node in or out of HavePodsWithRequiredAntiAffinityList
	// only when len(PodsWithRequiredAntiAffinity) flips (backend/cache/cache.go). A node therefore
	// cannot rejoin a list without changing state, changing state means a generation handed out after
	// the pass that saw it leave, and the watermark only holds generations from that pass's snapshot
	// - so the rejoined node is always newer than the watermark and is recomputed by the rule above
	// whatever this set says.
	//
	// What the set protects is the contract, not today's callers: Refresh takes whatever the caller
	// considers this pass's nodes, and a caller whose list is NOT a function of node state - a sampled
	// subset, a plugin specific filter, anything built outside the snapshot - can hand back a node
	// carrying the generation it left with. Read as "contributes nothing" that is a silent
	// under-count: no crash, no error, just numbers that are too small, which is the one failure mode
	// this whole design exists to remove.
	//
	// The price is one bounded map and one extra probe on the pass that skips a node, which measures
	// as free: guarding the probe with len(left) != 0 did not move the 2000 node steady pass (54.2us
	// without the guard, 56.6us with it, i.e. noise - the runtime already short-circuits a lookup in
	// an empty map), so the guard was dropped again.
	//
	// TestNodeMemoRecomputesANodeThatLeftAndCameBack constructs the situation directly, because no
	// caller produces it today.
	left map[string]int64
	// highest node generation seen by the last pass
	watermark int64
	// pass counter, used to drop entries whose node disappeared from the snapshot
	pass uint64
	// how many left over node names are remembered before the memo gives up and forgets everything
	leftCap int

	// Reused and Recomputed count nodes across all passes, and ParallelPasses counts the passes whose
	// compute part went to the parallelizer, for callers that want to expose the hit rate. They are
	// only touched from the scheduling goroutine.
	Reused         int64
	Recomputed     int64
	ParallelPasses int64
}

// memoWork is one node a pass has to recompute: where it is in the caller's list, the generation it
// carries now, and the contribution it is replacing, if any.
type memoWork[C any] struct {
	index      int
	name       string
	generation int64
	entry      *memoEntry[C]
}

// memoResult is the slot one memoWork is computed into. Each index writes its own slot, which is why
// the compute part needs no synchronization.
type memoResult[C any] struct {
	value C
	has   bool
}

type memoEntry[C any] struct {
	value C
	// generation of the node state the value was computed from
	generation int64
	// pass in which the node was last seen in the caller's list
	pass uint64
}

// DefaultLeftCap bounds the set of nodes that left the caller's list. Node names are small, but a
// cluster that keeps creating and deleting nodes would grow the set without end; past the bound the
// memo forgets everything and the next pass rebuilds it, which is exact and costs one cold pass.
const DefaultLeftCap = 4096

// NewNodeMemo returns an empty memo. Pass a Parallel to let the compute part of a pass run on the
// framework parallelizer; the zero Parallel keeps every part of it in the calling goroutine.
func NewNodeMemo[C any](parallel Parallel) *NodeMemo[C] {
	return &NodeMemo[C]{
		parallel: parallel,
		entries:  make(map[string]*memoEntry[C]),
		left:     make(map[string]int64),
		leftCap:  DefaultLeftCap,
	}
}

// Len reports how many nodes currently have a stored contribution.
func (m *NodeMemo[C]) Len() int { return len(m.entries) }

// There is deliberately no way to invalidate a memo from the outside. Everything a contribution
// depends on is either a node generation, which Refresh checks, or an input of the caller's own,
// which belongs in the key the caller keeps its memos under - a changed input is a different key and
// so a different memo, and the one left behind is dropped whole when the caller's LRU evicts it. An
// invalidation instead of a key would cost a cold pass on every change of input, and a cold pass is
// dearer than the full computation the memo replaces, so two pod shapes that alternate would end up
// slower than having no memo at all. The only forgetting that happens here is forgetEverything below,
// which the bound on left over node names forces and which the next pass rebuilds exactly.

// Refresh brings the memo up to date with nodes, calling compute only for the nodes whose state it
// does not already know, and reporting every replacement through onChange so the caller can fold the
// delta into its aggregate:
//
//	onChange(old, hadOld, cur, hasCur)
//
// hadOld is false for a node that contributed nothing before, hasCur false for one that stopped
// contributing. compute returning hasCur=false stores nothing, which keeps the memo sparse. onChange
// is always called from the calling goroutine, in the order the nodes appear in nodes.
//
// nodes must be the caller's view of this pass: a node the caller does not pass in is treated as
// gone and its contribution is reported as removed, so an aggregate can never keep counting a node
// that left the cluster or, for a filtered node list, stopped being relevant.
func (m *NodeMemo[C]) Refresh(ctx context.Context, nodes []fwk.NodeInfo, compute func(fwk.NodeInfo) (C, bool),
	onChange func(old C, hadOld bool, cur C, hasCur bool)) {
	m.pass++

	// Bound the set of left over node names before this pass decides anything. Forgetting at the end
	// instead would empty the aggregate this pass has just folded its results into, and the caller
	// reads that aggregate as soon as Refresh returns - in the same scheduling cycle it filters nodes
	// with, so an emptied one would mean "no node matches any term" rather than "start over".
	if len(m.left) > m.leftCap {
		m.forgetEverything(onChange)
	}

	highest := m.watermark

	// Part one, serial: work out which nodes have to be computed. Cheap - a generation read and a map
	// lookup per node - and the only part that touches the memo.
	todoCap := 16
	if m.watermark == 0 {
		todoCap = len(nodes) // first pass for this key: nothing about these nodes is known yet
	}
	todos := make([]memoWork[C], 0, todoCap)
	for i, nodeInfo := range nodes {
		node := nodeInfo.Node()
		if node == nil {
			// A node without a Node object contributes nothing, which is what the full computation
			// does with it too.
			continue
		}
		generation := nodeInfo.GetGeneration()
		if generation > highest {
			highest = generation
		}
		name := node.Name
		entry := m.entries[name]
		if entry != nil {
			if entry.generation == generation {
				// The very node state the stored contribution was computed from.
				entry.pass = m.pass
				m.Reused++
				continue
			}
			// Marking it as seen already keeps the pruning part below from mistaking a node that is
			// about to be replaced for one that is gone.
			entry.pass = m.pass
		} else if generation <= m.watermark {
			if _, left := m.left[name]; !left {
				// Not newer than anything a previous pass saw, and not a node that dropped out of the
				// list while contributing: a pass already looked at this exact state and stored
				// nothing, so it still contributes nothing.
				continue
			}
			// It left the list and is back, possibly carrying the generation it left with: recompute
			// rather than trust the absence of a contribution.
		}
		delete(m.left, name)
		todos = append(todos, memoWork[C]{index: i, name: name, generation: generation, entry: entry})
	}

	// Part two: the matching itself, one result slot per node.
	results := make([]memoResult[C], len(todos))
	runOne := func(k int) {
		results[k].value, results[k].has = compute(nodes[todos[k].index])
	}
	runAll := func() {
		for k := range todos {
			runOne(k)
		}
	}
	if m.parallel.Until != nil && len(todos) >= m.parallel.MinNodes {
		m.ParallelPasses++
		m.parallel.Until(ctx, len(todos), runOne)
		if ctx.Err() != nil {
			// A canceled context can leave pieces unrun, and a half computed pass must never reach
			// the aggregate: the nodes that were skipped would be stored as "contributes nothing"
			// under a generation the memo now believes it knows, so nothing would ever look at them
			// again. compute takes no context, so finishing the pass here is always possible.
			runAll()
		}
	} else {
		runAll()
	}

	// Part three, serial again: fold the results in and store them, in node order.
	for k, work := range todos {
		result := results[k]
		var oldValue C
		counted := work.entry != nil
		if counted {
			oldValue = work.entry.value
			delete(m.entries, work.name)
		}
		m.Recomputed++
		if result.has {
			m.entries[work.name] = &memoEntry[C]{value: result.value, generation: work.generation, pass: m.pass}
		}
		onChange(oldValue, counted, result.value, result.has)
	}
	for name, entry := range m.entries {
		if entry.pass == m.pass {
			continue
		}
		// The node is not in the caller's list any more. Its contribution leaves the aggregate, but
		// its name is remembered: the list may be a filtered view, and a node that comes back without
		// having changed carries the generation it left with.
		m.left[name] = entry.generation
		delete(m.entries, name)
		var zero C
		onChange(entry.value, true, zero, false)
	}
	m.watermark = highest
}

// forgetEverything drops every stored contribution, reporting the removals so that the caller's
// aggregate stays in step, and rewinds the watermark so that the pass recomputes all of it. It runs
// at the start of a pass, never at the end: what a pass hands back has to describe the nodes it was
// given.
func (m *NodeMemo[C]) forgetEverything(onChange func(old C, hadOld bool, cur C, hasCur bool)) {
	var zero C
	for _, entry := range m.entries {
		onChange(entry.value, true, zero, false)
	}
	m.entries = make(map[string]*memoEntry[C])
	m.left = make(map[string]int64)
	m.watermark = 0
}
