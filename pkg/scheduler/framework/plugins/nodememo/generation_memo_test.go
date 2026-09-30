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
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework"
)

// These tests price the two rules NodeMemo is built on - reuse a contribution only for the very
// generation it was computed from, and read a node no generation has moved past the watermark as
// empty - and the two escape hatches that keep them exact: the set of nodes that left the caller's
// list, and forgetting everything when that set grows without end. InterPodAffinity and
// PodTopologySpread drive the memo through their own tests; what is pinned here is the machinery
// itself, so that a change to it cannot be right for one caller and wrong for the next.

// countingMemo is the smallest caller that shows what a pass did: the contribution of a node is the
// number of pods on it, the aggregate their sum, and computed records the nodes compute was called
// for.
type countingMemo struct {
	t         *testing.T
	memo      *NodeMemo[int]
	aggregate int
	computed  []string
}

func newCountingMemo(t *testing.T, parallel Parallel) *countingMemo {
	t.Helper()
	return &countingMemo{t: t, memo: NewNodeMemo[int](parallel)}
}

// node builds a NodeInfo for name holding pods, with a generation of its own.
func node(name string, podCount int) fwk.NodeInfo {
	nodeInfo := framework.NewNodeInfo()
	nodeInfo.SetNode(&v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name, UID: types.UID(name)}})
	for i := range podCount {
		pod := &v1.Pod{ObjectMeta: metav1.ObjectMeta{
			Namespace: "testns", Name: name + "-pod-" + string(rune('a'+i)),
			UID: types.UID(name + "-pod-" + string(rune('a'+i))),
		}}
		pod.Spec.NodeName = name
		nodeInfo.AddPod(pod)
	}
	return nodeInfo
}

// pass runs one Refresh over nodes and returns the aggregate it left behind.
func (c *countingMemo) pass(nodes []fwk.NodeInfo) int {
	c.t.Helper()
	c.computed = nil
	c.memo.Refresh(context.Background(), nodes,
		func(nodeInfo fwk.NodeInfo) (int, bool) {
			count := len(nodeInfo.GetPods())
			c.computed = append(c.computed, nodeInfo.Node().Name)
			return count, count > 0
		},
		func(old int, hadOld bool, cur int, hasCur bool) {
			if hadOld {
				c.aggregate -= old
			}
			if hasCur {
				c.aggregate += cur
			}
		})
	return c.aggregate
}

func (c *countingMemo) wantComputed(step string, names ...string) {
	c.t.Helper()
	if len(c.computed) != len(names) {
		c.t.Errorf("%s: compute ran for %v, want %v", step, c.computed, names)
		return
	}
	for i, name := range names {
		if c.computed[i] != name {
			c.t.Errorf("%s: compute ran for %v, want %v", step, c.computed, names)
			return
		}
	}
}

func serialParallel() Parallel { return Parallel{} }

// TestNodeMemoReusesTheExactGeneration is the rule everything else rests on: a contribution is
// reused only while the node still has the generation it was computed from.
func TestNodeMemoReusesTheExactGeneration(t *testing.T) {
	c := newCountingMemo(t, serialParallel())
	nodeA, nodeB := node("a", 2), node("b", 1)

	if got := c.pass([]fwk.NodeInfo{nodeA, nodeB}); got != 3 {
		t.Fatalf("the first pass aggregated %d pods, want 3", got)
	}
	c.wantComputed("first pass", "a", "b")
	if c.memo.Recomputed != 2 || c.memo.Reused != 0 {
		t.Fatalf("the first pass recomputed %d and reused %d nodes, want 2 and 0", c.memo.Recomputed, c.memo.Reused)
	}

	if got := c.pass([]fwk.NodeInfo{nodeA, nodeB}); got != 3 {
		t.Errorf("an unchanged snapshot aggregated %d pods, want 3", got)
	}
	c.wantComputed("unchanged pass")
	if c.memo.Reused != 2 {
		t.Errorf("an unchanged snapshot reused %d contributions, want 2", c.memo.Reused)
	}

	// Moving the node to another object with the same name bumps its generation, which is what a
	// relabel does: the stored contribution no longer describes the node.
	relabelled := node("a", 5)
	if got := c.pass([]fwk.NodeInfo{relabelled, nodeB}); got != 6 {
		t.Errorf("after relabeling a node the aggregate is %d, want 6", got)
	}
	c.wantComputed("relabeled pass", "a")

	// And a snapshot can go BACKWARDS, which is why the rule is equality and not "not older than the
	// watermark": preemption's mutation session hands out clones (StartMutations) and puts the
	// originals back (EndMutations), generations included (backend/cache/snapshot.go), and
	// Snapshot.AssumePod restores the old number on purpose. Here the caller's list contains the
	// original nodeInfo again, so node a is seen with a generation older than the one the stored
	// contribution was computed from. Reusing it would keep counting the five pods of a state that
	// no longer exists.
	if got := c.pass([]fwk.NodeInfo{nodeA, nodeB}); got != 3 {
		t.Errorf("after the snapshot went back in time the aggregate is %d, want 3", got)
	}
	c.wantComputed("rolled back pass", "a")
}

// TestNodeMemoReadsKnownEmptyNodesAsEmpty is the second rule: a node that a previous pass looked at
// and stored nothing for, and that no generation has moved since, still contributes nothing. Without
// it every node of a large cluster that hosts no matching pod would be recomputed on every cycle,
// which is the whole cost the memo exists to save.
func TestNodeMemoReadsKnownEmptyNodesAsEmpty(t *testing.T) {
	c := newCountingMemo(t, serialParallel())
	empty, contributing := node("empty", 0), node("contributing", 3)

	if got := c.pass([]fwk.NodeInfo{empty, contributing}); got != 3 {
		t.Fatalf("the first pass aggregated %d pods, want 3", got)
	}
	if c.memo.Len() != 1 {
		t.Errorf("the memo stored %d contributions, want 1: a node that contributes nothing stays out of it", c.memo.Len())
	}

	// A node created after that pass has a generation above the watermark, so it has to be computed;
	// the empty node does not, even though the memo holds nothing for it.
	fresh := node("fresh", 1)
	if got := c.pass([]fwk.NodeInfo{empty, contributing, fresh}); got != 4 {
		t.Errorf("after a node joined the aggregate is %d, want 4", got)
	}
	c.wantComputed("second pass", "fresh")
}

// TestNodeMemoRecomputesANodeThatLeftAndCameBack covers the exception to that rule. A caller may
// hand in a filtered list, and a node can drop out of it and come back carrying the generation it
// left with: the memo has to remember that it contributed, instead of reading the absence of an
// entry as "contributes nothing".
func TestNodeMemoRecomputesANodeThatLeftAndCameBack(t *testing.T) {
	c := newCountingMemo(t, serialParallel())
	nodeA, nodeB := node("a", 2), node("b", 3)
	c.pass([]fwk.NodeInfo{nodeA, nodeB})

	if got := c.pass([]fwk.NodeInfo{nodeA}); got != 2 {
		t.Fatalf("with b out of the list the aggregate is %d, want 2", got)
	}
	// b comes back unchanged, and with the very generation it left with.
	if got := c.pass([]fwk.NodeInfo{nodeA, nodeB}); got != 5 {
		t.Errorf("with b back in the list the aggregate is %d, want 5: a node that left while contributing must be recomputed, not read as empty", got)
	}
	c.wantComputed("b is back", "b")
}

// TestNodeMemoForgetsEverythingWhenTooManyNodesLeft bounds the exception: a cluster that keeps
// creating and deleting nodes would grow the set of left over names without end, so past the bound
// the memo forgets everything and rebuilds it. That is exact, and costs one cold pass.
//
// The forgetting has to happen at the START of a pass, not at the end of the one that overflowed: a
// caller reads the aggregate as soon as Refresh returns, in the very cycle it filters nodes with, so
// a pass that emptied it on the way out would hand back "nothing matches anything" instead of its own
// result. For InterPodAffinity that is the difference between a correct Filter and admitting a pod
// onto a node that violates its required anti-affinity.
func TestNodeMemoForgetsEverythingWhenTooManyNodesLeft(t *testing.T) {
	c := newCountingMemo(t, serialParallel())
	c.memo.leftCap = 1
	nodeA, nodeB, nodeC := node("a", 1), node("b", 1), node("c", 1)
	if got := c.pass([]fwk.NodeInfo{nodeA, nodeB, nodeC}); got != 3 {
		t.Fatalf("the first pass aggregated %d pods, want 3", got)
	}

	// Two nodes leave, which is one more than the bound. This pass still has to report what it sees:
	// c is the only node left, and it contributes one pod.
	if got := c.pass([]fwk.NodeInfo{nodeC}); got != 1 {
		t.Errorf("the pass that overflowed the bound aggregated %d pods, want 1: what a pass hands back describes the nodes it was given, whatever it decides to forget for the next one", got)
	}
	// Nothing is recomputed: c is reused from the first pass, and a and b are pruned because this
	// pass was not given them.
	c.wantComputed("two nodes left")

	// The next pass is the one that forgets, and the rebuild is a cold pass that has to land on the
	// same numbers.
	if got := c.pass([]fwk.NodeInfo{nodeA, nodeB, nodeC}); got != 3 {
		t.Errorf("the pass after forgetting everything aggregated %d, want 3", got)
	}
	c.wantComputed("cold pass", "a", "b", "c")
	if c.memo.Len() != 3 {
		t.Errorf("after the rebuild the memo holds %d contributions, want 3", c.memo.Len())
	}
}

// TestNodeMemoFinishesACancelledParallelPass: a canceled context can leave pieces of the parallel
// part unrun, and a half computed pass must never reach the aggregate - the nodes that were skipped
// would be stored as "contributes nothing" under a generation the memo now believes it knows, so
// nothing would ever look at them again.
func TestNodeMemoFinishesACancelledParallelPass(t *testing.T) {
	// A parallelizer that runs nothing at all, which is what a canceled context leaves behind.
	c := newCountingMemo(t, Parallel{
		Until:    func(_ context.Context, _ int, _ func(index int)) {},
		MinNodes: 0,
	})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	nodes := []fwk.NodeInfo{node("a", 2), node("b", 3)}
	c.computed = nil
	c.memo.Refresh(ctx, nodes,
		func(nodeInfo fwk.NodeInfo) (int, bool) {
			count := len(nodeInfo.GetPods())
			return count, count > 0
		},
		func(old int, hadOld bool, cur int, hasCur bool) {
			if hadOld {
				c.aggregate -= old
			}
			if hasCur {
				c.aggregate += cur
			}
		})

	if c.aggregate != 5 {
		t.Errorf("a canceled pass aggregated %d pods, want 5: the pass has to be finished in the calling goroutine", c.aggregate)
	}
	if c.memo.Len() != 2 {
		t.Errorf("a canceled pass stored %d contributions, want 2", c.memo.Len())
	}
	if c.memo.ParallelPasses != 1 {
		t.Errorf("parallelPasses = %d, want 1: the pass did go to the parallelizer before it was abandoned", c.memo.ParallelPasses)
	}
}

// TestNodeMemoSkipsANodeWithoutObject: a node the snapshot has no object for contributes nothing,
// which is what the full computations do with it too.
func TestNodeMemoSkipsANodeWithoutObject(t *testing.T) {
	c := newCountingMemo(t, serialParallel())
	nodeless := framework.NewNodeInfo()
	if nodeless.Node() != nil {
		t.Fatalf("the fixture was supposed to build a NodeInfo without a Node object")
	}
	if got := c.pass([]fwk.NodeInfo{nodeless, node("a", 2)}); got != 2 {
		t.Errorf("the aggregate is %d, want 2", got)
	}
	c.wantComputed("node without an object", "a")
}

func TestMemoLRUEvictsTheLeastRecentlyUsed(t *testing.T) {
	lru := NewMemoLRU[string, *int](2)
	created := 0
	newValue := func() *int { created++; return &created }

	first := lru.GetOrCreate("a", newValue)
	lru.GetOrCreate("b", newValue)
	// Touching a has to make b the one that goes.
	if got := lru.GetOrCreate("a", newValue); got != first {
		t.Errorf("GetOrCreate(a) returned a new value, want the entry it already had")
	}
	if created != 2 {
		t.Errorf("GetOrCreate(a) created %d values in total, want 2", created)
	}
	lru.GetOrCreate("c", newValue)
	if _, ok := lru.Get("b"); ok {
		t.Errorf("b survived, want it evicted as the least recently used key")
	}
	if _, ok := lru.Get("a"); !ok {
		t.Errorf("a was evicted, want it kept: it was used after b")
	}
	if lru.Len() != 2 {
		t.Errorf("the LRU holds %d entries, want its size 2", lru.Len())
	}
}

func TestMemoLRUGetDoesNotCountAsAUse(t *testing.T) {
	lru := NewMemoLRU[string, *int](2)
	first := lru.GetOrCreate("a", func() *int { return new(int) })
	lru.GetOrCreate("b", func() *int { return new(int) })
	// Get is for inspection - the tests that assert on the memo counters use it - and must not keep a
	// alive: the caller that creates entries would otherwise find its LRU reordered by something it
	// cannot see. So the next eviction still has to take a, the least recently created key.
	if got, ok := lru.Get("a"); !ok || got != first {
		t.Fatalf("Get(a) = %v, %v, want the entry", got, ok)
	}
	lru.GetOrCreate("c", func() *int { return new(int) })
	if _, ok := lru.Get("a"); ok {
		t.Errorf("a survived an eviction after only being read, want it gone")
	}
	if _, ok := lru.Get("c"); !ok {
		t.Errorf("c is missing, want the entry GetOrCreate just made")
	}
}

func TestNewMemoLRUKeepsAtLeastOneEntry(t *testing.T) {
	// A size below one would evict the entry it just created, and every pass would be a cold one.
	for _, size := range []int{0, -1} {
		lru := NewMemoLRU[string, *int](size)
		value := new(int)
		if got := lru.GetOrCreate("a", func() *int { return value }); got != value {
			t.Fatalf("NewMemoLRU(%d).GetOrCreate returned another value", size)
		}
		if got := lru.GetOrCreate("a", func() *int { return new(int) }); got != value {
			t.Errorf("NewMemoLRU(%d) evicted the entry it just created", size)
		}
	}
}
