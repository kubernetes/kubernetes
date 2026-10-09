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

package podtopologyspread

import (
	"context"
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/component-helpers/scheduling/corev1/nodeaffinity"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework/parallelize"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodememo"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// benchSnapshot builds nodeCount nodes over eight zones, each holding podsPerNode pods the constraints
// select and one they do not, so that the matching has to look at every pod of every node. That walk is
// what the memo exists to shorten.
func benchSnapshot(b *testing.B, nodeCount, podsPerNode int) (*cache.Snapshot, []fwk.NodeInfo) {
	b.Helper()
	nodes := make([]*v1.Node, 0, nodeCount)
	pods := make([]*v1.Pod, 0, nodeCount*(podsPerNode+1))
	for i := range nodeCount {
		name := fmt.Sprintf("node-%d", i)
		nodes = append(nodes, st.MakeNode().Name(name).
			Label("zone", fmt.Sprintf("z%d", i%8)).Label(v1.LabelHostname, name).Obj())
		for j := range podsPerNode {
			pods = append(pods, st.MakePod().Namespace("bench").Name(fmt.Sprintf("%s-p%d", name, j)).
				UID(fmt.Sprintf("%s-p%d", name, j)).Label("app", "bench").Node(name).Obj())
		}
		pods = append(pods, st.MakePod().Namespace("bench").Name(name+"-other").UID(name+"-other").
			Label("app", "other").Node(name).Obj())
	}
	snapshot := cache.NewSnapshot(pods, nodes)
	infos, err := snapshot.NodeInfos().List()
	if err != nil {
		b.Fatal(err)
	}
	return snapshot, infos
}

// benchIncoming is a pod of a workload whose soft constraints select the pods labeled app=value: one
// zone scoped, which is the one PreScore counts, and one hostname scoped, which it does not - Score
// counts a node's own pods itself - and which is therefore the constraint countNodePods has to leave
// alone. Two values are two pod shapes: different constraints, same namespace, and, being pods of two
// Deployments, nothing in their labels to tell them apart either.
func benchIncoming(b *testing.B, value string) *v1.Pod {
	b.Helper()
	selector := &metav1.LabelSelector{MatchLabels: map[string]string{"app": value}}
	return st.MakePod().Namespace("bench").Name("incoming-"+value).UID("incoming-"+value).
		Label("app", "bench").
		SpreadConstraint(5, "zone", v1.ScheduleAnyway, selector, nil, nil, nil, nil).
		SpreadConstraint(1, v1.LabelHostname, v1.ScheduleAnyway, selector, nil, nil, nil, nil).Obj()
}

// benchPlugin builds a plugin by hand rather than through New: PreScore is driven below through
// initPreScoreState and the two counting functions directly, which is what a benchmark of the counting
// wants, and New would need an informer factory and a framework to produce the same thing.
func benchPlugin(snapshot *cache.Snapshot, memo bool) *PodTopologySpread {
	pl := &PodTopologySpread{
		parallelizer: parallelize.NewParallelizer(parallelize.DefaultParallelism),
		sharedLister: snapshot,
	}
	if memo {
		pl.scoringMemo = nodememo.NewMemoLRU[string, *spreadEntry](nodememo.DefaultLRUSize)
	}
	return pl
}

// benchState runs the half of PreScore that builds the per cycle maps, which both paths then fill. It
// is inside the measured loop because PreScore pays it on every cycle whichever path it takes, so
// leaving it out would flatter the memo - whose whole saving is in the counting - by the same amount on
// both sides of a subtraction that is not being made.
func benchState(b *testing.B, pl *PodTopologySpread, pod *v1.Pod, nodes []fwk.NodeInfo) *preScoreState {
	b.Helper()
	state := &preScoreState{IgnoredNodes: sets.New[string]()}
	if err := pl.initPreScoreState(state, pod, nodes, true); err != nil {
		b.Fatal(err)
	}
	if len(state.Constraints) == 0 {
		b.Fatal("the fixture produced no soft constraint, so there is nothing to count")
	}
	return state
}

// benchTouchOneNode moves one node, the way a rolling update does: the previous replica was just assumed
// onto it, so its generation moved and its contribution has to be recomputed while every other node's is
// reused. This is the steady state the memo is for.
func benchTouchOneNode(b *testing.B, snapshot *cache.Snapshot, i, nodeCount int) {
	b.Helper()
	nodeInfo, err := snapshot.NodeInfos().Get(fmt.Sprintf("node-%d", i%nodeCount))
	if err != nil {
		b.Fatal(err)
	}
	nodeInfo.SetNode(nodeInfo.Node())
}

// BenchmarkPreScoreCountsFullVsMemo prices the counting half of PreScore at three cluster sizes: what
// the community computation costs per cycle, what the first pass of a memo costs, and what every pass
// after it costs when one node moved.
func BenchmarkPreScoreCountsFullVsMemo(b *testing.B) {
	ctx := context.Background()
	const podsPerNode = 20
	for _, nodeCount := range []int{500, 2000, 5000} {
		snapshot, nodes := benchSnapshot(b, nodeCount, podsPerNode)
		memoPlugin := benchPlugin(snapshot, true)
		fullPlugin := benchPlugin(snapshot, false)
		pod := benchIncoming(b, "bench")

		b.Run(fmt.Sprintf("nodes=%d/pods=%d", nodeCount, nodeCount*(podsPerNode+1)), func(b *testing.B) {
			b.Run("full", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					state := benchState(b, fullPlugin, pod, nodes)
					fullPlugin.addTopologyCountsFull(ctx, pod, state, nodes, true,
						nodeaffinity.GetRequiredNodeAffinity(pod))
				}
			})
			b.Run("memo/cold", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					// A fresh memo per iteration, so every pass is a cold one: that is what the first pod of
					// a workload costs, and the pass the memo has to be no dearer than the full computation
					// on even at its worst.
					cold := benchPlugin(snapshot, true)
					state := benchState(b, cold, pod, nodes)
					cold.addTopologyCountsByMemo(ctx, pod, state, nodes, true, nodeaffinity.GetRequiredNodeAffinity(pod))
				}
			})
			b.Run("memo/steady", func(b *testing.B) {
				requiredNodeAffinity := nodeaffinity.GetRequiredNodeAffinity(pod)
				memoPlugin.addTopologyCountsByMemo(ctx, pod, benchState(b, memoPlugin, pod, nodes), nodes, true,
					requiredNodeAffinity)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					benchTouchOneNode(b, snapshot, i, nodeCount)
					memoPlugin.addTopologyCountsByMemo(ctx, pod, benchState(b, memoPlugin, pod, nodes), nodes, true,
						requiredNodeAffinity)
				}
			})
		})
	}
}

// BenchmarkPreScoreCountsAlternatingShapes prices two workloads being scored in turn, which is what
// decides whether the memo key is right. Each shape has to keep its own entry: sharing one key means
// invalidating it on every switch, and a cold pass costs more than the full computation it replaces. The
// two shapes here have the same namespace and the same labels - only their constraints differ - so a key
// built from the pod's labels would not tell them apart.
func BenchmarkPreScoreCountsAlternatingShapes(b *testing.B) {
	ctx := context.Background()
	const nodeCount, podsPerNode = 2000, 20
	snapshot, nodes := benchSnapshot(b, nodeCount, podsPerNode)
	memoPlugin := benchPlugin(snapshot, true)
	fullPlugin := benchPlugin(snapshot, false)
	shapes := []*v1.Pod{benchIncoming(b, "bench"), benchIncoming(b, "other")}

	b.Run(fmt.Sprintf("nodes=%d/pods=%d", nodeCount, nodeCount*(podsPerNode+1)), func(b *testing.B) {
		b.Run("full", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				pod := shapes[i%2]
				state := benchState(b, fullPlugin, pod, nodes)
				fullPlugin.addTopologyCountsFull(ctx, pod, state, nodes, true,
					nodeaffinity.GetRequiredNodeAffinity(pod))
			}
		})
		b.Run("memo", func(b *testing.B) {
			for _, pod := range shapes {
				memoPlugin.addTopologyCountsByMemo(ctx, pod, benchState(b, memoPlugin, pod, nodes), nodes, true,
					nodeaffinity.GetRequiredNodeAffinity(pod))
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchTouchOneNode(b, snapshot, i, nodeCount)
				pod := shapes[i%2]
				memoPlugin.addTopologyCountsByMemo(ctx, pod, benchState(b, memoPlugin, pod, nodes), nodes, true,
					nodeaffinity.GetRequiredNodeAffinity(pod))
			}
		})
	})
}
