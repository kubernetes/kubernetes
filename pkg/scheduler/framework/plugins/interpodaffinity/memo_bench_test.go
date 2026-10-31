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

package interpodaffinity

import (
	"context"
	"fmt"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/backend/cache"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodememo"
	plugintesting "k8s.io/kubernetes/pkg/scheduler/framework/plugins/testing"
	schedruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

// benchSnapshot builds nodeCount nodes over eight zones, each holding podsPerNode plain pods that the
// incoming pods' terms select, one pod with a required zone scoped anti-affinity term and one with a
// required affinity term. The last two are what put every node in the snapshot's filtered lists -
// HavePodsWithRequiredAntiAffinityList and HavePodsWithAffinityList - so that none of the four walks is
// short-circuited away on a node, and the matching has to look at every pod of every node. That walk is
// what the memo exists to shorten.
func benchSnapshot(b *testing.B, nodeCount, podsPerNode int) (*cache.Snapshot, []fwk.NodeInfo) {
	b.Helper()
	nodes := make([]*v1.Node, 0, nodeCount)
	pods := make([]*v1.Pod, 0, nodeCount*(podsPerNode+2))
	for i := range nodeCount {
		name := fmt.Sprintf("node-%d", i)
		nodes = append(nodes, st.MakeNode().Name(name).
			Label("zone", fmt.Sprintf("z%d", i%8)).Label(v1.LabelHostname, name).Obj())
		for j := range podsPerNode {
			pods = append(pods, st.MakePod().Namespace("bench").Name(fmt.Sprintf("%s-p%d", name, j)).
				UID(fmt.Sprintf("%s-p%d", name, j)).Label("app", "bench").Node(name).Obj())
		}
		pods = append(pods,
			st.MakePod().Namespace("bench").Name(name+"-anti").UID(name+"-anti").
				Label("app", "bench").Node(name).
				PodAntiAffinityExists("app", "zone", st.PodAntiAffinityWithRequiredReq).Obj(),
			st.MakePod().Namespace("bench").Name(name+"-aff").UID(name+"-aff").
				Label("app", "bench").Node(name).
				PodAffinityExists("app", "zone", st.PodAffinityWithRequiredReq).Obj())
	}
	snapshot := cache.NewSnapshot(pods, nodes)
	infos, err := snapshot.NodeInfos().List()
	if err != nil {
		b.Fatal(err)
	}
	return snapshot, infos
}

// benchIncoming is a pod of a workload whose terms select the pods labeled app=value: a required zone
// scoped affinity term, a required zone scoped anti-affinity term, and a preferred zone scoped
// anti-affinity term, so that PreFilter's two cluster wide walks and PreScore's all have something to
// match. Two values are two pod shapes - different terms, same namespace, and, being pods of two
// Deployments, nothing in their labels to tell them apart either.
func benchIncoming(b *testing.B, value string) *v1.Pod {
	b.Helper()
	selector := &metav1.LabelSelector{MatchLabels: map[string]string{"app": value}}
	return st.MakePod().Namespace("bench").Name("incoming-"+value).UID("incoming-"+value).
		Label("app", "bench").
		PodAffinity("zone", selector, st.PodAffinityWithRequiredReq).
		PodAntiAffinity("zone", selector, st.PodAntiAffinityWithRequiredReq).
		PodAntiAffinity("zone", selector, st.PodAntiAffinityWithPreferredReq).Obj()
}

// benchIncomingHostScoped is a pod whose only required affinity term is hostname scoped, which is the
// shape that makes PreFilter ask for the global count of the pods matching it - the third walk, and the
// one a pod with any cluster wide term never reaches.
func benchIncomingHostScoped(b *testing.B) *v1.Pod {
	b.Helper()
	return st.MakePod().Namespace("bench").Name("incoming-host").UID("incoming-host").
		Label("app", "bench").
		PodAffinityExists("app", v1.LabelHostname, st.PodAffinityWithRequiredReq).Obj()
}

// benchPlugins builds the plugin New would build, and a copy of it with every memo taken out, which
// therefore takes the full computation. Both are driven through the real extension points, so what is
// measured is what a scheduling cycle pays.
func benchPlugins(b *testing.B, snapshot *cache.Snapshot, fastPath bool) (memo, full *InterPodAffinity) {
	b.Helper()
	ctx := context.Background()
	p := plugintesting.SetupPluginWithInformers(ctx, b,
		schedruntime.FactoryAdapter(feature.Features{EnableInterPodAffinityHostnameFastPath: fastPath}, New),
		&config.InterPodAffinityArgs{}, snapshot, nil)
	memoPlugin := p.(*InterPodAffinity)
	fullPlugin := *memoPlugin
	fullPlugin.filteringExistingMemo = nil
	fullPlugin.filteringIncomingMemo = nil
	fullPlugin.filteringHostScopedAffinityMemo = nil
	fullPlugin.scoringMemo = nil
	return memoPlugin, &fullPlugin
}

// benchCold returns a copy of pl with empty memos, so that the next pass is a cold one and every node
// has to be computed. That is what the first pod of a workload costs, and it is the pass the memo has to
// be no dearer than the full computation on even at its worst: a cold pass walks every node the full
// computation walks, and then stores what it found.
func benchCold(pl *InterPodAffinity) *InterPodAffinity {
	cold := *pl
	cold.filteringExistingMemo = nodememo.NewMemoLRU[string, *existingCountsEntry](nodememo.DefaultLRUSize)
	cold.filteringIncomingMemo = nodememo.NewMemoLRU[string, *incomingCountsEntry](nodememo.DefaultLRUSize)
	cold.filteringHostScopedAffinityMemo = nodememo.NewMemoLRU[string, *hostScopedAffinityEntry](nodememo.DefaultLRUSize)
	cold.scoringMemo = nodememo.NewMemoLRU[string, *scoringEntry](nodememo.DefaultLRUSize)
	return &cold
}

// benchTouchOneNode moves one node, the way a rolling update does: the previous replica was just
// assumed onto it, so its generation moved and its contribution has to be recomputed while every other
// node's is reused. This is the steady state the memo is for.
func benchTouchOneNode(b *testing.B, snapshot *cache.Snapshot, i int, nodeCount int) {
	b.Helper()
	nodeInfo, err := snapshot.NodeInfos().Get(fmt.Sprintf("node-%d", i%nodeCount))
	if err != nil {
		b.Fatal(err)
	}
	nodeInfo.SetNode(nodeInfo.Node())
}

// BenchmarkInterPodAffinityFullVsMemo prices the four cluster wide walks - PreFilter's three and
// PreScore's - at three cluster sizes: what the community computation costs per cycle, what the first
// pass of a memo costs, and what every pass after it costs when one node moved.
func BenchmarkInterPodAffinityFullVsMemo(b *testing.B) {
	const podsPerNode = 20
	for _, nodeCount := range []int{500, 2000, 5000} {
		snapshot, nodes := benchSnapshot(b, nodeCount, podsPerNode)
		memoPlugin, fullPlugin := benchPlugins(b, snapshot, false)
		pod := benchIncoming(b, "bench")

		b.Run(fmt.Sprintf("nodes=%d/pods=%d", nodeCount, nodeCount*(podsPerNode+2)), func(b *testing.B) {
			for _, tc := range []struct {
				name string
				run  func(b *testing.B, pl *InterPodAffinity)
			}{
				{"PreFilter", func(b *testing.B, pl *InterPodAffinity) {
					state := framework.NewCycleState()
					if _, status := pl.PreFilter(context.Background(), state, pod, nodes); !status.IsSuccess() {
						b.Fatal(status)
					}
				}},
				{"PreScore", func(b *testing.B, pl *InterPodAffinity) {
					state := framework.NewCycleState()
					if status := pl.PreScore(context.Background(), state, pod, nodes); !status.IsSuccess() {
						b.Fatal(status)
					}
				}},
			} {
				b.Run(tc.name+"/full", func(b *testing.B) {
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						tc.run(b, fullPlugin)
					}
				})
				b.Run(tc.name+"/memo/cold", func(b *testing.B) {
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						tc.run(b, benchCold(memoPlugin))
					}
				})
				b.Run(tc.name+"/memo/steady", func(b *testing.B) {
					// Warm the entry, so that what is measured is a pass over a snapshot one node of which
					// moved, not the pass that built it.
					tc.run(b, memoPlugin)
					b.ReportAllocs()
					b.ResetTimer()
					for i := 0; i < b.N; i++ {
						benchTouchOneNode(b, snapshot, i, nodeCount)
						tc.run(b, memoPlugin)
					}
				})
			}
		})
	}
}

// BenchmarkInterPodAffinityHostScopedFullVsMemo is the third walk on its own: a pod whose only required
// affinity term is hostname scoped, for which PreFilter - with InterPodAffinityHostnameFastPath on, which
// is what splits host scoped terms off in the first place - counts the matching pods in the whole cluster
// instead of building a per topology domain map that a hostname scoped term would give one entry per node
// for. The gate has to be set globally and not only on the plugin: the snapshot decides which of its
// filtered lists to populate while it is being built.
func BenchmarkInterPodAffinityHostScopedFullVsMemo(b *testing.B) {
	featuregatetesting.SetFeatureGateDuringTest(b, utilfeature.DefaultFeatureGate,
		features.InterPodAffinityHostnameFastPath, true)
	const podsPerNode = 20
	for _, nodeCount := range []int{500, 2000, 5000} {
		snapshot, nodes := benchSnapshot(b, nodeCount, podsPerNode)
		memoPlugin, fullPlugin := benchPlugins(b, snapshot, true)
		pod := benchIncomingHostScoped(b)

		b.Run(fmt.Sprintf("nodes=%d/pods=%d", nodeCount, nodeCount*(podsPerNode+2)), func(b *testing.B) {
			run := func(pl *InterPodAffinity) {
				state := framework.NewCycleState()
				if _, status := pl.PreFilter(context.Background(), state, pod, nodes); !status.IsSuccess() {
					b.Fatal(status)
				}
			}
			b.Run("full", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					run(fullPlugin)
				}
			})
			b.Run("memo/cold", func(b *testing.B) {
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					run(benchCold(memoPlugin))
				}
			})
			b.Run("memo/steady", func(b *testing.B) {
				run(memoPlugin)
				b.ReportAllocs()
				b.ResetTimer()
				for i := 0; i < b.N; i++ {
					benchTouchOneNode(b, snapshot, i, nodeCount)
					run(memoPlugin)
				}
			})
		})
	}
}

// BenchmarkInterPodAffinityAlternatingShapes prices two workloads being scheduled in turn, which is what
// decides whether the memo key is right. Each shape has to keep its own entry: sharing one key means
// invalidating it on every switch, and a cold pass costs more than the full computation it replaces, so
// a coarser key would leave the memo slower than having none. The two shapes here have the same
// namespace and the same labels - only their terms differ - so a key built from the pod's labels would
// not tell them apart.
func BenchmarkInterPodAffinityAlternatingShapes(b *testing.B) {
	const nodeCount, podsPerNode = 2000, 20
	snapshot, nodes := benchSnapshot(b, nodeCount, podsPerNode)
	memoPlugin, fullPlugin := benchPlugins(b, snapshot, false)
	shapes := []*v1.Pod{benchIncoming(b, "bench"), benchIncoming(b, "other")}

	b.Run(fmt.Sprintf("nodes=%d/pods=%d", nodeCount, nodeCount*(podsPerNode+2)), func(b *testing.B) {
		run := func(pl *InterPodAffinity, pod *v1.Pod) {
			state := framework.NewCycleState()
			if _, status := pl.PreFilter(context.Background(), state, pod, nodes); !status.IsSuccess() {
				b.Fatal(status)
			}
		}
		b.Run("full", func(b *testing.B) {
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				run(fullPlugin, shapes[i%2])
			}
		})
		b.Run("memo", func(b *testing.B) {
			for _, pod := range shapes {
				run(memoPlugin, pod)
			}
			b.ReportAllocs()
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				benchTouchOneNode(b, snapshot, i, nodeCount)
				run(memoPlugin, shapes[i%2])
			}
		})
	})
}
