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

package dynamicresources

import (
	"context"
	"fmt"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	schedulingv1beta1 "k8s.io/api/scheduling/v1beta1"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/events"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpreemption"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/preemption"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
)

type podsFilter struct {
	name   string
	reject func(pods podNames) bool
}

type podNames map[string]bool

func (pl *podsFilter) Name() string {
	return pl.name
}

func (pl *podsFilter) Filter(_ context.Context, _ fwk.CycleState, _ *v1.Pod, nodeInfo fwk.NodeInfo) *fwk.Status {
	pods := podNames{}
	for _, podInfo := range nodeInfo.GetPods() {
		pods[podInfo.GetPod().Name] = true
	}
	if pl.reject != nil && pl.reject(pods) {
		return fwk.NewStatus(fwk.Unschedulable, "rejected by "+pl.name)
	}
	return nil
}

type dryRunContext struct {
	testCtx   *testContext
	fh        framework.Framework
	pl        *defaultpreemption.DefaultPreemption
	snapshot  *internalcache.Snapshot
	preemptor *v1.Pod
	state     fwk.CycleState
}

// setupDryRun wires DynamicResources between two mock Filter plugins ("Before"
// and "After", standing in for plugins like NodeResourcesFit without creating
// an import cycle) and runs PreFilter for preemptor.
func setupDryRun(tCtx ktesting.TContext, testCtx *testContext, preemptor *v1.Pod, pods, nominated []*v1.Pod, podGroups []*schedulingv1beta1.PodGroup, before, after func(pods podNames) bool) *dryRunContext {
	tCtx.Helper()
	genericWorkload := len(podGroups) > 0
	featuregatetesting.SetFeatureGateDuringTest(tCtx, utilfeature.DefaultFeatureGate, features.GenericWorkload, genericWorkload)
	newPlugin := func(ctx context.Context, _ apiruntime.Object, fh fwk.Handle) (fwk.Plugin, error) {
		return New(ctx, &config.DynamicResourcesArgs{}, fh, feature.Features{EnableDRAPreemption: true, EnableGenericWorkload: genericWorkload})
	}
	filterPlugin := func(pl *podsFilter) tf.RegisterPluginFunc {
		return tf.RegisterFilterPlugin(pl.name, func(context.Context, apiruntime.Object, fwk.Handle) (fwk.Plugin, error) {
			return pl, nil
		})
	}
	nodes := []*v1.Node{workerNode, workerNode2}
	podGroupManager := internalcache.New(tCtx, nil, genericWorkload, false)
	for _, podGroup := range podGroups {
		podGroupManager.AddGenericPodGroup(fwk.NewGenericPodGroup(podGroup))
	}
	snapshot := internalcache.NewTestSnapshotWithPodGroups(pods, nodes, podGroups, nil)
	features := feature.Features{EnableGenericWorkload: genericWorkload}
	// PodNominator ignores nominated pods that are not also present in the scheduling queue.
	nominator := internalqueue.NewTestQueueWithInformerFactory(tCtx, (&queuesort.PrioritySort{}).Less, testCtx.informerFactory)
	for _, pod := range nominated {
		nominator.Add(tCtx, pod)
		podInfo, err := framework.NewPodInfo(pod)
		tCtx.ExpectNoError(err)
		nominator.AddNominatedPod(tCtx.Logger(), podInfo, &fwk.NominatingInfo{NominatingMode: fwk.ModeOverride, NominatedNodeName: nodeName})
	}
	fh, err := tf.NewFramework(tCtx, []tf.RegisterPluginFunc{
		tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		filterPlugin(&podsFilter{name: "Before", reject: before}),
		tf.RegisterPluginAsExtensions(Name, newPlugin, "PreFilter", "Filter"),
		filterPlugin(&podsFilter{name: "After", reject: after}),
	}, "",
		runtime.WithClientSet(testCtx.client),
		runtime.WithInformerFactory(testCtx.informerFactory),
		runtime.WithEventRecorder(&events.FakeRecorder{}),
		runtime.WithSharedDRAManager(testCtx.draManager),
		runtime.WithPodGroupManager(podGroupManager),
		runtime.WithPodNominator(nominator),
		runtime.WithSnapshotSharedLister(snapshot),
		runtime.WithMutableSnapshotLister(snapshot),
		runtime.WithPreemptionManager(func(fh fwk.Handle) fwk.PreemptionManager {
			return preemption.NewPreemptionManager(fh, features)
		}),
	)
	tCtx.ExpectNoError(err, "create framework")
	tCtx.Cleanup(func() { runtime.WaitForShutdown(fh) })
	pl, err := defaultpreemption.New(tCtx, &config.DefaultPreemptionArgs{MinCandidateNodesPercentage: 10, MinCandidateNodesAbsolute: 100}, fh, features)
	tCtx.ExpectNoError(err, "create DefaultPreemption")

	state := framework.NewCycleState()
	_, status, _ := fh.RunPreFilterPlugins(tCtx, state, preemptor)
	require.True(tCtx, status.IsSuccess(), "PreFilter: %v", status)
	return &dryRunContext{testCtx: testCtx, fh: fh, pl: pl, snapshot: snapshot, preemptor: preemptor, state: state}
}

func (c *dryRunContext) selectVictims(tCtx ktesting.TContext, nodeName string) []string {
	tCtx.Helper()
	nodeInfo, err := c.snapshot.NodeInfos().Get(nodeName)
	tCtx.ExpectNoError(err, "get node")
	possibleVictims, err := c.pl.Evaluator.GetVictimsOnNode(tCtx, nodeInfo)
	tCtx.ExpectNoError(err, "get possible victims")
	pods, _, status := c.pl.SelectVictimsOnNode(tCtx, c.state.Clone(), c.preemptor, nodeInfo.Snapshot(), possibleVictims, nil)
	require.True(tCtx, status.IsSuccess(), "select victims on %s: %v", nodeName, status)
	var names []string
	for _, pod := range pods {
		names = append(names, pod.Name)
	}
	slices.Sort(names)
	return names
}

func (c *dryRunContext) prepare(tCtx ktesting.TContext, nodeName string) (devices, released []string) {
	tCtx.Helper()
	hold, _ := c.testCtx.p.PrepareNomination(tCtx, c.state, c.preemptor, nodeName).(*allocation)
	require.NotNil(tCtx, hold, "hold")
	require.Equal(tCtx, nodeName, hold.nodeName, "node of the hold")
	for _, held := range hold.claims {
		for _, result := range held.allocation.Devices.Results {
			devices = append(devices, result.Pool+"/"+result.Device)
		}
	}
	for _, claim := range hold.released {
		released = append(released, claim.claim.Name)
	}
	slices.Sort(released)
	return devices, released
}

func TestPrepareNominationWithDefaultPreemption(t *testing.T) {
	testPrepareNominationWithDefaultPreemption(ktesting.Init(t))
}

// testPrepareNominationWithDefaultPreemption verifies that PrepareNomination
// captures the allocation from the final victim set chosen by
// DefaultPreemption's dry run, even when intermediate reprieve attempts ran
// Filter later or when another Filter plugin rejected a candidate state.
func testPrepareNominationWithDefaultPreemption(tCtx ktesting.TContext) {
	preemptor := preemptionPod("preemptor", "preemptor-claim").Priority(100).Obj()
	v1Pod := preemptionPod("v1", "v1-claim").Node(nodeName).Priority(10).Obj()
	v2Pod := preemptionPod("v2", "v2-claim").Node(nodeName).Priority(20).Obj()
	// v3 has no claims (simulating a pod blocking CPU/memory rather than DRA).
	v3Pod := preemptionPod("v3").Node(nodeName).Priority(30).Obj()
	holder := preemptionPod("holder").Priority(200).Obj()
	setup := func(tCtx ktesting.TContext, numDevices int) *testContext {
		slice := st.MakeResourceSlice(nodeName, driver)
		for i := range numDevices {
			slice = slice.Device(fmt.Sprintf("gpu-%d", i))
		}
		return setupVictims(tCtx, []*resourceapi.ResourceClaim{
			preemptionClaim("preemptor-claim").Obj(),
			reserve(preemptionClaim("v1-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), v1Pod),
			reserve(preemptionClaim("v2-claim").Allocation(preemptionAllocation("gpu-1")).Obj(), v2Pod),
		}, []apiruntime.Object{slice.Obj(), preemptor, holder, v1Pod, v2Pod, v3Pod})
	}
	pods := []*v1.Pod{v1Pod, v2Pod, v3Pod}

	tCtx.Run("plugin-before-rejects", func(tCtx ktesting.TContext) {
		// "Before" rejects whenever v3 coexists with v1 or v2, so v3 is
		// reprieved while v1 and v2 remain victims.
		testCtx := setup(tCtx, 2)
		c := setupDryRun(tCtx, testCtx, preemptor, pods, nil, nil, func(pods podNames) bool {
			return pods["v3"] && (pods["v1"] || pods["v2"])
		}, nil)
		require.Equal(tCtx, []string{"v1", "v2"}, c.selectVictims(tCtx, nodeName), "victims")
		devices, released := c.prepare(tCtx, nodeName)
		require.Equal(tCtx, []string{nodeName + "/gpu-0"}, devices, "devices of the final state")
		require.Equal(tCtx, []string{"v1-claim", "v2-claim"}, released, "released claims of the final state")
	})

	tCtx.Run("plugin-after-rejects", func(tCtx ktesting.TContext) {
		// DynamicResources succeeds when v2 or v1 is added back (leaving gpu-1
		// as the most recent Filter result), but "After" rejects both reprieves,
		// so the final state still removes both v1 and v2. Nominating holder
		// also exercises pass-1-only rejections.
		testCtx := setup(tCtx, 2)
		c := setupDryRun(tCtx, testCtx, preemptor, pods, []*v1.Pod{holder}, nil, nil, func(pods podNames) bool {
			return pods["v1"] || pods["v2"]
		})
		require.Equal(tCtx, []string{"v1", "v2"}, c.selectVictims(tCtx, nodeName), "victims")
		devices, released := c.prepare(tCtx, nodeName)
		require.Equal(tCtx, []string{nodeName + "/gpu-0"}, devices, "devices of the final state")
		require.Equal(tCtx, []string{"v1-claim", "v2-claim"}, released, "released claims of the final state")
	})

	tCtx.Run("last-add-back-stays", func(tCtx ktesting.TContext) {
		// gpu-2 is already free and only v3 blocks scheduling, so both v2 and
		// v1 are reprieved and PrepareNomination must use the state after
		// adding v1 back.
		testCtx := setup(tCtx, 3)
		c := setupDryRun(tCtx, testCtx, preemptor, pods, nil, nil, func(pods podNames) bool {
			return pods["v3"]
		}, nil)
		require.Equal(tCtx, []string{"v3"}, c.selectVictims(tCtx, nodeName), "victims")
		devices, released := c.prepare(tCtx, nodeName)
		require.Equal(tCtx, []string{nodeName + "/gpu-2"}, devices, "devices of the final state")
		require.Empty(tCtx, released, "released claims of the final state")
	})

	tCtx.Run("podgroup-victim-on-two-nodes", func(tCtx ktesting.TContext) {
		// PodGroup {a (on node2), g (on node)} is evicted as a unit. Removing
		// pod a via node's dry run must not overwrite node2's own dry-run final
		// state, regardless of execution order. On node2, reprieving b is
		// rejected by "After" while reprieving the PodGroup succeeds on b's gpu-1.
		podGroup := st.MakePodGroup().Name("group").Namespace(namespace).UID("group").Priority(10).DisruptionModeAll().Obj()
		podA := preemptionPod("a", "a-claim").Node(node2Name).Priority(10).PodGroupName("group").Obj()
		podG := preemptionPod("g", "g-claim").Node(nodeName).Priority(10).PodGroupName("group").Obj()
		podB := preemptionPod("b", "b-claim").Node(node2Name).Priority(20).Obj()
		for name, order := range map[string][]string{"node-first": {nodeName, node2Name}, "node2-first": {node2Name, nodeName}} {
			tCtx.Run(name, func(tCtx ktesting.TContext) {
				testCtx := setupVictims(tCtx, []*resourceapi.ResourceClaim{
					preemptionClaim("preemptor-claim").Obj(),
					reserve(preemptionClaim("a-claim").Allocation(poolAllocation(node2Name, "gpu-0")).Obj(), podA),
					reserve(preemptionClaim("b-claim").Allocation(poolAllocation(node2Name, "gpu-1")).Obj(), podB),
					reserve(preemptionClaim("g-claim").Allocation(preemptionAllocation("gpu-0")).Obj(), podG),
				}, []apiruntime.Object{
					st.MakeResourceSlice(nodeName, driver).Device("gpu-0").Obj(),
					st.MakeResourceSlice(node2Name, driver).Device("gpu-0").Device("gpu-1").Obj(),
					preemptor, podA, podB, podG,
				})
				c := setupDryRun(tCtx, testCtx, preemptor, []*v1.Pod{podA, podB, podG}, nil, []*schedulingv1beta1.PodGroup{podGroup}, nil, func(pods podNames) bool {
					return pods["b"]
				})
				wantVictims := map[string][]string{nodeName: {"a", "g"}, node2Name: {"b"}}
				for _, nodeName := range order {
					require.Equal(tCtx, wantVictims[nodeName], c.selectVictims(tCtx, nodeName), "victims on %s", nodeName)
				}
				devices, released := c.prepare(tCtx, node2Name)
				require.Equal(tCtx, []string{node2Name + "/gpu-1"}, devices, "devices of the final state of node2")
				require.Equal(tCtx, []string{"b-claim"}, released, "released claims of the final state of node2")
				devices, released = c.prepare(tCtx, nodeName)
				require.Equal(tCtx, []string{nodeName + "/gpu-0"}, devices, "devices of the final state of node")
				require.Equal(tCtx, []string{"g-claim"}, released, "released claims of the final state of node")
			})
		}
	})
}
