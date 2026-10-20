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

package scheduler

import (
	"context"
	"reflect"
	"sync/atomic"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/events"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/dynamic-resource-allocation/structured"
	"k8s.io/dynamic-resource-allocation/structured/schedulerapi"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	schedulerconfig "k8s.io/kubernetes/pkg/scheduler/apis/config"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/utils/ptr"
)

type sharingAffinitySchedulerActivator struct {
	fwk.PodActivator
	uid   types.UID
	calls atomic.Int32
}

func (a *sharingAffinitySchedulerActivator) Activate(logger klog.Logger, pods map[string]*v1.Pod) {
	for _, pod := range pods {
		if pod.UID == a.uid {
			a.calls.Add(1)
			break
		}
	}
	a.PodActivator.Activate(logger, pods)
}

func TestSharingAffinitySchedulerLevel(t *testing.T) {
	featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, featuregatetesting.FeatureOverrides{
		features.DRAConsumableCapacity:  true,
		features.DRASharingAffinity:     true,
		features.SchedulerAsyncAPICalls: false,
	})

	for _, tc := range []struct {
		name        string
		healthyNode bool
		wantNode    string
	}{
		{name: "excluded-node-first-healthy-node-second", healthyNode: true, wantNode: "node-2"},
		{name: "incomplete-pool-slice-completion-real-queue", wantNode: "node-1"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithTimeout(ctx, 30*time.Second)
			defer cancel()
			const driver = "sharing.example.com"
			const namespace = "default"
			const schedulerName = "sharing-affinity"
			incompletePool := schedulerapi.PoolID{Driver: driver, Pool: "pool-1"}
			ownerDevice := structured.MakeDeviceID(driver, "pool-1", "device")
			claim := &resourceapi.ResourceClaim{
				ObjectMeta: metav1.ObjectMeta{Name: "new-claim", Namespace: namespace, UID: "new-claim", ResourceVersion: "1"},
				Spec: resourceapi.ResourceClaimSpec{Devices: resourceapi.DeviceClaim{
					Requests: []resourceapi.DeviceRequest{{
						Name: "req",
						Exactly: &resourceapi.ExactDeviceRequest{
							DeviceClassName: "network", AllocationMode: resourceapi.DeviceAllocationModeExactCount, Count: 1,
						},
					}},
					Config: []resourceapi.DeviceClaimConfiguration{{
						Requests: []string{"req"},
						DeviceConfiguration: resourceapi.DeviceConfiguration{Opaque: &resourceapi.OpaqueDeviceConfiguration{
							Driver:     driver,
							Parameters: runtime.RawExtension{Raw: []byte(`{"apiVersion":"config.example.com/v1","kind":"Network","subnet":"blue"}`)},
						}},
					}},
				}},
			}
			// An allocated owner without a Pod must still be reconstructed.
			owner := claim.DeepCopy()
			owner.Name, owner.UID = "owner", "owner"
			owner.Status.Allocation = &resourceapi.AllocationResult{
				Devices: resourceapi.DeviceAllocationResult{Results: []resourceapi.DeviceRequestAllocationResult{{
					Request: "req", Driver: driver, Pool: "pool-1", Device: "device", ShareID: ptr.To(types.UID("owner-share")),
				}}},
			}
			slice := &resourceapi.ResourceSlice{
				ObjectMeta: metav1.ObjectMeta{Name: "pool-1-devices", UID: "pool-1-devices", ResourceVersion: "1"},
				Spec: resourceapi.ResourceSliceSpec{
					Driver: driver, NodeName: ptr.To("node-1"),
					Pool: resourceapi.ResourcePool{Name: "pool-1", Generation: 1, ResourceSliceCount: 2},
					SharingAffinity: []resourceapi.SharingAffinityExtractor{{
						Name: "subnet", APIGroup: "config.example.com", Kind: "Network", Expression: `parameters.subnet`,
					}},
					Devices: []resourceapi.Device{{Name: "device", AllowMultipleAllocations: ptr.To(true)}},
				},
			}
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "consumer", Namespace: namespace, UID: "consumer", ResourceVersion: "1"},
				Spec: v1.PodSpec{
					SchedulerName:  schedulerName,
					Containers:     []v1.Container{{Name: "container", Image: "image"}},
					ResourceClaims: []v1.PodResourceClaim{{Name: "resource", ResourceClaimName: ptr.To(claim.Name)}},
				},
			}
			objects := []runtime.Object{claim, owner, slice, &resourceapi.DeviceClass{ObjectMeta: metav1.ObjectMeta{Name: "network"}}}
			if tc.healthyNode {
				healthy := slice.DeepCopy()
				healthy.Name, healthy.UID = "pool-2-devices", "pool-2-devices"
				healthy.Spec.NodeName = ptr.To("node-2")
				healthy.Spec.Pool.Name, healthy.Spec.Pool.ResourceSliceCount = "pool-2", 1
				objects = append(objects, healthy)
			}
			client := clientsetfake.NewClientset(objects...)
			factory := informers.NewSharedInformerFactory(client, 0)
			broadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
			defer broadcaster.Shutdown()
			components, err := NewFrameworkComponents(ctx, client, factory,
				WithParallelism(1),
				WithProfiles(schedulerconfig.KubeSchedulerProfile{
					SchedulerName: schedulerName,
					Plugins: &schedulerconfig.Plugins{
						QueueSort:  schedulerconfig.PluginSet{Enabled: []schedulerconfig.Plugin{{Name: queuesort.Name}}},
						MultiPoint: schedulerconfig.PluginSet{Enabled: []schedulerconfig.Plugin{{Name: dynamicresources.Name}}},
						Bind:       schedulerconfig.PluginSet{Enabled: []schedulerconfig.Plugin{{Name: defaultbinder.Name}}},
					},
					PluginConfig: []schedulerconfig.PluginConfig{{
						Name: dynamicresources.Name, Args: &schedulerconfig.DynamicResourcesArgs{},
					}},
				}))
			if err != nil {
				t.Fatal(err)
			}
			snapshot := internalcache.NewEmptySnapshot()
			activator := &sharingAffinitySchedulerActivator{uid: pod.UID}
			profiles, err := NewFrameworkMap(ctx, components, func(name string) events.EventRecorderLogger {
				return broadcaster.NewRecorder(scheme.Scheme, name)
			}, snapshot, frameworkruntime.WithPodActivator(activator))
			if err != nil {
				t.Fatal(err)
			}
			defer profiles.Close()
			profile := profiles[schedulerName]
			hints, err := buildQueueingHintMap(ctx, profile.EnqueueExtensions())
			if err != nil {
				t.Fatal(err)
			}
			hintsByProfile := internalqueue.QueueingHintMapPerProfile{schedulerName: hints}
			preEnqueue := make(map[string]fwk.PreEnqueuePlugin)
			for _, plugin := range profile.PreEnqueuePlugins() {
				preEnqueue[plugin.Name()] = plugin
			}
			queue := internalqueue.NewSchedulingQueue(profile.QueueSortFunc(), factory,
				internalqueue.WithMetricsRecorder(components.metricsRecorder),
				internalqueue.WithPodLister(factory.Core().V1().Pods().Lister()),
				internalqueue.WithPreEnqueuePluginMap(map[string]map[string]fwk.PreEnqueuePlugin{schedulerName: preEnqueue}),
				internalqueue.WithQueueingHintMapPerProfile(hintsByProfile),
				internalqueue.WithPodInitialBackoffDuration(0),
				internalqueue.WithPodMaxBackoffDuration(0))
			defer queue.Close()
			activator.PodActivator = queue
			profile.SetPodNominator(queue)
			sched := &Scheduler{
				Cache: components.cache, client: client, NextEntity: queue.Pop,
				SchedulingQueue: queue, Profiles: profiles, nodeInfoSnapshot: snapshot,
				StopEverything: ctx.Done(), logger: logger,
			}
			initTestAlgorithm(t, sched)
			sched.applyDefaultHandlers()
			if err := addAllEventHandlers(sched, factory, nil, components.resourceClaimCache, components.resourceSliceTracker,
				components.draManager, unionedGVKs(hintsByProfile)); err != nil {
				t.Fatal(err)
			}
			bound := make(chan *v1.Binding, 2)
			client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if action.GetSubresource() != "binding" {
					return false, nil, nil
				}
				binding := action.(clienttesting.CreateAction).GetObject().(*v1.Binding)
				pods := v1.SchemeGroupVersion.WithResource("pods")
				obj, err := client.Tracker().Get(pods, binding.Namespace, binding.Name)
				if err != nil {
					return true, nil, err
				}
				scheduled := obj.(*v1.Pod).DeepCopy()
				scheduled.Spec.NodeName = binding.Target.Name
				scheduled.ResourceVersion = "2"
				if err := client.Tracker().Update(pods, scheduled, binding.Namespace); err != nil {
					return true, nil, err
				}
				bound <- binding.DeepCopy()
				return true, binding, nil
			})
			factory.Start(ctx.Done())
			defer func() {
				cancel()
				factory.Shutdown()
			}()
			for informer, synced := range factory.WaitForCacheSync(ctx.Done()) {
				if !synced {
					t.Fatalf("informer %v did not sync", informer)
				}
			}
			if err := sched.WaitForHandlersSync(ctx); err != nil {
				t.Fatal(err)
			}
			queue.Run(logger)
			poll := func(description string, condition func() bool) {
				t.Helper()
				if err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 5*time.Second, true, func(context.Context) (bool, error) {
					return condition(), nil
				}); err != nil {
					t.Fatalf("%s: %v", description, err)
				}
			}
			// Sequential node publication plus one filter worker makes node 1 first,
			// instead of depending on informer delivery or map iteration order.
			nodes := []string{"node-1"}
			if tc.healthyNode {
				nodes = append(nodes, "node-2")
			}
			for i, name := range nodes {
				if _, err := client.CoreV1().Nodes().Create(ctx, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}}, metav1.CreateOptions{}); err != nil {
					t.Fatal(err)
				}
				poll("node cached", func() bool {
					if err := sched.Cache.UpdateSnapshot(logger, snapshot); err != nil {
						return false
					}
					got, err := snapshot.List()
					return err == nil && len(got) == i+1
				})
			}
			nodeInfos, err := snapshot.List()
			if err != nil {
				t.Fatal(err)
			}
			for i, node := range nodeInfos {
				if node.Node().Name != nodes[i] {
					t.Fatalf("filter order[%d] = %q, want %q", i, node.Node().Name, nodes[i])
				}
			}
			poll("allocated owner cached", func() bool {
				state, err := components.draManager.ResourceClaims().GatherAllocatedState()
				return err == nil && state.AllocatedSharedDeviceIDs.Has(ownerDevice)
			})
			if _, err := client.CoreV1().Pods(namespace).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
				t.Fatal(err)
			}
			poll("pod in active queue", func() bool { return len(queue.PodsInActiveQ()) == 1 })
			sched.ScheduleOne(ctx)
			var state *structured.AllocatedState
			poll("consistent allocated state after scheduling", func() bool {
				var err error
				state, err = components.draManager.ResourceClaims().GatherAllocatedState()
				return err == nil
			})
			if !state.DeviceSharingAffinity.StalePools.Has(incompletePool) {
				t.Error("incomplete affinity pool is not excluded in StalePools")
			}

			if !tc.healthyNode {
				poll("pod parked as unschedulable, not an Error/backoff retry", func() bool {
					unschedulable := queue.UnschedulablePods()
					return len(unschedulable) == 1 && unschedulable[0].UID == pod.UID
				})
				select {
				case binding := <-bound:
					t.Fatalf("incomplete pool bound pod to %q", binding.Target.Name)
				default:
				}
				unallocated, err := client.ResourceV1().ResourceClaims(namespace).Get(ctx, claim.Name, metav1.GetOptions{})
				if err != nil {
					t.Fatal(err)
				}
				if unallocated.Status.Allocation != nil {
					t.Fatalf("incomplete pool allocated claim: %#v", unallocated.Status.Allocation)
				}
				if got := activator.calls.Load(); got != 0 {
					t.Fatalf("incomplete pool activated waiting pod %d times", got)
				}
				missing := slice.DeepCopy()
				missing.Name, missing.UID = "pool-1-completion", "pool-1-completion"
				missing.Spec.Devices = nil
				missing.Spec.SharingAffinity = nil
				if _, err := client.ResourceV1().ResourceSlices().Create(ctx, missing, metav1.CreateOptions{}); err != nil {
					t.Fatal(err)
				}
				poll("slice completion reconciles existing owner", func() bool {
					state, err := components.draManager.ResourceClaims().GatherAllocatedState()
					if err != nil || state.DeviceSharingAffinity.StalePools.Has(incompletePool) {
						return false
					}
					affinity := state.DeviceSharingAffinity.DeviceStates[ownerDevice]
					return affinity.Status == structured.AffinityStatusLocked && reflect.DeepEqual(affinity.LockedAffinity, map[string]string{"subnet": "blue"})
				})
				poll("slice completion activates pod through the real queue", func() bool { return len(queue.PodsInActiveQ()) == 1 })
				sched.ScheduleOne(ctx)
			}
			select {
			case binding := <-bound:
				if binding.Name != pod.Name || binding.Namespace != namespace || binding.Target.Name != tc.wantNode {
					t.Fatalf("binding = %s/%s -> %s, want %s/%s -> %s", binding.Namespace, binding.Name, binding.Target.Name, namespace, pod.Name, tc.wantNode)
				}
			case <-time.After(5 * time.Second):
				t.Fatalf("pod did not bind to %s", tc.wantNode)
			}
			poll("bound pod observed by informer", func() bool {
				scheduled, err := factory.Core().V1().Pods().Lister().Pods(namespace).Get(pod.Name)
				return err == nil && scheduled.Spec.NodeName == tc.wantNode
			})
			allocated, err := client.ResourceV1().ResourceClaims(namespace).Get(ctx, claim.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if allocated.Status.Allocation == nil || len(allocated.Status.Allocation.Devices.Results) != 1 {
				t.Fatalf("bound pod has no single-device claim allocation: %#v", allocated.Status.Allocation)
			}
			result := allocated.Status.Allocation.Devices.Results[0]
			wantPool := "pool-1"
			if tc.healthyNode {
				wantPool = "pool-2"
			}
			if result.Driver != driver || result.Pool != wantPool || result.Device != "device" || result.ShareID == nil {
				t.Fatalf("allocation = %#v, want shared %s/%s/device", result, driver, wantPool)
			}
			retained, err := client.ResourceV1().ResourceClaims(namespace).Get(ctx, owner.Name, metav1.GetOptions{})
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(retained, owner) {
				t.Error("scheduling mutated the existing allocated owner")
			}
			poll("existing shared owner remains accounted", func() bool {
				state, err := components.draManager.ResourceClaims().GatherAllocatedState()
				return err == nil && state.AllocatedSharedDeviceIDs.Has(ownerDevice)
			})
			if !tc.healthyNode {
				// A successful bind is not enough if reconciliation keeps waking
				// the same waiter after its exclusion has already resolved.
				err := wait.PollUntilContextTimeout(ctx, 10*time.Millisecond, 300*time.Millisecond, true, func(context.Context) (bool, error) {
					return activator.calls.Load() > 1, nil
				})
				if err == nil {
					t.Errorf("slice completion activated the pod %d times, want at most one", activator.calls.Load())
				} else if ctx.Err() != nil {
					t.Fatalf("test context ended while checking activation count: %v", ctx.Err())
				}
			}
		})
	}
}
