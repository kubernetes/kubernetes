/*
Copyright 2019 The Kubernetes Authors.

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
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"

	appsv1 "k8s.io/api/apps/v1"
	batchv1 "k8s.io/api/batch/v1"
	v1 "k8s.io/api/core/v1"
	resourceapi "k8s.io/api/resource/v1"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	schedulingapi "k8s.io/api/scheduling/v1alpha1"
	storagev1 "k8s.io/api/storage/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/version"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/dynamic/dynamicinformer"
	dyfake "k8s.io/client-go/dynamic/fake"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/tools/cache"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	resourceslicetracker "k8s.io/dynamic-resource-allocation/resourceslice/tracker"
	"k8s.io/klog/v2"
	"k8s.io/klog/v2/ktesting"
	fwk "k8s.io/kube-scheduler/framework"
	"k8s.io/kubernetes/pkg/features"
	apidispatcher "k8s.io/kubernetes/pkg/scheduler/backend/api_dispatcher"
	internalcache "k8s.io/kubernetes/pkg/scheduler/backend/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/backend/queue"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	apicalls "k8s.io/kubernetes/pkg/scheduler/framework/api_calls"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/dynamicresources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeaffinity"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodename"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	"k8s.io/kubernetes/pkg/scheduler/metrics"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	tf "k8s.io/kubernetes/pkg/scheduler/testing/framework"
	"k8s.io/kubernetes/pkg/scheduler/util"
	"k8s.io/kubernetes/pkg/scheduler/util/assumecache"
)

func TestEventHandlers_MoveToActiveOnNominatedNodeUpdate(t *testing.T) {
	highPriorityPod :=
		st.MakePod().Name("hpp").Namespace("ns1").UID("hppns1").Priority(highPriority).SchedulerName(testSchedulerName).Obj()

	medNominatedPriorityPod :=
		st.MakePod().Name("mpp").Namespace("ns2").UID("mppns1").Priority(midPriority).SchedulerName(testSchedulerName).NominatedNodeName("node1").Obj()
	medPriorityPod :=
		st.MakePod().Name("smpp").Namespace("ns3").UID("mppns2").Priority(midPriority).SchedulerName(testSchedulerName).Obj()

	lowPriorityPod :=
		st.MakePod().Name("lpp").Namespace("ns4").UID("lppns1").Priority(lowPriority).SchedulerName(testSchedulerName).Obj()

	unschedulablePods := []*v1.Pod{highPriorityPod, medNominatedPriorityPod, medPriorityPod, lowPriorityPod}

	// Make pods schedulable on Delete event when QHints are enabled, but not when nominated node appears.
	queueHintForPodDelete := func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
		oldPod, _, err := util.As[*v1.Pod](oldObj, newObj)
		if err != nil {
			t.Errorf("Failed to convert objects to pods: %v", err)
		}
		if oldPod.Status.NominatedNodeName == "" {
			return fwk.QueueSkip, nil
		}
		return fwk.Queue, nil
	}
	queueingHintMap := internalqueue.QueueingHintMapPerProfile{
		testSchedulerName: {
			framework.EventAssignedPodDelete: {
				{
					PluginName:     "fooPlugin1",
					QueueingHintFn: queueHintForPodDelete,
				},
			},
		},
	}

	tests := []struct {
		name                  string
		updateFunc            func(s *Scheduler)
		wantInActiveOrBackoff sets.Set[string]
	}{
		{
			name: "Update of a nominated node name to a different value should trigger rescheduling of lower priority pods",
			updateFunc: func(s *Scheduler) {
				updatedPod := medNominatedPriorityPod.DeepCopy()
				updatedPod.Status.NominatedNodeName = "node2"
				updatedPod.ResourceVersion = "1"
				s.updatePodInSchedulingQueue(medNominatedPriorityPod, updatedPod)
			},
			wantInActiveOrBackoff: sets.New(lowPriorityPod.Name, medPriorityPod.Name, medNominatedPriorityPod.Name),
		},
		{
			name: "Removal of a nominated node name should trigger rescheduling of lower priority pods",
			updateFunc: func(s *Scheduler) {
				updatedPod := medNominatedPriorityPod.DeepCopy()
				updatedPod.Status.NominatedNodeName = ""
				updatedPod.ResourceVersion = "1"
				s.updatePodInSchedulingQueue(medNominatedPriorityPod, updatedPod)
			},
			wantInActiveOrBackoff: sets.New(lowPriorityPod.Name, medPriorityPod.Name, medNominatedPriorityPod.Name),
		},
		{
			name: "Removal of a pod that had nominated node name should trigger rescheduling of lower priority pods",
			updateFunc: func(s *Scheduler) {
				s.deletePodFromSchedulingQueue(medNominatedPriorityPod, false)
			},
			wantInActiveOrBackoff: sets.New(lowPriorityPod.Name, medPriorityPod.Name),
		},
		{
			name: "Addition of a nominated node name to the high priority pod that did not have it before shouldn't trigger rescheduling",
			updateFunc: func(s *Scheduler) {
				updatedPod := highPriorityPod.DeepCopy()
				updatedPod.Status.NominatedNodeName = "node2"
				updatedPod.ResourceVersion = "1"
				s.updatePodInSchedulingQueue(highPriorityPod, updatedPod)
			},
			wantInActiveOrBackoff: sets.New[string](),
		},
	}

	for _, tt := range tests {
		for _, qHintEnabled := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s, with queuehint(%v)", tt.name, qHintEnabled), func(t *testing.T) {
				if !qHintEnabled {
					featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.33"))
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerQueueingHints, false)
				}

				logger, ctx := ktesting.NewTestContext(t)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()

				var objs []runtime.Object
				for _, pod := range unschedulablePods {
					objs = append(objs, pod)
				}
				client := fake.NewClientset(objs...)
				informerFactory := informers.NewSharedInformerFactory(client, 0)

				// apiDispatcher is unused in the test, but intializing it anyway.
				apiDispatcher := apidispatcher.New(client, 16, apicalls.Relevances)
				apiDispatcher.Run(logger)
				defer apiDispatcher.Close()

				recorder := metrics.NewMetricsAsyncRecorder(3, 20*time.Microsecond, ctx.Done())
				queue := internalqueue.NewPriorityQueue(
					newDefaultQueueSort(),
					informerFactory,
					internalqueue.WithMetricsRecorder(recorder),
					internalqueue.WithQueueingHintMapPerProfile(queueingHintMap),
					internalqueue.WithAPIDispatcher(apiDispatcher),
					// disable backoff queue
					internalqueue.WithPodInitialBackoffDuration(0),
					internalqueue.WithPodMaxBackoffDuration(0))
				schedulerCache := internalcache.New(ctx, nil)

				// Put test pods into unschedulable queue
				for _, pod := range unschedulablePods {
					queue.Add(logger, pod)
					poppedPod, err := queue.Pop(logger)
					if err != nil {
						t.Fatalf("Pop failed: %v", err)
					}
					poppedPod.UnschedulablePlugins = sets.New("fooPlugin1")
					if err := queue.AddUnschedulableIfNotPresent(logger, poppedPod, queue.SchedulingCycle()); err != nil {
						t.Errorf("Unexpected error from AddUnschedulableIfNotPresent: %v", err)
					}
				}

				s, _, err := initScheduler(ctx, schedulerCache, queue, apiDispatcher, client, informerFactory)
				if err != nil {
					t.Fatalf("Failed to initialize test scheduler: %v", err)
				}

				if len(s.SchedulingQueue.PodsInActiveQ()) > 0 {
					t.Errorf("No pods were expected to be in the activeQ before the update, but there were %v", s.SchedulingQueue.PodsInActiveQ())
				}
				tt.updateFunc(s)

				podsInActiveOrBackoff := s.SchedulingQueue.PodsInActiveQ()
				podsInActiveOrBackoff = append(podsInActiveOrBackoff, s.SchedulingQueue.PodsInBackoffQ()...)
				if len(podsInActiveOrBackoff) != len(tt.wantInActiveOrBackoff) {
					t.Errorf("Different number of pods were expected to be in the activeQ or backoffQ, but found actual %v vs. expected %v", podsInActiveOrBackoff, tt.wantInActiveOrBackoff)
				}
				for _, pod := range podsInActiveOrBackoff {
					if !tt.wantInActiveOrBackoff.Has(pod.Name) {
						t.Errorf("Found unexpected pod in activeQ or backoffQ: %s", pod.Name)
					}
				}
			})
		}
	}
}

func newDefaultQueueSort() fwk.LessFunc {
	sort := &queuesort.PrioritySort{}
	return sort.Less
}

func TestUpdateAssignedPodInCache(t *testing.T) {
	nodeName := "node"

	tests := []struct {
		name   string
		oldPod *v1.Pod
		newPod *v1.Pod
	}{
		{
			name:   "pod updated with the same UID",
			oldPod: withPodName(podWithPort("oldUID", nodeName, 80), "pod"),
			newPod: withPodName(podWithPort("oldUID", nodeName, 8080), "pod"),
		},
		{
			name:   "pod updated with different UIDs",
			oldPod: withPodName(podWithPort("oldUID", nodeName, 80), "pod"),
			newPod: withPodName(podWithPort("newUID", nodeName, 8080), "pod"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()
			sched := &Scheduler{
				Cache:           internalcache.New(ctx, nil),
				SchedulingQueue: internalqueue.NewTestQueue(ctx, nil),
				logger:          logger,
			}
			sched.addAssignedPodToCache(tt.oldPod)
			sched.updateAssignedPodInCache(tt.oldPod, tt.newPod)

			if tt.oldPod.UID != tt.newPod.UID {
				if pod, err := sched.Cache.GetPod(tt.oldPod); err == nil {
					t.Errorf("Get pod UID %v from cache but it should not happen", pod.UID)
				}
			}
			pod, err := sched.Cache.GetPod(tt.newPod)
			if err != nil {
				t.Errorf("Failed to get pod from scheduler: %v", err)
			}
			if pod.UID != tt.newPod.UID {
				t.Errorf("Want pod UID %v, got %v", tt.newPod.UID, pod.UID)
			}
		})
	}
}

func withPodName(pod *v1.Pod, name string) *v1.Pod {
	pod.Name = name
	return pod
}

func TestPreCheckForNode(t *testing.T) {
	logger, _ := ktesting.NewTestContext(t)

	cpu4 := map[v1.ResourceName]string{v1.ResourceCPU: "4"}
	cpu8 := map[v1.ResourceName]string{v1.ResourceCPU: "8"}
	cpu16 := map[v1.ResourceName]string{v1.ResourceCPU: "16"}
	tests := []struct {
		name               string
		nodeFn             func() *v1.Node
		existingPods, pods []*v1.Pod
		want               []bool
		qHintEnabled       bool
	}{
		{
			name: "regular node, pods with a single constraint",
			nodeFn: func() *v1.Node {
				return st.MakeNode().Name("fake-node").Label("hostname", "fake-node").Capacity(cpu8).Obj()
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p").HostPort(80).Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").Req(cpu4).Obj(),
				st.MakePod().Name("p2").Req(cpu16).Obj(),
				st.MakePod().Name("p3").Req(cpu4).Req(cpu8).Obj(),
				st.MakePod().Name("p4").NodeAffinityIn("hostname", []string{"fake-node"}, st.NodeSelectorTypeMatchExpressions).Obj(),
				st.MakePod().Name("p5").NodeAffinityNotIn("hostname", []string{"fake-node"}).Obj(),
				st.MakePod().Name("p6").Obj(),
				st.MakePod().Name("p7").Node("invalid-node").Obj(),
				st.MakePod().Name("p8").HostPort(8080).Obj(),
				st.MakePod().Name("p9").HostPort(80).Obj(),
			},
			want: []bool{true, false, false, true, false, true, false, true, false},
		},
		{
			name: "no filtering when QHint is enabled",
			nodeFn: func() *v1.Node {
				return st.MakeNode().Name("fake-node").Label("hostname", "fake-node").Capacity(cpu8).Obj()
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p").HostPort(80).Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").Req(cpu4).Obj(),
				st.MakePod().Name("p2").Req(cpu16).Obj(),
				st.MakePod().Name("p3").Req(cpu4).Req(cpu8).Obj(),
				st.MakePod().Name("p4").NodeAffinityIn("hostname", []string{"fake-node"}, st.NodeSelectorTypeMatchExpressions).Obj(),
				st.MakePod().Name("p5").NodeAffinityNotIn("hostname", []string{"fake-node"}).Obj(),
				st.MakePod().Name("p6").Obj(),
				st.MakePod().Name("p7").Node("invalid-node").Obj(),
				st.MakePod().Name("p8").HostPort(8080).Obj(),
				st.MakePod().Name("p9").HostPort(80).Obj(),
			},
			qHintEnabled: true,
			want:         []bool{true, true, true, true, true, true, true, true, true},
		},
		{
			name: "tainted node, pods with a single constraint",
			nodeFn: func() *v1.Node {
				node := st.MakeNode().Name("fake-node").Obj()
				node.Spec.Taints = []v1.Taint{
					{Key: "foo", Effect: v1.TaintEffectNoSchedule},
					{Key: "bar", Effect: v1.TaintEffectPreferNoSchedule},
				}
				return node
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").Obj(),
				st.MakePod().Name("p2").Toleration("foo").Obj(),
				st.MakePod().Name("p3").Toleration("bar").Obj(),
				st.MakePod().Name("p4").Toleration("bar").Toleration("foo").Obj(),
			},
			want: []bool{false, true, false, true},
		},
		{
			name: "regular node, pods with multiple constraints",
			nodeFn: func() *v1.Node {
				return st.MakeNode().Name("fake-node").Label("hostname", "fake-node").Capacity(cpu8).Obj()
			},
			existingPods: []*v1.Pod{
				st.MakePod().Name("p").HostPort(80).Obj(),
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").Req(cpu4).NodeAffinityNotIn("hostname", []string{"fake-node"}).Obj(),
				st.MakePod().Name("p2").Req(cpu16).NodeAffinityIn("hostname", []string{"fake-node"}, st.NodeSelectorTypeMatchExpressions).Obj(),
				st.MakePod().Name("p3").Req(cpu8).NodeAffinityIn("hostname", []string{"fake-node"}, st.NodeSelectorTypeMatchExpressions).Obj(),
				st.MakePod().Name("p4").HostPort(8080).Node("invalid-node").Obj(),
				st.MakePod().Name("p5").Req(cpu4).NodeAffinityIn("hostname", []string{"fake-node"}, st.NodeSelectorTypeMatchExpressions).HostPort(80).Obj(),
			},
			want: []bool{false, false, true, false, false},
		},
		{
			name: "tainted node, pods with multiple constraints",
			nodeFn: func() *v1.Node {
				node := st.MakeNode().Name("fake-node").Label("hostname", "fake-node").Capacity(cpu8).Obj()
				node.Spec.Taints = []v1.Taint{
					{Key: "foo", Effect: v1.TaintEffectNoSchedule},
					{Key: "bar", Effect: v1.TaintEffectPreferNoSchedule},
				}
				return node
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").Req(cpu4).Toleration("bar").Obj(),
				st.MakePod().Name("p2").Req(cpu4).Toleration("bar").Toleration("foo").Obj(),
				st.MakePod().Name("p3").Req(cpu16).Toleration("foo").Obj(),
				st.MakePod().Name("p3").Req(cpu16).Toleration("bar").Obj(),
			},
			want: []bool{false, true, false, false},
		},
		{
			name: "tainted node with NoExecute effect, pods with tolerations",
			nodeFn: func() *v1.Node {
				node := st.MakeNode().Name("fake-node").Label("hostname", "fake-node").Capacity(cpu8).Obj()
				node.Spec.Taints = []v1.Taint{
					{Key: "foo", Effect: v1.TaintEffectPreferNoSchedule},
					{Key: "baz", Effect: v1.TaintEffectNoExecute},
				}
				return node
			},
			pods: []*v1.Pod{
				st.MakePod().Name("p1").Obj(),
				st.MakePod().Name("p2").Obj(),
				st.MakePod().Name("p3").Toleration("foo").Obj(),
				st.MakePod().Name("p4").Toleration("baz").Obj(),
				st.MakePod().Name("p5").Obj(),
				st.MakePod().Name("p6").Toleration("bar").Toleration("baz").Obj(),
			},
			want: []bool{false, false, false, true, false, true},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if !tt.qHintEnabled {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.33"))
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerQueueingHints, false)
			}

			nodeInfo := framework.NewNodeInfo(tt.existingPods...)
			nodeInfo.SetNode(tt.nodeFn())
			preCheckFn := preCheckForNode(logger, nodeInfo)

			got := make([]bool, 0, len(tt.pods))
			for _, pod := range tt.pods {
				got = append(got, preCheckFn == nil || preCheckFn(pod))
			}

			if diff := cmp.Diff(tt.want, got); diff != "" {
				t.Errorf("Unexpected diff (-want, +got):\n%s", diff)
			}
		})
	}
}

// test for informers of resources we care about is registered
func TestAddAllEventHandlers(t *testing.T) {
	tests := []struct {
		name                      string
		gvkMap                    map[fwk.EventResource]fwk.ActionType
		enableDRA                 bool
		enableDRADeviceTaints     bool
		enableDRADeviceTaintRules bool
		enableDRAExtendedResource bool
		enableGenericWorkload     bool
		expectStaticInformers     map[reflect.Type]bool
		expectDynamicInformers    map[schema.GroupVersionResource]bool
	}{
		{
			name:   "default handlers in framework",
			gvkMap: map[fwk.EventResource]fwk.ActionType{},
			expectStaticInformers: map[reflect.Type]bool{
				reflect.TypeOf(&v1.Pod{}):       true,
				reflect.TypeOf(&v1.Node{}):      true,
				reflect.TypeOf(&v1.Namespace{}): true,
			},
			expectDynamicInformers: map[schema.GroupVersionResource]bool{},
		},
		{
			name: "DRA events disabled",
			gvkMap: map[fwk.EventResource]fwk.ActionType{
				fwk.ResourceClaim: fwk.Add,
				fwk.ResourceSlice: fwk.Add,
				fwk.DeviceClass:   fwk.Add,
			},
			expectStaticInformers: map[reflect.Type]bool{
				reflect.TypeOf(&v1.Pod{}):       true,
				reflect.TypeOf(&v1.Node{}):      true,
				reflect.TypeOf(&v1.Namespace{}): true,
			},
			expectDynamicInformers: map[schema.GroupVersionResource]bool{},
		},
		{
			name: "core DRA events enabled",
			gvkMap: map[fwk.EventResource]fwk.ActionType{
				fwk.ResourceClaim: fwk.Add,
				fwk.ResourceSlice: fwk.Add,
				fwk.DeviceClass:   fwk.Add,
			},
			enableDRA: true,
			expectStaticInformers: map[reflect.Type]bool{
				reflect.TypeOf(&v1.Pod{}):                    true,
				reflect.TypeOf(&v1.Node{}):                   true,
				reflect.TypeOf(&v1.Namespace{}):              true,
				reflect.TypeOf(&resourceapi.ResourceClaim{}): true,
				reflect.TypeOf(&resourceapi.ResourceSlice{}): true,
				reflect.TypeOf(&resourceapi.DeviceClass{}):   true,
			},
			expectDynamicInformers: map[schema.GroupVersionResource]bool{},
		},
		{
			name: "device taints partially enabled",
			gvkMap: map[fwk.EventResource]fwk.ActionType{
				fwk.ResourceClaim: fwk.Add,
				fwk.ResourceSlice: fwk.Add,
				fwk.DeviceClass:   fwk.Add,
			},
			enableDRA:             true,
			enableDRADeviceTaints: true,
			expectStaticInformers: map[reflect.Type]bool{
				reflect.TypeOf(&v1.Pod{}):                    true,
				reflect.TypeOf(&v1.Node{}):                   true,
				reflect.TypeOf(&v1.Namespace{}):              true,
				reflect.TypeOf(&resourceapi.ResourceClaim{}): true,
				reflect.TypeOf(&resourceapi.ResourceSlice{}): true,
				reflect.TypeOf(&resourceapi.DeviceClass{}):   true,
			},
			expectDynamicInformers: map[schema.GroupVersionResource]bool{},
		},
		{
			name: "all DRA events enabled",
			gvkMap: map[fwk.EventResource]fwk.ActionType{
				fwk.ResourceClaim: fwk.Add,
				fwk.ResourceSlice: fwk.Add,
				fwk.DeviceClass:   fwk.Add,
			},
			enableDRA:                 true,
			enableDRADeviceTaints:     true,
			enableDRADeviceTaintRules: true,
			expectStaticInformers: map[reflect.Type]bool{
				reflect.TypeOf(&v1.Pod{}):                           true,
				reflect.TypeOf(&v1.Node{}):                          true,
				reflect.TypeOf(&v1.Namespace{}):                     true,
				reflect.TypeOf(&resourceapi.ResourceClaim{}):        true,
				reflect.TypeOf(&resourceapi.ResourceSlice{}):        true,
				reflect.TypeOf(&resourcealphaapi.DeviceTaintRule{}): true,
				reflect.TypeOf(&resourceapi.DeviceClass{}):          true,
			},
			expectDynamicInformers: map[schema.GroupVersionResource]bool{},
		},
		{
			name: "Workload events disabled",
			gvkMap: map[fwk.EventResource]fwk.ActionType{
				fwk.Workload: fwk.Add,
			},
			expectStaticInformers: map[reflect.Type]bool{
				reflect.TypeOf(&v1.Pod{}):       true,
				reflect.TypeOf(&v1.Node{}):      true,
				reflect.TypeOf(&v1.Namespace{}): true,
			},
			expectDynamicInformers: map[schema.GroupVersionResource]bool{},
		},
		{
			name: "Workload events enabled",
			gvkMap: map[fwk.EventResource]fwk.ActionType{
				fwk.Workload: fwk.Add,
			},
			enableDRA:             true,
			enableGenericWorkload: true,
			expectStaticInformers: map[reflect.Type]bool{
				reflect.TypeOf(&v1.Pod{}):                    true,
				reflect.TypeOf(&v1.Node{}):                   true,
				reflect.TypeOf(&v1.Namespace{}):              true,
				reflect.TypeOf(&resourceapi.ResourceClaim{}): true,
				reflect.TypeOf(&resourceapi.ResourceSlice{}): true,
				reflect.TypeOf(&schedulingapi.Workload{}):    true,
			},
			expectDynamicInformers: map[schema.GroupVersionResource]bool{},
		},
		{
			name: "add GVKs handlers defined in framework dynamically",
			gvkMap: map[fwk.EventResource]fwk.ActionType{
				"Pod":                               fwk.Add | fwk.Delete,
				"PersistentVolume":                  fwk.Delete,
				"storage.k8s.io/CSIStorageCapacity": fwk.Update,
			},
			expectStaticInformers: map[reflect.Type]bool{
				reflect.TypeOf(&v1.Pod{}):                       true,
				reflect.TypeOf(&v1.Node{}):                      true,
				reflect.TypeOf(&v1.Namespace{}):                 true,
				reflect.TypeOf(&v1.PersistentVolume{}):          true,
				reflect.TypeOf(&storagev1.CSIStorageCapacity{}): true,
			},
			expectDynamicInformers: map[schema.GroupVersionResource]bool{},
		},
		{
			name: "add GVKs handlers defined in plugins dynamically",
			gvkMap: map[fwk.EventResource]fwk.ActionType{
				"daemonsets.v1.apps": fwk.Add | fwk.Delete,
				"cronjobs.v1.batch":  fwk.Delete,
			},
			expectStaticInformers: map[reflect.Type]bool{
				reflect.TypeOf(&v1.Pod{}):       true,
				reflect.TypeOf(&v1.Node{}):      true,
				reflect.TypeOf(&v1.Namespace{}): true,
			},
			expectDynamicInformers: map[schema.GroupVersionResource]bool{
				{Group: "apps", Version: "v1", Resource: "daemonsets"}: true,
				{Group: "batch", Version: "v1", Resource: "cronjobs"}:  true,
			},
		},
		{
			name: "add GVKs handlers defined in plugins dynamically, with one illegal GVK form",
			gvkMap: map[fwk.EventResource]fwk.ActionType{
				"daemonsets.v1.apps":    fwk.Add | fwk.Delete,
				"custommetrics.v1beta1": fwk.Update,
			},
			expectStaticInformers: map[reflect.Type]bool{
				reflect.TypeOf(&v1.Pod{}):       true,
				reflect.TypeOf(&v1.Node{}):      true,
				reflect.TypeOf(&v1.Namespace{}): true,
			},
			expectDynamicInformers: map[schema.GroupVersionResource]bool{
				{Group: "apps", Version: "v1", Resource: "daemonsets"}: true,
			},
		},
	}

	scheme := runtime.NewScheme()
	var localSchemeBuilder = runtime.SchemeBuilder{
		appsv1.AddToScheme,
		batchv1.AddToScheme,
	}
	localSchemeBuilder.AddToScheme(scheme)

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			overrides := featuregatetesting.FeatureOverrides{
				features.DynamicResourceAllocation: tt.enableDRA,
				features.DRADeviceTaints:           tt.enableDRADeviceTaints,
				features.DRAExtendedResource:       tt.enableDRAExtendedResource,
			}
			if !tt.enableDRA {
				featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.34"))
			} else {
				// Making this depend on the emulated version avoids "cannot set feature gate DRADeviceTaintRules to false, feature is PreAlpha at emulated version 1.34".
				overrides[features.DRADeviceTaintRules] = tt.enableDRADeviceTaintRules
				overrides[features.GenericWorkload] = tt.enableGenericWorkload
			}
			featuregatetesting.SetFeatureGatesDuringTest(t, utilfeature.DefaultFeatureGate, overrides)

			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			informerFactory := informers.NewSharedInformerFactory(fake.NewClientset(), 0)
			schedulingQueue := internalqueue.NewTestQueueWithInformerFactory(ctx, nil, informerFactory)
			testSched := Scheduler{
				StopEverything:  ctx.Done(),
				SchedulingQueue: schedulingQueue,
				logger:          logger,
			}

			dynclient := dyfake.NewSimpleDynamicClient(scheme)
			dynInformerFactory := dynamicinformer.NewDynamicSharedInformerFactory(dynclient, 0)
			var resourceClaimCache *assumecache.AssumeCache
			var resourceSliceTracker *resourceslicetracker.Tracker
			var draManager fwk.SharedDRAManager
			if utilfeature.DefaultFeatureGate.Enabled(features.DynamicResourceAllocation) {
				resourceClaimInformer := informerFactory.Resource().V1().ResourceClaims().Informer()
				resourceClaimCache = assumecache.NewAssumeCache(logger, resourceClaimInformer, "ResourceClaim", "", nil)
				var err error
				opts := resourceslicetracker.Options{
					EnableDeviceTaintRules: utilfeature.DefaultFeatureGate.Enabled(features.DRADeviceTaintRules),
					SliceInformer:          informerFactory.Resource().V1().ResourceSlices(),
				}
				if opts.EnableDeviceTaintRules {
					opts.TaintInformer = informerFactory.Resource().V1alpha3().DeviceTaintRules()
					opts.ClassInformer = informerFactory.Resource().V1().DeviceClasses()

				}
				resourceSliceTracker, err = resourceslicetracker.StartTracker(ctx, opts)
				if err != nil {
					t.Fatalf("couldn't start resource slice tracker: %v", err)
				}

				if tt.enableDRAExtendedResource {
					draManager = dynamicresources.NewDRAManager(ctx, resourceClaimCache, resourceSliceTracker, informerFactory)
				}
			}

			if err := addAllEventHandlers(&testSched, informerFactory, dynInformerFactory, resourceClaimCache, resourceSliceTracker, draManager, tt.gvkMap); err != nil {
				t.Fatalf("Add event handlers failed, error = %v", err)
			}

			informerFactory.Start(testSched.StopEverything)
			dynInformerFactory.Start(testSched.StopEverything)
			staticInformers := informerFactory.WaitForCacheSync(testSched.StopEverything)
			dynamicInformers := dynInformerFactory.WaitForCacheSync(testSched.StopEverything)

			if diff := cmp.Diff(tt.expectStaticInformers, staticInformers); diff != "" {
				t.Errorf("Unexpected diff (-want, +got):\n%s", diff)
			}
			if diff := cmp.Diff(tt.expectDynamicInformers, dynamicInformers); diff != "" {
				t.Errorf("Unexpected diff (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestAdmissionCheck(t *testing.T) {
	nodeaffinityError := AdmissionResult{Name: nodeaffinity.Name, Reason: nodeaffinity.ErrReasonPod}
	nodenameError := AdmissionResult{Name: nodename.Name, Reason: nodename.ErrReason}
	nodeportsError := AdmissionResult{Name: nodeports.Name, Reason: nodeports.ErrReason}
	podOverheadError := AdmissionResult{InsufficientResource: &noderesources.InsufficientResource{ResourceName: v1.ResourceCPU, Reason: "Insufficient cpu", Requested: 2000, Used: 7000, Capacity: 8000}}
	extendedResourceError := AdmissionResult{InsufficientResource: &noderesources.InsufficientResource{ResourceName: "foo.com/bar", Reason: "Insufficient foo.com/bar", Requested: 1, Unresolvable: true}}
	cpu := map[v1.ResourceName]string{v1.ResourceCPU: "8"}
	extendedResource := map[v1.ResourceName]string{"foo.com/bar": "1"}
	tests := []struct {
		name                      string
		node                      *v1.Node
		existingPods              []*v1.Pod
		pod                       *v1.Pod
		wantAdmissionResults      [][]AdmissionResult
		enableDRAExtendedResource bool
	}{
		{
			name: "check nodeAffinity and nodeports, nodeAffinity need fail quickly if includeAllFailures is false",
			node: st.MakeNode().Name("fake-node").Label("foo", "bar").Obj(),
			pod:  st.MakePod().Name("pod2").HostPort(80).NodeSelector(map[string]string{"foo": "bar1"}).Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("pod1").HostPort(80).Obj(),
			},
			wantAdmissionResults: [][]AdmissionResult{{nodeaffinityError, nodeportsError}, {nodeaffinityError}},
		},
		{
			name: "check PodOverhead and nodeAffinity, PodOverhead need fail quickly if includeAllFailures is false",
			node: st.MakeNode().Name("fake-node").Label("foo", "bar").Capacity(cpu).Obj(),
			pod:  st.MakePod().Name("pod2").Container("c").Overhead(v1.ResourceList{v1.ResourceCPU: resource.MustParse("1")}).Req(map[v1.ResourceName]string{v1.ResourceCPU: "1"}).NodeSelector(map[string]string{"foo": "bar1"}).Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("pod1").Req(map[v1.ResourceName]string{v1.ResourceCPU: "7"}).Node("fake-node").Obj(),
			},
			wantAdmissionResults: [][]AdmissionResult{{podOverheadError, nodeaffinityError}, {podOverheadError}},
		},
		{
			name: "check nodename and nodeports, nodename need fail quickly if includeAllFailures is false",
			node: st.MakeNode().Name("fake-node").Obj(),
			pod:  st.MakePod().Name("pod2").HostPort(80).Node("fake-node1").Obj(),
			existingPods: []*v1.Pod{
				st.MakePod().Name("pod1").HostPort(80).Node("fake-node").Obj(),
			},
			wantAdmissionResults: [][]AdmissionResult{{nodenameError, nodeportsError}, {nodenameError}},
		},
		{
			name:                 "check extended resource handling when node Allocatable doesn't have the resource",
			node:                 st.MakeNode().Name("fake-node").Obj(),
			pod:                  st.MakePod().Name("pod1").Req(extendedResource).Obj(),
			wantAdmissionResults: [][]AdmissionResult{{extendedResourceError}, {extendedResourceError}},
		},
		{
			name:                      "check extended resource handling when node Allocatable doesn't have the resource and DRAExtendedResource is enabled",
			node:                      st.MakeNode().Name("fake-node").Obj(),
			pod:                       st.MakePod().Name("pod1").Req(extendedResource).Obj(),
			wantAdmissionResults:      [][]AdmissionResult{{extendedResourceError}, {extendedResourceError}},
			enableDRAExtendedResource: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.DRAExtendedResource, tt.enableDRAExtendedResource)
			nodeInfo := framework.NewNodeInfo(tt.existingPods...)
			nodeInfo.SetNode(tt.node)

			flags := []bool{true, false}
			for i := range flags {
				admissionResults := AdmissionCheck(tt.pod, nodeInfo, flags[i])

				if diff := cmp.Diff(tt.wantAdmissionResults[i], admissionResults); diff != "" {
					t.Errorf("Unexpected admissionResults (-want, +got):\n%s", diff)
				}
			}
		})
	}
}

func TestAddPod(t *testing.T) {
	tests := []struct {
		name          string
		pod           *v1.Pod
		expectInQueue bool
		expectInCache bool
	}{
		{
			name:          "add unscheduled pod",
			pod:           st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").SchedulerName("supported-scheduler").Obj(),
			expectInQueue: true,
		},
		{
			name: "add unscheduled pod with other scheduler name",
			pod:  st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").SchedulerName("other-scheduler").Obj(),
		},
		{
			name:          "add scheduled pod",
			pod:           st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Node("node1").Obj(),
			expectInCache: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			sched := &Scheduler{
				Cache:           internalcache.New(ctx, nil),
				SchedulingQueue: internalqueue.NewTestQueue(ctx, nil),
				logger:          logger,
				Profiles: profile.Map{
					"supported-scheduler": nil,
				},
			}

			sched.addPod(tt.pod)

			_, ok := sched.SchedulingQueue.GetPod(tt.pod.Name, tt.pod.Namespace)
			if tt.expectInQueue && !ok {
				t.Errorf("Expected pod to be in scheduling queue")
			} else if !tt.expectInQueue && ok {
				t.Errorf("Expected pod not to be in scheduling queue")
			}
			_, err := sched.Cache.GetPod(tt.pod)
			if tt.expectInCache && err != nil {
				t.Errorf("Expected pod to be in cache: %v", err)
			} else if !tt.expectInCache && err == nil {
				t.Errorf("Expected pod not to be in cache")
			}
		})
	}
}

func TestUpdatePod(t *testing.T) {
	pod := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").SchedulerName("supported-scheduler").Obj()
	updatedPod := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Labels(map[string]string{"foo": "bar"}).ResourceVersion("2").SchedulerName("supported-scheduler").Obj()

	podWithDeletionTimestamp := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Terminating().ResourceVersion("2").SchedulerName("supported-scheduler").Obj()

	otherPod := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").SchedulerName("other-scheduler").Obj()
	updatedOtherPod := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Labels(map[string]string{"foo": "bar"}).ResourceVersion("2").SchedulerName("other-scheduler").Obj()

	scheduledPod := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Node("node1").SchedulerName("supported-scheduler").Obj()
	updatedScheduledPod := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Labels(map[string]string{"foo": "bar"}).ResourceVersion("2").Node("node1").SchedulerName("supported-scheduler").Obj()

	otherScheduledPod := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Node("node1").SchedulerName("other-scheduler").Obj()
	updatedOtherScheduledPod := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Labels(map[string]string{"foo": "bar"}).ResourceVersion("2").Node("node1").SchedulerName("other-scheduler").Obj()

	scheduledPodOtherNode := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Node("node2").SchedulerName("supported-scheduler").Obj()

	tests := []struct {
		name          string
		oldPod        *v1.Pod
		assumedPod    *v1.Pod
		newPod        *v1.Pod
		expectInQueue *v1.Pod
		expectInCache *v1.Pod
	}{
		{
			name:          "update unscheduled pod",
			oldPod:        pod,
			newPod:        updatedPod,
			expectInQueue: updatedPod,
		},
		{
			name:   "update unscheduled pod with other scheduler name",
			oldPod: otherPod,
			newPod: updatedOtherPod,
		},
		{
			name:          "update assumed pod",
			oldPod:        pod,
			assumedPod:    scheduledPod,
			newPod:        updatedPod,
			expectInCache: scheduledPod,
		},
		{
			name:          "update scheduled pod",
			oldPod:        scheduledPod,
			newPod:        updatedScheduledPod,
			expectInCache: updatedScheduledPod,
		},
		{
			name:          "update scheduled pod with other scheduler name",
			oldPod:        otherScheduledPod,
			newPod:        updatedOtherScheduledPod,
			expectInCache: updatedOtherScheduledPod,
		},
		{
			name:          "bind unscheduled pod",
			oldPod:        pod,
			newPod:        scheduledPod,
			expectInCache: scheduledPod,
		},
		{
			name:          "bind unscheduled pod with other scheduler name",
			oldPod:        pod,
			newPod:        scheduledPod,
			expectInCache: scheduledPod,
		},
		{
			name:          "bind assumed pod",
			oldPod:        pod,
			assumedPod:    scheduledPod,
			newPod:        scheduledPod,
			expectInCache: scheduledPod,
		},
		{
			name:          "bind assumed pod to a different node",
			oldPod:        pod,
			assumedPod:    scheduledPod,
			newPod:        scheduledPodOtherNode,
			expectInCache: scheduledPodOtherNode,
		},
		{
			name:       "delete assumed pod with deletion timestamp",
			oldPod:     pod,
			assumedPod: scheduledPod,
			newPod:     podWithDeletionTimestamp,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			registerPluginFuncs := []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			waitingPods := frameworkruntime.NewWaitingPodsMap()
			schedFramework, err := tf.NewFramework(
				ctx,
				registerPluginFuncs,
				"supported-scheduler",
				frameworkruntime.WithWaitingPods(waitingPods),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}
			sched := &Scheduler{
				Cache:           internalcache.New(ctx, nil),
				SchedulingQueue: internalqueue.NewTestQueue(ctx, nil),
				logger:          logger,
				Profiles: profile.Map{
					"supported-scheduler": schedFramework,
				},
			}

			if tt.assumedPod != nil {
				err := sched.Cache.AssumePod(logger, tt.assumedPod)
				if err != nil {
					t.Fatalf("Failed to assume pod: %v", err)
				}
			} else {
				sched.addPod(tt.oldPod)
			}

			sched.updatePod(tt.oldPod, tt.newPod)

			qPod, ok := sched.SchedulingQueue.GetPod(tt.newPod.Name, tt.newPod.Namespace)
			if tt.expectInQueue != nil {
				if !ok {
					t.Errorf("Expected pod to be in scheduling queue")
				} else if diff := cmp.Diff(tt.expectInQueue, qPod.Pod); diff != "" {
					t.Errorf("Unexpected pod after update (-want,+got):\n%s", diff)
				}
			} else if ok {
				t.Errorf("Expected pod not to be in scheduling queue")
			}
			pod, err := sched.Cache.GetPod(tt.newPod)
			if tt.expectInCache != nil {
				if err != nil {
					t.Errorf("Expected pod to be in cache: %v", err)
				} else if diff := cmp.Diff(tt.expectInCache, pod); diff != "" {
					t.Errorf("Unexpected pod after update (-want,+got):\n%s", diff)
				}
			} else if err == nil {
				t.Errorf("Expected pod not to be in cache")
			}
		})
	}
}

func TestUpdatePod_WakeUpPodsOnExternalScheduling(t *testing.T) {
	highPriorityPod :=
		st.MakePod().Name("hpp").Namespace("ns1").UID("hppns1").Priority(highPriority).SchedulerName(testSchedulerName).Obj()
	medPriorityPod :=
		st.MakePod().Name("smpp").Namespace("ns3").UID("mppns2").Priority(midPriority).SchedulerName(testSchedulerName).Obj()
	lowPriorityPod :=
		st.MakePod().Name("lpp").Namespace("ns4").UID("lppns1").Priority(lowPriority).SchedulerName(testSchedulerName).Obj()

	pod := st.MakePod().Name("pod1").UID("pod1").SchedulerName(testSchedulerName).Obj()
	assumedPodOnNodeA := pod.DeepCopy()
	assumedPodOnNodeA.Spec.NodeName = "node-a"
	boundPodOnNodeB := pod.DeepCopy()
	boundPodOnNodeB.Spec.NodeName = "node-b"

	medPriorityNominatedPodOnNodeA := pod.DeepCopy()
	medPriorityNominatedPodOnNodeA.Status.NominatedNodeName = "node-a"
	medPriorityNominatedPodOnNodeA.Spec.Priority = &midPriority

	unschedulablePods := []*v1.Pod{highPriorityPod, medPriorityPod, lowPriorityPod}

	// Make pods schedulable on Delete event when QHints are enabled
	queueHintForPodDelete := func(logger klog.Logger, pod *v1.Pod, oldObj, newObj interface{}) (fwk.QueueingHint, error) {
		return fwk.Queue, nil
	}
	queueingHintMap := internalqueue.QueueingHintMapPerProfile{
		testSchedulerName: {
			framework.EventAssignedPodDelete: {
				{
					PluginName:     "fooPlugin1",
					QueueingHintFn: queueHintForPodDelete,
				},
			},
		},
	}

	tests := []struct {
		name                  string
		oldPod                *v1.Pod
		newPod                *v1.Pod
		assumedPod            *v1.Pod
		wantInActiveOrBackoff sets.Set[string]
	}{
		{
			name:                  "assumed pod externally bound to a different node should wake up other pods",
			oldPod:                pod,
			newPod:                boundPodOnNodeB,
			assumedPod:            assumedPodOnNodeA,
			wantInActiveOrBackoff: sets.New(lowPriorityPod.Name, medPriorityPod.Name, highPriorityPod.Name),
		},
		{
			name:                  "nominated pod externally bound to a different node should wake up other pods with lower or equaL priority",
			oldPod:                medPriorityNominatedPodOnNodeA,
			newPod:                boundPodOnNodeB,
			wantInActiveOrBackoff: sets.New(lowPriorityPod.Name, medPriorityPod.Name),
		},
	}

	for _, tt := range tests {
		for _, qHintEnabled := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s, with queuehint(%v)", tt.name, qHintEnabled), func(t *testing.T) {
				if !qHintEnabled {
					featuregatetesting.SetFeatureGateEmulationVersionDuringTest(t, utilfeature.DefaultFeatureGate, version.MustParse("1.33"))
					featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.SchedulerQueueingHints, false)
				}

				logger, ctx := ktesting.NewTestContext(t)
				ctx, cancel := context.WithCancel(ctx)
				defer cancel()

				var objs []runtime.Object
				for _, pod := range unschedulablePods {
					objs = append(objs, pod)
				}
				client := fake.NewClientset(objs...)
				informerFactory := informers.NewSharedInformerFactory(client, 0)

				// apiDispatcher is unused in the test, but intializing it anyway.
				apiDispatcher := apidispatcher.New(client, 16, apicalls.Relevances)
				apiDispatcher.Run(logger)
				defer apiDispatcher.Close()

				recorder := metrics.NewMetricsAsyncRecorder(3, 20*time.Microsecond, ctx.Done())
				queue := internalqueue.NewPriorityQueue(
					newDefaultQueueSort(),
					informerFactory,
					internalqueue.WithMetricsRecorder(recorder),
					internalqueue.WithQueueingHintMapPerProfile(queueingHintMap),
					internalqueue.WithAPIDispatcher(apiDispatcher),
					// disable backoff queue
					internalqueue.WithPodInitialBackoffDuration(0),
					internalqueue.WithPodMaxBackoffDuration(0))
				schedulerCache := internalcache.New(ctx, 30*time.Second, nil)

				// Put test pods into unschedulable queue
				for _, pod := range unschedulablePods {
					queue.Add(logger, pod)
					poppedPod, err := queue.Pop(logger)
					if err != nil {
						t.Fatalf("Pop failed: %v", err)
					}
					poppedPod.UnschedulablePlugins = sets.New("fooPlugin1")
					if err := queue.AddUnschedulableIfNotPresent(logger, poppedPod, queue.SchedulingCycle()); err != nil {
						t.Errorf("Unexpected error from AddUnschedulableIfNotPresent: %v", err)
					}
				}

				s, _, err := initScheduler(ctx, schedulerCache, queue, apiDispatcher, client, informerFactory)
				if err != nil {
					t.Fatalf("Failed to initialize test scheduler: %v", err)
				}

				if tt.assumedPod != nil {
					err := schedulerCache.AssumePod(logger, tt.assumedPod)
					if err != nil {
						t.Fatalf("Failed to assume pod: %v", err)
					}
				}

				if len(s.SchedulingQueue.PodsInActiveQ()) > 0 {
					t.Errorf("No pods were expected to be in the activeQ before the update, but there were %v", s.SchedulingQueue.PodsInActiveQ())
				}

				s.updatePod(tt.oldPod, tt.newPod)

				podsInActiveOrBackoff := s.SchedulingQueue.PodsInActiveQ()
				podsInActiveOrBackoff = append(podsInActiveOrBackoff, s.SchedulingQueue.PodsInBackoffQ()...)
				if len(podsInActiveOrBackoff) != len(tt.wantInActiveOrBackoff) {
					t.Errorf("Different number of pods were expected to be in the activeQ or backoffQ, but found actual %v vs. expected %v", podsInActiveOrBackoff, tt.wantInActiveOrBackoff)
				}
				for _, pod := range podsInActiveOrBackoff {
					if !tt.wantInActiveOrBackoff.Has(pod.Name) {
						t.Errorf("Found unexpected pod in activeQ or backoffQ: %s", pod.Name)
					}
				}
			})
		}
	}
}

func TestDeletePod(t *testing.T) {
	pod := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").SchedulerName("supported-scheduler").Obj()
	otherPod := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").SchedulerName("other-scheduler").Obj()
	scheduledPod := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Node("node1").SchedulerName("supported-scheduler").Obj()
	otherScheduledPod := st.MakePod().Name("pod1").Namespace("ns1").UID("pod1").Node("node1").SchedulerName("other-scheduler").Obj()

	tests := []struct {
		name            string
		initialPod      *v1.Pod
		assumed         bool
		waitingOnPermit bool
		podToDelete     any
	}{
		{
			name:        "delete unscheduled pod",
			initialPod:  pod,
			podToDelete: pod,
		},
		{
			name:        "delete unscheduled pod with other scheduler name",
			initialPod:  otherPod,
			podToDelete: otherPod,
		},
		{
			name:        "delete unscheduled pod with unknown state",
			initialPod:  pod,
			podToDelete: cache.DeletedFinalStateUnknown{Obj: pod},
		},
		{
			name:        "delete assumed pod",
			initialPod:  scheduledPod,
			assumed:     true,
			podToDelete: pod,
		},
		{
			name:        "delete scheduled pod",
			initialPod:  scheduledPod,
			podToDelete: scheduledPod,
		},
		{
			name:        "delete scheduled pod with other scheduler name",
			initialPod:  otherScheduledPod,
			podToDelete: otherScheduledPod,
		},
		{
			name:        "delete scheduled pod with unknown state",
			initialPod:  scheduledPod,
			podToDelete: cache.DeletedFinalStateUnknown{Obj: scheduledPod},
		},
		{
			name:        "delete scheduled pod with unknown older state",
			initialPod:  scheduledPod,
			podToDelete: cache.DeletedFinalStateUnknown{Obj: pod},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			logger, ctx := ktesting.NewTestContext(t)
			ctx, cancel := context.WithCancel(ctx)
			defer cancel()

			registerPluginFuncs := []tf.RegisterPluginFunc{
				tf.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				tf.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}
			waitingPods := frameworkruntime.NewWaitingPodsMap()
			schedFramework, err := tf.NewFramework(
				ctx,
				registerPluginFuncs,
				"supported-scheduler",
				frameworkruntime.WithWaitingPods(waitingPods),
			)
			if err != nil {
				t.Fatalf("Failed to create framework: %v", err)
			}
			sched := &Scheduler{
				Cache:           internalcache.New(ctx, nil),
				SchedulingQueue: internalqueue.NewTestQueue(ctx, nil),
				logger:          logger,
				Profiles: profile.Map{
					"supported-scheduler": schedFramework,
				},
			}

			if tt.assumed {
				err := sched.Cache.AssumePod(logger, tt.initialPod)
				if err != nil {
					t.Fatalf("Failed to assume pod: %v", err)
				}
			} else {
				sched.addPod(tt.initialPod)
			}

			sched.deletePod(tt.podToDelete)

			_, err = sched.Cache.GetPod(tt.initialPod)
			if err == nil {
				t.Errorf("Unexpected pod in cache after removal")
			}
			_, ok := sched.SchedulingQueue.GetPod(tt.initialPod.Name, tt.initialPod.Namespace)
			if ok {
				t.Errorf("Unexpected pod in scheduling queue after removal")
			}
		})
	}
}
