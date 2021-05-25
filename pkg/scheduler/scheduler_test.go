/*
Copyright 2014 The Kubernetes Authors.

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
	"errors"
	"fmt"
	"io/ioutil"
	"math"
	"os"
	"path"
	"reflect"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	eventsv1 "k8s.io/api/events/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	apiruntime "k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	clienttesting "k8s.io/client-go/testing"
	clientcache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	pvutil "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/util"
	"k8s.io/kubernetes/pkg/controller/volume/scheduling"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/core"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/feature"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/podtopologyspread"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	fakecache "k8s.io/kubernetes/pkg/scheduler/internal/cache/fake"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

func podWithID(id, desiredHost string) *v1.Pod {
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: id,
			UID:  types.UID(id),
		},
		Spec: v1.PodSpec{
			NodeName:      desiredHost,
			SchedulerName: testSchedulerName,
		},
	}
}

func deletingPod(id string) *v1.Pod {
	deletionTimestamp := metav1.Now()
	return &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:              id,
			UID:               types.UID(id),
			DeletionTimestamp: &deletionTimestamp,
		},
		Spec: v1.PodSpec{
			NodeName:      "",
			SchedulerName: testSchedulerName,
		},
	}
}

func podWithPort(id, desiredHost string, port int) *v1.Pod {
	pod := podWithID(id, desiredHost)
	pod.Spec.Containers = []v1.Container{
		{Name: "ctr", Ports: []v1.ContainerPort{{HostPort: int32(port)}}},
	}
	return pod
}

func podWithResources(id, desiredHost string, limits v1.ResourceList, requests v1.ResourceList) *v1.Pod {
	pod := podWithID(id, desiredHost)
	pod.Spec.Containers = []v1.Container{
		{Name: "ctr", Resources: v1.ResourceRequirements{Limits: limits, Requests: requests}},
	}
	return pod
}

type mockScheduler struct {
	result core.ScheduleResult
	err    error
}

type mockMainScheduler struct {
	Scheduler
	result ScheduleResult
	err    error
}

func (mockSched *mockMainScheduler) Schedule(ctx context.Context, fwk framework.Framework, state *framework.CycleState, pod *v1.Pod) (result ScheduleResult, err error) {
	//return ScheduleResult{SuggestedHost: "machine1", EvaluatedNodes: 1, FeasibleNodes: 1}, nil

	return mockSched.result, mockSched.err
}

func (es mockScheduler) Schedule(ctx context.Context, fwk framework.Framework, state *framework.CycleState, pod *v1.Pod) (core.ScheduleResult, error) {
	return es.result, es.err
}

func (es mockScheduler) Extenders() []framework.Extender {
	return nil
}

func (es mockScheduler) FindNodesThatFitPod(ctx context.Context, fwk framework.Framework, state *framework.CycleState, pod *v1.Pod) ([]*v1.Node, framework.Diagnosis, error) {
	diagnosis := framework.Diagnosis{
		NodeToStatusMap:      make(framework.NodeToStatusMap),
		UnschedulablePlugins: sets.NewString(),
	}
	return nil, diagnosis, nil
}

func (es mockScheduler) NodeInfoSnapshot() *internalcache.Snapshot {
	nodeNames := []string{"machine1"}
	nodes := make([]*v1.Node, 0, len(nodeNames))
	for _, nodeName := range nodeNames {
		nodes = append(nodes, &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: nodeName, UID: types.UID("machine1")}})
	}

	pods := make([]*v1.Pod, 0, 1)
	pods = append(pods, &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID("foo")}})
	snapshot := internalcache.NewSnapshot(pods, nodes)
	return snapshot
}

func (es mockScheduler) PrioritizeNodes(
	ctx context.Context,
	fwk framework.Framework,
	state *framework.CycleState,
	pod *v1.Pod,
	nodes []*v1.Node,
) (framework.NodeScoreList, error) {
	return nil, nil
}

func (es mockScheduler) SelectHost(nodeScoreList framework.NodeScoreList) (string, error) {
	return "", nil
}

func (es mockScheduler) Snapshot() error {
	return nil
}

func TestSchedulerCreation(t *testing.T) {
	invalidRegistry := map[string]frameworkruntime.PluginFactory{
		defaultbinder.Name: defaultbinder.New,
	}
	validRegistry := map[string]frameworkruntime.PluginFactory{
		"Foo": defaultbinder.New,
	}
	cases := []struct {
		name         string
		opts         []Option
		wantErr      string
		wantProfiles []string
	}{
		{
			name:         "default scheduler",
			wantProfiles: []string{"default-scheduler"},
		},
		{
			name:         "valid out-of-tree registry",
			opts:         []Option{WithFrameworkOutOfTreeRegistry(validRegistry)},
			wantProfiles: []string{"default-scheduler"},
		},
		{
			name:         "repeated plugin name in out-of-tree plugin",
			opts:         []Option{WithFrameworkOutOfTreeRegistry(invalidRegistry)},
			wantProfiles: []string{"default-scheduler"},
			wantErr:      "a plugin named DefaultBinder already exists",
		},
		{
			name: "multiple profiles",
			opts: []Option{WithProfiles(
				schedulerapi.KubeSchedulerProfile{SchedulerName: "foo"},
				schedulerapi.KubeSchedulerProfile{SchedulerName: "bar"},
			)},
			wantProfiles: []string{"bar", "foo"},
		},
		{
			name: "Repeated profiles",
			opts: []Option{WithProfiles(
				schedulerapi.KubeSchedulerProfile{SchedulerName: "foo"},
				schedulerapi.KubeSchedulerProfile{SchedulerName: "bar"},
				schedulerapi.KubeSchedulerProfile{SchedulerName: "foo"},
			)},
			wantErr: "duplicate profile with scheduler name \"foo\"",
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})

			stopCh := make(chan struct{})
			defer close(stopCh)
			s, err := New(
				client,
				informerFactory,
				profile.NewRecorderFactory(eventBroadcaster),
				stopCh,
				tc.opts...,
			)

			if len(tc.wantErr) != 0 {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("got error %q, want %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Failed to create scheduler: %v", err)
			}
			profiles := make([]string, 0, len(s.Profiles))
			for name := range s.Profiles {
				profiles = append(profiles, name)
			}
			sort.Strings(profiles)
			if diff := cmp.Diff(tc.wantProfiles, profiles); diff != "" {
				t.Errorf("unexpected profiles (-want, +got):\n%s", diff)
			}
		})
	}
}

func TestSchedulerScheduleOne(t *testing.T) {
	testNode := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}}
	client := clientsetfake.NewSimpleClientset(&testNode)
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	errS := errors.New("scheduler")
	errB := errors.New("binder")
	preBindErr := errors.New("on PreBind")

	table := []struct {
		name                string
		injectBindError     error
		sendPod             *v1.Pod
		algo                core.ScheduleAlgorithm
		scheduleResult      ScheduleResult
		err                 error
		registerPluginFuncs []st.RegisterPluginFunc
		expectErrorPod      *v1.Pod
		expectForgetPod     *v1.Pod
		expectAssumedPod    *v1.Pod
		expectError         error
		expectBind          *v1.Binding
		eventReason         string
	}{
		{
			name:           "error reserve pod",
			sendPod:        podWithID("foo", ""),
			algo:           mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			scheduleResult: ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1},
			registerPluginFuncs: []st.RegisterPluginFunc{
				st.RegisterReservePlugin("FakeReserve", st.NewFakeReservePlugin(framework.NewStatus(framework.Error, "reserve error"))),
			},
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			expectAssumedPod: podWithID("foo", testNode.Name),
			expectError:      fmt.Errorf(`running Reserve plugin "FakeReserve": %w`, errors.New("reserve error")),
			eventReason:      "FailedScheduling",
		},
		{
			name:           "error permit pod",
			sendPod:        podWithID("foo", ""),
			algo:           mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			scheduleResult: ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1},
			registerPluginFuncs: []st.RegisterPluginFunc{
				st.RegisterPermitPlugin("FakePermit", st.NewFakePermitPlugin(framework.NewStatus(framework.Error, "permit error"), time.Minute)),
			},
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			expectAssumedPod: podWithID("foo", testNode.Name),
			expectError:      fmt.Errorf(`running Permit plugin "FakePermit": %w`, errors.New("permit error")),
			eventReason:      "FailedScheduling",
		},
		{
			name:           "error prebind pod",
			sendPod:        podWithID("foo", ""),
			algo:           mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			scheduleResult: ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1},
			registerPluginFuncs: []st.RegisterPluginFunc{
				st.RegisterPreBindPlugin("FakePreBind", st.NewFakePreBindPlugin(framework.AsStatus(preBindErr))),
			},
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			expectAssumedPod: podWithID("foo", testNode.Name),
			expectError:      fmt.Errorf(`running PreBind plugin "FakePreBind": %w`, preBindErr),
			eventReason:      "FailedScheduling",
		},
		{
			name:             "bind assumed pod scheduled",
			sendPod:          podWithID("foo", ""),
			algo:             mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			scheduleResult:   ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1},
			expectBind:       &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: testNode.Name}},
			expectAssumedPod: podWithID("foo", testNode.Name),
			eventReason:      "Scheduled",
		},
		{
			name:           "error pod failed scheduling",
			sendPod:        podWithID("foo", ""),
			algo:           mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, errS},
			scheduleResult: ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1},
			err:            errS,
			expectError:    errS,
			expectErrorPod: podWithID("foo", ""),
			eventReason:    "FailedScheduling",
		},
		{
			name:             "error bind forget pod failed scheduling",
			sendPod:          podWithID("foo", ""),
			algo:             mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			scheduleResult:   ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1},
			expectBind:       &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: testNode.Name}},
			expectAssumedPod: podWithID("foo", testNode.Name),
			injectBindError:  errB,
			expectError:      fmt.Errorf(`binding rejected: %w`, fmt.Errorf("running Bind plugin %q: %w", "DefaultBinder", errors.New("binder"))),
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			eventReason:      "FailedScheduling",
		},
		{
			name:        "deleting pod",
			sendPod:     deletingPod("foo"),
			algo:        mockScheduler{core.ScheduleResult{}, nil},
			eventReason: "FailedScheduling",
		},
	}

	stop := make(chan struct{})
	defer close(stop)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	informerFactory.Start(stop)
	informerFactory.WaitForCacheSync(stop)

	for _, item := range table {
		t.Run(item.name, func(t *testing.T) {
			var gotError error
			var gotPod *v1.Pod
			var gotForgetPod *v1.Pod
			var gotAssumedPod *v1.Pod
			var gotBinding *v1.Binding
			sCache := &fakecache.Cache{
				ForgetFunc: func(pod *v1.Pod) {
					gotForgetPod = pod
				},
				AssumeFunc: func(pod *v1.Pod) {
					gotAssumedPod = pod
				},
				IsAssumedPodFunc: func(pod *v1.Pod) bool {
					if pod == nil || gotAssumedPod == nil {
						return false
					}
					return pod.UID == gotAssumedPod.UID
				},
			}
			client := clientsetfake.NewSimpleClientset(item.sendPod)
			client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if action.GetSubresource() != "binding" {
					return false, nil, nil
				}
				gotBinding = action.(clienttesting.CreateAction).GetObject().(*v1.Binding)
				return true, gotBinding, item.injectBindError
			})
			registerPluginFuncs := append(item.registerPluginFuncs,
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			)
			fwk, err := st.NewFramework(registerPluginFuncs,
				testSchedulerName,
				frameworkruntime.WithClientSet(client),
				frameworkruntime.WithEventRecorder(eventBroadcaster.NewRecorder(scheme.Scheme, testSchedulerName)))
			if err != nil {
				t.Fatal(err)
			}

			// s := &Scheduler{
			// 	SchedulerCache: sCache,
			// 	Algorithm:      item.algo,
			// 	client:         client,
			// 	Error: func(p *framework.QueuedPodInfo, err error) {
			// 		gotPod = p.Pod
			// 		gotError = err
			// 	},
			// 	NextPod: func() *framework.QueuedPodInfo {
			// 		return &framework.QueuedPodInfo{PodInfo: framework.NewPodInfo(item.sendPod)}
			// 	},
			// 	Profiles: profile.Map{
			// 		testSchedulerName: fwk,
			// 	},
			// }

			s := mockMainScheduler{
				Scheduler{
					SchedulerCache: sCache,
					Algorithm:      item.algo,
					client:         client,
					Error: func(p *framework.QueuedPodInfo, err error) {
						gotPod = p.Pod
						gotError = err
					},
					NextPod: func() *framework.QueuedPodInfo {
						return &framework.QueuedPodInfo{PodInfo: framework.NewPodInfo(item.sendPod)}
					},
					Profiles: profile.Map{
						testSchedulerName: fwk,
					},
				},
				item.scheduleResult,
				item.err,
			}

			// *** interface function hajack
			s.i = &s

			called := make(chan struct{})
			stopFunc := eventBroadcaster.StartEventWatcher(func(obj runtime.Object) {
				e, _ := obj.(*eventsv1.Event)
				if e.Reason != item.eventReason {
					t.Errorf("got event %v, want %v", e.Reason, item.eventReason)
				}
				close(called)
			})
			s.scheduleOne(context.Background())
			<-called
			if e, a := item.expectAssumedPod, gotAssumedPod; !reflect.DeepEqual(e, a) {
				t.Errorf("assumed pod: wanted %v, got %v", e, a)
			}
			if e, a := item.expectErrorPod, gotPod; !reflect.DeepEqual(e, a) {
				t.Errorf("error pod: wanted %v, got %v", e, a)
			}
			if e, a := item.expectForgetPod, gotForgetPod; !reflect.DeepEqual(e, a) {
				t.Errorf("forget pod: wanted %v, got %v", e, a)
			}
			if e, a := item.expectError, gotError; !reflect.DeepEqual(e, a) {
				t.Errorf("error: wanted %v, got %v", e, a)
			}
			if diff := cmp.Diff(item.expectBind, gotBinding); diff != "" {
				t.Errorf("got binding diff (-want, +got): %s", diff)
			}
			stopFunc()
		})
	}
}

type fakeNodeSelectorArgs struct {
	NodeName string `json:"nodeName"`
}

type fakeNodeSelector struct {
	fakeNodeSelectorArgs
}

func newFakeNodeSelector(args runtime.Object, _ framework.Handle) (framework.Plugin, error) {
	pl := &fakeNodeSelector{}
	if err := frameworkruntime.DecodeInto(args, &pl.fakeNodeSelectorArgs); err != nil {
		return nil, err
	}
	return pl, nil
}

func (s *fakeNodeSelector) Name() string {
	return "FakeNodeSelector"
}

func (s *fakeNodeSelector) Filter(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	if nodeInfo.Node().Name != s.NodeName {
		return framework.NewStatus(framework.UnschedulableAndUnresolvable)
	}
	return nil
}

func TestSchedulerMultipleProfilesScheduling(t *testing.T) {
	nodes := []runtime.Object{
		st.MakeNode().Name("machine1").UID("machine1").Obj(),
		st.MakeNode().Name("machine2").UID("machine2").Obj(),
		st.MakeNode().Name("machine3").UID("machine3").Obj(),
	}
	pods := []*v1.Pod{
		st.MakePod().Name("pod1").UID("pod1").SchedulerName("match-machine3").Obj(),
		st.MakePod().Name("pod2").UID("pod2").SchedulerName("match-machine2").Obj(),
		st.MakePod().Name("pod3").UID("pod3").SchedulerName("match-machine2").Obj(),
		st.MakePod().Name("pod4").UID("pod4").SchedulerName("match-machine3").Obj(),
	}
	wantBindings := map[string]string{
		"pod1": "machine3",
		"pod2": "machine2",
		"pod3": "machine2",
		"pod4": "machine3",
	}
	wantControllers := map[string]string{
		"pod1": "match-machine3",
		"pod2": "match-machine2",
		"pod3": "match-machine2",
		"pod4": "match-machine3",
	}

	// Set up scheduler for the 3 nodes.
	// We use a fake filter that only allows one particular node. We create two
	// profiles, each with a different node in the filter configuration.
	objs := append([]runtime.Object{
		&v1.Namespace{ObjectMeta: metav1.ObjectMeta{Name: ""}}}, nodes...)
	client := clientsetfake.NewSimpleClientset(objs...)
	broadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	informerFactory := informers.NewSharedInformerFactory(client, 0)
	sched, err := New(
		client,
		informerFactory,
		profile.NewRecorderFactory(broadcaster),
		ctx.Done(),
		WithProfiles(
			schedulerapi.KubeSchedulerProfile{SchedulerName: "match-machine2",
				Plugins: &schedulerapi.Plugins{
					Filter: schedulerapi.PluginSet{
						Enabled:  []schedulerapi.Plugin{{Name: "FakeNodeSelector"}},
						Disabled: []schedulerapi.Plugin{{Name: "*"}},
					}},
				PluginConfig: []schedulerapi.PluginConfig{
					{Name: "FakeNodeSelector",
						Args: &runtime.Unknown{Raw: []byte(`{"nodeName":"machine2"}`)},
					},
				},
			},
			schedulerapi.KubeSchedulerProfile{
				SchedulerName: "match-machine3",
				Plugins: &schedulerapi.Plugins{
					Filter: schedulerapi.PluginSet{
						Enabled:  []schedulerapi.Plugin{{Name: "FakeNodeSelector"}},
						Disabled: []schedulerapi.Plugin{{Name: "*"}},
					}},
				PluginConfig: []schedulerapi.PluginConfig{
					{Name: "FakeNodeSelector",
						Args: &runtime.Unknown{Raw: []byte(`{"nodeName":"machine3"}`)},
					},
				},
			},
		),
		WithFrameworkOutOfTreeRegistry(frameworkruntime.Registry{
			"FakeNodeSelector": newFakeNodeSelector,
		}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// Capture the bindings and events' controllers.
	var wg sync.WaitGroup
	wg.Add(2 * len(pods))
	bindings := make(map[string]string)
	client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		if action.GetSubresource() != "binding" {
			return false, nil, nil
		}
		binding := action.(clienttesting.CreateAction).GetObject().(*v1.Binding)
		bindings[binding.Name] = binding.Target.Name
		wg.Done()
		return true, binding, nil
	})
	controllers := make(map[string]string)
	stopFn := broadcaster.StartEventWatcher(func(obj runtime.Object) {
		e, ok := obj.(*eventsv1.Event)
		if !ok || e.Reason != "Scheduled" {
			return
		}
		controllers[e.Regarding.Name] = e.ReportingController
		wg.Done()
	})
	defer stopFn()

	// Run scheduler.
	informerFactory.Start(ctx.Done())
	informerFactory.WaitForCacheSync(ctx.Done())
	go sched.Run(ctx)

	// Send pods to be scheduled.
	for _, p := range pods {
		_, err := client.CoreV1().Pods("").Create(ctx, p, metav1.CreateOptions{})
		if err != nil {
			t.Fatal(err)
		}
	}
	wg.Wait()

	// Verify correct bindings and reporting controllers.
	if diff := cmp.Diff(wantBindings, bindings); diff != "" {
		t.Errorf("pods were scheduled incorrectly (-want, +got):\n%s", diff)
	}
	if diff := cmp.Diff(wantControllers, controllers); diff != "" {
		t.Errorf("events were reported with wrong controllers (-want, +got):\n%s", diff)
	}
}

func TestSchedulerNoPhantomPodAfterExpire(t *testing.T) {
	stop := make(chan struct{})
	defer close(stop)
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	scache := internalcache.New(100*time.Millisecond, stop)
	pod := podWithPort("pod.Name", "", 8080)
	node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}}
	scache.AddNode(&node)
	client := clientsetfake.NewSimpleClientset(&node)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	fns := []st.RegisterPluginFunc{
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		st.RegisterPluginAsExtensions(nodeports.Name, nodeports.New, "Filter", "PreFilter"),
	}
	scheduler, bindingChan, errChan := setupTestSchedulerWithOnePodOnNode(t, queuedPodStore, scache, informerFactory, stop, pod, &node, fns...)

	waitPodExpireChan := make(chan struct{})
	timeout := make(chan struct{})
	go func() {
		for {
			select {
			case <-timeout:
				return
			default:
			}
			pods, err := scache.PodCount()
			if err != nil {
				errChan <- fmt.Errorf("cache.List failed: %v", err)
				return
			}
			if pods == 0 {
				close(waitPodExpireChan)
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
	}()
	// waiting for the assumed pod to expire
	select {
	case err := <-errChan:
		t.Fatal(err)
	case <-waitPodExpireChan:
	case <-time.After(wait.ForeverTestTimeout):
		close(timeout)
		t.Fatalf("timeout timeout in waiting pod expire after %v", wait.ForeverTestTimeout)
	}

	// We use conflicted pod ports to incur fit predicate failure if first pod not removed.
	secondPod := podWithPort("bar", "", 8080)
	queuedPodStore.Add(secondPod)
	scheduler.scheduleOne(context.Background())
	select {
	case b := <-bindingChan:
		expectBinding := &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Name: "bar", UID: types.UID("bar")},
			Target:     v1.ObjectReference{Kind: "Node", Name: node.Name},
		}
		if !reflect.DeepEqual(expectBinding, b) {
			t.Errorf("binding want=%v, get=%v", expectBinding, b)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout in binding after %v", wait.ForeverTestTimeout)
	}
}

func TestSchedulerNoPhantomPodAfterDelete(t *testing.T) {
	stop := make(chan struct{})
	defer close(stop)
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	scache := internalcache.New(10*time.Minute, stop)
	firstPod := podWithPort("pod.Name", "", 8080)
	node := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}}
	scache.AddNode(&node)
	client := clientsetfake.NewSimpleClientset(&node)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	fns := []st.RegisterPluginFunc{
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		st.RegisterPluginAsExtensions(nodeports.Name, nodeports.New, "Filter", "PreFilter"),
	}
	scheduler, bindingChan, errChan := setupTestSchedulerWithOnePodOnNode(t, queuedPodStore, scache, informerFactory, stop, firstPod, &node, fns...)

	// We use conflicted pod ports to incur fit predicate failure.
	secondPod := podWithPort("bar", "", 8080)
	queuedPodStore.Add(secondPod)
	// queuedPodStore: [bar:8080]
	// cache: [(assumed)foo:8080]

	scheduler.scheduleOne(context.Background())
	select {
	case err := <-errChan:
		expectErr := &framework.FitError{
			Pod:         secondPod,
			NumAllNodes: 1,
			Diagnosis: framework.Diagnosis{
				NodeToStatusMap: framework.NodeToStatusMap{
					node.Name: framework.NewStatus(framework.Unschedulable, nodeports.ErrReason).WithFailedPlugin(nodeports.Name),
				},
				UnschedulablePlugins: sets.NewString(nodeports.Name),
			},
		}
		if !reflect.DeepEqual(expectErr, err) {
			t.Errorf("err want=%v, get=%v", expectErr, err)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout in fitting after %v", wait.ForeverTestTimeout)
	}

	// We mimic the workflow of cache behavior when a pod is removed by user.
	// Note: if the schedulernodeinfo timeout would be super short, the first pod would expire
	// and would be removed itself (without any explicit actions on schedulernodeinfo). Even in that case,
	// explicitly AddPod will as well correct the behavior.
	firstPod.Spec.NodeName = node.Name
	if err := scache.AddPod(firstPod); err != nil {
		t.Fatalf("err: %v", err)
	}
	if err := scache.RemovePod(firstPod); err != nil {
		t.Fatalf("err: %v", err)
	}

	queuedPodStore.Add(secondPod)
	scheduler.scheduleOne(context.Background())
	select {
	case b := <-bindingChan:
		expectBinding := &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Name: "bar", UID: types.UID("bar")},
			Target:     v1.ObjectReference{Kind: "Node", Name: node.Name},
		}
		if !reflect.DeepEqual(expectBinding, b) {
			t.Errorf("binding want=%v, get=%v", expectBinding, b)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout in binding after %v", wait.ForeverTestTimeout)
	}
}

// queuedPodStore: pods queued before processing.
// cache: scheduler cache that might contain assumed pods.
func setupTestSchedulerWithOnePodOnNode(t *testing.T, queuedPodStore *clientcache.FIFO, scache internalcache.Cache,
	informerFactory informers.SharedInformerFactory, stop chan struct{}, pod *v1.Pod, node *v1.Node, fns ...st.RegisterPluginFunc) (*Scheduler, chan *v1.Binding, chan error) {

	scheduler, bindingChan, errChan := setupTestScheduler(queuedPodStore, scache, informerFactory, nil, fns...)

	informerFactory.Start(stop)
	informerFactory.WaitForCacheSync(stop)

	queuedPodStore.Add(pod)
	// queuedPodStore: [foo:8080]
	// cache: []

	scheduler.scheduleOne(context.Background())
	// queuedPodStore: []
	// cache: [(assumed)foo:8080]

	select {
	case b := <-bindingChan:
		expectBinding := &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{Name: pod.Name, UID: types.UID(pod.Name)},
			Target:     v1.ObjectReference{Kind: "Node", Name: node.Name},
		}
		if !reflect.DeepEqual(expectBinding, b) {
			t.Errorf("binding want=%v, get=%v", expectBinding, b)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
	}
	return scheduler, bindingChan, errChan
}

func TestSchedulerFailedSchedulingReasons(t *testing.T) {
	stop := make(chan struct{})
	defer close(stop)
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	scache := internalcache.New(10*time.Minute, stop)

	// Design the baseline for the pods, and we will make nodes that don't fit it later.
	var cpu = int64(4)
	var mem = int64(500)
	podWithTooBigResourceRequests := podWithResources("bar", "", v1.ResourceList{
		v1.ResourceCPU:    *(resource.NewQuantity(cpu, resource.DecimalSI)),
		v1.ResourceMemory: *(resource.NewQuantity(mem, resource.DecimalSI)),
	}, v1.ResourceList{
		v1.ResourceCPU:    *(resource.NewQuantity(cpu, resource.DecimalSI)),
		v1.ResourceMemory: *(resource.NewQuantity(mem, resource.DecimalSI)),
	})

	// create several nodes which cannot schedule the above pod
	var nodes []*v1.Node
	var objects []runtime.Object
	for i := 0; i < 100; i++ {
		uid := fmt.Sprintf("machine%v", i)
		node := v1.Node{
			ObjectMeta: metav1.ObjectMeta{Name: uid, UID: types.UID(uid)},
			Status: v1.NodeStatus{
				Capacity: v1.ResourceList{
					v1.ResourceCPU:    *(resource.NewQuantity(cpu/2, resource.DecimalSI)),
					v1.ResourceMemory: *(resource.NewQuantity(mem/5, resource.DecimalSI)),
					v1.ResourcePods:   *(resource.NewQuantity(10, resource.DecimalSI)),
				},
				Allocatable: v1.ResourceList{
					v1.ResourceCPU:    *(resource.NewQuantity(cpu/2, resource.DecimalSI)),
					v1.ResourceMemory: *(resource.NewQuantity(mem/5, resource.DecimalSI)),
					v1.ResourcePods:   *(resource.NewQuantity(10, resource.DecimalSI)),
				}},
		}
		scache.AddNode(&node)
		nodes = append(nodes, &node)
		objects = append(objects, &node)
	}
	client := clientsetfake.NewSimpleClientset(objects...)
	informerFactory := informers.NewSharedInformerFactory(client, 0)

	// Create expected failure reasons for all the nodes. Hopefully they will get rolled up into a non-spammy summary.
	failedNodeStatues := framework.NodeToStatusMap{}
	for _, node := range nodes {
		failedNodeStatues[node.Name] = framework.NewStatus(
			framework.Unschedulable,
			fmt.Sprintf("Insufficient %v", v1.ResourceCPU),
			fmt.Sprintf("Insufficient %v", v1.ResourceMemory),
		).WithFailedPlugin(noderesources.FitName)
	}
	fns := []st.RegisterPluginFunc{
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		st.RegisterPluginAsExtensions(noderesources.FitName, func(plArgs apiruntime.Object, fh framework.Handle) (framework.Plugin, error) {
			return noderesources.NewFit(plArgs, fh, feature.Features{})
		}, "Filter", "PreFilter"),
	}
	scheduler, _, errChan := setupTestScheduler(queuedPodStore, scache, informerFactory, nil, fns...)

	informerFactory.Start(stop)
	informerFactory.WaitForCacheSync(stop)

	queuedPodStore.Add(podWithTooBigResourceRequests)
	scheduler.scheduleOne(context.Background())
	select {
	case err := <-errChan:
		expectErr := &framework.FitError{
			Pod:         podWithTooBigResourceRequests,
			NumAllNodes: len(nodes),
			Diagnosis: framework.Diagnosis{
				NodeToStatusMap:      failedNodeStatues,
				UnschedulablePlugins: sets.NewString(noderesources.FitName),
			},
		}
		if len(fmt.Sprint(expectErr)) > 150 {
			t.Errorf("message is too spammy ! %v ", len(fmt.Sprint(expectErr)))
		}
		if !reflect.DeepEqual(expectErr, err) {
			t.Errorf("\n err \nWANT=%+v,\nGOT=%+v", expectErr, err)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
	}
}

// queuedPodStore: pods queued before processing.
// scache: scheduler cache that might contain assumed pods.
func setupTestScheduler(queuedPodStore *clientcache.FIFO, scache internalcache.Cache, informerFactory informers.SharedInformerFactory, broadcaster events.EventBroadcaster, fns ...st.RegisterPluginFunc) (*Scheduler, chan *v1.Binding, chan error) {
	bindingChan := make(chan *v1.Binding, 1)
	client := clientsetfake.NewSimpleClientset()
	client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
		var b *v1.Binding
		if action.GetSubresource() == "binding" {
			b := action.(clienttesting.CreateAction).GetObject().(*v1.Binding)
			bindingChan <- b
		}
		return true, b, nil
	})

	var recorder events.EventRecorder
	if broadcaster != nil {
		recorder = broadcaster.NewRecorder(scheme.Scheme, testSchedulerName)
	} else {
		recorder = &events.FakeRecorder{}
	}

	fwk, _ := st.NewFramework(
		fns,
		testSchedulerName,
		frameworkruntime.WithClientSet(client),
		frameworkruntime.WithEventRecorder(recorder),
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithPodNominator(internalqueue.NewPodNominator()),
	)

	algo := core.NewGenericScheduler(
		scache,
		internalcache.NewEmptySnapshot(),
		[]framework.Extender{},
		schedulerapi.DefaultPercentageOfNodesToScore,
	)

	errChan := make(chan error, 1)
	sched := &Scheduler{
		SchedulerCache: scache,
		Algorithm:      algo,
		NextPod: func() *framework.QueuedPodInfo {
			return &framework.QueuedPodInfo{PodInfo: framework.NewPodInfo(clientcache.Pop(queuedPodStore).(*v1.Pod))}
		},
		Error: func(p *framework.QueuedPodInfo, err error) {
			errChan <- err
		},
		Profiles: profile.Map{
			testSchedulerName: fwk,
		},
		client: client,
	}
	sched.i = sched

	return sched, bindingChan, errChan
}

func setupTestSchedulerWithVolumeBinding(volumeBinder scheduling.SchedulerVolumeBinder, stop <-chan struct{}, broadcaster events.EventBroadcaster) (*Scheduler, chan *v1.Binding, chan error) {
	testNode := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "machine1", UID: types.UID("machine1")}}
	queuedPodStore := clientcache.NewFIFO(clientcache.MetaNamespaceKeyFunc)
	pod := podWithID("foo", "")
	pod.Namespace = "foo-ns"
	pod.Spec.Volumes = append(pod.Spec.Volumes, v1.Volume{Name: "testVol",
		VolumeSource: v1.VolumeSource{PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{ClaimName: "testPVC"}}})
	queuedPodStore.Add(pod)
	scache := internalcache.New(10*time.Minute, stop)
	scache.AddNode(&testNode)
	testPVC := v1.PersistentVolumeClaim{ObjectMeta: metav1.ObjectMeta{Name: "testPVC", Namespace: pod.Namespace, UID: types.UID("testPVC")}}
	client := clientsetfake.NewSimpleClientset(&testNode, &testPVC)
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	pvcInformer := informerFactory.Core().V1().PersistentVolumeClaims()
	pvcInformer.Informer().GetStore().Add(&testPVC)

	fns := []st.RegisterPluginFunc{
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		st.RegisterPluginAsExtensions(volumebinding.Name, func(plArgs runtime.Object, handle framework.Handle) (framework.Plugin, error) {
			return &volumebinding.VolumeBinding{Binder: volumeBinder, PVCLister: pvcInformer.Lister()}, nil
		}, "PreFilter", "Filter", "Reserve", "PreBind"),
	}
	s, bindingChan, errChan := setupTestScheduler(queuedPodStore, scache, informerFactory, broadcaster, fns...)
	informerFactory.Start(stop)
	informerFactory.WaitForCacheSync(stop)
	return s, bindingChan, errChan
}

// This is a workaround because golint complains that errors cannot
// end with punctuation.  However, the real predicate error message does
// end with a period.
func makePredicateError(failReason string) error {
	s := fmt.Sprintf("0/1 nodes are available: %v.", failReason)
	return fmt.Errorf(s)
}

func TestSchedulerWithVolumeBinding(t *testing.T) {
	findErr := fmt.Errorf("find err")
	assumeErr := fmt.Errorf("assume err")
	bindErr := fmt.Errorf("bind err")
	client := clientsetfake.NewSimpleClientset()

	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})

	// This can be small because we wait for pod to finish scheduling first
	chanTimeout := 2 * time.Second

	table := []struct {
		name               string
		expectError        error
		expectPodBind      *v1.Binding
		expectAssumeCalled bool
		expectBindCalled   bool
		eventReason        string
		volumeBinderConfig *scheduling.FakeVolumeBinderConfig
	}{
		{
			name: "all bound",
			volumeBinderConfig: &scheduling.FakeVolumeBinderConfig{
				AllBound: true,
			},
			expectAssumeCalled: true,
			expectPodBind:      &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "foo-ns", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: "machine1"}},
			eventReason:        "Scheduled",
		},
		{
			name: "bound/invalid pv affinity",
			volumeBinderConfig: &scheduling.FakeVolumeBinderConfig{
				AllBound:    true,
				FindReasons: scheduling.ConflictReasons{scheduling.ErrReasonNodeConflict},
			},
			eventReason: "FailedScheduling",
			expectError: makePredicateError("1 node(s) had volume node affinity conflict"),
		},
		{
			name: "unbound/no matches",
			volumeBinderConfig: &scheduling.FakeVolumeBinderConfig{
				FindReasons: scheduling.ConflictReasons{scheduling.ErrReasonBindConflict},
			},
			eventReason: "FailedScheduling",
			expectError: makePredicateError("1 node(s) didn't find available persistent volumes to bind"),
		},
		{
			name: "bound and unbound unsatisfied",
			volumeBinderConfig: &scheduling.FakeVolumeBinderConfig{
				FindReasons: scheduling.ConflictReasons{scheduling.ErrReasonBindConflict, scheduling.ErrReasonNodeConflict},
			},
			eventReason: "FailedScheduling",
			expectError: makePredicateError("1 node(s) didn't find available persistent volumes to bind, 1 node(s) had volume node affinity conflict"),
		},
		{
			name:               "unbound/found matches/bind succeeds",
			volumeBinderConfig: &scheduling.FakeVolumeBinderConfig{},
			expectAssumeCalled: true,
			expectBindCalled:   true,
			expectPodBind:      &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "foo-ns", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: "machine1"}},
			eventReason:        "Scheduled",
		},
		{
			name: "predicate error",
			volumeBinderConfig: &scheduling.FakeVolumeBinderConfig{
				FindErr: findErr,
			},
			eventReason: "FailedScheduling",
			expectError: fmt.Errorf("running %q filter plugin: %v", volumebinding.Name, findErr),
		},
		{
			name: "assume error",
			volumeBinderConfig: &scheduling.FakeVolumeBinderConfig{
				AssumeErr: assumeErr,
			},
			expectAssumeCalled: true,
			eventReason:        "FailedScheduling",
			expectError:        fmt.Errorf("running Reserve plugin %q: %w", volumebinding.Name, assumeErr),
		},
		{
			name: "bind error",
			volumeBinderConfig: &scheduling.FakeVolumeBinderConfig{
				BindErr: bindErr,
			},
			expectAssumeCalled: true,
			expectBindCalled:   true,
			eventReason:        "FailedScheduling",
			expectError:        fmt.Errorf("running PreBind plugin %q: %w", volumebinding.Name, bindErr),
		},
	}

	for _, item := range table {
		t.Run(item.name, func(t *testing.T) {
			stop := make(chan struct{})
			fakeVolumeBinder := scheduling.NewFakeVolumeBinder(item.volumeBinderConfig)
			s, bindingChan, errChan := setupTestSchedulerWithVolumeBinding(fakeVolumeBinder, stop, eventBroadcaster)
			eventChan := make(chan struct{})
			stopFunc := eventBroadcaster.StartEventWatcher(func(obj runtime.Object) {
				e, _ := obj.(*eventsv1.Event)
				if e, a := item.eventReason, e.Reason; e != a {
					t.Errorf("expected %v, got %v", e, a)
				}
				close(eventChan)
			})
			s.scheduleOne(context.Background())
			// Wait for pod to succeed or fail scheduling
			select {
			case <-eventChan:
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatalf("scheduling timeout after %v", wait.ForeverTestTimeout)
			}
			stopFunc()
			// Wait for scheduling to return an error or succeed binding.
			var (
				gotErr  error
				gotBind *v1.Binding
			)
			select {
			case gotErr = <-errChan:
			case gotBind = <-bindingChan:
			case <-time.After(chanTimeout):
				t.Fatalf("did not receive pod binding or error after %v", chanTimeout)
			}
			if item.expectError != nil {
				if gotErr == nil || item.expectError.Error() != gotErr.Error() {
					t.Errorf("err \nWANT=%+v,\nGOT=%+v", item.expectError, gotErr)
				}
			} else if gotErr != nil {
				t.Errorf("err \nWANT=%+v,\nGOT=%+v", item.expectError, gotErr)
			}
			if !cmp.Equal(item.expectPodBind, gotBind) {
				t.Errorf("err \nWANT=%+v,\nGOT=%+v", item.expectPodBind, gotBind)
			}

			if item.expectAssumeCalled != fakeVolumeBinder.AssumeCalled {
				t.Errorf("expectedAssumeCall %v", item.expectAssumeCalled)
			}

			if item.expectBindCalled != fakeVolumeBinder.BindCalled {
				t.Errorf("expectedBindCall %v", item.expectBindCalled)
			}

			close(stop)
		})
	}
}

func TestInitPolicyFromFile(t *testing.T) {
	dir, err := ioutil.TempDir(os.TempDir(), "policy")
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	defer os.RemoveAll(dir)

	for i, test := range []struct {
		policy             string
		expectedPredicates sets.String
	}{
		// Test json format policy file
		{
			policy: `{
				"kind" : "Policy",
				"apiVersion" : "v1",
				"predicates" : [
					{"name" : "PredicateOne"},
					{"name" : "PredicateTwo"}
				],
				"priorities" : [
					{"name" : "PriorityOne", "weight" : 1},
					{"name" : "PriorityTwo", "weight" : 5}
				]
			}`,
			expectedPredicates: sets.NewString(
				"PredicateOne",
				"PredicateTwo",
			),
		},
		// Test yaml format policy file
		{
			policy: `apiVersion: v1
kind: Policy
predicates:
- name: PredicateOne
- name: PredicateTwo
priorities:
- name: PriorityOne
  weight: 1
- name: PriorityTwo
  weight: 5
`,
			expectedPredicates: sets.NewString(
				"PredicateOne",
				"PredicateTwo",
			),
		},
	} {
		file := fmt.Sprintf("scheduler-policy-config-file-%d", i)
		fullPath := path.Join(dir, file)

		if err := ioutil.WriteFile(fullPath, []byte(test.policy), 0644); err != nil {
			t.Fatalf("Failed writing a policy config file: %v", err)
		}

		policy := &schedulerapi.Policy{}

		if err := initPolicyFromFile(fullPath, policy); err != nil {
			t.Fatalf("Failed writing a policy config file: %v", err)
		}

		// Verify that the policy is initialized correctly.
		schedPredicates := sets.NewString()
		for _, p := range policy.Predicates {
			schedPredicates.Insert(p.Name)
		}
		schedPrioritizers := sets.NewString()
		for _, p := range policy.Priorities {
			schedPrioritizers.Insert(p.Name)
		}
		if !schedPredicates.Equal(test.expectedPredicates) {
			t.Errorf("Expected predicates %v, got %v", test.expectedPredicates, schedPredicates)
		}
	}
}

func TestSchedulerBinding(t *testing.T) {
	table := []struct {
		podName      string
		extenders    []framework.Extender
		wantBinderID int
		name         string
	}{
		{
			name:    "the extender is not a binder",
			podName: "pod0",
			extenders: []framework.Extender{
				&fakeExtender{isBinder: false, interestedPodName: "pod0"},
			},
			wantBinderID: -1, // default binding.
		},
		{
			name:    "one of the extenders is a binder and interested in pod",
			podName: "pod0",
			extenders: []framework.Extender{
				&fakeExtender{isBinder: false, interestedPodName: "pod0"},
				&fakeExtender{isBinder: true, interestedPodName: "pod0"},
			},
			wantBinderID: 1,
		},
		{
			name:    "one of the extenders is a binder, but not interested in pod",
			podName: "pod1",
			extenders: []framework.Extender{
				&fakeExtender{isBinder: false, interestedPodName: "pod1"},
				&fakeExtender{isBinder: true, interestedPodName: "pod0"},
			},
			wantBinderID: -1, // default binding.
		},
	}

	for _, test := range table {
		t.Run(test.name, func(t *testing.T) {
			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: test.podName,
				},
			}
			defaultBound := false
			client := clientsetfake.NewSimpleClientset(pod)
			client.PrependReactor("create", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
				if action.GetSubresource() == "binding" {
					defaultBound = true
				}
				return false, nil, nil
			})
			fwk, err := st.NewFramework([]st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			}, "", frameworkruntime.WithClientSet(client), frameworkruntime.WithEventRecorder(&events.FakeRecorder{}))
			if err != nil {
				t.Fatal(err)
			}
			stop := make(chan struct{})
			defer close(stop)
			scache := internalcache.New(100*time.Millisecond, stop)
			algo := core.NewGenericScheduler(
				scache,
				nil,
				test.extenders,
				0,
			)
			sched := Scheduler{
				Algorithm:      algo,
				SchedulerCache: scache,
			}
			err = sched.bind(context.Background(), fwk, pod, "node", nil)
			if err != nil {
				t.Error(err)
			}

			// Checking default binding.
			if wantBound := test.wantBinderID == -1; defaultBound != wantBound {
				t.Errorf("got bound with default binding: %v, want %v", defaultBound, wantBound)
			}

			// Checking extenders binding.
			for i, ext := range test.extenders {
				wantBound := i == test.wantBinderID
				if gotBound := ext.(*fakeExtender).gotBind; gotBound != wantBound {
					t.Errorf("got bound with extender #%d: %v, want %v", i, gotBound, wantBound)
				}
			}

		})
	}
}

func TestUpdatePod(t *testing.T) {
	tests := []struct {
		name                     string
		currentPodConditions     []v1.PodCondition
		newPodCondition          *v1.PodCondition
		currentNominatedNodeName string
		newNominatedNodeName     string
		expectedPatchRequests    int
		expectedPatchDataPattern string
	}{
		{
			name:                 "Should make patch request to add pod condition when there are none currently",
			currentPodConditions: []v1.PodCondition{},
			newPodCondition: &v1.PodCondition{
				Type:               "newType",
				Status:             "newStatus",
				LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 1, 1, 1, 1, time.UTC)),
				LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 1, 1, 1, 1, time.UTC)),
				Reason:             "newReason",
				Message:            "newMessage",
			},
			expectedPatchRequests:    1,
			expectedPatchDataPattern: `{"status":{"conditions":\[{"lastProbeTime":"2020-05-13T01:01:01Z","lastTransitionTime":".*","message":"newMessage","reason":"newReason","status":"newStatus","type":"newType"}]}}`,
		},
		{
			name: "Should make patch request to add a new pod condition when there is already one with another type",
			currentPodConditions: []v1.PodCondition{
				{
					Type:               "someOtherType",
					Status:             "someOtherTypeStatus",
					LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 11, 0, 0, 0, 0, time.UTC)),
					LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 10, 0, 0, 0, 0, time.UTC)),
					Reason:             "someOtherTypeReason",
					Message:            "someOtherTypeMessage",
				},
			},
			newPodCondition: &v1.PodCondition{
				Type:               "newType",
				Status:             "newStatus",
				LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 1, 1, 1, 1, time.UTC)),
				LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 1, 1, 1, 1, time.UTC)),
				Reason:             "newReason",
				Message:            "newMessage",
			},
			expectedPatchRequests:    1,
			expectedPatchDataPattern: `{"status":{"\$setElementOrder/conditions":\[{"type":"someOtherType"},{"type":"newType"}],"conditions":\[{"lastProbeTime":"2020-05-13T01:01:01Z","lastTransitionTime":".*","message":"newMessage","reason":"newReason","status":"newStatus","type":"newType"}]}}`,
		},
		{
			name: "Should make patch request to update an existing pod condition",
			currentPodConditions: []v1.PodCondition{
				{
					Type:               "currentType",
					Status:             "currentStatus",
					LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 0, 0, 0, 0, time.UTC)),
					LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
					Reason:             "currentReason",
					Message:            "currentMessage",
				},
			},
			newPodCondition: &v1.PodCondition{
				Type:               "currentType",
				Status:             "newStatus",
				LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 1, 1, 1, 1, time.UTC)),
				LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 1, 1, 1, 1, time.UTC)),
				Reason:             "newReason",
				Message:            "newMessage",
			},
			expectedPatchRequests:    1,
			expectedPatchDataPattern: `{"status":{"\$setElementOrder/conditions":\[{"type":"currentType"}],"conditions":\[{"lastProbeTime":"2020-05-13T01:01:01Z","lastTransitionTime":".*","message":"newMessage","reason":"newReason","status":"newStatus","type":"currentType"}]}}`,
		},
		{
			name: "Should make patch request to update an existing pod condition, but the transition time should remain unchanged because the status is the same",
			currentPodConditions: []v1.PodCondition{
				{
					Type:               "currentType",
					Status:             "currentStatus",
					LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 0, 0, 0, 0, time.UTC)),
					LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
					Reason:             "currentReason",
					Message:            "currentMessage",
				},
			},
			newPodCondition: &v1.PodCondition{
				Type:               "currentType",
				Status:             "currentStatus",
				LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 1, 1, 1, 1, time.UTC)),
				LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
				Reason:             "newReason",
				Message:            "newMessage",
			},
			expectedPatchRequests:    1,
			expectedPatchDataPattern: `{"status":{"\$setElementOrder/conditions":\[{"type":"currentType"}],"conditions":\[{"lastProbeTime":"2020-05-13T01:01:01Z","message":"newMessage","reason":"newReason","type":"currentType"}]}}`,
		},
		{
			name: "Should not make patch request if pod condition already exists and is identical and nominated node name is not set",
			currentPodConditions: []v1.PodCondition{
				{
					Type:               "currentType",
					Status:             "currentStatus",
					LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 0, 0, 0, 0, time.UTC)),
					LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
					Reason:             "currentReason",
					Message:            "currentMessage",
				},
			},
			newPodCondition: &v1.PodCondition{
				Type:               "currentType",
				Status:             "currentStatus",
				LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 0, 0, 0, 0, time.UTC)),
				LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
				Reason:             "currentReason",
				Message:            "currentMessage",
			},
			currentNominatedNodeName: "node1",
			expectedPatchRequests:    0,
		},
		{
			name: "Should make patch request if pod condition already exists and is identical but nominated node name is set and different",
			currentPodConditions: []v1.PodCondition{
				{
					Type:               "currentType",
					Status:             "currentStatus",
					LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 0, 0, 0, 0, time.UTC)),
					LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
					Reason:             "currentReason",
					Message:            "currentMessage",
				},
			},
			newPodCondition: &v1.PodCondition{
				Type:               "currentType",
				Status:             "currentStatus",
				LastProbeTime:      metav1.NewTime(time.Date(2020, 5, 13, 0, 0, 0, 0, time.UTC)),
				LastTransitionTime: metav1.NewTime(time.Date(2020, 5, 12, 0, 0, 0, 0, time.UTC)),
				Reason:             "currentReason",
				Message:            "currentMessage",
			},
			newNominatedNodeName:     "node1",
			expectedPatchRequests:    1,
			expectedPatchDataPattern: `{"status":{"nominatedNodeName":"node1"}}`,
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			actualPatchRequests := 0
			var actualPatchData string
			cs := &clientsetfake.Clientset{}
			cs.AddReactor("patch", "pods", func(action clienttesting.Action) (bool, runtime.Object, error) {
				actualPatchRequests++
				patch := action.(clienttesting.PatchAction)
				actualPatchData = string(patch.GetPatch())
				// For this test, we don't care about the result of the patched pod, just that we got the expected
				// patch request, so just returning &v1.Pod{} here is OK because scheduler doesn't use the response.
				return true, &v1.Pod{}, nil
			})

			pod := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Status: v1.PodStatus{
					Conditions:        test.currentPodConditions,
					NominatedNodeName: test.currentNominatedNodeName,
				},
			}

			if err := updatePod(cs, pod, test.newPodCondition, test.newNominatedNodeName); err != nil {
				t.Fatalf("Error calling update: %v", err)
			}

			if actualPatchRequests != test.expectedPatchRequests {
				t.Fatalf("Actual patch requests (%d) does not equal expected patch requests (%d), actual patch data: %v", actualPatchRequests, test.expectedPatchRequests, actualPatchData)
			}

			regex, err := regexp.Compile(test.expectedPatchDataPattern)
			if err != nil {
				t.Fatalf("Error compiling regexp for %v: %v", test.expectedPatchDataPattern, err)
			}

			if test.expectedPatchRequests > 0 && !regex.MatchString(actualPatchData) {
				t.Fatalf("Patch data mismatch: Actual was %v, but expected to match regexp %v", actualPatchData, test.expectedPatchDataPattern)
			}
		})
	}
}

// Below are the tests migrated from generic_scheduler_test.go

var (
	errPrioritize = fmt.Errorf("priority map encounters an error")
)

type numericMapPlugin struct{}

func newNumericMapPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &numericMapPlugin{}, nil
	}
}

func (pl *numericMapPlugin) Name() string {
	return "NumericMap"
}

func (pl *numericMapPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeName string) (int64, *framework.Status) {
	score, err := strconv.Atoi(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("Error converting nodename to int: %+v", nodeName))
	}
	return int64(score), nil
}

func (pl *numericMapPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

type reverseNumericMapPlugin struct{}

func newReverseNumericMapPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &reverseNumericMapPlugin{}, nil
	}
}

func (pl *reverseNumericMapPlugin) Name() string {
	return "ReverseNumericMap"
}

func (pl *reverseNumericMapPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeName string) (int64, *framework.Status) {
	score, err := strconv.Atoi(nodeName)
	if err != nil {
		return 0, framework.NewStatus(framework.Error, fmt.Sprintf("Error converting nodename to int: %+v", nodeName))
	}
	return int64(score), nil
}

func (pl *reverseNumericMapPlugin) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

func (pl *reverseNumericMapPlugin) NormalizeScore(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeScores framework.NodeScoreList) *framework.Status {
	var maxScore float64
	minScore := math.MaxFloat64

	for _, hostPriority := range nodeScores {
		maxScore = math.Max(maxScore, float64(hostPriority.Score))
		minScore = math.Min(minScore, float64(hostPriority.Score))
	}
	for i, hostPriority := range nodeScores {
		nodeScores[i] = framework.NodeScore{
			Name:  hostPriority.Name,
			Score: int64(maxScore + minScore - float64(hostPriority.Score)),
		}
	}
	return nil
}

type trueMapPlugin struct{}

func newTrueMapPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &trueMapPlugin{}, nil
	}
}

func (pl *trueMapPlugin) Name() string {
	return "TrueMap"
}

func (pl *trueMapPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, _ string) (int64, *framework.Status) {
	return 1, nil
}

func (pl *trueMapPlugin) ScoreExtensions() framework.ScoreExtensions {
	return pl
}

func (pl *trueMapPlugin) NormalizeScore(_ context.Context, _ *framework.CycleState, _ *v1.Pod, nodeScores framework.NodeScoreList) *framework.Status {
	for _, host := range nodeScores {
		if host.Name == "" {
			return framework.NewStatus(framework.Error, "unexpected empty host name")
		}
	}
	return nil
}

type falseMapPlugin struct{}

func newFalseMapPlugin() frameworkruntime.PluginFactory {
	return func(_ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
		return &falseMapPlugin{}, nil
	}
}

func (pl *falseMapPlugin) Name() string {
	return "FalseMap"
}

func (pl *falseMapPlugin) Score(_ context.Context, _ *framework.CycleState, _ *v1.Pod, _ string) (int64, *framework.Status) {
	return 0, framework.AsStatus(errPrioritize)
}

func (pl *falseMapPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

type noPodsFilterPlugin struct{}

// Name returns name of the plugin.
func (pl *noPodsFilterPlugin) Name() string {
	return "NoPodsFilter"
}

// Filter invoked at the filter extension point.
func (pl *noPodsFilterPlugin) Filter(_ context.Context, _ *framework.CycleState, pod *v1.Pod, nodeInfo *framework.NodeInfo) *framework.Status {
	if len(nodeInfo.Pods) == 0 {
		return nil
	}
	return framework.NewStatus(framework.Unschedulable, st.ErrReasonFake)
}

// NewNoPodsFilterPlugin initializes a noPodsFilterPlugin and returns it.
func NewNoPodsFilterPlugin(_ runtime.Object, _ framework.Handle) (framework.Plugin, error) {
	return &noPodsFilterPlugin{}, nil
}

func TestGenericScheduler(t *testing.T) {
	tests := []struct {
		name            string
		registerPlugins []st.RegisterPluginFunc
		nodes           []string
		pvcs            []v1.PersistentVolumeClaim
		pod             *v1.Pod
		pods            []*v1.Pod
		expectedHosts   sets.String
		wErr            error
	}{
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("FalseFilter", st.NewFalseFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			name:  "test 1",
			wErr: &framework.FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 2,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"machine1": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("FalseFilter"),
						"machine2": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("FalseFilter"),
					},
					UnschedulablePlugins: sets.NewString("FalseFilter"),
				},
			},
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"machine1", "machine2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")}},
			expectedHosts: sets.NewString("machine1", "machine2"),
			name:          "test 2",
			wErr:          nil,
		},
		{
			// Fits on a machine where the pod ID matches the machine name
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"machine1", "machine2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "machine2", UID: types.UID("machine2")}},
			expectedHosts: sets.NewString("machine2"),
			name:          "test 3",
			wErr:          nil,
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")}},
			expectedHosts: sets.NewString("3"),
			name:          "test 4",
			wErr:          nil,
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			expectedHosts: sets.NewString("2"),
			name:          "test 5",
			wErr:          nil,
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterScorePlugin("ReverseNumericMap", newReverseNumericMapPlugin(), 2),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"3", "2", "1"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			expectedHosts: sets.NewString("1"),
			name:          "test 6",
			wErr:          nil,
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterFilterPlugin("FalseFilter", st.NewFalseFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"3", "2", "1"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			name:  "test 7",
			wErr: &framework.FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 3,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"3": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("FalseFilter"),
						"2": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("FalseFilter"),
						"1": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("FalseFilter"),
					},
					UnschedulablePlugins: sets.NewString("FalseFilter"),
				},
			},
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("NoPodsFilter", NewNoPodsFilterPlugin),
				st.RegisterFilterPlugin("MatchFilter", st.NewMatchFilterPlugin),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")},
					Spec: v1.PodSpec{
						NodeName: "2",
					},
					Status: v1.PodStatus{
						Phase: v1.PodRunning,
					},
				},
			},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
			nodes: []string{"1", "2"},
			name:  "test 8",
			wErr: &framework.FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2", UID: types.UID("2")}},
				NumAllNodes: 2,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"1": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("MatchFilter"),
						"2": framework.NewStatus(framework.Unschedulable, st.ErrReasonFake).WithFailedPlugin("NoPodsFilter"),
					},
					UnschedulablePlugins: sets.NewString("MatchFilter", "NoPodsFilter"),
				},
			},
		},
		{
			// Pod with existing PVC
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(volumebinding.Name, volumebinding.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pvcs: []v1.PersistentVolumeClaim{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "existingPVC", UID: types.UID("existingPVC"), Namespace: v1.NamespaceDefault},
					Spec:       v1.PersistentVolumeClaimSpec{VolumeName: "existingPV"},
				},
			},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore"), Namespace: v1.NamespaceDefault},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							VolumeSource: v1.VolumeSource{
								PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
									ClaimName: "existingPVC",
								},
							},
						},
					},
				},
			},
			expectedHosts: sets.NewString("machine1", "machine2"),
			name:          "existing PVC",
			wErr:          nil,
		},
		{
			// Pod with non existing PVC
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(volumebinding.Name, volumebinding.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							VolumeSource: v1.VolumeSource{
								PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
									ClaimName: "unknownPVC",
								},
							},
						},
					},
				},
			},
			name: "unknown PVC",
			wErr: &framework.FitError{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore")},
					Spec: v1.PodSpec{
						Volumes: []v1.Volume{
							{
								VolumeSource: v1.VolumeSource{
									PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
										ClaimName: "unknownPVC",
									},
								},
							},
						},
					},
				},
				NumAllNodes: 2,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"machine1": framework.NewStatus(framework.UnschedulableAndUnresolvable, `persistentvolumeclaim "unknownPVC" not found`).WithFailedPlugin(volumebinding.Name),
						"machine2": framework.NewStatus(framework.UnschedulableAndUnresolvable, `persistentvolumeclaim "unknownPVC" not found`).WithFailedPlugin(volumebinding.Name),
					},
					UnschedulablePlugins: sets.NewString(volumebinding.Name),
				},
			},
		},
		{
			// Pod with deleting PVC
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(volumebinding.Name, volumebinding.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pvcs:  []v1.PersistentVolumeClaim{{ObjectMeta: metav1.ObjectMeta{Name: "existingPVC", UID: types.UID("existingPVC"), Namespace: v1.NamespaceDefault, DeletionTimestamp: &metav1.Time{}}}},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore"), Namespace: v1.NamespaceDefault},
				Spec: v1.PodSpec{
					Volumes: []v1.Volume{
						{
							VolumeSource: v1.VolumeSource{
								PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
									ClaimName: "existingPVC",
								},
							},
						},
					},
				},
			},
			name: "deleted PVC",
			wErr: &framework.FitError{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{Name: "ignore", UID: types.UID("ignore"), Namespace: v1.NamespaceDefault},
					Spec: v1.PodSpec{
						Volumes: []v1.Volume{
							{
								VolumeSource: v1.VolumeSource{
									PersistentVolumeClaim: &v1.PersistentVolumeClaimVolumeSource{
										ClaimName: "existingPVC",
									},
								},
							},
						},
					},
				},
				NumAllNodes: 2,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"machine1": framework.NewStatus(framework.UnschedulableAndUnresolvable, `persistentvolumeclaim "existingPVC" is being deleted`).WithFailedPlugin(volumebinding.Name),
						"machine2": framework.NewStatus(framework.UnschedulableAndUnresolvable, `persistentvolumeclaim "existingPVC" is being deleted`).WithFailedPlugin(volumebinding.Name),
					},
					UnschedulablePlugins: sets.NewString(volumebinding.Name),
				},
			},
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterScorePlugin("FalseMap", newFalseMapPlugin(), 1),
				st.RegisterScorePlugin("TrueMap", newTrueMapPlugin(), 2),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"2", "1"},
			pod:   &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "2"}},
			name:  "test error with priority map",
			wErr:  fmt.Errorf("running Score plugins: %w", fmt.Errorf(`plugin "FalseMap" failed with: %w`, errPrioritize)),
		},
		{
			name: "test podtopologyspread plugin - 2 nodes with maxskew=1",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(
					podtopologyspread.Name,
					podtopologyspread.New,
					"PreFilter",
					"Filter",
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2"},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "p", UID: types.UID("p"), Labels: map[string]string{"foo": ""}},
				Spec: v1.PodSpec{
					TopologySpreadConstraints: []v1.TopologySpreadConstraint{
						{
							MaxSkew:           1,
							TopologyKey:       "hostname",
							WhenUnsatisfiable: v1.DoNotSchedule,
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "foo",
										Operator: metav1.LabelSelectorOpExists,
									},
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: types.UID("pod1"), Labels: map[string]string{"foo": ""}},
					Spec: v1.PodSpec{
						NodeName: "machine1",
					},
					Status: v1.PodStatus{
						Phase: v1.PodRunning,
					},
				},
			},
			expectedHosts: sets.NewString("machine2"),
			wErr:          nil,
		},
		{
			name: "test podtopologyspread plugin - 3 nodes with maxskew=2",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPluginAsExtensions(
					podtopologyspread.Name,
					podtopologyspread.New,
					"PreFilter",
					"Filter",
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes: []string{"machine1", "machine2", "machine3"},
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "p", UID: types.UID("p"), Labels: map[string]string{"foo": ""}},
				Spec: v1.PodSpec{
					TopologySpreadConstraints: []v1.TopologySpreadConstraint{
						{
							MaxSkew:           2,
							TopologyKey:       "hostname",
							WhenUnsatisfiable: v1.DoNotSchedule,
							LabelSelector: &metav1.LabelSelector{
								MatchExpressions: []metav1.LabelSelectorRequirement{
									{
										Key:      "foo",
										Operator: metav1.LabelSelectorOpExists,
									},
								},
							},
						},
					},
				},
			},
			pods: []*v1.Pod{
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod1a", UID: types.UID("pod1a"), Labels: map[string]string{"foo": ""}},
					Spec: v1.PodSpec{
						NodeName: "machine1",
					},
					Status: v1.PodStatus{
						Phase: v1.PodRunning,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod1b", UID: types.UID("pod1b"), Labels: map[string]string{"foo": ""}},
					Spec: v1.PodSpec{
						NodeName: "machine1",
					},
					Status: v1.PodStatus{
						Phase: v1.PodRunning,
					},
				},
				{
					ObjectMeta: metav1.ObjectMeta{Name: "pod2", UID: types.UID("pod2"), Labels: map[string]string{"foo": ""}},
					Spec: v1.PodSpec{
						NodeName: "machine2",
					},
					Status: v1.PodStatus{
						Phase: v1.PodRunning,
					},
				},
			},
			expectedHosts: sets.NewString("machine2", "machine3"),
			wErr:          nil,
		},
		{
			name: "test with filter plugin returning Unschedulable status",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(
					"FakeFilter",
					st.NewFakeFilterPlugin(map[string]framework.Code{"3": framework.Unschedulable}),
				),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"3"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-filter", UID: types.UID("test-filter")}},
			expectedHosts: nil,
			wErr: &framework.FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-filter", UID: types.UID("test-filter")}},
				NumAllNodes: 1,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"3": framework.NewStatus(framework.Unschedulable, "injecting failure for pod test-filter").WithFailedPlugin("FakeFilter"),
					},
					UnschedulablePlugins: sets.NewString("FakeFilter"),
				},
			},
		},
		{
			name: "test with filter plugin returning UnschedulableAndUnresolvable status",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(
					"FakeFilter",
					st.NewFakeFilterPlugin(map[string]framework.Code{"3": framework.UnschedulableAndUnresolvable}),
				),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"3"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-filter", UID: types.UID("test-filter")}},
			expectedHosts: nil,
			wErr: &framework.FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-filter", UID: types.UID("test-filter")}},
				NumAllNodes: 1,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"3": framework.NewStatus(framework.UnschedulableAndUnresolvable, "injecting failure for pod test-filter").WithFailedPlugin("FakeFilter"),
					},
					UnschedulablePlugins: sets.NewString("FakeFilter"),
				},
			},
		},
		{
			name: "test with partial failed filter plugin",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterFilterPlugin(
					"FakeFilter",
					st.NewFakeFilterPlugin(map[string]framework.Code{"1": framework.Unschedulable}),
				),
				st.RegisterScorePlugin("NumericMap", newNumericMapPlugin(), 1),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"1", "2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-filter", UID: types.UID("test-filter")}},
			expectedHosts: nil,
			wErr:          nil,
		},
		{
			name: "test prefilter plugin returning Unschedulable status",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(
					"FakePreFilter",
					st.NewFakePreFilterPlugin(framework.NewStatus(framework.UnschedulableAndUnresolvable, "injected unschedulable status")),
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"1", "2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-prefilter", UID: types.UID("test-prefilter")}},
			expectedHosts: nil,
			wErr: &framework.FitError{
				Pod:         &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-prefilter", UID: types.UID("test-prefilter")}},
				NumAllNodes: 2,
				Diagnosis: framework.Diagnosis{
					NodeToStatusMap: framework.NodeToStatusMap{
						"1": framework.NewStatus(framework.UnschedulableAndUnresolvable, "injected unschedulable status").WithFailedPlugin("FakePreFilter"),
						"2": framework.NewStatus(framework.UnschedulableAndUnresolvable, "injected unschedulable status").WithFailedPlugin("FakePreFilter"),
					},
					UnschedulablePlugins: sets.NewString("FakePreFilter"),
				},
			},
		},
		{
			name: "test prefilter plugin returning error status",
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterPreFilterPlugin(
					"FakePreFilter",
					st.NewFakePreFilterPlugin(framework.NewStatus(framework.Error, "injected error status")),
				),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			nodes:         []string{"1", "2"},
			pod:           &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test-prefilter", UID: types.UID("test-prefilter")}},
			expectedHosts: nil,
			wErr:          fmt.Errorf(`running PreFilter plugin "FakePreFilter": %w`, errors.New("injected error status")),
		},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cache := internalcache.New(time.Duration(0), wait.NeverStop)
			for _, pod := range test.pods {
				cache.AddPod(pod)
			}
			var nodes []*v1.Node
			for _, name := range test.nodes {
				node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{"hostname": name}}}
				nodes = append(nodes, node)
				cache.AddNode(node)
			}

			ctx := context.Background()
			cs := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(cs, 0)
			for _, pvc := range test.pvcs {
				metav1.SetMetaDataAnnotation(&pvc.ObjectMeta, pvutil.AnnBindCompleted, "true")
				cs.CoreV1().PersistentVolumeClaims(pvc.Namespace).Create(ctx, &pvc, metav1.CreateOptions{})
				if pvName := pvc.Spec.VolumeName; pvName != "" {
					pv := v1.PersistentVolume{ObjectMeta: metav1.ObjectMeta{Name: pvName}}
					cs.CoreV1().PersistentVolumes().Create(ctx, &pv, metav1.CreateOptions{})
				}
			}
			snapshot := internalcache.NewSnapshot(test.pods, nodes)
			fwk, err := st.NewFramework(
				test.registerPlugins, "",
				frameworkruntime.WithSnapshotSharedLister(snapshot),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithPodNominator(internalqueue.NewPodNominator()),
			)
			if err != nil {
				t.Fatal(err)
			}

			// scheduler := NewGenericScheduler(
			// 	cache,
			// 	snapshot,
			// 	[]framework.Extender{},
			// 	schedulerapi.DefaultPercentageOfNodesToScore,
			// )
			algo := core.NewGenericScheduler(
				cache,
				snapshot,
				nil,
				0,
			)
			s := Scheduler{
				SchedulerCache: cache,
				Algorithm:      algo,
				Error:          nil,
			}
			informerFactory.Start(ctx.Done())
			informerFactory.WaitForCacheSync(ctx.Done())

			result, err := s.Schedule(ctx, fwk, framework.NewCycleState(), test.pod)

			// TODO(#94696): replace reflect.DeepEqual with cmp.Diff().
			if err != test.wErr {
				gotFitErr, gotOK := err.(*framework.FitError)
				wantFitErr, wantOK := test.wErr.(*framework.FitError)
				if gotOK != wantOK {
					t.Errorf("Expected err to be FitError: %v, but got %v", wantOK, gotOK)
				} else if gotOK && !reflect.DeepEqual(gotFitErr, wantFitErr) {
					t.Errorf("Unexpected fitError. Want %v, but got %v", wantFitErr, gotFitErr)
				}
			}
			if test.expectedHosts != nil && !test.expectedHosts.Has(result.SuggestedHost) {
				t.Errorf("Expected: %s, got: %s", test.expectedHosts, result.SuggestedHost)
			}
			if test.wErr == nil && len(test.nodes) != result.EvaluatedNodes {
				t.Errorf("Expected EvaluatedNodes: %d, got: %d", len(test.nodes), result.EvaluatedNodes)
			}
		})
	}
}

func createNode(name string) *v1.Node {
	return &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}}
}

func TestGenericSchedulerWithExtenders(t *testing.T) {
	tests := []struct {
		name            string
		registerPlugins []st.RegisterPluginFunc
		extenders       []st.FakeExtender
		nodes           []string
		expectedResult  ScheduleResult
		expectsErr      bool
	}{
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []st.FakeExtender{
				{
					Predicates: []st.FitPredicate{st.TruePredicateExtender},
				},
				{
					Predicates: []st.FitPredicate{st.ErrorPredicateExtender},
				},
			},
			nodes:      []string{"node1", "node2"},
			expectsErr: true,
			name:       "test 1",
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []st.FakeExtender{
				{
					Predicates: []st.FitPredicate{st.TruePredicateExtender},
				},
				{
					Predicates: []st.FitPredicate{st.FalsePredicateExtender},
				},
			},
			nodes:      []string{"node1", "node2"},
			expectsErr: true,
			name:       "test 2",
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []st.FakeExtender{
				{
					Predicates: []st.FitPredicate{st.TruePredicateExtender},
				},
				{
					Predicates: []st.FitPredicate{st.Node1PredicateExtender},
				},
			},
			nodes: []string{"node1", "node2"},
			expectedResult: ScheduleResult{
				SuggestedHost:  "node1",
				EvaluatedNodes: 2,
				FeasibleNodes:  1,
			},
			name: "test 3",
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []st.FakeExtender{
				{
					Predicates: []st.FitPredicate{st.Node2PredicateExtender},
				},
				{
					Predicates: []st.FitPredicate{st.Node1PredicateExtender},
				},
			},
			nodes:      []string{"node1", "node2"},
			expectsErr: true,
			name:       "test 4",
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []st.FakeExtender{
				{
					Predicates:   []st.FitPredicate{st.TruePredicateExtender},
					Prioritizers: []st.PriorityConfig{{Function: st.ErrorPrioritizerExtender, Weight: 10}},
					Weight:       1,
				},
			},
			nodes: []string{"node1"},
			expectedResult: ScheduleResult{
				SuggestedHost:  "node1",
				EvaluatedNodes: 1,
				FeasibleNodes:  1,
			},
			name: "test 5",
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []st.FakeExtender{
				{
					Predicates:   []st.FitPredicate{st.TruePredicateExtender},
					Prioritizers: []st.PriorityConfig{{Function: st.Node1PrioritizerExtender, Weight: 10}},
					Weight:       1,
				},
				{
					Predicates:   []st.FitPredicate{st.TruePredicateExtender},
					Prioritizers: []st.PriorityConfig{{Function: st.Node2PrioritizerExtender, Weight: 10}},
					Weight:       5,
				},
			},
			nodes: []string{"node1", "node2"},
			expectedResult: ScheduleResult{
				SuggestedHost:  "node2",
				EvaluatedNodes: 2,
				FeasibleNodes:  2,
			},
			name: "test 6",
		},
		{
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterScorePlugin("Node2Prioritizer", st.NewNode2PrioritizerPlugin(), 20),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []st.FakeExtender{
				{
					Predicates:   []st.FitPredicate{st.TruePredicateExtender},
					Prioritizers: []st.PriorityConfig{{Function: st.Node1PrioritizerExtender, Weight: 10}},
					Weight:       1,
				},
			},
			nodes: []string{"node1", "node2"},
			expectedResult: ScheduleResult{
				SuggestedHost:  "node2",
				EvaluatedNodes: 2,
				FeasibleNodes:  2,
			}, // node2 has higher score
			name: "test 7",
		},
		{
			// Scheduler is expected to not send pod to extender in
			// Filter/Prioritize phases if the extender is not interested in
			// the pod.
			//
			// If scheduler sends the pod by mistake, the test would fail
			// because of the errors from errorPredicateExtender and/or
			// errorPrioritizerExtender.
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterScorePlugin("Node2Prioritizer", st.NewNode2PrioritizerPlugin(), 1),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []st.FakeExtender{
				{
					Predicates:   []st.FitPredicate{st.ErrorPredicateExtender},
					Prioritizers: []st.PriorityConfig{{Function: st.ErrorPrioritizerExtender, Weight: 10}},
					UnInterested: true,
				},
			},
			nodes:      []string{"node1", "node2"},
			expectsErr: false,
			expectedResult: ScheduleResult{
				SuggestedHost:  "node2",
				EvaluatedNodes: 2,
				FeasibleNodes:  2,
			}, // node2 has higher score
			name: "test 8",
		},
		{
			// Scheduling is expected to not fail in
			// Filter/Prioritize phases if the extender is not available and ignorable.
			//
			// If scheduler did not ignore the extender, the test would fail
			// because of the errors from errorPredicateExtender.
			registerPlugins: []st.RegisterPluginFunc{
				st.RegisterFilterPlugin("TrueFilter", st.NewTrueFilterPlugin),
				st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
				st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
			},
			extenders: []st.FakeExtender{
				{
					Predicates: []st.FitPredicate{st.ErrorPredicateExtender},
					Ignorable:  true,
				},
				{
					Predicates: []st.FitPredicate{st.Node1PredicateExtender},
				},
			},
			nodes:      []string{"node1", "node2"},
			expectsErr: false,
			expectedResult: ScheduleResult{
				SuggestedHost:  "node1",
				EvaluatedNodes: 2,
				FeasibleNodes:  1,
			},
			name: "test 9",
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			cache := internalcache.New(time.Duration(0), wait.NeverStop)
			// for _, pod := range test.pods {
			// 	cache.AddPod(pod)
			// }
			var nodes []*v1.Node
			for _, name := range test.nodes {
				node := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name, Labels: map[string]string{"hostname": name}}}
				nodes = append(nodes, node)
				cache.AddNode(node)
			}

			client := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			var extenders []framework.Extender
			for ii := range test.extenders {
				extenders = append(extenders, &test.extenders[ii])
			}
			for _, name := range test.nodes {
				cache.AddNode(createNode(name))
			}
			fwk, err := st.NewFramework(
				test.registerPlugins, "",
				frameworkruntime.WithClientSet(client),
				frameworkruntime.WithInformerFactory(informerFactory),
				frameworkruntime.WithPodNominator(internalqueue.NewPodNominator()),
			)
			if err != nil {
				t.Fatal(err)
			}

			snapshot := internalcache.NewSnapshot(nil, nodes)
			algo := core.NewGenericScheduler(
				cache,
				snapshot,
				extenders,
				0,
			)
			s := Scheduler{
				SchedulerCache: cache,
				Algorithm:      algo,
				Error:          nil,
			}

			podIgnored := &v1.Pod{}
			result, err := s.Schedule(context.Background(), fwk, framework.NewCycleState(), podIgnored)
			if test.expectsErr {
				if err == nil {
					t.Errorf("Unexpected non-error, result %+v", result)
				}
			} else {
				if err != nil {
					t.Errorf("Unexpected error: %v", err)
					return
				}

				if !reflect.DeepEqual(result, test.expectedResult) {
					t.Errorf("Expected: %+v, Saw: %+v", test.expectedResult, result)
				}
			}
		})
	}
}
