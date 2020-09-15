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
	"os"
	"path"
	"reflect"
	"regexp"
	"sort"
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
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	clienttesting "k8s.io/client-go/testing"
	clientcache "k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	"k8s.io/kubernetes/pkg/controller/volume/scheduling"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/core"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodeports"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
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

func (es mockScheduler) Schedule(ctx context.Context, profile *profile.Profile, state *framework.CycleState, pod *v1.Pod) (core.ScheduleResult, error) {
	return es.result, es.err
}

func (es mockScheduler) Extenders() []framework.Extender {
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
			s, err := New(client,
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

	table := []struct {
		name                string
		injectBindError     error
		sendPod             *v1.Pod
		algo                core.ScheduleAlgorithm
		registerPluginFuncs []st.RegisterPluginFunc
		expectErrorPod      *v1.Pod
		expectForgetPod     *v1.Pod
		expectAssumedPod    *v1.Pod
		expectError         error
		expectBind          *v1.Binding
		eventReason         string
	}{
		{
			name:    "error reserve pod",
			sendPod: podWithID("foo", ""),
			algo:    mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			registerPluginFuncs: []st.RegisterPluginFunc{
				st.RegisterReservePlugin("FakeReserve", st.NewFakeReservePlugin(framework.NewStatus(framework.Error, "reserve error"))),
			},
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			expectAssumedPod: podWithID("foo", testNode.Name),
			expectError:      errors.New(`error while running Reserve in "FakeReserve" reserve plugin for pod "foo": reserve error`),
			eventReason:      "FailedScheduling",
		},
		{
			name:    "error permit pod",
			sendPod: podWithID("foo", ""),
			algo:    mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			registerPluginFuncs: []st.RegisterPluginFunc{
				st.RegisterPermitPlugin("FakePermit", st.NewFakePermitPlugin(framework.NewStatus(framework.Error, "permit error"), time.Minute)),
			},
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			expectAssumedPod: podWithID("foo", testNode.Name),
			expectError:      errors.New(`error while running "FakePermit" permit plugin for pod "foo": permit error`),
			eventReason:      "FailedScheduling",
		},
		{
			name:    "error prebind pod",
			sendPod: podWithID("foo", ""),
			algo:    mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			registerPluginFuncs: []st.RegisterPluginFunc{
				st.RegisterPreBindPlugin("FakePreBind", st.NewFakePreBindPlugin(framework.NewStatus(framework.Error, "prebind error"))),
			},
			expectErrorPod:   podWithID("foo", testNode.Name),
			expectForgetPod:  podWithID("foo", testNode.Name),
			expectAssumedPod: podWithID("foo", testNode.Name),
			expectError:      errors.New(`error while running "FakePreBind" prebind plugin for pod "foo": prebind error`),
			eventReason:      "FailedScheduling",
		},
		{
			name:             "bind assumed pod scheduled",
			sendPod:          podWithID("foo", ""),
			algo:             mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			expectBind:       &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: testNode.Name}},
			expectAssumedPod: podWithID("foo", testNode.Name),
			eventReason:      "Scheduled",
		},
		{
			name:           "error pod failed scheduling",
			sendPod:        podWithID("foo", ""),
			algo:           mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, errS},
			expectError:    errS,
			expectErrorPod: podWithID("foo", ""),
			eventReason:    "FailedScheduling",
		},
		{
			name:             "error bind forget pod failed scheduling",
			sendPod:          podWithID("foo", ""),
			algo:             mockScheduler{core.ScheduleResult{SuggestedHost: testNode.Name, EvaluatedNodes: 1, FeasibleNodes: 1}, nil},
			expectBind:       &v1.Binding{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: types.UID("foo")}, Target: v1.ObjectReference{Kind: "Node", Name: testNode.Name}},
			expectAssumedPod: podWithID("foo", testNode.Name),
			injectBindError:  errB,
			expectError:      errors.New("Binding rejected: plugin \"DefaultBinder\" failed to bind pod \"/foo\": binder"),
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
			fwk, err := st.NewFramework(registerPluginFuncs, frameworkruntime.WithClientSet(client))
			if err != nil {
				t.Fatal(err)
			}

			s := &Scheduler{
				SchedulerCache: sCache,
				Algorithm:      item.algo,
				client:         client,
				Error: func(p *framework.QueuedPodInfo, err error) {
					gotPod = p.Pod
					gotError = err
				},
				NextPod: func() *framework.QueuedPodInfo {
					return &framework.QueuedPodInfo{Pod: item.sendPod}
				},
				Profiles: profile.Map{
					testSchedulerName: &profile.Profile{
						Framework: fwk,
						Recorder:  eventBroadcaster.NewRecorder(scheme.Scheme, testSchedulerName),
						Name:      testSchedulerName,
					},
				},
			}
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

func newFakeNodeSelector(args runtime.Object, _ framework.FrameworkHandle) (framework.Plugin, error) {
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
	client := clientsetfake.NewSimpleClientset(nodes...)
	broadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	informerFactory := informers.NewSharedInformerFactory(client, 0)
	sched, err := New(client,
		informerFactory,
		profile.NewRecorderFactory(broadcaster),
		ctx.Done(),
		WithProfiles(
			schedulerapi.KubeSchedulerProfile{SchedulerName: "match-machine2",
				Plugins: &schedulerapi.Plugins{
					Filter: &schedulerapi.PluginSet{
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
					Filter: &schedulerapi.PluginSet{
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
		expectErr := &core.FitError{
			Pod:         secondPod,
			NumAllNodes: 1,
			FilteredNodesStatuses: framework.NodeToStatusMap{
				node.Name: framework.NewStatus(
					framework.Unschedulable,
					nodeports.ErrReason,
				),
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

	// Design the baseline for the pods, and we will make nodes that dont fit it later.
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
		)
	}
	fns := []st.RegisterPluginFunc{
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
		st.RegisterPluginAsExtensions(noderesources.FitName, noderesources.NewFit, "Filter", "PreFilter"),
	}
	scheduler, _, errChan := setupTestScheduler(queuedPodStore, scache, informerFactory, nil, fns...)

	informerFactory.Start(stop)
	informerFactory.WaitForCacheSync(stop)

	queuedPodStore.Add(podWithTooBigResourceRequests)
	scheduler.scheduleOne(context.Background())
	select {
	case err := <-errChan:
		expectErr := &core.FitError{
			Pod:                   podWithTooBigResourceRequests,
			NumAllNodes:           len(nodes),
			FilteredNodesStatuses: failedNodeStatues,
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

	fwk, _ := st.NewFramework(
		fns,
		frameworkruntime.WithClientSet(client),
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithPodNominator(internalqueue.NewPodNominator()),
	)
	prof := &profile.Profile{
		Framework: fwk,
		Recorder:  &events.FakeRecorder{},
		Name:      testSchedulerName,
	}
	if broadcaster != nil {
		prof.Recorder = broadcaster.NewRecorder(scheme.Scheme, testSchedulerName)
	}
	profiles := profile.Map{
		testSchedulerName: prof,
	}

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
			return &framework.QueuedPodInfo{Pod: clientcache.Pop(queuedPodStore).(*v1.Pod)}
		},
		Error: func(p *framework.QueuedPodInfo, err error) {
			errChan <- err
		},
		Profiles: profiles,
		client:   client,
	}

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
		st.RegisterPluginAsExtensions(volumebinding.Name, func(plArgs runtime.Object, handle framework.FrameworkHandle) (framework.Plugin, error) {
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
			expectError: fmt.Errorf("running %q filter plugin for pod %q: %v", volumebinding.Name, "foo", findErr),
		},
		{
			name: "assume error",
			volumeBinderConfig: &scheduling.FakeVolumeBinderConfig{
				AssumeErr: assumeErr,
			},
			expectAssumeCalled: true,
			eventReason:        "FailedScheduling",
			expectError:        fmt.Errorf("error while running Reserve in %q reserve plugin for pod %q: %v", volumebinding.Name, "foo", assumeErr),
		},
		{
			name: "bind error",
			volumeBinderConfig: &scheduling.FakeVolumeBinderConfig{
				BindErr: bindErr,
			},
			expectAssumeCalled: true,
			expectBindCalled:   true,
			eventReason:        "FailedScheduling",
			expectError:        fmt.Errorf("error while running %q prebind plugin for pod %q: %v", volumebinding.Name, "foo", bindErr),
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
			}, frameworkruntime.WithClientSet(client))
			if err != nil {
				t.Fatal(err)
			}
			prof := &profile.Profile{
				Framework: fwk,
				Recorder:  &events.FakeRecorder{},
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
			err = sched.bind(context.Background(), prof, pod, "node", nil)
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
