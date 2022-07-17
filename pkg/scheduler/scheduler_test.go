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
	"fmt"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/events"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	frameworkruntime "k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	testingclock "k8s.io/utils/clock/testing"
)

func TestSchedulerCreation(t *testing.T) {
	invalidRegistry := map[string]frameworkruntime.PluginFactory{
		defaultbinder.Name: defaultbinder.New,
	}
	validRegistry := map[string]frameworkruntime.PluginFactory{
		"Foo": defaultbinder.New,
	}
	cases := []struct {
		name          string
		opts          []Option
		wantErr       string
		wantProfiles  []string
		wantExtenders []string
	}{
		{
			name: "valid out-of-tree registry",
			opts: []Option{
				WithFrameworkOutOfTreeRegistry(validRegistry),
				WithProfiles(
					schedulerapi.KubeSchedulerProfile{
						SchedulerName: "default-scheduler",
						Plugins: &schedulerapi.Plugins{
							QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
							Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
						},
					},
				)},
			wantProfiles: []string{"default-scheduler"},
		},
		{
			name: "repeated plugin name in out-of-tree plugin",
			opts: []Option{
				WithFrameworkOutOfTreeRegistry(invalidRegistry),
				WithProfiles(
					schedulerapi.KubeSchedulerProfile{
						SchedulerName: "default-scheduler",
						Plugins: &schedulerapi.Plugins{
							QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
							Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
						},
					},
				)},
			wantProfiles: []string{"default-scheduler"},
			wantErr:      "a plugin named DefaultBinder already exists",
		},
		{
			name: "multiple profiles",
			opts: []Option{
				WithProfiles(
					schedulerapi.KubeSchedulerProfile{
						SchedulerName: "foo",
						Plugins: &schedulerapi.Plugins{
							QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
							Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
						},
					},
					schedulerapi.KubeSchedulerProfile{
						SchedulerName: "bar",
						Plugins: &schedulerapi.Plugins{
							QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
							Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
						},
					},
				)},
			wantProfiles: []string{"bar", "foo"},
		},
		{
			name: "Repeated profiles",
			opts: []Option{
				WithProfiles(
					schedulerapi.KubeSchedulerProfile{
						SchedulerName: "foo",
						Plugins: &schedulerapi.Plugins{
							QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
							Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
						},
					},
					schedulerapi.KubeSchedulerProfile{
						SchedulerName: "bar",
						Plugins: &schedulerapi.Plugins{
							QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
							Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
						},
					},
					schedulerapi.KubeSchedulerProfile{
						SchedulerName: "foo",
						Plugins: &schedulerapi.Plugins{
							QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
							Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
						},
					},
				)},
			wantErr: "duplicate profile with scheduler name \"foo\"",
		},
		{
			name: "With extenders",
			opts: []Option{
				WithProfiles(
					schedulerapi.KubeSchedulerProfile{
						SchedulerName: "default-scheduler",
						Plugins: &schedulerapi.Plugins{
							QueueSort: schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "PrioritySort"}}},
							Bind:      schedulerapi.PluginSet{Enabled: []schedulerapi.Plugin{{Name: "DefaultBinder"}}},
						},
					},
				),
				WithExtenders(
					schedulerapi.Extender{
						URLPrefix: "http://extender.kube-system/",
					},
				),
			},
			wantProfiles:  []string{"default-scheduler"},
			wantExtenders: []string{"http://extender.kube-system/"},
		},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})

			stopCh := make(chan struct{})
			defer close(stopCh)
			s, err := New(
				client,
				informerFactory,
				nil,
				profile.NewRecorderFactory(eventBroadcaster),
				stopCh,
				tc.opts...,
			)

			// Errors
			if len(tc.wantErr) != 0 {
				if err == nil || !strings.Contains(err.Error(), tc.wantErr) {
					t.Errorf("got error %q, want %q", err, tc.wantErr)
				}
				return
			}
			if err != nil {
				t.Fatalf("Failed to create scheduler: %v", err)
			}

			// Profiles
			profiles := make([]string, 0, len(s.Profiles))
			for name := range s.Profiles {
				profiles = append(profiles, name)
			}
			sort.Strings(profiles)
			if diff := cmp.Diff(tc.wantProfiles, profiles); diff != "" {
				t.Errorf("unexpected profiles (-want, +got):\n%s", diff)
			}

			// Extenders
			if len(tc.wantExtenders) != 0 {
				// Scheduler.Extenders
				extenders := make([]string, 0, len(s.Extenders))
				for _, e := range s.Extenders {
					extenders = append(extenders, e.Name())
				}
				if diff := cmp.Diff(tc.wantExtenders, extenders); diff != "" {
					t.Errorf("unexpected extenders (-want, +got):\n%s", diff)
				}

				// framework.Handle.Extenders()
				for _, p := range s.Profiles {
					extenders := make([]string, 0, len(p.Extenders()))
					for _, e := range p.Extenders() {
						extenders = append(extenders, e.Name())
					}
					if diff := cmp.Diff(tc.wantExtenders, extenders); diff != "" {
						t.Errorf("unexpected extenders (-want, +got):\n%s", diff)
					}
				}
			}
		})
	}
}

func TestFailureHandler(t *testing.T) {
	testPod := st.MakePod().Name("test-pod").Namespace(v1.NamespaceDefault).Obj()
	testPodUpdated := testPod.DeepCopy()
	testPodUpdated.Labels = map[string]string{"foo": ""}

	tests := []struct {
		name                       string
		injectErr                  error
		podUpdatedDuringScheduling bool // pod is updated during a scheduling cycle
		podDeletedDuringScheduling bool // pod is deleted during a scheduling cycle
		expect                     *v1.Pod
	}{
		{
			name:                       "pod is updated during a scheduling cycle",
			injectErr:                  nil,
			podUpdatedDuringScheduling: true,
			expect:                     testPodUpdated,
		},
		{
			name:      "pod is not updated during a scheduling cycle",
			injectErr: nil,
			expect:    testPod,
		},
		{
			name:                       "pod is deleted during a scheduling cycle",
			injectErr:                  nil,
			podDeletedDuringScheduling: true,
			expect:                     nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			client := fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testPod}})
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			podInformer := informerFactory.Core().V1().Pods()
			// Need to add/update/delete testPod to the store.
			podInformer.Informer().GetStore().Add(testPod)

			queue := internalqueue.NewPriorityQueue(nil, informerFactory, internalqueue.WithClock(testingclock.NewFakeClock(time.Now())))
			schedulerCache := internalcache.New(30*time.Second, ctx.Done())

			queue.Add(testPod)
			queue.Pop()

			if tt.podUpdatedDuringScheduling {
				podInformer.Informer().GetStore().Update(testPodUpdated)
				queue.Update(testPod, testPodUpdated)
			}
			if tt.podDeletedDuringScheduling {
				podInformer.Informer().GetStore().Delete(testPod)
				queue.Delete(testPod)
			}

			s, fwk, err := initScheduler(ctx.Done(), schedulerCache, queue, client, informerFactory)
			if err != nil {
				t.Fatal(err)
			}

			testPodInfo := &framework.QueuedPodInfo{PodInfo: framework.NewPodInfo(testPod)}
			s.FailureHandler(ctx, fwk, testPodInfo, tt.injectErr, v1.PodReasonUnschedulable, nil)

			var got *v1.Pod
			if tt.podUpdatedDuringScheduling {
				head, e := queue.Pop()
				if e != nil {
					t.Fatalf("Cannot pop pod from the activeQ: %v", e)
				}
				got = head.Pod
			} else {
				got = getPodFromPriorityQueue(queue, testPod)
			}

			if diff := cmp.Diff(tt.expect, got); diff != "" {
				t.Errorf("Unexpected pod (-want, +got): %s", diff)
			}
		})
	}
}

func TestFailureHandler_NodeNotFound(t *testing.T) {
	nodeFoo := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	nodeBar := &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}
	testPod := st.MakePod().Name("test-pod").Namespace(v1.NamespaceDefault).Obj()
	tests := []struct {
		name             string
		nodes            []v1.Node
		nodeNameToDelete string
		injectErr        error
		expectNodeNames  sets.String
	}{
		{
			name:             "node is deleted during a scheduling cycle",
			nodes:            []v1.Node{*nodeFoo, *nodeBar},
			nodeNameToDelete: "foo",
			injectErr:        apierrors.NewNotFound(v1.Resource("node"), nodeFoo.Name),
			expectNodeNames:  sets.NewString("bar"),
		},
		{
			name:            "node is not deleted but NodeNotFound is received incorrectly",
			nodes:           []v1.Node{*nodeFoo, *nodeBar},
			injectErr:       apierrors.NewNotFound(v1.Resource("node"), nodeFoo.Name),
			expectNodeNames: sets.NewString("foo", "bar"),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()

			client := fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testPod}}, &v1.NodeList{Items: tt.nodes})
			informerFactory := informers.NewSharedInformerFactory(client, 0)
			podInformer := informerFactory.Core().V1().Pods()
			// Need to add testPod to the store.
			podInformer.Informer().GetStore().Add(testPod)

			queue := internalqueue.NewPriorityQueue(nil, informerFactory, internalqueue.WithClock(testingclock.NewFakeClock(time.Now())))
			schedulerCache := internalcache.New(30*time.Second, ctx.Done())

			for i := range tt.nodes {
				node := tt.nodes[i]
				// Add node to schedulerCache no matter it's deleted in API server or not.
				schedulerCache.AddNode(&node)
				if node.Name == tt.nodeNameToDelete {
					client.CoreV1().Nodes().Delete(ctx, node.Name, metav1.DeleteOptions{})
				}
			}

			s, fwk, err := initScheduler(ctx.Done(), schedulerCache, queue, client, informerFactory)
			if err != nil {
				t.Fatal(err)
			}

			testPodInfo := &framework.QueuedPodInfo{PodInfo: framework.NewPodInfo(testPod)}
			s.FailureHandler(ctx, fwk, testPodInfo, tt.injectErr, v1.PodReasonUnschedulable, nil)

			gotNodes := schedulerCache.Dump().Nodes
			gotNodeNames := sets.NewString()
			for _, nodeInfo := range gotNodes {
				gotNodeNames.Insert(nodeInfo.Node().Name)
			}
			if diff := cmp.Diff(tt.expectNodeNames, gotNodeNames); diff != "" {
				t.Errorf("Unexpected nodes (-want, +got): %s", diff)
			}
		})
	}
}

func TestFailureHandler_PodAlreadyBound(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	nodeFoo := v1.Node{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	testPod := st.MakePod().Name("test-pod").Namespace(v1.NamespaceDefault).Node("foo").Obj()

	client := fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testPod}}, &v1.NodeList{Items: []v1.Node{nodeFoo}})
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	podInformer := informerFactory.Core().V1().Pods()
	// Need to add testPod to the store.
	podInformer.Informer().GetStore().Add(testPod)

	queue := internalqueue.NewPriorityQueue(nil, informerFactory, internalqueue.WithClock(testingclock.NewFakeClock(time.Now())))
	schedulerCache := internalcache.New(30*time.Second, ctx.Done())

	// Add node to schedulerCache no matter it's deleted in API server or not.
	schedulerCache.AddNode(&nodeFoo)

	s, fwk, err := initScheduler(ctx.Done(), schedulerCache, queue, client, informerFactory)
	if err != nil {
		t.Fatal(err)
	}

	testPodInfo := &framework.QueuedPodInfo{PodInfo: framework.NewPodInfo(testPod)}
	s.FailureHandler(ctx, fwk, testPodInfo, fmt.Errorf("binding rejected: timeout"), v1.PodReasonUnschedulable, nil)

	pod := getPodFromPriorityQueue(queue, testPod)
	if pod != nil {
		t.Fatalf("Unexpected pod: %v should not be in PriorityQueue when the NodeName of pod is not empty", pod.Name)
	}
}

// getPodFromPriorityQueue is the function used in the TestDefaultErrorFunc test to get
// the specific pod from the given priority queue. It returns the found pod in the priority queue.
func getPodFromPriorityQueue(queue *internalqueue.PriorityQueue, pod *v1.Pod) *v1.Pod {
	podList := queue.PendingPods()
	if len(podList) == 0 {
		return nil
	}

	queryPodKey, err := cache.MetaNamespaceKeyFunc(pod)
	if err != nil {
		return nil
	}

	for _, foundPod := range podList {
		foundPodKey, err := cache.MetaNamespaceKeyFunc(foundPod)
		if err != nil {
			return nil
		}

		if foundPodKey == queryPodKey {
			return foundPod
		}
	}

	return nil
}

func initScheduler(stop <-chan struct{}, cache internalcache.Cache, queue internalqueue.SchedulingQueue,
	client kubernetes.Interface, informerFactory informers.SharedInformerFactory) (*Scheduler, framework.Framework, error) {
	registerPluginFuncs := []st.RegisterPluginFunc{
		st.RegisterQueueSortPlugin(queuesort.Name, queuesort.New),
		st.RegisterBindPlugin(defaultbinder.Name, defaultbinder.New),
	}
	eventBroadcaster := events.NewBroadcaster(&events.EventSinkImpl{Interface: client.EventsV1()})
	fwk, err := st.NewFramework(registerPluginFuncs,
		testSchedulerName,
		frameworkruntime.WithClientSet(client),
		frameworkruntime.WithInformerFactory(informerFactory),
		frameworkruntime.WithEventRecorder(eventBroadcaster.NewRecorder(scheme.Scheme, testSchedulerName)),
	)
	if err != nil {
		return nil, nil, err
	}

	s := newScheduler(
		cache,
		nil,
		nil,
		stop,
		queue,
		profile.Map{testSchedulerName: fwk},
		client,
		nil,
		0,
	)

	return s, fwk, nil
}
