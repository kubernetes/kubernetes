/*
Copyright 2015 The Kubernetes Authors.

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

package core

import (
	"context"
	"reflect"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	clientsetfake "k8s.io/client-go/kubernetes/fake"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/runtime"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/profile"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
)

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
			client := clientsetfake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			var extenders []framework.Extender
			for ii := range test.extenders {
				extenders = append(extenders, &test.extenders[ii])
			}
			cache := internalcache.New(time.Duration(0), wait.NeverStop)
			for _, name := range test.nodes {
				cache.AddNode(createNode(name))
			}

			fwk, err := st.NewFramework(
				test.registerPlugins,
				runtime.WithClientSet(client),
				runtime.WithInformerFactory(informerFactory),
				runtime.WithPodNominator(internalqueue.NewPodNominator()),
			)
			if err != nil {
				t.Fatal(err)
			}
			prof := &profile.Profile{
				Framework: fwk,
			}

			scheduler := NewGenericScheduler(
				cache,
				emptySnapshot,
				extenders,
				schedulerapi.DefaultPercentageOfNodesToScore)
			podIgnored := &v1.Pod{}
			result, err := scheduler.Schedule(context.Background(), prof, framework.NewCycleState(), podIgnored)
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

func createNode(name string) *v1.Node {
	return &v1.Node{ObjectMeta: metav1.ObjectMeta{Name: name}}
}

func TestIsInterested(t *testing.T) {
	mem := &HTTPExtender{
		managedResources: sets.NewString(),
	}
	mem.managedResources.Insert("memory")

	for _, tc := range []struct {
		label    string
		extender *HTTPExtender
		pod      *v1.Pod
		want     bool
	}{
		{
			label: "Empty managed resources",
			extender: &HTTPExtender{
				managedResources: sets.NewString(),
			},
			pod:  &v1.Pod{},
			want: true,
		},
		{
			label:    "Managed memory, empty resources",
			extender: mem,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "app",
						},
					},
				},
			},
			want: false,
		},
		{
			label:    "Managed memory, container memory",
			extender: mem,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "app",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{"memory": resource.Quantity{}},
								Limits:   v1.ResourceList{"memory": resource.Quantity{}},
							},
						},
					},
				},
			},
			want: true,
		},
		{
			label:    "Managed memory, init container memory",
			extender: mem,
			pod: &v1.Pod{
				Spec: v1.PodSpec{
					Containers: []v1.Container{
						{
							Name: "app",
						},
					},
					InitContainers: []v1.Container{
						{
							Name: "init",
							Resources: v1.ResourceRequirements{
								Requests: v1.ResourceList{"memory": resource.Quantity{}},
								Limits:   v1.ResourceList{"memory": resource.Quantity{}},
							},
						},
					},
				},
			},
			want: true,
		},
	} {
		t.Run(tc.label, func(t *testing.T) {
			if got := tc.extender.IsInterested(tc.pod); got != tc.want {
				t.Fatalf("IsInterested(%v) = %v, wanted %v", tc.pod, got, tc.want)
			}
		})
	}
}
