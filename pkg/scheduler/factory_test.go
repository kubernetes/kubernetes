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
	"encoding/json"
	"errors"
	"reflect"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/clock"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	kubefeatures "k8s.io/kubernetes/pkg/features"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/apis/config/scheme"
	extenderv1 "k8s.io/kubernetes/pkg/scheduler/apis/extender/v1"
	frameworkplugins "k8s.io/kubernetes/pkg/scheduler/framework/plugins"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultbinder"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/nodelabel"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/queuesort"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins/serviceaffinity"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	internalcache "k8s.io/kubernetes/pkg/scheduler/internal/cache"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	"k8s.io/kubernetes/pkg/scheduler/listers"
	schedulernodeinfo "k8s.io/kubernetes/pkg/scheduler/nodeinfo"
)

const (
	disablePodPreemption             = false
	bindTimeoutSeconds               = 600
	podInitialBackoffDurationSeconds = 1
	podMaxBackoffDurationSeconds     = 10
)

func TestCreate(t *testing.T) {
	client := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight, stopCh)
	factory.createFromProvider(schedulerapi.SchedulerDefaultProviderName)
}

// Test configures a scheduler from a policies defined in a file
// It combines some configurable predicate/priorities with some pre-defined ones
func TestCreateFromConfig(t *testing.T) {
	var configData []byte
	var policy schedulerapi.Policy

	client := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight, stopCh)

	configData = []byte(`{
		"kind" : "Policy",
		"apiVersion" : "v1",
		"predicates" : [
			{"name" : "TestZoneAffinity", "argument" : {"serviceAffinity" : {"labels" : ["zone"]}}},
			{"name" : "TestZoneAffinity", "argument" : {"serviceAffinity" : {"labels" : ["foo"]}}},
			{"name" : "TestRequireZone", "argument" : {"labelsPresence" : {"labels" : ["zone"], "presence" : true}}},
			{"name" : "TestNoFooLabel", "argument" : {"labelsPresence" : {"labels" : ["foo"], "presence" : false}}},
			{"name" : "PodFitsResources"},
			{"name" : "PodFitsHostPorts"}
		],
		"priorities" : [
			{"name" : "RackSpread", "weight" : 3, "argument" : {"serviceAntiAffinity" : {"label" : "rack"}}},
			{"name" : "ZoneSpread", "weight" : 3, "argument" : {"serviceAntiAffinity" : {"label" : "zone"}}},
			{"name" : "LabelPreference1", "weight" : 3, "argument" : {"labelPreference" : {"label" : "l1", "presence": true}}},
			{"name" : "LabelPreference2", "weight" : 3, "argument" : {"labelPreference" : {"label" : "l2", "presence": false}}},
			{"name" : "NodeAffinityPriority", "weight" : 2},
			{"name" : "ImageLocalityPriority", "weight" : 1}		]
	}`)
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), configData, &policy); err != nil {
		t.Errorf("Invalid configuration: %v", err)
	}

	sched, err := factory.createFromConfig(policy)
	if err != nil {
		t.Fatalf("createFromConfig failed: %v", err)
	}
	hpa := factory.hardPodAffinitySymmetricWeight
	if hpa != v1.DefaultHardPodAffinitySymmetricWeight {
		t.Errorf("Wrong hardPodAffinitySymmetricWeight, ecpected: %d, got: %d", v1.DefaultHardPodAffinitySymmetricWeight, hpa)
	}
	queueSortPls := sched.Framework.ListPlugins()["QueueSortPlugin"]
	wantQueuePls := []schedulerapi.Plugin{{Name: queuesort.Name}}
	if diff := cmp.Diff(wantQueuePls, queueSortPls); diff != "" {
		t.Errorf("Unexpected QueueSort plugins (-want, +got): %s", diff)
	}
	bindPls := sched.Framework.ListPlugins()["BindPlugin"]
	wantBindPls := []schedulerapi.Plugin{{Name: defaultbinder.Name}}
	if diff := cmp.Diff(wantBindPls, bindPls); diff != "" {
		t.Errorf("Unexpected Bind plugins (-want, +got): %s", diff)
	}

	// Verify that node label predicate/priority are converted to framework plugins.
	wantArgs := `{"Name":"NodeLabel","Args":{"presentLabels":["zone"],"absentLabels":["foo"],"presentLabelsPreference":["l1"],"absentLabelsPreference":["l2"]}}`
	verifyPluginConvertion(t, nodelabel.Name, []string{"FilterPlugin", "ScorePlugin"}, sched, factory, 6, wantArgs)
	// Verify that service affinity custom predicate/priority is converted to framework plugin.
	wantArgs = `{"Name":"ServiceAffinity","Args":{"labels":["zone","foo"],"antiAffinityLabelsPreference":["rack","zone"]}}`
	verifyPluginConvertion(t, serviceaffinity.Name, []string{"FilterPlugin", "ScorePlugin"}, sched, factory, 6, wantArgs)
}

func verifyPluginConvertion(t *testing.T, name string, extentionPoints []string, sched *Scheduler, configurator *Configurator, wantWeight int32, wantArgs string) {
	for _, extensionPoint := range extentionPoints {
		plugin, ok := findPlugin(name, extensionPoint, sched)
		if !ok {
			t.Fatalf("%q plugin does not exist in framework.", name)
		}
		if extensionPoint == "ScorePlugin" {
			if plugin.Weight != wantWeight {
				t.Errorf("Wrong weight. Got: %v, want: %v", plugin.Weight, wantWeight)
			}
		}
		// Verify that the policy config is converted to plugin config.
		pluginConfig := findPluginConfig(name, configurator)
		encoding, err := json.Marshal(pluginConfig)
		if err != nil {
			t.Errorf("Failed to marshal %+v: %v", pluginConfig, err)
		}
		if string(encoding) != wantArgs {
			t.Errorf("Config for %v plugin mismatch. got: %v, want: %v", name, string(encoding), wantArgs)
		}
	}
}

func findPlugin(name, extensionPoint string, sched *Scheduler) (schedulerapi.Plugin, bool) {
	for _, pl := range sched.Framework.ListPlugins()[extensionPoint] {
		if pl.Name == name {
			return pl, true
		}
	}
	return schedulerapi.Plugin{}, false
}

func findPluginConfig(name string, configurator *Configurator) schedulerapi.PluginConfig {
	for _, c := range configurator.pluginConfig {
		if c.Name == name {
			return c
		}
	}
	return schedulerapi.PluginConfig{}
}

func TestCreateFromConfigWithHardPodAffinitySymmetricWeight(t *testing.T) {
	var configData []byte
	var policy schedulerapi.Policy

	client := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight, stopCh)

	configData = []byte(`{
		"kind" : "Policy",
		"apiVersion" : "v1",
		"predicates" : [
			{"name" : "TestZoneAffinity", "argument" : {"serviceAffinity" : {"labels" : ["zone"]}}},
			{"name" : "TestRequireZone", "argument" : {"labelsPresence" : {"labels" : ["zone"], "presence" : true}}},
			{"name" : "PodFitsResources"},
			{"name" : "PodFitsHostPorts"}
		],
		"priorities" : [
			{"name" : "RackSpread", "weight" : 3, "argument" : {"serviceAntiAffinity" : {"label" : "rack"}}},
			{"name" : "NodeAffinityPriority", "weight" : 2},
			{"name" : "ImageLocalityPriority", "weight" : 1}
		],
		"hardPodAffinitySymmetricWeight" : 10
	}`)
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), configData, &policy); err != nil {
		t.Errorf("Invalid configuration: %v", err)
	}
	factory.createFromConfig(policy)
	hpa := factory.hardPodAffinitySymmetricWeight
	if hpa != 10 {
		t.Errorf("Wrong hardPodAffinitySymmetricWeight, ecpected: %d, got: %d", 10, hpa)
	}
}

func TestCreateFromEmptyConfig(t *testing.T) {
	var configData []byte
	var policy schedulerapi.Policy

	client := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight, stopCh)

	configData = []byte(`{}`)
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), configData, &policy); err != nil {
		t.Errorf("Invalid configuration: %v", err)
	}

	factory.createFromConfig(policy)
}

// Test configures a scheduler from a policy that does not specify any
// predicate/priority.
// The predicate/priority from DefaultProvider will be used.
func TestCreateFromConfigWithUnspecifiedPredicatesOrPriorities(t *testing.T) {
	client := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight, stopCh)

	configData := []byte(`{
		"kind" : "Policy",
		"apiVersion" : "v1"
	}`)
	var policy schedulerapi.Policy
	if err := runtime.DecodeInto(scheme.Codecs.UniversalDecoder(), configData, &policy); err != nil {
		t.Fatalf("Invalid configuration: %v", err)
	}

	sched, err := factory.createFromConfig(policy)
	if err != nil {
		t.Fatalf("Failed to create scheduler from configuration: %v", err)
	}
	if _, exist := findPlugin("NodeResourcesFit", "FilterPlugin", sched); !exist {
		t.Errorf("Expected plugin NodeResourcesFit")
	}
}

func TestDefaultErrorFunc(t *testing.T) {
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
		Spec:       apitesting.V1DeepEqualSafePodSpec(),
	}
	testPodInfo := &framework.PodInfo{Pod: testPod}
	client := fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testPod}})
	stopCh := make(chan struct{})
	defer close(stopCh)

	timestamp := time.Now()
	queue := internalqueue.NewPriorityQueue(nil, internalqueue.WithClock(clock.NewFakeClock(timestamp)))
	schedulerCache := internalcache.New(30*time.Second, stopCh)
	errFunc := MakeDefaultErrorFunc(client, queue, schedulerCache)

	// Trigger error handling again to put the pod in unschedulable queue
	errFunc(testPodInfo, nil)

	// Try up to a minute to retrieve the error pod from priority queue
	foundPodFlag := false
	maxIterations := 10 * 60
	for i := 0; i < maxIterations; i++ {
		time.Sleep(100 * time.Millisecond)
		got := getPodfromPriorityQueue(queue, testPod)
		if got == nil {
			continue
		}

		testClientGetPodRequest(client, t, testPod.Namespace, testPod.Name)

		if e, a := testPod, got; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}

		foundPodFlag = true
		break
	}

	if !foundPodFlag {
		t.Errorf("Failed to get pod from the unschedulable queue after waiting for a minute: %v", testPod)
	}

	// Remove the pod from priority queue to test putting error
	// pod in backoff queue.
	queue.Delete(testPod)

	// Trigger a move request
	queue.MoveAllToActiveOrBackoffQueue("test")

	// Trigger error handling again to put the pod in backoff queue
	errFunc(testPodInfo, nil)

	foundPodFlag = false
	for i := 0; i < maxIterations; i++ {
		time.Sleep(100 * time.Millisecond)
		// The pod should be found from backoff queue at this time
		got := getPodfromPriorityQueue(queue, testPod)
		if got == nil {
			continue
		}

		testClientGetPodRequest(client, t, testPod.Namespace, testPod.Name)

		if e, a := testPod, got; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}

		foundPodFlag = true
		break
	}

	if !foundPodFlag {
		t.Errorf("Failed to get pod from the backoff queue after waiting for a minute: %v", testPod)
	}
}

// getPodfromPriorityQueue is the function used in the TestDefaultErrorFunc test to get
// the specific pod from the given priority queue. It returns the found pod in the priority queue.
func getPodfromPriorityQueue(queue *internalqueue.PriorityQueue, pod *v1.Pod) *v1.Pod {
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

// testClientGetPodRequest function provides a routine used by TestDefaultErrorFunc test.
// It tests whether the fake client can receive request and correctly "get" the namespace
// and name of the error pod.
func testClientGetPodRequest(client *fake.Clientset, t *testing.T, podNs string, podName string) {
	requestReceived := false
	actions := client.Actions()
	for _, a := range actions {
		if a.GetVerb() == "get" {
			getAction, ok := a.(clienttesting.GetAction)
			if !ok {
				t.Errorf("Can't cast action object to GetAction interface")
				break
			}
			name := getAction.GetName()
			ns := a.GetNamespace()
			if name != podName || ns != podNs {
				t.Errorf("Expected name %s namespace %s, got %s %s",
					podName, podNs, name, ns)
			}
			requestReceived = true
		}
	}
	if !requestReceived {
		t.Errorf("Get pod request not received")
	}
}

func newConfigFactoryWithFrameworkRegistry(
	client clientset.Interface, hardPodAffinitySymmetricWeight int32, stopCh <-chan struct{},
	registry framework.Registry) *Configurator {
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	snapshot := internalcache.NewEmptySnapshot()
	return &Configurator{
		client:                         client,
		informerFactory:                informerFactory,
		podInformer:                    informerFactory.Core().V1().Pods(),
		hardPodAffinitySymmetricWeight: hardPodAffinitySymmetricWeight,
		disablePreemption:              disablePodPreemption,
		percentageOfNodesToScore:       schedulerapi.DefaultPercentageOfNodesToScore,
		bindTimeoutSeconds:             bindTimeoutSeconds,
		podInitialBackoffSeconds:       podInitialBackoffDurationSeconds,
		podMaxBackoffSeconds:           podMaxBackoffDurationSeconds,
		StopEverything:                 stopCh,
		enableNonPreempting:            utilfeature.DefaultFeatureGate.Enabled(kubefeatures.NonPreemptingPriority),
		registry:                       registry,
		plugins:                        nil,
		pluginConfig:                   []schedulerapi.PluginConfig{},
		nodeInfoSnapshot:               snapshot,
	}
}

func newConfigFactory(
	client clientset.Interface, hardPodAffinitySymmetricWeight int32, stopCh <-chan struct{}) *Configurator {
	return newConfigFactoryWithFrameworkRegistry(client, hardPodAffinitySymmetricWeight, stopCh,
		frameworkplugins.NewInTreeRegistry())
}

type fakeExtender struct {
	isBinder          bool
	interestedPodName string
	ignorable         bool
	gotBind           bool
}

func (f *fakeExtender) Name() string {
	return "fakeExtender"
}

func (f *fakeExtender) IsIgnorable() bool {
	return f.ignorable
}

func (f *fakeExtender) ProcessPreemption(
	pod *v1.Pod,
	nodeToVictims map[*v1.Node]*extenderv1.Victims,
	nodeInfos listers.NodeInfoLister,
) (map[*v1.Node]*extenderv1.Victims, error) {
	return nil, nil
}

func (f *fakeExtender) SupportsPreemption() bool {
	return false
}

func (f *fakeExtender) Filter(pod *v1.Pod, nodes []*v1.Node) (filteredNodes []*v1.Node, failedNodesMap extenderv1.FailedNodesMap, err error) {
	return nil, nil, nil
}

func (f *fakeExtender) Prioritize(
	pod *v1.Pod,
	nodes []*v1.Node,
) (hostPriorities *extenderv1.HostPriorityList, weight int64, err error) {
	return nil, 0, nil
}

func (f *fakeExtender) Bind(binding *v1.Binding) error {
	if f.isBinder {
		f.gotBind = true
		return nil
	}
	return errors.New("not a binder")
}

func (f *fakeExtender) IsBinder() bool {
	return f.isBinder
}

func (f *fakeExtender) IsInterested(pod *v1.Pod) bool {
	return pod != nil && pod.Name == f.interestedPodName
}

type TestPlugin struct {
	name string
}

var _ framework.ScorePlugin = &TestPlugin{}
var _ framework.FilterPlugin = &TestPlugin{}

func (t *TestPlugin) Name() string {
	return t.name
}

func (t *TestPlugin) Score(ctx context.Context, state *framework.CycleState, p *v1.Pod, nodeName string) (int64, *framework.Status) {
	return 1, nil
}

func (t *TestPlugin) ScoreExtensions() framework.ScoreExtensions {
	return nil
}

func (t *TestPlugin) Filter(ctx context.Context, state *framework.CycleState, pod *v1.Pod, nodeInfo *schedulernodeinfo.NodeInfo) *framework.Status {
	return nil
}
