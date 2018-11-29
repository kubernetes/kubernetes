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

package factory

import (
	"errors"
	"fmt"
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/fake"
	fakeV1 "k8s.io/client-go/kubernetes/typed/core/v1/fake"
	clienttesting "k8s.io/client-go/testing"
	"k8s.io/client-go/tools/cache"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	latestschedulerapi "k8s.io/kubernetes/pkg/scheduler/api/latest"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
	fakecache "k8s.io/kubernetes/pkg/scheduler/internal/cache/fake"
	internalqueue "k8s.io/kubernetes/pkg/scheduler/internal/queue"
	schedulertesting "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/pkg/scheduler/util"
)

const (
	enableEquivalenceCache = true
	disablePodPreemption   = false
	bindTimeoutSeconds     = 600
)

func TestCreate(t *testing.T) {
	client := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight, stopCh)
	factory.Create()
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

	// Pre-register some predicate and priority functions
	RegisterFitPredicate("PredicateOne", PredicateOne)
	RegisterFitPredicate("PredicateTwo", PredicateTwo)
	RegisterPriorityFunction("PriorityOne", PriorityOne, 1)
	RegisterPriorityFunction("PriorityTwo", PriorityTwo, 1)

	configData = []byte(`{
		"kind" : "Policy",
		"apiVersion" : "v1",
		"predicates" : [
			{"name" : "TestZoneAffinity", "argument" : {"serviceAffinity" : {"labels" : ["zone"]}}},
			{"name" : "TestRequireZone", "argument" : {"labelsPresence" : {"labels" : ["zone"], "presence" : true}}},
			{"name" : "PredicateOne"},
			{"name" : "PredicateTwo"}
		],
		"priorities" : [
			{"name" : "RackSpread", "weight" : 3, "argument" : {"serviceAntiAffinity" : {"label" : "rack"}}},
			{"name" : "PriorityOne", "weight" : 2},
			{"name" : "PriorityTwo", "weight" : 1}		]
	}`)
	if err := runtime.DecodeInto(latestschedulerapi.Codec, configData, &policy); err != nil {
		t.Errorf("Invalid configuration: %v", err)
	}

	factory.CreateFromConfig(policy)
	hpa := factory.GetHardPodAffinitySymmetricWeight()
	if hpa != v1.DefaultHardPodAffinitySymmetricWeight {
		t.Errorf("Wrong hardPodAffinitySymmetricWeight, ecpected: %d, got: %d", v1.DefaultHardPodAffinitySymmetricWeight, hpa)
	}
}

func TestCreateFromConfigWithHardPodAffinitySymmetricWeight(t *testing.T) {
	var configData []byte
	var policy schedulerapi.Policy

	client := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight, stopCh)

	// Pre-register some predicate and priority functions
	RegisterFitPredicate("PredicateOne", PredicateOne)
	RegisterFitPredicate("PredicateTwo", PredicateTwo)
	RegisterPriorityFunction("PriorityOne", PriorityOne, 1)
	RegisterPriorityFunction("PriorityTwo", PriorityTwo, 1)

	configData = []byte(`{
		"kind" : "Policy",
		"apiVersion" : "v1",
		"predicates" : [
			{"name" : "TestZoneAffinity", "argument" : {"serviceAffinity" : {"labels" : ["zone"]}}},
			{"name" : "TestRequireZone", "argument" : {"labelsPresence" : {"labels" : ["zone"], "presence" : true}}},
			{"name" : "PredicateOne"},
			{"name" : "PredicateTwo"}
		],
		"priorities" : [
			{"name" : "RackSpread", "weight" : 3, "argument" : {"serviceAntiAffinity" : {"label" : "rack"}}},
			{"name" : "PriorityOne", "weight" : 2},
			{"name" : "PriorityTwo", "weight" : 1}
		],
		"hardPodAffinitySymmetricWeight" : 10
	}`)
	if err := runtime.DecodeInto(latestschedulerapi.Codec, configData, &policy); err != nil {
		t.Errorf("Invalid configuration: %v", err)
	}
	factory.CreateFromConfig(policy)
	hpa := factory.GetHardPodAffinitySymmetricWeight()
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
	if err := runtime.DecodeInto(latestschedulerapi.Codec, configData, &policy); err != nil {
		t.Errorf("Invalid configuration: %v", err)
	}

	factory.CreateFromConfig(policy)
}

// Test configures a scheduler from a policy that does not specify any
// predicate/priority.
// The predicate/priority from DefaultProvider will be used.
func TestCreateFromConfigWithUnspecifiedPredicatesOrPriorities(t *testing.T) {
	client := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight, stopCh)

	RegisterFitPredicate("PredicateOne", PredicateOne)
	RegisterPriorityFunction("PriorityOne", PriorityOne, 1)

	RegisterAlgorithmProvider(DefaultProvider, sets.NewString("PredicateOne"), sets.NewString("PriorityOne"))

	configData := []byte(`{
		"kind" : "Policy",
		"apiVersion" : "v1"
	}`)
	var policy schedulerapi.Policy
	if err := runtime.DecodeInto(latestschedulerapi.Codec, configData, &policy); err != nil {
		t.Fatalf("Invalid configuration: %v", err)
	}

	config, err := factory.CreateFromConfig(policy)
	if err != nil {
		t.Fatalf("Failed to create scheduler from configuration: %v", err)
	}
	if _, found := config.Algorithm.Predicates()["PredicateOne"]; !found {
		t.Errorf("Expected predicate PredicateOne from %q", DefaultProvider)
	}
	if len(config.Algorithm.Prioritizers()) != 1 || config.Algorithm.Prioritizers()[0].Name != "PriorityOne" {
		t.Errorf("Expected priority PriorityOne from %q", DefaultProvider)
	}
}

// Test configures a scheduler from a policy that contains empty
// predicate/priority.
// Empty predicate/priority sets will be used.
func TestCreateFromConfigWithEmptyPredicatesOrPriorities(t *testing.T) {
	client := fake.NewSimpleClientset()
	stopCh := make(chan struct{})
	defer close(stopCh)
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight, stopCh)

	RegisterFitPredicate("PredicateOne", PredicateOne)
	RegisterPriorityFunction("PriorityOne", PriorityOne, 1)

	RegisterAlgorithmProvider(DefaultProvider, sets.NewString("PredicateOne"), sets.NewString("PriorityOne"))

	configData := []byte(`{
		"kind" : "Policy",
		"apiVersion" : "v1",
		"predicates" : [],
		"priorities" : []
	}`)
	var policy schedulerapi.Policy
	if err := runtime.DecodeInto(latestschedulerapi.Codec, configData, &policy); err != nil {
		t.Fatalf("Invalid configuration: %v", err)
	}

	config, err := factory.CreateFromConfig(policy)
	if err != nil {
		t.Fatalf("Failed to create scheduler from configuration: %v", err)
	}
	if len(config.Algorithm.Predicates()) != 0 {
		t.Error("Expected empty predicate sets")
	}
	if len(config.Algorithm.Prioritizers()) != 0 {
		t.Error("Expected empty priority sets")
	}
}

func PredicateOne(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	return true, nil, nil
}

func PredicateTwo(pod *v1.Pod, meta algorithm.PredicateMetadata, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	return true, nil, nil
}

func PriorityOne(pod *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	return []schedulerapi.HostPriority{}, nil
}

func PriorityTwo(pod *v1.Pod, nodeNameToInfo map[string]*schedulercache.NodeInfo, nodes []*v1.Node) (schedulerapi.HostPriorityList, error) {
	return []schedulerapi.HostPriority{}, nil
}

func TestDefaultErrorFunc(t *testing.T) {
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "bar"},
		Spec:       apitesting.V1DeepEqualSafePodSpec(),
	}
	client := fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testPod}})
	stopCh := make(chan struct{})
	defer close(stopCh)
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight, stopCh)
	queue := &internalqueue.FIFO{FIFO: cache.NewFIFO(cache.MetaNamespaceKeyFunc)}
	podBackoff := util.CreatePodBackoff(1*time.Millisecond, 1*time.Second)
	errFunc := factory.MakeDefaultErrorFunc(podBackoff, queue)

	errFunc(testPod, nil)

	for {
		// This is a terrible way to do this but I plan on replacing this
		// whole error handling system in the future. The test will time
		// out if something doesn't work.
		time.Sleep(10 * time.Millisecond)
		got, exists, _ := queue.Get(testPod)
		if !exists {
			continue
		}
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
				if name != "foo" || ns != "bar" {
					t.Errorf("Expected name %s namespace %s, got %s %s",
						"foo", "bar", name, ns)
				}
				requestReceived = true
			}
		}
		if !requestReceived {
			t.Errorf("Get pod request not received")
		}
		if e, a := testPod, got; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
		break
	}
}

func TestNodeEnumerator(t *testing.T) {
	testList := &v1.NodeList{
		Items: []v1.Node{
			{ObjectMeta: metav1.ObjectMeta{Name: "foo"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "bar"}},
			{ObjectMeta: metav1.ObjectMeta{Name: "baz"}},
		},
	}
	me := nodeEnumerator{testList}

	if e, a := 3, me.Len(); e != a {
		t.Fatalf("expected %v, got %v", e, a)
	}
	for i := range testList.Items {
		t.Run(fmt.Sprintf("node enumerator/%v", i), func(t *testing.T) {
			gotObj := me.Get(i)
			if e, a := testList.Items[i].Name, gotObj.(*v1.Node).Name; e != a {
				t.Errorf("Expected %v, got %v", e, a)
			}
			if e, a := &testList.Items[i], gotObj; !reflect.DeepEqual(e, a) {
				t.Errorf("Expected %#v, got %v#", e, a)
			}
		})
	}
}

func TestBind(t *testing.T) {
	table := []struct {
		name    string
		binding *v1.Binding
	}{
		{
			name: "binding can bind and validate request",
			binding: &v1.Binding{
				ObjectMeta: metav1.ObjectMeta{
					Namespace: metav1.NamespaceDefault,
					Name:      "foo",
				},
				Target: v1.ObjectReference{
					Name: "foohost.kubernetes.mydomain.com",
				},
			},
		},
	}

	for _, test := range table {
		t.Run(test.name, func(t *testing.T) {
			testBind(test.binding, t)
		})
	}
}

func testBind(binding *v1.Binding, t *testing.T) {
	testPod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: binding.GetName(), Namespace: metav1.NamespaceDefault},
		Spec:       apitesting.V1DeepEqualSafePodSpec(),
	}
	client := fake.NewSimpleClientset(&v1.PodList{Items: []v1.Pod{*testPod}})

	b := binder{client}

	if err := b.Bind(binding); err != nil {
		t.Errorf("Unexpected error: %v", err)
		return
	}

	pod := client.CoreV1().Pods(metav1.NamespaceDefault).(*fakeV1.FakePods)

	bind, err := pod.GetBinding(binding.GetName())
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
		return
	}

	expectedBody := runtime.EncodeOrDie(schedulertesting.Test.Codec(), binding)
	bind.APIVersion = ""
	bind.Kind = ""
	body := runtime.EncodeOrDie(schedulertesting.Test.Codec(), bind)
	if expectedBody != body {
		t.Errorf("Expected body %s, Got %s", expectedBody, body)
	}
}

func TestInvalidHardPodAffinitySymmetricWeight(t *testing.T) {
	client := fake.NewSimpleClientset()
	// factory of "default-scheduler"
	stopCh := make(chan struct{})
	factory := newConfigFactory(client, -1, stopCh)
	defer close(stopCh)
	_, err := factory.Create()
	if err == nil {
		t.Errorf("expected err: invalid hardPodAffinitySymmetricWeight, got nothing")
	}
}

func TestInvalidFactoryArgs(t *testing.T) {
	client := fake.NewSimpleClientset()

	testCases := []struct {
		name                           string
		hardPodAffinitySymmetricWeight int32
		expectErr                      string
	}{
		{
			name:                           "symmetric weight below range",
			hardPodAffinitySymmetricWeight: -1,
			expectErr:                      "invalid hardPodAffinitySymmetricWeight: -1, must be in the range 0-100",
		},
		{
			name:                           "symmetric weight above range",
			hardPodAffinitySymmetricWeight: 101,
			expectErr:                      "invalid hardPodAffinitySymmetricWeight: 101, must be in the range 0-100",
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			stopCh := make(chan struct{})
			factory := newConfigFactory(client, test.hardPodAffinitySymmetricWeight, stopCh)
			defer close(stopCh)
			_, err := factory.Create()
			if err == nil {
				t.Errorf("expected err: %s, got nothing", test.expectErr)
			}
		})
	}

}

func TestSkipPodUpdate(t *testing.T) {
	table := []struct {
		pod              *v1.Pod
		isAssumedPodFunc func(*v1.Pod) bool
		getPodFunc       func(*v1.Pod) *v1.Pod
		expected         bool
		name             string
	}{
		{
			name: "Non-assumed pod",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name: "pod-0",
				},
			},
			isAssumedPodFunc: func(*v1.Pod) bool { return false },
			getPodFunc: func(*v1.Pod) *v1.Pod {
				return &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name: "pod-0",
					},
				}
			},
			expected: false,
		},
		{
			name: "with changes on ResourceVersion, Spec.NodeName and/or Annotations",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            "pod-0",
					Annotations:     map[string]string{"a": "b"},
					ResourceVersion: "0",
				},
				Spec: v1.PodSpec{
					NodeName: "node-0",
				},
			},
			isAssumedPodFunc: func(*v1.Pod) bool {
				return true
			},
			getPodFunc: func(*v1.Pod) *v1.Pod {
				return &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:            "pod-0",
						Annotations:     map[string]string{"c": "d"},
						ResourceVersion: "1",
					},
					Spec: v1.PodSpec{
						NodeName: "node-1",
					},
				}
			},
			expected: true,
		},
		{
			name: "with changes on Labels",
			pod: &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:   "pod-0",
					Labels: map[string]string{"a": "b"},
				},
			},
			isAssumedPodFunc: func(*v1.Pod) bool {
				return true
			},
			getPodFunc: func(*v1.Pod) *v1.Pod {
				return &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{
						Name:   "pod-0",
						Labels: map[string]string{"c": "d"},
					},
				}
			},
			expected: false,
		},
	}
	for _, test := range table {
		t.Run(test.name, func(t *testing.T) {
			c := &configFactory{
				schedulerCache: &fakecache.Cache{
					IsAssumedPodFunc: test.isAssumedPodFunc,
					GetPodFunc:       test.getPodFunc,
				},
			}
			got := c.skipPodUpdate(test.pod)
			if got != test.expected {
				t.Errorf("skipPodUpdate() = %t, expected = %t", got, test.expected)
			}
		})
	}
}

func newConfigFactory(client clientset.Interface, hardPodAffinitySymmetricWeight int32, stopCh <-chan struct{}) Configurator {
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	return NewConfigFactory(&ConfigFactoryArgs{
		v1.DefaultSchedulerName,
		client,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Apps().V1().ReplicaSets(),
		informerFactory.Apps().V1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		informerFactory.Storage().V1().StorageClasses(),
		hardPodAffinitySymmetricWeight,
		enableEquivalenceCache,
		disablePodPreemption,
		schedulerapi.DefaultPercentageOfNodesToScore,
		bindTimeoutSeconds,
		stopCh,
	})
}

type fakeExtender struct {
	isBinder          bool
	interestedPodName string
	ignorable         bool
}

func (f *fakeExtender) Name() string {
	return "fakeExtender"
}

func (f *fakeExtender) IsIgnorable() bool {
	return f.ignorable
}

func (f *fakeExtender) ProcessPreemption(
	pod *v1.Pod,
	nodeToVictims map[*v1.Node]*schedulerapi.Victims,
	nodeNameToInfo map[string]*schedulercache.NodeInfo,
) (map[*v1.Node]*schedulerapi.Victims, error) {
	return nil, nil
}

func (f *fakeExtender) SupportsPreemption() bool {
	return false
}

func (f *fakeExtender) Filter(
	pod *v1.Pod,
	nodes []*v1.Node,
	nodeNameToInfo map[string]*schedulercache.NodeInfo,
) (filteredNodes []*v1.Node, failedNodesMap schedulerapi.FailedNodesMap, err error) {
	return nil, nil, nil
}

func (f *fakeExtender) Prioritize(
	pod *v1.Pod,
	nodes []*v1.Node,
) (hostPriorities *schedulerapi.HostPriorityList, weight int, err error) {
	return nil, 0, nil
}

func (f *fakeExtender) Bind(binding *v1.Binding) error {
	if f.isBinder {
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

func TestGetBinderFunc(t *testing.T) {
	table := []struct {
		podName            string
		extenders          []algorithm.SchedulerExtender
		expectedBinderType string
		name               string
	}{
		{
			name:    "the extender is not a binder",
			podName: "pod0",
			extenders: []algorithm.SchedulerExtender{
				&fakeExtender{isBinder: false, interestedPodName: "pod0"},
			},
			expectedBinderType: "*factory.binder",
		},
		{
			name:    "one of the extenders is a binder and interested in pod",
			podName: "pod0",
			extenders: []algorithm.SchedulerExtender{
				&fakeExtender{isBinder: false, interestedPodName: "pod0"},
				&fakeExtender{isBinder: true, interestedPodName: "pod0"},
			},
			expectedBinderType: "*factory.fakeExtender",
		},
		{
			name:    "one of the extenders is a binder, but not interested in pod",
			podName: "pod1",
			extenders: []algorithm.SchedulerExtender{
				&fakeExtender{isBinder: false, interestedPodName: "pod1"},
				&fakeExtender{isBinder: true, interestedPodName: "pod0"},
			},
			expectedBinderType: "*factory.binder",
		},
	}

	for _, test := range table {
		t.Run(test.name, func(t *testing.T) {
			testGetBinderFunc(test.expectedBinderType, test.podName, test.extenders, t)
		})
	}
}

func testGetBinderFunc(expectedBinderType, podName string, extenders []algorithm.SchedulerExtender, t *testing.T) {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
	}

	f := &configFactory{}
	binderFunc := f.getBinderFunc(extenders)
	binder := binderFunc(pod)

	binderType := fmt.Sprintf("%s", reflect.TypeOf(binder))
	if binderType != expectedBinderType {
		t.Errorf("Expected binder %q but got %q", expectedBinderType, binderType)
	}
}

func TestNodeAllocatableChanged(t *testing.T) {
	newQuantity := func(value int64) resource.Quantity {
		return *resource.NewQuantity(value, resource.BinarySI)
	}
	for _, c := range []struct {
		Name           string
		Changed        bool
		OldAllocatable v1.ResourceList
		NewAllocatable v1.ResourceList
	}{
		{
			Name:           "no allocatable resources changed",
			Changed:        false,
			OldAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024)},
			NewAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024)},
		},
		{
			Name:           "new node has more allocatable resources",
			Changed:        true,
			OldAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024)},
			NewAllocatable: v1.ResourceList{v1.ResourceMemory: newQuantity(1024), v1.ResourceStorage: newQuantity(1024)},
		},
	} {
		oldNode := &v1.Node{Status: v1.NodeStatus{Allocatable: c.OldAllocatable}}
		newNode := &v1.Node{Status: v1.NodeStatus{Allocatable: c.NewAllocatable}}
		changed := nodeAllocatableChanged(newNode, oldNode)
		if changed != c.Changed {
			t.Errorf("nodeAllocatableChanged should be %t, got %t", c.Changed, changed)
		}
	}
}

func TestNodeLabelsChanged(t *testing.T) {
	for _, c := range []struct {
		Name      string
		Changed   bool
		OldLabels map[string]string
		NewLabels map[string]string
	}{
		{
			Name:      "no labels changed",
			Changed:   false,
			OldLabels: map[string]string{"foo": "bar"},
			NewLabels: map[string]string{"foo": "bar"},
		},
		// Labels changed.
		{
			Name:      "new node has more labels",
			Changed:   true,
			OldLabels: map[string]string{"foo": "bar"},
			NewLabels: map[string]string{"foo": "bar", "test": "value"},
		},
	} {
		oldNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: c.OldLabels}}
		newNode := &v1.Node{ObjectMeta: metav1.ObjectMeta{Labels: c.NewLabels}}
		changed := nodeLabelsChanged(newNode, oldNode)
		if changed != c.Changed {
			t.Errorf("Test case %q failed: should be %t, got %t", c.Name, c.Changed, changed)
		}
	}
}

func TestNodeTaintsChanged(t *testing.T) {
	for _, c := range []struct {
		Name      string
		Changed   bool
		OldTaints []v1.Taint
		NewTaints []v1.Taint
	}{
		{
			Name:      "no taint changed",
			Changed:   false,
			OldTaints: []v1.Taint{{Key: "key", Value: "value"}},
			NewTaints: []v1.Taint{{Key: "key", Value: "value"}},
		},
		{
			Name:      "taint value changed",
			Changed:   true,
			OldTaints: []v1.Taint{{Key: "key", Value: "value1"}},
			NewTaints: []v1.Taint{{Key: "key", Value: "value2"}},
		},
	} {
		oldNode := &v1.Node{Spec: v1.NodeSpec{Taints: c.OldTaints}}
		newNode := &v1.Node{Spec: v1.NodeSpec{Taints: c.NewTaints}}
		changed := nodeTaintsChanged(newNode, oldNode)
		if changed != c.Changed {
			t.Errorf("Test case %q failed: should be %t, not %t", c.Name, c.Changed, changed)
		}
	}
}

func TestNodeConditionsChanged(t *testing.T) {
	nodeConditionType := reflect.TypeOf(v1.NodeCondition{})
	if nodeConditionType.NumField() != 6 {
		t.Errorf("NodeCondition type has changed. The nodeConditionsChanged() function must be reevaluated.")
	}

	for _, c := range []struct {
		Name          string
		Changed       bool
		OldConditions []v1.NodeCondition
		NewConditions []v1.NodeCondition
	}{
		{
			Name:          "no condition changed",
			Changed:       false,
			OldConditions: []v1.NodeCondition{{Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue}},
			NewConditions: []v1.NodeCondition{{Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue}},
		},
		{
			Name:          "only LastHeartbeatTime changed",
			Changed:       false,
			OldConditions: []v1.NodeCondition{{Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue, LastHeartbeatTime: metav1.Unix(1, 0)}},
			NewConditions: []v1.NodeCondition{{Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue, LastHeartbeatTime: metav1.Unix(2, 0)}},
		},
		{
			Name:          "new node has more healthy conditions",
			Changed:       true,
			OldConditions: []v1.NodeCondition{},
			NewConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
		},
		{
			Name:          "new node has less unhealthy conditions",
			Changed:       true,
			OldConditions: []v1.NodeCondition{{Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue}},
			NewConditions: []v1.NodeCondition{},
		},
		{
			Name:          "condition status changed",
			Changed:       true,
			OldConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}},
			NewConditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}},
		},
	} {
		oldNode := &v1.Node{Status: v1.NodeStatus{Conditions: c.OldConditions}}
		newNode := &v1.Node{Status: v1.NodeStatus{Conditions: c.NewConditions}}
		changed := nodeConditionsChanged(newNode, oldNode)
		if changed != c.Changed {
			t.Errorf("Test case %q failed: should be %t, got %t", c.Name, c.Changed, changed)
		}
	}
}
