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
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	utiltesting "k8s.io/client-go/util/testing"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/pkg/scheduler/api"
	latestschedulerapi "k8s.io/kubernetes/pkg/scheduler/api/latest"
	schedulercache "k8s.io/kubernetes/pkg/scheduler/cache"
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
	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight)
	factory.Create()
}

// Test configures a scheduler from a policies defined in a file
// It combines some configurable predicate/priorities with some pre-defined ones
func TestCreateFromConfig(t *testing.T) {
	var configData []byte
	var policy schedulerapi.Policy

	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight)

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

	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight)

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

	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight)

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
	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight)

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
	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight)

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
	handler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: runtime.EncodeOrDie(schedulertesting.Test.Codec(), testPod),
		T:            t,
	}
	mux := http.NewServeMux()

	// FakeHandler mustn't be sent requests other than the one you want to test.
	mux.Handle(schedulertesting.Test.ResourcePath(string(v1.ResourcePods), "bar", "foo"), &handler)
	server := httptest.NewServer(mux)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	factory := newConfigFactory(client, v1.DefaultHardPodAffinitySymmetricWeight)
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
		handler.ValidateRequest(t, schedulertesting.Test.ResourcePath(string(v1.ResourcePods), "bar", "foo"), "GET", nil)
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
	handler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	b := binder{client}

	if err := b.Bind(binding); err != nil {
		t.Errorf("Unexpected error: %v", err)
		return
	}
	expectedBody := runtime.EncodeOrDie(schedulertesting.Test.Codec(), binding)
	handler.ValidateRequest(t,
		schedulertesting.Test.SubResourcePath(string(v1.ResourcePods), metav1.NamespaceDefault, "foo", "binding"),
		"POST", &expectedBody)
}

func TestInvalidHardPodAffinitySymmetricWeight(t *testing.T) {
	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})
	// factory of "default-scheduler"
	factory := newConfigFactory(client, -1)
	_, err := factory.Create()
	if err == nil {
		t.Errorf("expected err: invalid hardPodAffinitySymmetricWeight, got nothing")
	}
}

func TestInvalidFactoryArgs(t *testing.T) {
	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}}})

	testCases := []struct {
		name                           string
		hardPodAffinitySymmetricWeight int32
		expectErr                      string
	}{
		{
			name: "symmetric weight below range",
			hardPodAffinitySymmetricWeight: -1,
			expectErr:                      "invalid hardPodAffinitySymmetricWeight: -1, must be in the range 0-100",
		},
		{
			name: "symmetric weight above range",
			hardPodAffinitySymmetricWeight: 101,
			expectErr:                      "invalid hardPodAffinitySymmetricWeight: 101, must be in the range 0-100",
		},
	}

	for _, test := range testCases {
		t.Run(test.name, func(t *testing.T) {
			factory := newConfigFactory(client, test.hardPodAffinitySymmetricWeight)
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
				schedulerCache: &schedulertesting.FakeCache{
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

func newConfigFactory(client *clientset.Clientset, hardPodAffinitySymmetricWeight int32) scheduler.Configurator {
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
	})
}

type fakeExtender struct {
	isBinder          bool
	interestedPodName string
	ignorable         bool
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
