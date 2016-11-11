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
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"
	"time"

	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/client/cache"
	clientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_5"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	latestschedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api/latest"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
)

func TestCreate(t *testing.T) {
	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}})
	factory := NewConfigFactory(client, v1.DefaultSchedulerName, v1.DefaultHardPodAffinitySymmetricWeight, v1.DefaultFailureDomains)
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
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}})
	factory := NewConfigFactory(client, v1.DefaultSchedulerName, v1.DefaultHardPodAffinitySymmetricWeight, v1.DefaultFailureDomains)

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
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}})
	factory := NewConfigFactory(client, v1.DefaultSchedulerName, v1.DefaultHardPodAffinitySymmetricWeight, v1.DefaultFailureDomains)

	configData = []byte(`{}`)
	if err := runtime.DecodeInto(latestschedulerapi.Codec, configData, &policy); err != nil {
		t.Errorf("Invalid configuration: %v", err)
	}

	factory.CreateFromConfig(policy)
}

func PredicateOne(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
	return true, nil, nil
}

func PredicateTwo(pod *v1.Pod, meta interface{}, nodeInfo *schedulercache.NodeInfo) (bool, []algorithm.PredicateFailureReason, error) {
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
		ObjectMeta: v1.ObjectMeta{Name: "foo", Namespace: "bar"},
		Spec:       apitesting.V1DeepEqualSafePodSpec(),
	}
	handler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: runtime.EncodeOrDie(testapi.Default.Codec(), testPod),
		T:            t,
	}
	mux := http.NewServeMux()

	// FakeHandler musn't be sent requests other than the one you want to test.
	mux.Handle(testapi.Default.ResourcePath("pods", "bar", "foo"), &handler)
	server := httptest.NewServer(mux)
	defer server.Close()
	factory := NewConfigFactory(clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}}), v1.DefaultSchedulerName, v1.DefaultHardPodAffinitySymmetricWeight, v1.DefaultFailureDomains)
	queue := cache.NewFIFO(cache.MetaNamespaceKeyFunc)
	podBackoff := podBackoff{
		perPodBackoff:   map[types.NamespacedName]*backoffEntry{},
		clock:           &fakeClock{},
		defaultDuration: 1 * time.Millisecond,
		maxDuration:     1 * time.Second,
	}
	errFunc := factory.makeDefaultErrorFunc(&podBackoff, queue)

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
		handler.ValidateRequest(t, testapi.Default.ResourcePath("pods", "bar", "foo"), "GET", nil)
		if e, a := testPod, got; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
		break
	}
}

func TestNodeEnumerator(t *testing.T) {
	testList := &v1.NodeList{
		Items: []v1.Node{
			{ObjectMeta: v1.ObjectMeta{Name: "foo"}},
			{ObjectMeta: v1.ObjectMeta{Name: "bar"}},
			{ObjectMeta: v1.ObjectMeta{Name: "baz"}},
		},
	}
	me := nodeEnumerator{testList}

	if e, a := 3, me.Len(); e != a {
		t.Fatalf("expected %v, got %v", e, a)
	}
	for i := range testList.Items {
		gotObj := me.Get(i)
		if e, a := testList.Items[i].Name, gotObj.(*v1.Node).Name; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
		if e, a := &testList.Items[i], gotObj; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %#v, got %v#", e, a)
		}
	}
}

type fakeClock struct {
	t time.Time
}

func (f *fakeClock) Now() time.Time {
	return f.t
}

func TestBind(t *testing.T) {
	table := []struct {
		binding *v1.Binding
	}{
		{binding: &v1.Binding{
			ObjectMeta: v1.ObjectMeta{
				Namespace: v1.NamespaceDefault,
				Name:      "foo",
			},
			Target: v1.ObjectReference{
				Name: "foohost.kubernetes.mydomain.com",
			},
		}},
	}

	for _, item := range table {
		handler := utiltesting.FakeHandler{
			StatusCode:   200,
			ResponseBody: "",
			T:            t,
		}
		server := httptest.NewServer(&handler)
		defer server.Close()
		client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}})
		b := binder{client}

		if err := b.Bind(item.binding); err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		expectedBody := runtime.EncodeOrDie(testapi.Default.Codec(), item.binding)
		handler.ValidateRequest(t, testapi.Default.ResourcePath("bindings", v1.NamespaceDefault, ""), "POST", &expectedBody)
	}
}

func TestBackoff(t *testing.T) {
	clock := fakeClock{}
	backoff := podBackoff{
		perPodBackoff:   map[types.NamespacedName]*backoffEntry{},
		clock:           &clock,
		defaultDuration: 1 * time.Second,
		maxDuration:     60 * time.Second,
	}

	tests := []struct {
		podID            types.NamespacedName
		expectedDuration time.Duration
		advanceClock     time.Duration
	}{
		{
			podID:            types.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 1 * time.Second,
		},
		{
			podID:            types.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 2 * time.Second,
		},
		{
			podID:            types.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 4 * time.Second,
		},
		{
			podID:            types.NamespacedName{Namespace: "default", Name: "bar"},
			expectedDuration: 1 * time.Second,
			advanceClock:     120 * time.Second,
		},
		// 'foo' should have been gc'd here.
		{
			podID:            types.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 1 * time.Second,
		},
	}

	for _, test := range tests {
		duration := backoff.getEntry(test.podID).getBackoff(backoff.maxDuration)
		if duration != test.expectedDuration {
			t.Errorf("expected: %s, got %s for %s", test.expectedDuration.String(), duration.String(), test.podID)
		}
		clock.t = clock.t.Add(test.advanceClock)
		backoff.gc()
	}
	fooID := types.NamespacedName{Namespace: "default", Name: "foo"}
	backoff.perPodBackoff[fooID].backoff = 60 * time.Second
	duration := backoff.getEntry(fooID).getBackoff(backoff.maxDuration)
	if duration != 60*time.Second {
		t.Errorf("expected: 60, got %s", duration.String())
	}
	// Verify that we split on namespaces correctly, same name, different namespace
	fooID.Namespace = "other"
	duration = backoff.getEntry(fooID).getBackoff(backoff.maxDuration)
	if duration != 1*time.Second {
		t.Errorf("expected: 1, got %s", duration.String())
	}
}

// TestResponsibleForPod tests if a pod with an annotation that should cause it to
// be picked up by the default scheduler, is in fact picked by the default scheduler
// Two schedulers are made in the test: one is default scheduler and other scheduler
// is of name "foo-scheduler". A pod must be picked up by at most one of the two
// schedulers.
func TestResponsibleForPod(t *testing.T) {
	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}})
	// factory of "default-scheduler"
	factoryDefaultScheduler := NewConfigFactory(client, v1.DefaultSchedulerName, v1.DefaultHardPodAffinitySymmetricWeight, v1.DefaultFailureDomains)
	// factory of "foo-scheduler"
	factoryFooScheduler := NewConfigFactory(client, "foo-scheduler", v1.DefaultHardPodAffinitySymmetricWeight, v1.DefaultFailureDomains)
	// scheduler annotations to be tested
	schedulerAnnotationFitsDefault := map[string]string{"scheduler.alpha.kubernetes.io/name": "default-scheduler"}
	schedulerAnnotationFitsFoo := map[string]string{"scheduler.alpha.kubernetes.io/name": "foo-scheduler"}
	schedulerAnnotationFitsNone := map[string]string{"scheduler.alpha.kubernetes.io/name": "bar-scheduler"}
	tests := []struct {
		pod             *v1.Pod
		pickedByDefault bool
		pickedByFoo     bool
	}{
		{
			// pod with no annotation "scheduler.alpha.kubernetes.io/name=<scheduler-name>" should be
			// picked by the default scheduler, NOT by the one of name "foo-scheduler"
			pod:             &v1.Pod{ObjectMeta: v1.ObjectMeta{Name: "foo", Namespace: "bar"}},
			pickedByDefault: true,
			pickedByFoo:     false,
		},
		{
			// pod with annotation "scheduler.alpha.kubernetes.io/name=default-scheduler" should be picked
			// by the scheduler of name "default-scheduler", NOT by the one of name "foo-scheduler"
			pod:             &v1.Pod{ObjectMeta: v1.ObjectMeta{Name: "foo", Namespace: "bar", Annotations: schedulerAnnotationFitsDefault}},
			pickedByDefault: true,
			pickedByFoo:     false,
		},
		{
			// pod with annotataion "scheduler.alpha.kubernetes.io/name=foo-scheduler" should be NOT
			// be picked by the scheduler of name "default-scheduler", but by the one of name "foo-scheduler"
			pod:             &v1.Pod{ObjectMeta: v1.ObjectMeta{Name: "foo", Namespace: "bar", Annotations: schedulerAnnotationFitsFoo}},
			pickedByDefault: false,
			pickedByFoo:     true,
		},
		{
			// pod with annotataion "scheduler.alpha.kubernetes.io/name=foo-scheduler" should be NOT
			// be picked by niether the scheduler of name "default-scheduler" nor the one of name "foo-scheduler"
			pod:             &v1.Pod{ObjectMeta: v1.ObjectMeta{Name: "foo", Namespace: "bar", Annotations: schedulerAnnotationFitsNone}},
			pickedByDefault: false,
			pickedByFoo:     false,
		},
	}

	for _, test := range tests {
		podOfDefault := factoryDefaultScheduler.responsibleForPod(test.pod)
		podOfFoo := factoryFooScheduler.responsibleForPod(test.pod)
		results := []bool{podOfDefault, podOfFoo}
		expected := []bool{test.pickedByDefault, test.pickedByFoo}
		if !reflect.DeepEqual(results, expected) {
			t.Errorf("expected: {%v, %v}, got {%v, %v}", test.pickedByDefault, test.pickedByFoo, podOfDefault, podOfFoo)
		}
	}
}

func TestInvalidHardPodAffinitySymmetricWeight(t *testing.T) {
	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	// TODO: Uncomment when fix #19254
	// defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}})
	// factory of "default-scheduler"
	factory := NewConfigFactory(client, v1.DefaultSchedulerName, -1, v1.DefaultFailureDomains)
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
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &registered.GroupOrDie(v1.GroupName).GroupVersion}})

	testCases := []struct {
		hardPodAffinitySymmetricWeight int
		failureDomains                 string
		expectErr                      string
	}{
		{
			hardPodAffinitySymmetricWeight: -1,
			failureDomains:                 v1.DefaultFailureDomains,
			expectErr:                      "invalid hardPodAffinitySymmetricWeight: -1, must be in the range 0-100",
		},
		{
			hardPodAffinitySymmetricWeight: 101,
			failureDomains:                 v1.DefaultFailureDomains,
			expectErr:                      "invalid hardPodAffinitySymmetricWeight: 101, must be in the range 0-100",
		},
		{
			hardPodAffinitySymmetricWeight: 0,
			failureDomains:                 "INVALID_FAILURE_DOMAINS",
			expectErr:                      "invalid failure domain: INVALID_FAILURE_DOMAINS",
		},
	}

	for _, test := range testCases {
		factory := NewConfigFactory(client, v1.DefaultSchedulerName, test.hardPodAffinitySymmetricWeight, test.failureDomains)
		_, err := factory.Create()
		if err == nil {
			t.Errorf("expected err: %s, got nothing", test.expectErr)
		}
	}

}

func TestNodeConditionPredicate(t *testing.T) {
	nodeFunc := getNodeConditionPredicate()
	nodeList := &v1.NodeList{
		Items: []v1.Node{
			// node1 considered
			{ObjectMeta: v1.ObjectMeta{Name: "node1"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}}}},
			// node2 ignored - node not Ready
			{ObjectMeta: v1.ObjectMeta{Name: "node2"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}}}},
			// node3 ignored - node out of disk
			{ObjectMeta: v1.ObjectMeta{Name: "node3"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue}}}},
			// node4 considered
			{ObjectMeta: v1.ObjectMeta{Name: "node4"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeOutOfDisk, Status: v1.ConditionFalse}}}},

			// node5 ignored - node out of disk
			{ObjectMeta: v1.ObjectMeta{Name: "node5"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}, {Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue}}}},
			// node6 considered
			{ObjectMeta: v1.ObjectMeta{Name: "node6"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionTrue}, {Type: v1.NodeOutOfDisk, Status: v1.ConditionFalse}}}},
			// node7 ignored - node out of disk, node not Ready
			{ObjectMeta: v1.ObjectMeta{Name: "node7"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}, {Type: v1.NodeOutOfDisk, Status: v1.ConditionTrue}}}},
			// node8 ignored - node not Ready
			{ObjectMeta: v1.ObjectMeta{Name: "node8"}, Status: v1.NodeStatus{Conditions: []v1.NodeCondition{{Type: v1.NodeReady, Status: v1.ConditionFalse}, {Type: v1.NodeOutOfDisk, Status: v1.ConditionFalse}}}},

			// node9 ignored - node unschedulable
			{ObjectMeta: v1.ObjectMeta{Name: "node9"}, Spec: v1.NodeSpec{Unschedulable: true}},
			// node10 considered
			{ObjectMeta: v1.ObjectMeta{Name: "node10"}, Spec: v1.NodeSpec{Unschedulable: false}},
			// node11 considered
			{ObjectMeta: v1.ObjectMeta{Name: "node11"}},
		},
	}

	nodeNames := []string{}
	for _, node := range nodeList.Items {
		if nodeFunc(&node) {
			nodeNames = append(nodeNames, node.Name)
		}
	}
	expectedNodes := []string{"node1", "node4", "node6", "node10", "node11"}
	if !reflect.DeepEqual(expectedNodes, nodeNames) {
		t.Errorf("expected: %v, got %v", expectedNodes, nodeNames)
	}
}
