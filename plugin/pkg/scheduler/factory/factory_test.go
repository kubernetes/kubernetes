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

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/informers"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	utiltesting "k8s.io/client-go/util/testing"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/plugin/pkg/scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api"
	latestschedulerapi "k8s.io/kubernetes/plugin/pkg/scheduler/api/latest"
	"k8s.io/kubernetes/plugin/pkg/scheduler/core"
	"k8s.io/kubernetes/plugin/pkg/scheduler/schedulercache"
	schedulertesting "k8s.io/kubernetes/plugin/pkg/scheduler/testing"
	"k8s.io/kubernetes/plugin/pkg/scheduler/util"
)

const enableEquivalenceCache = true

func TestCreate(t *testing.T) {
	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	factory := NewConfigFactory(
		v1.DefaultSchedulerName,
		client,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		v1.DefaultHardPodAffinitySymmetricWeight,
		enableEquivalenceCache,
	)
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
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	factory := NewConfigFactory(
		v1.DefaultSchedulerName,
		client,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		v1.DefaultHardPodAffinitySymmetricWeight,
		enableEquivalenceCache,
	)

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
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	factory := NewConfigFactory(
		v1.DefaultSchedulerName,
		client,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		v1.DefaultHardPodAffinitySymmetricWeight,
		enableEquivalenceCache,
	)

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
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	factory := NewConfigFactory(
		v1.DefaultSchedulerName,
		client,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		v1.DefaultHardPodAffinitySymmetricWeight,
		enableEquivalenceCache,
	)

	configData = []byte(`{}`)
	if err := runtime.DecodeInto(latestschedulerapi.Codec, configData, &policy); err != nil {
		t.Errorf("Invalid configuration: %v", err)
	}

	factory.CreateFromConfig(policy)
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
		ResponseBody: runtime.EncodeOrDie(util.Test.Codec(), testPod),
		T:            t,
	}
	mux := http.NewServeMux()

	// FakeHandler musn't be sent requests other than the one you want to test.
	mux.Handle(util.Test.ResourcePath(string(v1.ResourcePods), "bar", "foo"), &handler)
	server := httptest.NewServer(mux)
	defer server.Close()
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	factory := NewConfigFactory(
		v1.DefaultSchedulerName,
		client,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		v1.DefaultHardPodAffinitySymmetricWeight,
		enableEquivalenceCache,
	)
	queue := &core.FIFO{FIFO: cache.NewFIFO(cache.MetaNamespaceKeyFunc)}
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
		handler.ValidateRequest(t, util.Test.ResourcePath(string(v1.ResourcePods), "bar", "foo"), "GET", nil)
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
		gotObj := me.Get(i)
		if e, a := testList.Items[i].Name, gotObj.(*v1.Node).Name; e != a {
			t.Errorf("Expected %v, got %v", e, a)
		}
		if e, a := &testList.Items[i], gotObj; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %#v, got %v#", e, a)
		}
	}
}

func TestBind(t *testing.T) {
	table := []struct {
		binding *v1.Binding
	}{
		{binding: &v1.Binding{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: metav1.NamespaceDefault,
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
		client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
		b := binder{client}

		if err := b.Bind(item.binding); err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}
		expectedBody := runtime.EncodeOrDie(util.Test.Codec(), item.binding)
		handler.ValidateRequest(t,
			util.Test.SubResourcePath(string(v1.ResourcePods), metav1.NamespaceDefault, "foo", "binding"),
			"POST", &expectedBody)
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
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})
	// factory of "default-scheduler"
	informerFactory := informers.NewSharedInformerFactory(client, 0)
	factory := NewConfigFactory(
		v1.DefaultSchedulerName,
		client,
		informerFactory.Core().V1().Nodes(),
		informerFactory.Core().V1().Pods(),
		informerFactory.Core().V1().PersistentVolumes(),
		informerFactory.Core().V1().PersistentVolumeClaims(),
		informerFactory.Core().V1().ReplicationControllers(),
		informerFactory.Extensions().V1beta1().ReplicaSets(),
		informerFactory.Apps().V1beta1().StatefulSets(),
		informerFactory.Core().V1().Services(),
		informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
		-1,
		enableEquivalenceCache,
	)
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
	client := clientset.NewForConfigOrDie(&restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: &legacyscheme.Registry.GroupOrDie(v1.GroupName).GroupVersion}})

	testCases := []struct {
		hardPodAffinitySymmetricWeight int
		expectErr                      string
	}{
		{
			hardPodAffinitySymmetricWeight: -1,
			expectErr:                      "invalid hardPodAffinitySymmetricWeight: -1, must be in the range 0-100",
		},
		{
			hardPodAffinitySymmetricWeight: 101,
			expectErr:                      "invalid hardPodAffinitySymmetricWeight: 101, must be in the range 0-100",
		},
	}

	for _, test := range testCases {
		informerFactory := informers.NewSharedInformerFactory(client, 0)
		factory := NewConfigFactory(
			v1.DefaultSchedulerName,
			client,
			informerFactory.Core().V1().Nodes(),
			informerFactory.Core().V1().Pods(),
			informerFactory.Core().V1().PersistentVolumes(),
			informerFactory.Core().V1().PersistentVolumeClaims(),
			informerFactory.Core().V1().ReplicationControllers(),
			informerFactory.Extensions().V1beta1().ReplicaSets(),
			informerFactory.Apps().V1beta1().StatefulSets(),
			informerFactory.Core().V1().Services(),
			informerFactory.Policy().V1beta1().PodDisruptionBudgets(),
			test.hardPodAffinitySymmetricWeight,
			enableEquivalenceCache,
		)
		_, err := factory.Create()
		if err == nil {
			t.Errorf("expected err: %s, got nothing", test.expectErr)
		}
	}

}

func TestSkipPodUpdate(t *testing.T) {
	for _, test := range []struct {
		pod              *v1.Pod
		isAssumedPodFunc func(*v1.Pod) bool
		getPodFunc       func(*v1.Pod) *v1.Pod
		expected         bool
	}{
		// Non-assumed pod should not be skipped.
		{
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
		// Pod update (with changes on ResourceVersion, Spec.NodeName and/or
		// Annotations) for an already assumed pod should be skipped.
		{
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
		// Pod update (with changes on Labels) for an already assumed pod
		// should not be skipped.
		{
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
	} {
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
	}
}
