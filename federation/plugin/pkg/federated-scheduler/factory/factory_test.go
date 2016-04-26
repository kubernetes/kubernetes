/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/api/testapi"
	apiunversioned "k8s.io/kubernetes/pkg/api/unversioned"
	extensions "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/client/cache"
	"k8s.io/kubernetes/federation/client/clientset_generated/release_1_3"
	kubeclientset "k8s.io/kubernetes/pkg/client/clientset_generated/release_1_3"
	"k8s.io/kubernetes/pkg/client/restclient"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/types"
	utiltesting "k8s.io/kubernetes/pkg/util/testing"
	"k8s.io/kubernetes/federation/apis/federation/unversioned"
	federation "k8s.io/kubernetes/federation/apis/federation/v1alpha1"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/algorithm"
	schedulerapi "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/api"
	latestschedulerapi "k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/api/latest"
	"k8s.io/kubernetes/federation/plugin/pkg/federated-scheduler/schedulercache"
)

func TestCreate(t *testing.T) {
	handler := utiltesting.FakeHandler{
		StatusCode:   500,
		ResponseBody: "",
		T:            t,
	}
	server := httptest.NewServer(&handler)

	kubeconfig := restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Extensions.GroupVersion()}}
	kubeClientSet := kubeclientset.NewForConfigOrDie(&kubeconfig)
	federatedClientSet := release_1_3.NewForConfigOrDie(&kubeconfig)
	factory := NewConfigFactory(federatedClientSet, kubeClientSet, api.DefaultSchedulerName)

	factory.Create()
}

// Test configures a federated-scheduler from a policies defined in a file
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
	kubeconfig := restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Extensions.GroupVersion()}}
	kubeClientSet := kubeclientset.NewForConfigOrDie(&kubeconfig)
	federatedClientSet := release_1_3.NewForConfigOrDie(&kubeconfig)
	factory := NewConfigFactory(federatedClientSet, kubeClientSet, api.DefaultSchedulerName)

	// Pre-register some predicate and priority functions
	RegisterFitPredicate("PredicateOne", PredicateOne)
	RegisterFitPredicate("PredicateTwo", PredicateTwo)
	RegisterPriorityFunction("PriorityOne", PriorityOne, 1)
	RegisterPriorityFunction("PriorityTwo", PriorityTwo, 1)

	configData = []byte(`{
		"kind" : "Policy",
		"apiVersion" : "v1",
		"predicates" : [
			{"name" : "PredicateOne"},
			{"name" : "PredicateTwo"}
		],
		"priorities" : [
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

	kubeconfig := restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Extensions.GroupVersion()}}
	kubeClientSet := kubeclientset.NewForConfigOrDie(&kubeconfig)
	federatedClientSet := release_1_3.NewForConfigOrDie(&kubeconfig)
	factory := NewConfigFactory(federatedClientSet, kubeClientSet, api.DefaultSchedulerName)

	configData = []byte(`{}`)
	if err := runtime.DecodeInto(latestschedulerapi.Codec, configData, &policy); err != nil {
		t.Errorf("Invalid configuration: %v", err)
	}

	factory.CreateFromConfig(policy)
}

func PredicateOne(replicaSet *extensions.ReplicaSet, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	return true, nil
}

func PredicateTwo(replicaSet *extensions.ReplicaSet, clusterName string, clusterInfo *schedulercache.ClusterInfo) (bool, error) {
	return true, nil
}

func PriorityOne(replicaSet *extensions.ReplicaSet, clusterNameToInfo map[string]*schedulercache.ClusterInfo, clusterLister algorithm.ClusterLister) (schedulerapi.ClusterPriorityList, error) {
	return []schedulerapi.ClusterPriority{}, nil
}

func PriorityTwo(replicaSet *extensions.ReplicaSet, clusterNameToInfo map[string]*schedulercache.ClusterInfo, clusterLister algorithm.ClusterLister) (schedulerapi.ClusterPriorityList, error) {
	return []schedulerapi.ClusterPriority{}, nil
}

func TestDefaultErrorFunc(t *testing.T) {
	testReplicaSet := &extensions.ReplicaSet{
		TypeMeta: apiunversioned.TypeMeta {
			Kind: "ReplicaSet",
			APIVersion: "extensions/v1beta1",
		},
		ObjectMeta: v1.ObjectMeta {
			Name: "foo",
			Namespace: "bar",
		},
	}
	codec, _ := api.Codecs.SerializerForFileExtension("json")

	handler := utiltesting.FakeHandler{
		StatusCode:   200,
		ResponseBody: runtime.EncodeOrDie(codec, testReplicaSet),
		T:            t,
	}
	mux := http.NewServeMux()
	// FakeHandler mustn't be sent requests other than the one you want to test.
	mux.Handle(testapi.Extensions.ResourcePath("replicasets", "bar", "foo"), &handler)
	server := httptest.NewServer(mux)
	defer server.Close()
	kubeconfig := restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Extensions.GroupVersion()}}
	federatedconfig := restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Federation.GroupVersion()}}
	kubeClientSet := kubeclientset.NewForConfigOrDie(&kubeconfig)
	federatedClientSet := release_1_3.NewForConfigOrDie(&federatedconfig)
	factory := NewConfigFactory(federatedClientSet, kubeClientSet, api.DefaultSchedulerName)
	queue := cache.NewFIFO(cache.MetaNamespaceKeyFunc)

	replicaSetBackoff := rsBackoff{
		perRsBackoff:   map[types.NamespacedName]*backoffEntry{},
		clock:           &fakeClock{},
		defaultDuration: 1 * time.Millisecond,
		maxDuration:     1 * time.Second,
	}
	errFunc := factory.makeDefaultErrorFunc(&replicaSetBackoff, queue)
	errFunc(testReplicaSet, nil)
	for {
		// This is a terrible way to do this but I plan on replacing this
		// whole error handling system in the future. The test will time
		// out if something doesn't work.
		time.Sleep(10 * time.Millisecond)
		got, exists, _ := queue.Get(testReplicaSet)
		if !exists {
			continue
		}
		handler.ValidateRequest(t, testapi.Extensions.ResourcePath("replicaSets", "bar", "foo"), "GET", nil)
		if e, a := testReplicaSet, got; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected %v, got %v", e, a)
		}
		break
	}
}

func TestClusterEnumerator(t *testing.T) {
	testList := &federation.ClusterList{
		Items: []federation.Cluster{
			{ObjectMeta: v1.ObjectMeta{Name: "foo"}},
			{ObjectMeta: v1.ObjectMeta{Name: "bar"}},
			{ObjectMeta: v1.ObjectMeta{Name: "baz"}},
		},
	}
	me := clusterEnumerator{testList}

	if e, a := 3, me.Len(); e != a {
		t.Fatalf("expected %v, got %v", e, a)
	}
	for i := range testList.Items {
		gotObj := me.Get(i)
		if e, a := testList.Items[i].Name, gotObj.(*federation.Cluster).Name; e != a {
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
	grace := int64(30)
	table := []struct {
		binding *extensions.ReplicaSet
	}{
		{binding: &extensions.ReplicaSet{
			TypeMeta: apiunversioned.TypeMeta{
				Kind:"SubReplicaSet",
				APIVersion:"federation/v1alpha1",
			},
			ObjectMeta: v1.ObjectMeta {
				Name: "rs-foo",
				GenerateName: "rs-foo",
				Namespace: "bar",
			},
			Spec: extensions.ReplicaSetSpec{
				Template: v1.PodTemplateSpec{
					Spec: v1.PodSpec{
						RestartPolicy:                 v1.RestartPolicyAlways,
						DNSPolicy:                     v1.DNSClusterFirst,
						TerminationGracePeriodSeconds: &grace,
						SecurityContext:               &v1.PodSecurityContext{},
					},
				},
			},
		}},
	}

	for _, item := range table {
		handler := utiltesting.FakeHandler{
			StatusCode:   200,
			ResponseBody: "{\"result\":\"ok\"}",
			T:            t,
		}
		server := httptest.NewServer(&handler)

		federatedconfig := restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Federation.GroupVersion()}}
		kubeconfig := restclient.Config{Host: server.URL, ContentConfig: restclient.ContentConfig{GroupVersion: testapi.Extensions.GroupVersion()}}
		federatedClientSet := release_1_3.NewForConfigOrDie(&federatedconfig)
		kubeClientSet := kubeclientset.NewForConfigOrDie(&kubeconfig)
		b := binder{federatedClientSet,kubeClientSet}

		item.binding.Annotations = map[string]string{}
		item.binding.Annotations[unversioned.TargetClusterKey] = "foo"
		item.binding.Annotations[unversioned.FederationReplicaSetKey] = item.binding.Name

		if err := b.Bind(item.binding); err != nil {
			t.Errorf("Unexpected error: %v", err)
			continue
		}

		codec, _ := api.Codecs.SerializerForFileExtension("json")

		expectedBody := runtime.EncodeOrDie(codec, item.binding)
		//handler.ResponseBody
		handler.ValidateRequest(t, testapi.Extensions.ResourcePath("replicasets", "bar", "rs-foo"), "PUT", &expectedBody)
	}
}

func TestBackoff(t *testing.T) {
	clock := fakeClock{}
	backoff := rsBackoff{
		perRsBackoff:   map[types.NamespacedName]*backoffEntry{},
		clock:           &clock,
		defaultDuration: 1 * time.Second,
		maxDuration:     60 * time.Second,
	}

	tests := []struct {
		replicaSetID            types.NamespacedName
		expectedDuration time.Duration
		advanceClock     time.Duration
	}{
		{
			replicaSetID:            types.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 1 * time.Second,
		},
		{
			replicaSetID:            types.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 2 * time.Second,
		},
		{
			replicaSetID:            types.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 4 * time.Second,
		},
		{
			replicaSetID:            types.NamespacedName{Namespace: "default", Name: "bar"},
			expectedDuration: 1 * time.Second,
			advanceClock:     120 * time.Second,
		},
		// 'foo' should have been gc'd here.
		{
			replicaSetID:            types.NamespacedName{Namespace: "default", Name: "foo"},
			expectedDuration: 1 * time.Second,
		},
	}

	for _, test := range tests {
		duration := backoff.getEntry(test.replicaSetID).getBackoff(backoff.maxDuration)
		if duration != test.expectedDuration {
			t.Errorf("expected: %s, got %s for %s", test.expectedDuration.String(), duration.String(), test.replicaSetID)
		}
		clock.t = clock.t.Add(test.advanceClock)
		backoff.gc()
	}
	fooID := types.NamespacedName{Namespace: "default", Name: "foo"}
	backoff.perRsBackoff[fooID].backoff = 60 * time.Second
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
