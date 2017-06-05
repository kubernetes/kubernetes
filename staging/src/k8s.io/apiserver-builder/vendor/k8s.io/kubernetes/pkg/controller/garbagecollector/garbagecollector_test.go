/*
Copyright 2016 The Kubernetes Authors.

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

package garbagecollector

import (
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"sync"
	"testing"

	"github.com/stretchr/testify/assert"

	_ "k8s.io/kubernetes/pkg/api/install"

	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/json"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/client-go/dynamic"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/util/workqueue"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/controller/garbagecollector/metaonly"
)

func TestNewGarbageCollector(t *testing.T) {
	config := &restclient.Config{}
	config.ContentConfig.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: metaonly.NewMetadataCodecFactory()}
	metaOnlyClientPool := dynamic.NewClientPool(config, api.Registry.RESTMapper(), dynamic.LegacyAPIPathResolverFunc)
	config.ContentConfig.NegotiatedSerializer = nil
	clientPool := dynamic.NewClientPool(config, api.Registry.RESTMapper(), dynamic.LegacyAPIPathResolverFunc)
	podResource := map[schema.GroupVersionResource]struct{}{
		schema.GroupVersionResource{Version: "v1", Resource: "pods"}: {},
		// no monitor will be constructed for non-core resource, the GC construction will not fail.
		schema.GroupVersionResource{Group: "tpr.io", Version: "v1", Resource: "unknown"}: {},
	}
	gc, err := NewGarbageCollector(metaOnlyClientPool, clientPool, api.Registry.RESTMapper(), podResource)
	if err != nil {
		t.Fatal(err)
	}
	assert.Equal(t, 1, len(gc.dependencyGraphBuilder.monitors))
}

// fakeAction records information about requests to aid in testing.
type fakeAction struct {
	method string
	path   string
	query  string
}

// String returns method=path to aid in testing
func (f *fakeAction) String() string {
	return strings.Join([]string{f.method, f.path}, "=")
}

type FakeResponse struct {
	statusCode int
	content    []byte
}

// fakeActionHandler holds a list of fakeActions received
type fakeActionHandler struct {
	// statusCode and content returned by this handler for different method + path.
	response map[string]FakeResponse

	lock    sync.Mutex
	actions []fakeAction
}

// ServeHTTP logs the action that occurred and always returns the associated status code
func (f *fakeActionHandler) ServeHTTP(response http.ResponseWriter, request *http.Request) {
	f.lock.Lock()
	defer f.lock.Unlock()

	f.actions = append(f.actions, fakeAction{method: request.Method, path: request.URL.Path, query: request.URL.RawQuery})
	fakeResponse, ok := f.response[request.Method+request.URL.Path]
	if !ok {
		fakeResponse.statusCode = 200
		fakeResponse.content = []byte("{\"kind\": \"List\"}")
	}
	response.Header().Set("Content-Type", "application/json")
	response.WriteHeader(fakeResponse.statusCode)
	response.Write(fakeResponse.content)
}

// testServerAndClientConfig returns a server that listens and a config that can reference it
func testServerAndClientConfig(handler func(http.ResponseWriter, *http.Request)) (*httptest.Server, *restclient.Config) {
	srv := httptest.NewServer(http.HandlerFunc(handler))
	config := &restclient.Config{
		Host: srv.URL,
	}
	return srv, config
}

func setupGC(t *testing.T, config *restclient.Config) *GarbageCollector {
	config.ContentConfig.NegotiatedSerializer = serializer.DirectCodecFactory{CodecFactory: metaonly.NewMetadataCodecFactory()}
	metaOnlyClientPool := dynamic.NewClientPool(config, api.Registry.RESTMapper(), dynamic.LegacyAPIPathResolverFunc)
	config.ContentConfig.NegotiatedSerializer = nil
	clientPool := dynamic.NewClientPool(config, api.Registry.RESTMapper(), dynamic.LegacyAPIPathResolverFunc)
	podResource := map[schema.GroupVersionResource]struct{}{schema.GroupVersionResource{Version: "v1", Resource: "pods"}: {}}
	gc, err := NewGarbageCollector(metaOnlyClientPool, clientPool, api.Registry.RESTMapper(), podResource)
	if err != nil {
		t.Fatal(err)
	}
	return gc
}

func getPod(podName string, ownerReferences []metav1.OwnerReference) *v1.Pod {
	return &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:            podName,
			Namespace:       "ns1",
			OwnerReferences: ownerReferences,
		},
	}
}

func serilizeOrDie(t *testing.T, object interface{}) []byte {
	data, err := json.Marshal(object)
	if err != nil {
		t.Fatal(err)
	}
	return data
}

// test the attemptToDeleteItem function making the expected actions.
func TestAttemptToDeleteItem(t *testing.T) {
	pod := getPod("ToBeDeletedPod", []metav1.OwnerReference{
		{
			Kind:       "ReplicationController",
			Name:       "owner1",
			UID:        "123",
			APIVersion: "v1",
		},
	})
	testHandler := &fakeActionHandler{
		response: map[string]FakeResponse{
			"GET" + "/api/v1/namespaces/ns1/replicationcontrollers/owner1": {
				404,
				[]byte{},
			},
			"GET" + "/api/v1/namespaces/ns1/pods/ToBeDeletedPod": {
				200,
				serilizeOrDie(t, pod),
			},
		},
	}
	srv, clientConfig := testServerAndClientConfig(testHandler.ServeHTTP)
	defer srv.Close()
	gc := setupGC(t, clientConfig)
	item := &node{
		identity: objectReference{
			OwnerReference: metav1.OwnerReference{
				Kind:       pod.Kind,
				APIVersion: pod.APIVersion,
				Name:       pod.Name,
				UID:        pod.UID,
			},
			Namespace: pod.Namespace,
		},
		// owners are intentionally left empty. The attemptToDeleteItem routine should get the latest item from the server.
		owners: nil,
	}
	err := gc.attemptToDeleteItem(item)
	if err != nil {
		t.Errorf("Unexpected Error: %v", err)
	}
	expectedActionSet := sets.NewString()
	expectedActionSet.Insert("GET=/api/v1/namespaces/ns1/replicationcontrollers/owner1")
	expectedActionSet.Insert("DELETE=/api/v1/namespaces/ns1/pods/ToBeDeletedPod")
	expectedActionSet.Insert("GET=/api/v1/namespaces/ns1/pods/ToBeDeletedPod")

	actualActionSet := sets.NewString()
	for _, action := range testHandler.actions {
		actualActionSet.Insert(action.String())
	}
	if !expectedActionSet.Equal(actualActionSet) {
		t.Errorf("expected actions:\n%v\n but got:\n%v\nDifference:\n%v", expectedActionSet,
			actualActionSet, expectedActionSet.Difference(actualActionSet))
	}
}

// verifyGraphInvariants verifies that all of a node's owners list the node as a
// dependent and vice versa. uidToNode has all the nodes in the graph.
func verifyGraphInvariants(scenario string, uidToNode map[types.UID]*node, t *testing.T) {
	for myUID, node := range uidToNode {
		for dependentNode := range node.dependents {
			found := false
			for _, owner := range dependentNode.owners {
				if owner.UID == myUID {
					found = true
					break
				}
			}
			if !found {
				t.Errorf("scenario: %s: node %s has node %s as a dependent, but it's not present in the latter node's owners list", scenario, node.identity, dependentNode.identity)
			}
		}

		for _, owner := range node.owners {
			ownerNode, ok := uidToNode[owner.UID]
			if !ok {
				// It's possible that the owner node doesn't exist
				continue
			}
			if _, ok := ownerNode.dependents[node]; !ok {
				t.Errorf("node %s has node %s as an owner, but it's not present in the latter node's dependents list", node.identity, ownerNode.identity)
			}
		}
	}
}

func createEvent(eventType eventType, selfUID string, owners []string) event {
	var ownerReferences []metav1.OwnerReference
	for i := 0; i < len(owners); i++ {
		ownerReferences = append(ownerReferences, metav1.OwnerReference{UID: types.UID(owners[i])})
	}
	return event{
		eventType: eventType,
		obj: &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				UID:             types.UID(selfUID),
				OwnerReferences: ownerReferences,
			},
		},
	}
}

func TestProcessEvent(t *testing.T) {
	var testScenarios = []struct {
		name string
		// a series of events that will be supplied to the
		// GraphBuilder.eventQueue.
		events []event
	}{
		{
			name: "test1",
			events: []event{
				createEvent(addEvent, "1", []string{}),
				createEvent(addEvent, "2", []string{"1"}),
				createEvent(addEvent, "3", []string{"1", "2"}),
			},
		},
		{
			name: "test2",
			events: []event{
				createEvent(addEvent, "1", []string{}),
				createEvent(addEvent, "2", []string{"1"}),
				createEvent(addEvent, "3", []string{"1", "2"}),
				createEvent(addEvent, "4", []string{"2"}),
				createEvent(deleteEvent, "2", []string{"doesn't matter"}),
			},
		},
		{
			name: "test3",
			events: []event{
				createEvent(addEvent, "1", []string{}),
				createEvent(addEvent, "2", []string{"1"}),
				createEvent(addEvent, "3", []string{"1", "2"}),
				createEvent(addEvent, "4", []string{"3"}),
				createEvent(updateEvent, "2", []string{"4"}),
			},
		},
		{
			name: "reverse test2",
			events: []event{
				createEvent(addEvent, "4", []string{"2"}),
				createEvent(addEvent, "3", []string{"1", "2"}),
				createEvent(addEvent, "2", []string{"1"}),
				createEvent(addEvent, "1", []string{}),
				createEvent(deleteEvent, "2", []string{"doesn't matter"}),
			},
		},
	}

	for _, scenario := range testScenarios {
		dependencyGraphBuilder := &GraphBuilder{
			graphChanges: workqueue.NewRateLimitingQueue(workqueue.DefaultControllerRateLimiter()),
			uidToNode: &concurrentUIDToNode{
				uidToNodeLock: sync.RWMutex{},
				uidToNode:     make(map[types.UID]*node),
			},
			attemptToDelete:  workqueue.NewRateLimitingQueue(workqueue.DefaultControllerRateLimiter()),
			absentOwnerCache: NewUIDCache(2),
		}
		for i := 0; i < len(scenario.events); i++ {
			dependencyGraphBuilder.graphChanges.Add(&scenario.events[i])
			dependencyGraphBuilder.processGraphChanges()
			verifyGraphInvariants(scenario.name, dependencyGraphBuilder.uidToNode.uidToNode, t)
		}
	}
}

// TestDependentsRace relies on golang's data race detector to check if there is
// data race among in the dependents field.
func TestDependentsRace(t *testing.T) {
	gc := setupGC(t, &restclient.Config{})

	const updates = 100
	owner := &node{dependents: make(map[*node]struct{})}
	ownerUID := types.UID("owner")
	gc.dependencyGraphBuilder.uidToNode.Write(owner)
	go func() {
		for i := 0; i < updates; i++ {
			dependent := &node{}
			gc.dependencyGraphBuilder.addDependentToOwners(dependent, []metav1.OwnerReference{{UID: ownerUID}})
			gc.dependencyGraphBuilder.removeDependentFromOwners(dependent, []metav1.OwnerReference{{UID: ownerUID}})
		}
	}()
	go func() {
		gc.attemptToOrphan.Add(owner)
		for i := 0; i < updates; i++ {
			gc.attemptToOrphanWorker()
		}
	}()
}

// test the list and watch functions correctly converts the ListOptions
func TestGCListWatcher(t *testing.T) {
	testHandler := &fakeActionHandler{}
	srv, clientConfig := testServerAndClientConfig(testHandler.ServeHTTP)
	defer srv.Close()
	clientPool := dynamic.NewClientPool(clientConfig, api.Registry.RESTMapper(), dynamic.LegacyAPIPathResolverFunc)
	podResource := schema.GroupVersionResource{Version: "v1", Resource: "pods"}
	client, err := clientPool.ClientForGroupVersionResource(podResource)
	if err != nil {
		t.Fatal(err)
	}
	lw := listWatcher(client, podResource)
	if _, err := lw.Watch(metav1.ListOptions{ResourceVersion: "1"}); err != nil {
		t.Fatal(err)
	}
	if _, err := lw.List(metav1.ListOptions{ResourceVersion: "1"}); err != nil {
		t.Fatal(err)
	}
	if e, a := 2, len(testHandler.actions); e != a {
		t.Errorf("expect %d requests, got %d", e, a)
	}
	if e, a := "resourceVersion=1&watch=true", testHandler.actions[0].query; e != a {
		t.Errorf("expect %s, got %s", e, a)
	}
	if e, a := "resourceVersion=1", testHandler.actions[1].query; e != a {
		t.Errorf("expect %s, got %s", e, a)
	}
}

func podToGCNode(pod *v1.Pod) *node {
	return &node{
		identity: objectReference{
			OwnerReference: metav1.OwnerReference{
				Kind:       pod.Kind,
				APIVersion: pod.APIVersion,
				Name:       pod.Name,
				UID:        pod.UID,
			},
			Namespace: pod.Namespace,
		},
		// owners are intentionally left empty. The attemptToDeleteItem routine should get the latest item from the server.
		owners: nil,
	}
}

func TestAbsentUIDCache(t *testing.T) {
	rc1Pod1 := getPod("rc1Pod1", []metav1.OwnerReference{
		{
			Kind:       "ReplicationController",
			Name:       "rc1",
			UID:        "1",
			APIVersion: "v1",
		},
	})
	rc1Pod2 := getPod("rc1Pod2", []metav1.OwnerReference{
		{
			Kind:       "ReplicationController",
			Name:       "rc1",
			UID:        "1",
			APIVersion: "v1",
		},
	})
	rc2Pod1 := getPod("rc2Pod1", []metav1.OwnerReference{
		{
			Kind:       "ReplicationController",
			Name:       "rc2",
			UID:        "2",
			APIVersion: "v1",
		},
	})
	rc3Pod1 := getPod("rc3Pod1", []metav1.OwnerReference{
		{
			Kind:       "ReplicationController",
			Name:       "rc3",
			UID:        "3",
			APIVersion: "v1",
		},
	})
	testHandler := &fakeActionHandler{
		response: map[string]FakeResponse{
			"GET" + "/api/v1/namespaces/ns1/pods/rc1Pod1": {
				200,
				serilizeOrDie(t, rc1Pod1),
			},
			"GET" + "/api/v1/namespaces/ns1/pods/rc1Pod2": {
				200,
				serilizeOrDie(t, rc1Pod2),
			},
			"GET" + "/api/v1/namespaces/ns1/pods/rc2Pod1": {
				200,
				serilizeOrDie(t, rc2Pod1),
			},
			"GET" + "/api/v1/namespaces/ns1/pods/rc3Pod1": {
				200,
				serilizeOrDie(t, rc3Pod1),
			},
			"GET" + "/api/v1/namespaces/ns1/replicationcontrollers/rc1": {
				404,
				[]byte{},
			},
			"GET" + "/api/v1/namespaces/ns1/replicationcontrollers/rc2": {
				404,
				[]byte{},
			},
			"GET" + "/api/v1/namespaces/ns1/replicationcontrollers/rc3": {
				404,
				[]byte{},
			},
		},
	}
	srv, clientConfig := testServerAndClientConfig(testHandler.ServeHTTP)
	defer srv.Close()
	gc := setupGC(t, clientConfig)
	gc.absentOwnerCache = NewUIDCache(2)
	gc.attemptToDeleteItem(podToGCNode(rc1Pod1))
	gc.attemptToDeleteItem(podToGCNode(rc2Pod1))
	// rc1 should already be in the cache, no request should be sent. rc1 should be promoted in the UIDCache
	gc.attemptToDeleteItem(podToGCNode(rc1Pod2))
	// after this call, rc2 should be evicted from the UIDCache
	gc.attemptToDeleteItem(podToGCNode(rc3Pod1))
	// check cache
	if !gc.absentOwnerCache.Has(types.UID("1")) {
		t.Errorf("expected rc1 to be in the cache")
	}
	if gc.absentOwnerCache.Has(types.UID("2")) {
		t.Errorf("expected rc2 to not exist in the cache")
	}
	if !gc.absentOwnerCache.Has(types.UID("3")) {
		t.Errorf("expected rc3 to be in the cache")
	}
	// check the request sent to the server
	count := 0
	for _, action := range testHandler.actions {
		if action.String() == "GET=/api/v1/namespaces/ns1/replicationcontrollers/rc1" {
			count++
		}
	}
	if count != 1 {
		t.Errorf("expected only 1 GET rc1 request, got %d", count)
	}
}

func TestDeleteOwnerRefPatch(t *testing.T) {
	original := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "100",
			OwnerReferences: []metav1.OwnerReference{
				{UID: "1"},
				{UID: "2"},
				{UID: "3"},
			},
		},
	}
	originalData := serilizeOrDie(t, original)
	expected := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "100",
			OwnerReferences: []metav1.OwnerReference{
				{UID: "1"},
			},
		},
	}
	patch := deleteOwnerRefPatch("100", "2", "3")
	patched, err := strategicpatch.StrategicMergePatch(originalData, patch, v1.Pod{})
	if err != nil {
		t.Fatal(err)
	}
	var got v1.Pod
	if err := json.Unmarshal(patched, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(expected, got) {
		t.Errorf("expected: %#v,\ngot: %#v", expected, got)
	}
}

func TestUnblockOwnerReference(t *testing.T) {
	trueVar := true
	falseVar := false
	original := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "100",
			OwnerReferences: []metav1.OwnerReference{
				{UID: "1", BlockOwnerDeletion: &trueVar},
				{UID: "2", BlockOwnerDeletion: &falseVar},
				{UID: "3"},
			},
		},
	}
	originalData := serilizeOrDie(t, original)
	expected := v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			UID: "100",
			OwnerReferences: []metav1.OwnerReference{
				{UID: "1", BlockOwnerDeletion: &falseVar},
				{UID: "2", BlockOwnerDeletion: &falseVar},
				{UID: "3"},
			},
		},
	}
	accessor, err := meta.Accessor(&original)
	if err != nil {
		t.Fatal(err)
	}
	n := node{
		owners: accessor.GetOwnerReferences(),
	}
	patch, err := n.patchToUnblockOwnerReferences()
	if err != nil {
		t.Fatal(err)
	}
	patched, err := strategicpatch.StrategicMergePatch(originalData, patch, v1.Pod{})
	if err != nil {
		t.Fatal(err)
	}
	var got v1.Pod
	if err := json.Unmarshal(patched, &got); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(expected, got) {
		t.Errorf("expected: %#v,\ngot: %#v", expected, got)
		t.Errorf("expected: %#v,\ngot: %#v", expected.OwnerReferences, got.OwnerReferences)
		for _, ref := range got.OwnerReferences {
			t.Errorf("ref.UID=%s, ref.BlockOwnerDeletion=%v", ref.UID, *ref.BlockOwnerDeletion)
		}
	}
}
