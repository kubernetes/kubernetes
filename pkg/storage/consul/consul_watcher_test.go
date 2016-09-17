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

package consul

import (
	"strconv"
	"testing"
	"time"

	"golang.org/x/net/context"

	"k8s.io/kubernetes/pkg/storage/consul/consultest"
	consultesting "k8s.io/kubernetes/pkg/storage/consul/testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/watch"

	consulapi "github.com/hashicorp/consul/api"
)

func newConsulHelper(client consulapi.Client, codec runtime.Codec, prefix string, quorum bool, config consulapi.Config) consulHelper {
	return *NewConsulStorage(client, codec, prefix, quorum, config).(*consulHelper)
}

func TestOldWatcherEvents(t *testing.T) {
	codec := testapi.Default.Codec()
	server := consultesting.NewConsulTestClientServer(t)
	defer server.Terminate(t)
	key := "/some/key"
	h := newConsulHelper(*server.Client, codec, consultest.PathPrefix(), true, *server.ClientConfig)

	// Test normal case
	pod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	returnObj := &api.Pod{}
	err := h.Create(context.TODO(), key, pod, returnObj, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	addedPod := &api.Pod{}
	err = h.Get(context.TODO(), key, addedPod, false)
	if err != nil {
		t.Fatalf("Failed to load pod")
	}

	// Update an existing node. #1
	callbackCalled := false
	objUpdate := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "bar"}}
	err = h.GuaranteedUpdate(context.TODO(), key, returnObj, true, nil, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		callbackCalled = true

		if in.(*api.Pod).Name != "foo" {
			t.Errorf("Callback input was not current set value")
		}

		return objUpdate, nil
	}))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	//Get the resource version from where we later want to start watching
	objAfterFirstUpdate := &api.Pod{}
	err = h.Get(context.TODO(), key, objAfterFirstUpdate, false)
	if err != nil {
		t.Errorf("Failed to load object: %+v", err)
	}

	// Update an existing node. #2
	callbackCalled = false
	objUpdate = &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foobar"}}
	err = h.GuaranteedUpdate(context.TODO(), key, returnObj, true, nil, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		callbackCalled = true

		if in.(*api.Pod).Name != "bar" {
			t.Errorf("Callback input was not current set value")
		}

		return objUpdate, nil
	}))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	// Update an existing node. #3
	callbackCalled = false
	objUpdate = &api.Pod{ObjectMeta: api.ObjectMeta{Name: "barfoo"}}
	err = h.GuaranteedUpdate(context.TODO(), key, returnObj, true, nil, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		callbackCalled = true

		if in.(*api.Pod).Name != "foobar" {
			t.Errorf("Callback input was not current set value")
		}

		return objUpdate, nil
	}))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	watching, err := h.Watch(context.TODO(), key, objAfterFirstUpdate.ObjectMeta.ResourceVersion, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	// watching is explicitly closed below.

	var lastVersion int
	var currentVersion int

	lastVersion, err = strconv.Atoi(addedPod.ObjectMeta.ResourceVersion)
	if err != nil {
		t.Errorf("Failed to parse ResourceVersion: %+v", err)
	}

	// Delete the node
	lastObj := &api.Pod{}
	err = h.Delete(context.TODO(), key, lastObj, nil)

	// Recreate the node
	recreatedPod := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "potato"}}
	returnObj = &api.Pod{}
	err = h.Create(context.TODO(), key, recreatedPod, returnObj, 0)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	readdedPod := &api.Pod{}
	err = h.Get(context.TODO(), key, readdedPod, false)
	if err != nil {
		t.Fatalf("Failed to load pod")
	}

	if readdedPod.Name != recreatedPod.Name {
		t.Errorf("the retrieved pod shoulf have the same name as the created one")
	}

	//wait a bit to get events processed
	time.Sleep(time.Duration(5) * time.Second)

	//we start watching after the first update and so expect events for two more updates
	event := <-watching.ResultChan()
	if e, a := watch.Modified, event.Type; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	currentVersion, err = strconv.Atoi(event.Object.(*api.Pod).ObjectMeta.ResourceVersion)
	if err != nil {
		t.Errorf("Failed to parse ResourceVersion: %+v of %+v", err, *event.Object.(*api.Pod))
	}
	if currentVersion <= lastVersion {
		t.Errorf("RessourceVersion should have increased after an object update")
	}
	lastVersion = currentVersion

	event = <-watching.ResultChan()
	if e, a := watch.Modified, event.Type; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	currentVersion, err = strconv.Atoi(event.Object.(*api.Pod).ObjectMeta.ResourceVersion)
	if err != nil {
		t.Errorf("Failed to parse ResourceVersion: %+v", err)
	}
	if currentVersion <= lastVersion {
		t.Errorf("RessourceVersion should have increased after an object update: %d - %d", currentVersion, lastVersion)
	}
	lastVersion = currentVersion

	event = <-watching.ResultChan()
	if e, a := watch.Deleted, event.Type; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	currentVersion, err = strconv.Atoi(event.Object.(*api.Pod).ObjectMeta.ResourceVersion)
	if err != nil {
		t.Errorf("Failed to parse ResourceVersion: %+v", err)
	}
	if currentVersion != lastVersion {
		t.Errorf("RessourceVersion should NOT have increased after an object deletion: %d - %d", currentVersion, lastVersion)
	}
	lastVersion = currentVersion

	event = <-watching.ResultChan()
	if e, a := watch.Added, event.Type; e != a {
		t.Errorf("Expected %v, got %v", e, a)
	}
	currentVersion, err = strconv.Atoi(event.Object.(*api.Pod).ObjectMeta.ResourceVersion)
	if err != nil {
		t.Errorf("Failed to parse ResourceVersion: %+v", err)
	}
	if currentVersion <= lastVersion {
		t.Errorf("RessourceVersion should have increased after an object creation: %d - %d", currentVersion, lastVersion)
	}
	lastVersion = currentVersion

	//event channel should be empty now
	select {
	case event, _ := <-watching.ResultChan():
		t.Fatalf("Unexpected event: %+v", *event.Object.(*api.Pod))
	default:
		// fall through, expected behavior
	}

	watching.Stop()
}
