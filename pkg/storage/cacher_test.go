/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package storage_test

import (
	"fmt"
	"reflect"
	"testing"
	"time"

	"github.com/coreos/go-etcd/etcd"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/wait"
	"k8s.io/kubernetes/pkg/watch"
)

func newTestCacher(client tools.EtcdClient) *storage.Cacher {
	prefix := "pods"
	config := storage.CacherConfig{
		CacheCapacity:  10,
		Versioner:      etcdstorage.APIObjectVersioner{},
		Storage:        etcdstorage.NewEtcdStorage(client, testapi.Default.Codec(), etcdtest.PathPrefix()),
		Type:           &api.Pod{},
		ResourcePrefix: prefix,
		KeyFunc:        func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc(prefix, obj) },
		NewListFunc:    func() runtime.Object { return &api.PodList{} },
		StopChannel:    util.NeverStop,
	}
	return storage.NewCacher(config)
}

func makeTestPod(name string) *api.Pod {
	return &api.Pod{
		ObjectMeta: api.ObjectMeta{Namespace: "ns", Name: name},
		Spec:       apitesting.DeepEqualSafePodSpec(),
	}
}

func waitForUpToDateCache(cacher *storage.Cacher, resourceVersion uint64) error {
	ready := func() (bool, error) {
		result, err := cacher.LastSyncResourceVersion()
		if err != nil {
			return false, err
		}
		return result == resourceVersion, nil
	}
	return wait.Poll(10*time.Millisecond, util.ForeverTestTimeout, ready)
}

func TestListFromMemory(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	prefixedKey := etcdtest.AddPrefix("pods")
	fakeClient.ExpectNotFoundGet(prefixedKey)
	cacher := newTestCacher(fakeClient)
	fakeClient.WaitForWatchCompletion()

	podFoo := makeTestPod("foo")
	podBar := makeTestPod("bar")
	podBaz := makeTestPod("baz")

	podFooPrime := makeTestPod("foo")
	podFooPrime.Spec.NodeName = "fakeNode"

	testCases := []*etcd.Response{
		{
			Action: "create",
			Node: &etcd.Node{
				Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
				CreatedIndex:  1,
				ModifiedIndex: 1,
			},
		},
		{
			Action: "create",
			Node: &etcd.Node{
				Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podBar)),
				CreatedIndex:  2,
				ModifiedIndex: 2,
			},
		},
		{
			Action: "create",
			Node: &etcd.Node{
				Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podBaz)),
				CreatedIndex:  3,
				ModifiedIndex: 3,
			},
		},
		{
			Action: "set",
			Node: &etcd.Node{
				Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFooPrime)),
				CreatedIndex:  1,
				ModifiedIndex: 4,
			},
			PrevNode: &etcd.Node{
				Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
				CreatedIndex:  1,
				ModifiedIndex: 1,
			},
		},
		{
			Action: "delete",
			Node: &etcd.Node{
				CreatedIndex:  1,
				ModifiedIndex: 5,
			},
			PrevNode: &etcd.Node{
				Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podBar)),
				CreatedIndex:  1,
				ModifiedIndex: 1,
			},
		},
	}

	// Propagate some data to etcd.
	for _, test := range testCases {
		fakeClient.WatchResponse <- test
	}
	if err := waitForUpToDateCache(cacher, 5); err != nil {
		t.Errorf("watch cache didn't propagated correctly: %v", err)
	}

	result := &api.PodList{}
	if err := cacher.ListFromMemory("pods/ns", result); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	if result.ListMeta.ResourceVersion != "5" {
		t.Errorf("incorrect resource version: %v", result.ListMeta.ResourceVersion)
	}
	if len(result.Items) != 2 {
		t.Errorf("unexpected list result: %d", len(result.Items))
	}
	keys := sets.String{}
	for _, item := range result.Items {
		keys.Insert(item.ObjectMeta.Name)
	}
	if !keys.HasAll("foo", "baz") {
		t.Errorf("unexpected list result: %#v", result)
	}
	for _, item := range result.Items {
		// unset fields that are set by the infrastructure
		item.ObjectMeta.ResourceVersion = ""
		item.ObjectMeta.CreationTimestamp = unversioned.Time{}

		var expected *api.Pod
		switch item.ObjectMeta.Name {
		case "foo":
			expected = podFooPrime
		case "baz":
			expected = podBaz
		default:
			t.Errorf("unexpected item: %v", item)
		}
		if e, a := *expected, item; !reflect.DeepEqual(e, a) {
			t.Errorf("expected: %#v, got: %#v", e, a)
		}
	}

	close(fakeClient.WatchResponse)
}

func TestWatch(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	prefixedKey := etcdtest.AddPrefix("pods")
	fakeClient.ExpectNotFoundGet(prefixedKey)
	cacher := newTestCacher(fakeClient)
	fakeClient.WaitForWatchCompletion()

	podFoo := makeTestPod("foo")
	podBar := makeTestPod("bar")

	testCases := []struct {
		object       *api.Pod
		etcdResponse *etcd.Response
		event        watch.EventType
		filtered     bool
	}{
		{
			object: podFoo,
			etcdResponse: &etcd.Response{
				Action: "create",
				Node: &etcd.Node{
					Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
					CreatedIndex:  2,
					ModifiedIndex: 2,
				},
			},
			event:    watch.Added,
			filtered: true,
		},
		{
			object: podBar,
			etcdResponse: &etcd.Response{
				Action: "create",
				Node: &etcd.Node{
					Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podBar)),
					CreatedIndex:  3,
					ModifiedIndex: 3,
				},
			},
			event:    watch.Added,
			filtered: false,
		},
		{
			object: podFoo,
			etcdResponse: &etcd.Response{
				Action: "set",
				Node: &etcd.Node{
					Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
					CreatedIndex:  2,
					ModifiedIndex: 4,
				},
				PrevNode: &etcd.Node{
					Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
					CreatedIndex:  2,
					ModifiedIndex: 2,
				},
			},
			event:    watch.Modified,
			filtered: true,
		},
	}

	// Set up Watch for object "podFoo".
	watcher, err := cacher.Watch("pods/ns/foo", 2, storage.Everything)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, test := range testCases {
		fakeClient.WatchResponse <- test.etcdResponse
		if test.filtered {
			event := <-watcher.ResultChan()
			if e, a := test.event, event.Type; e != a {
				t.Errorf("%v %v", e, a)
			}
			// unset fields that are set by the infrastructure
			obj := event.Object.(*api.Pod)
			obj.ObjectMeta.ResourceVersion = ""
			obj.ObjectMeta.CreationTimestamp = unversioned.Time{}
			if e, a := test.object, obj; !reflect.DeepEqual(e, a) {
				t.Errorf("expected: %#v, got: %#v", e, a)
			}
		}
	}

	// Check whether we get too-old error.
	_, err = cacher.Watch("pods/ns/foo", 1, storage.Everything)
	if err == nil {
		t.Errorf("exepcted 'error too old' error")
	}

	// Now test watch with initial state.
	initialWatcher, err := cacher.Watch("pods/ns/foo", 2, storage.Everything)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	for _, test := range testCases {
		if test.filtered {
			event := <-initialWatcher.ResultChan()
			if e, a := test.event, event.Type; e != a {
				t.Errorf("%v %v", e, a)
			}
			// unset fields that are set by the infrastructure
			obj := event.Object.(*api.Pod)
			obj.ObjectMeta.ResourceVersion = ""
			obj.ObjectMeta.CreationTimestamp = unversioned.Time{}
			if e, a := test.object, obj; !reflect.DeepEqual(e, a) {
				t.Errorf("expected: %#v, got: %#v", e, a)
			}
		}
	}

	// Now test watch from "now".
	nowWatcher, err := cacher.Watch("pods/ns/foo", 0, storage.Everything)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	select {
	case event := <-nowWatcher.ResultChan():
		if obj := event.Object.(*api.Pod); event.Type != watch.Added || obj.ResourceVersion != "4" {
			t.Errorf("unexpected event: %v", event)
		}
	case <-time.After(util.ForeverTestTimeout):
		t.Errorf("timed out waiting for an event")
	}
	// Emit a new event and check if it is observed by the watcher.
	fakeClient.WatchResponse <- &etcd.Response{
		Action: "set",
		Node: &etcd.Node{
			Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
			CreatedIndex:  2,
			ModifiedIndex: 5,
		},
		PrevNode: &etcd.Node{
			Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
			CreatedIndex:  2,
			ModifiedIndex: 4,
		},
	}
	event := <-nowWatcher.ResultChan()
	obj := event.Object.(*api.Pod)
	if event.Type != watch.Modified || obj.ResourceVersion != "5" {
		t.Errorf("unexpected event: %v", event)
	}

	close(fakeClient.WatchResponse)
}

func TestFiltering(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	prefixedKey := etcdtest.AddPrefix("pods")
	fakeClient.ExpectNotFoundGet(prefixedKey)
	cacher := newTestCacher(fakeClient)
	fakeClient.WaitForWatchCompletion()

	podFoo := makeTestPod("foo")
	podFoo.ObjectMeta.Labels = map[string]string{"filter": "foo"}
	podFooFiltered := makeTestPod("foo")

	testCases := []struct {
		object       *api.Pod
		etcdResponse *etcd.Response
		filtered     bool
		event        watch.EventType
	}{
		{
			object: podFoo,
			etcdResponse: &etcd.Response{
				Action: "create",
				Node: &etcd.Node{
					Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
					CreatedIndex:  1,
					ModifiedIndex: 1,
				},
			},
			filtered: true,
			event:    watch.Added,
		},
		{
			object: podFooFiltered,
			etcdResponse: &etcd.Response{
				Action: "set",
				Node: &etcd.Node{
					Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFooFiltered)),
					CreatedIndex:  1,
					ModifiedIndex: 2,
				},
				PrevNode: &etcd.Node{
					Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
					CreatedIndex:  1,
					ModifiedIndex: 1,
				},
			},
			filtered: true,
			// Deleted, because the new object doesn't match filter.
			event: watch.Deleted,
		},
		{
			object: podFoo,
			etcdResponse: &etcd.Response{
				Action: "set",
				Node: &etcd.Node{
					Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
					CreatedIndex:  1,
					ModifiedIndex: 3,
				},
				PrevNode: &etcd.Node{
					Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFooFiltered)),
					CreatedIndex:  1,
					ModifiedIndex: 2,
				},
			},
			filtered: true,
			// Added, because the previous object didn't match filter.
			event: watch.Added,
		},
		{
			object: podFoo,
			etcdResponse: &etcd.Response{
				Action: "set",
				Node: &etcd.Node{
					Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
					CreatedIndex:  1,
					ModifiedIndex: 4,
				},
				PrevNode: &etcd.Node{
					Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
					CreatedIndex:  1,
					ModifiedIndex: 3,
				},
			},
			filtered: true,
			event:    watch.Modified,
		},
		{
			object: podFoo,
			etcdResponse: &etcd.Response{
				Action: "delete",
				Node: &etcd.Node{
					CreatedIndex:  1,
					ModifiedIndex: 5,
				},
				PrevNode: &etcd.Node{
					Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
					CreatedIndex:  1,
					ModifiedIndex: 4,
				},
			},
			filtered: true,
			event:    watch.Deleted,
		},
	}

	// Set up Watch for object "podFoo" with label filter set.
	selector := labels.SelectorFromSet(labels.Set{"filter": "foo"})
	filter := func(obj runtime.Object) bool {
		metadata, err := meta.Accessor(obj)
		if err != nil {
			t.Errorf("unexpected error: %v", err)
			return false
		}
		return selector.Matches(labels.Set(metadata.Labels()))
	}
	watcher, err := cacher.Watch("pods/ns/foo", 1, filter)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	for _, test := range testCases {
		fakeClient.WatchResponse <- test.etcdResponse
		if test.filtered {
			event := <-watcher.ResultChan()
			if e, a := test.event, event.Type; e != a {
				t.Errorf("%v %v", e, a)
			}
			// unset fields that are set by the infrastructure
			obj := event.Object.(*api.Pod)
			obj.ObjectMeta.ResourceVersion = ""
			obj.ObjectMeta.CreationTimestamp = unversioned.Time{}
			if e, a := test.object, obj; !reflect.DeepEqual(e, a) {
				t.Errorf("expected: %#v, got: %#v", e, a)
			}
		}
	}

	close(fakeClient.WatchResponse)
}

func TestStorageError(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	prefixedKey := etcdtest.AddPrefix("pods")
	fakeClient.ExpectNotFoundGet(prefixedKey)
	cacher := newTestCacher(fakeClient)
	fakeClient.WaitForWatchCompletion()

	podFoo := makeTestPod("foo")

	// Set up Watch for object "podFoo".
	watcher, err := cacher.Watch("pods/ns/foo", 1, storage.Everything)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	fakeClient.WatchResponse <- &etcd.Response{
		Action: "create",
		Node: &etcd.Node{
			Value:         string(runtime.EncodeOrDie(testapi.Default.Codec(), podFoo)),
			CreatedIndex:  1,
			ModifiedIndex: 1,
		},
	}
	_ = <-watcher.ResultChan()

	// Injecting error is simulating error from etcd.
	// This is almost the same what would happen e.g. in case of
	// "error too old" when reconnecting to etcd watch.
	fakeClient.WatchInjectError <- fmt.Errorf("fake error")

	_, ok := <-watcher.ResultChan()
	if ok {
		t.Errorf("unexpected event")
	}
}
