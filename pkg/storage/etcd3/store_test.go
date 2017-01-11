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

package etcd3

import (
	"fmt"
	"reflect"
	"sync"
	"testing"

	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/storage"

	"github.com/coreos/etcd/integration"
	"golang.org/x/net/context"
	"k8s.io/apimachinery/pkg/watch"
)

func TestCreate(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	etcdClient := cluster.RandClient()

	key := "/testkey"
	out := &api.Pod{}
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}

	// verify that kv pair is empty before set
	getResp, err := etcdClient.KV.Get(ctx, key)
	if err != nil {
		t.Fatalf("etcdClient.KV.Get failed: %v", err)
	}
	if len(getResp.Kvs) != 0 {
		t.Fatalf("expecting empty result on key: %s", key)
	}

	err = store.Create(ctx, key, obj, out, 0)
	if err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	// basic tests of the output
	if obj.ObjectMeta.Name != out.ObjectMeta.Name {
		t.Errorf("pod name want=%s, get=%s", obj.ObjectMeta.Name, out.ObjectMeta.Name)
	}
	if out.ResourceVersion == "" {
		t.Errorf("output should have non-empty resource version")
	}

	// verify that kv pair is not empty after set
	getResp, err = etcdClient.KV.Get(ctx, key)
	if err != nil {
		t.Fatalf("etcdClient.KV.Get failed: %v", err)
	}
	if len(getResp.Kvs) == 0 {
		t.Fatalf("expecting non empty result on key: %s", key)
	}
}

func TestCreateWithTTL(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)

	input := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	key := "/somekey"

	out := &api.Pod{}
	if err := store.Create(ctx, key, input, out, 1); err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	w, err := store.Watch(ctx, key, out.ResourceVersion, storage.Everything)
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	testCheckEventType(t, watch.Deleted, w)
}

func TestCreateWithKeyExist(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	key, _ := testPropogateStore(ctx, t, store, obj)
	out := &api.Pod{}
	err := store.Create(ctx, key, obj, out, 0)
	if err == nil || !storage.IsNodeExist(err) {
		t.Errorf("expecting key exists error, but get: %s", err)
	}
}

func TestGet(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	key, storedObj := testPropogateStore(ctx, t, store, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}})

	tests := []struct {
		key               string
		ignoreNotFound    bool
		expectNotFoundErr bool
		expectedOut       *api.Pod
	}{{ // test get on existing item
		key:               key,
		ignoreNotFound:    false,
		expectNotFoundErr: false,
		expectedOut:       storedObj,
	}, { // test get on non-existing item with ignoreNotFound=false
		key:               "/non-existing",
		ignoreNotFound:    false,
		expectNotFoundErr: true,
	}, { // test get on non-existing item with ignoreNotFound=true
		key:               "/non-existing",
		ignoreNotFound:    true,
		expectNotFoundErr: false,
		expectedOut:       &api.Pod{},
	}}

	for i, tt := range tests {
		out := &api.Pod{}
		err := store.Get(ctx, tt.key, "", out, tt.ignoreNotFound)
		if tt.expectNotFoundErr {
			if err == nil || !storage.IsNotFound(err) {
				t.Errorf("#%d: expecting not found error, but get: %s", i, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("Get failed: %v", err)
		}
		if !reflect.DeepEqual(tt.expectedOut, out) {
			t.Errorf("#%d: pod want=%#v, get=%#v", i, tt.expectedOut, out)
		}
	}
}

func TestUnconditionalDelete(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	key, storedObj := testPropogateStore(ctx, t, store, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}})

	tests := []struct {
		key               string
		expectedObj       *api.Pod
		expectNotFoundErr bool
	}{{ // test unconditional delete on existing key
		key:               key,
		expectedObj:       storedObj,
		expectNotFoundErr: false,
	}, { // test unconditional delete on non-existing key
		key:               "/non-existing",
		expectedObj:       nil,
		expectNotFoundErr: true,
	}}

	for i, tt := range tests {
		out := &api.Pod{} // reset
		err := store.Delete(ctx, tt.key, out, nil)
		if tt.expectNotFoundErr {
			if err == nil || !storage.IsNotFound(err) {
				t.Errorf("#%d: expecting not found error, but get: %s", i, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("Delete failed: %v", err)
		}
		if !reflect.DeepEqual(tt.expectedObj, out) {
			t.Errorf("#%d: pod want=%#v, get=%#v", i, tt.expectedObj, out)
		}
	}
}

func TestConditionalDelete(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	key, storedObj := testPropogateStore(ctx, t, store, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo", UID: "A"}})

	tests := []struct {
		precondition        *storage.Preconditions
		expectInvalidObjErr bool
	}{{ // test conditional delete with UID match
		precondition:        storage.NewUIDPreconditions("A"),
		expectInvalidObjErr: false,
	}, { // test conditional delete with UID mismatch
		precondition:        storage.NewUIDPreconditions("B"),
		expectInvalidObjErr: true,
	}}

	for i, tt := range tests {
		out := &api.Pod{}
		err := store.Delete(ctx, key, out, tt.precondition)
		if tt.expectInvalidObjErr {
			if err == nil || !storage.IsInvalidObj(err) {
				t.Errorf("#%d: expecting invalid UID error, but get: %s", i, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("Delete failed: %v", err)
		}
		if !reflect.DeepEqual(storedObj, out) {
			t.Errorf("#%d: pod want=%#v, get=%#v", i, storedObj, out)
		}
		key, storedObj = testPropogateStore(ctx, t, store, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo", UID: "A"}})
	}
}

func TestGetToList(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	key, storedObj := testPropogateStore(ctx, t, store, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}})

	tests := []struct {
		key         string
		pred        storage.SelectionPredicate
		expectedOut []*api.Pod
	}{{ // test GetToList on existing key
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*api.Pod{storedObj},
	}, { // test GetToList on non-existing key
		key:         "/non-existing",
		pred:        storage.Everything,
		expectedOut: nil,
	}, { // test GetToList with matching pod name
		key: "/non-existing",
		pred: storage.SelectionPredicate{
			Label: labels.Everything(),
			Field: fields.ParseSelectorOrDie("metadata.name!=" + storedObj.Name),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*api.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, nil
			},
		},
		expectedOut: nil,
	}}

	for i, tt := range tests {
		out := &api.PodList{}
		err := store.GetToList(ctx, tt.key, "", tt.pred, out)
		if err != nil {
			t.Fatalf("GetToList failed: %v", err)
		}
		if len(out.Items) != len(tt.expectedOut) {
			t.Errorf("#%d: length of list want=%d, get=%d", i, len(tt.expectedOut), len(out.Items))
			continue
		}
		for j, wantPod := range tt.expectedOut {
			getPod := &out.Items[j]
			if !reflect.DeepEqual(wantPod, getPod) {
				t.Errorf("#%d: pod want=%#v, get=%#v", i, wantPod, getPod)
			}
		}
	}
}

func TestGuaranteedUpdate(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	key, storeObj := testPropogateStore(ctx, t, store, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo", UID: "A"}})

	tests := []struct {
		key                 string
		ignoreNotFound      bool
		precondition        *storage.Preconditions
		expectNotFoundErr   bool
		expectInvalidObjErr bool
		expectNoUpdate      bool
	}{{ // GuaranteedUpdate on non-existing key with ignoreNotFound=false
		key:                 "/non-existing",
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   true,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
	}, { // GuaranteedUpdate on non-existing key with ignoreNotFound=true
		key:                 "/non-existing",
		ignoreNotFound:      true,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
	}, { // GuaranteedUpdate on existing key
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      false,
	}, { // GuaranteedUpdate with same data
		key:                 key,
		ignoreNotFound:      false,
		precondition:        nil,
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      true,
	}, { // GuaranteedUpdate with UID match
		key:                 key,
		ignoreNotFound:      false,
		precondition:        storage.NewUIDPreconditions("A"),
		expectNotFoundErr:   false,
		expectInvalidObjErr: false,
		expectNoUpdate:      true,
	}, { // GuaranteedUpdate with UID mismatch
		key:                 key,
		ignoreNotFound:      false,
		precondition:        storage.NewUIDPreconditions("B"),
		expectNotFoundErr:   false,
		expectInvalidObjErr: true,
		expectNoUpdate:      true,
	}}

	for i, tt := range tests {
		out := &api.Pod{}
		name := fmt.Sprintf("foo-%d", i)
		if tt.expectNoUpdate {
			name = storeObj.Name
		}
		version := storeObj.ResourceVersion
		err := store.GuaranteedUpdate(ctx, tt.key, out, tt.ignoreNotFound, tt.precondition,
			storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
				if tt.expectNotFoundErr && tt.ignoreNotFound {
					if pod := obj.(*api.Pod); pod.Name != "" {
						t.Errorf("#%d: expecting zero value, but get=%#v", i, pod)
					}
				}
				pod := *storeObj
				pod.Name = name
				return &pod, nil
			}))

		if tt.expectNotFoundErr {
			if err == nil || !storage.IsNotFound(err) {
				t.Errorf("#%d: expecting not found error, but get: %v", i, err)
			}
			continue
		}
		if tt.expectInvalidObjErr {
			if err == nil || !storage.IsInvalidObj(err) {
				t.Errorf("#%d: expecting invalid UID error, but get: %s", i, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("GuaranteedUpdate failed: %v", err)
		}
		if out.ObjectMeta.Name != name {
			t.Errorf("#%d: pod name want=%s, get=%s", i, name, out.ObjectMeta.Name)
		}
		switch tt.expectNoUpdate {
		case true:
			if version != out.ResourceVersion {
				t.Errorf("#%d: expect no version change, before=%s, after=%s", i, version, out.ResourceVersion)
			}
		case false:
			if version == out.ResourceVersion {
				t.Errorf("#%d: expect version change, but get the same version=%s", i, version)
			}
		}
		storeObj = out
	}
}

func TestGuaranteedUpdateWithTTL(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)

	input := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	key := "/somekey"

	out := &api.Pod{}
	err := store.GuaranteedUpdate(ctx, key, out, true, nil,
		func(_ runtime.Object, _ storage.ResponseMeta) (runtime.Object, *uint64, error) {
			ttl := uint64(1)
			return input, &ttl, nil
		})
	if err != nil {
		t.Fatalf("Create failed: %v", err)
	}

	w, err := store.Watch(ctx, key, out.ResourceVersion, storage.Everything)
	if err != nil {
		t.Fatalf("Watch failed: %v", err)
	}
	testCheckEventType(t, watch.Deleted, w)
}

func TestGuaranteedUpdateWithConflict(t *testing.T) {
	ctx, store, cluster := testSetup(t)
	defer cluster.Terminate(t)
	key, _ := testPropogateStore(ctx, t, store, &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}})

	errChan := make(chan error, 1)
	var firstToFinish sync.WaitGroup
	var secondToEnter sync.WaitGroup
	firstToFinish.Add(1)
	secondToEnter.Add(1)

	go func() {
		err := store.GuaranteedUpdate(ctx, key, &api.Pod{}, false, nil,
			storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
				pod := obj.(*api.Pod)
				pod.Name = "foo-1"
				secondToEnter.Wait()
				return pod, nil
			}))
		firstToFinish.Done()
		errChan <- err
	}()

	updateCount := 0
	err := store.GuaranteedUpdate(ctx, key, &api.Pod{}, false, nil,
		storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
			if updateCount == 0 {
				secondToEnter.Done()
				firstToFinish.Wait()
			}
			updateCount++
			pod := obj.(*api.Pod)
			pod.Name = "foo-2"
			return pod, nil
		}))
	if err != nil {
		t.Fatalf("Second GuaranteedUpdate error %#v", err)
	}
	if err := <-errChan; err != nil {
		t.Fatalf("First GuaranteedUpdate error %#v", err)
	}

	if updateCount != 2 {
		t.Errorf("Should have conflict and called update func twice")
	}
}

func TestList(t *testing.T) {
	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	defer cluster.Terminate(t)
	store := newStore(cluster.RandClient(), false, testapi.Default.Codec(), "")
	ctx := context.Background()

	// Setup storage with the following structure:
	//  /
	//   - one-level/
	//  |            - test
	//  |
	//   - two-level/
	//               - 1/
	//              |   - test
	//              |
	//               - 2/
	//                  - test
	preset := []struct {
		key       string
		obj       *api.Pod
		storedObj *api.Pod
	}{{
		key: "/one-level/test",
		obj: &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}},
	}, {
		key: "/two-level/1/test",
		obj: &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}},
	}, {
		key: "/two-level/2/test",
		obj: &api.Pod{ObjectMeta: api.ObjectMeta{Name: "bar"}},
	}}

	for i, ps := range preset {
		preset[i].storedObj = &api.Pod{}
		err := store.Create(ctx, ps.key, ps.obj, preset[i].storedObj, 0)
		if err != nil {
			t.Fatalf("Set failed: %v", err)
		}
	}

	tests := []struct {
		prefix      string
		pred        storage.SelectionPredicate
		expectedOut []*api.Pod
	}{{ // test List on existing key
		prefix:      "/one-level/",
		pred:        storage.Everything,
		expectedOut: []*api.Pod{preset[0].storedObj},
	}, { // test List on non-existing key
		prefix:      "/non-existing/",
		pred:        storage.Everything,
		expectedOut: nil,
	}, { // test List with pod name matching
		prefix: "/one-level/",
		pred: storage.SelectionPredicate{
			Label: labels.Everything(),
			Field: fields.ParseSelectorOrDie("metadata.name!=" + preset[0].storedObj.Name),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*api.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, nil
			},
		},
		expectedOut: nil,
	}, { // test List with multiple levels of directories and expect flattened result
		prefix:      "/two-level/",
		pred:        storage.Everything,
		expectedOut: []*api.Pod{preset[1].storedObj, preset[2].storedObj},
	}}

	for i, tt := range tests {
		out := &api.PodList{}
		err := store.List(ctx, tt.prefix, "0", tt.pred, out)
		if err != nil {
			t.Fatalf("List failed: %v", err)
		}
		if len(tt.expectedOut) != len(out.Items) {
			t.Errorf("#%d: length of list want=%d, get=%d", i, len(tt.expectedOut), len(out.Items))
			continue
		}
		for j, wantPod := range tt.expectedOut {
			getPod := &out.Items[j]
			if !reflect.DeepEqual(wantPod, getPod) {
				t.Errorf("#%d: pod want=%#v, get=%#v", i, wantPod, getPod)
			}
		}
	}
}

func testSetup(t *testing.T) (context.Context, *store, *integration.ClusterV3) {
	cluster := integration.NewClusterV3(t, &integration.ClusterConfig{Size: 1})
	store := newStore(cluster.RandClient(), false, testapi.Default.Codec(), "")
	ctx := context.Background()
	return ctx, store, cluster
}

// testPropogateStore helps propogates store with objects, automates key generation, and returns
// keys and stored objects.
func testPropogateStore(ctx context.Context, t *testing.T, store *store, obj *api.Pod) (string, *api.Pod) {
	// Setup store with a key and grab the output for returning.
	key := "/testkey"
	setOutput := &api.Pod{}
	err := store.Create(ctx, key, obj, setOutput, 0)
	if err != nil {
		t.Fatalf("Set failed: %v", err)
	}
	return key, setOutput
}
