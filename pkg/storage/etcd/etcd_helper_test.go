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

package etcd

import (
	"path"
	"reflect"
	"sync"
	"testing"
	"time"

	etcd "github.com/coreos/etcd/client"
	"github.com/stretchr/testify/assert"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	storagetesting "k8s.io/kubernetes/pkg/storage/testing"
)

const validEtcdVersion = "etcd 2.0.9"

func testScheme(t *testing.T) (*runtime.Scheme, runtime.Codec) {
	scheme := runtime.NewScheme()
	scheme.Log(t)
	scheme.AddKnownTypes(*testapi.Default.GroupVersion(), &storagetesting.TestResource{})
	scheme.AddKnownTypes(testapi.Default.InternalGroupVersion(), &storagetesting.TestResource{})
	if err := scheme.AddConversionFuncs(
		func(in *storagetesting.TestResource, out *storagetesting.TestResource, s conversion.Scope) error {
			*out = *in
			return nil
		},
		func(in, out *time.Time, s conversion.Scope) error {
			*out = *in
			return nil
		},
	); err != nil {
		panic(err)
	}
	codec := serializer.NewCodecFactory(scheme).LegacyCodec(*testapi.Default.GroupVersion())
	return scheme, codec
}

func newEtcdHelper(client etcd.Client, codec runtime.Codec, prefix string) etcdHelper {
	return *NewEtcdStorage(client, codec, prefix, false, etcdtest.DeserializationCacheSize).(*etcdHelper)
}

// Returns an encoded version of api.Pod with the given name.
func getEncodedPod(name string) string {
	pod, _ := runtime.Encode(testapi.Default.Codec(), &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: name},
	})
	return string(pod)
}

func createObj(t *testing.T, helper etcdHelper, name string, obj, out runtime.Object, ttl uint64) error {
	err := helper.Create(context.TODO(), name, obj, out, ttl)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	return err
}

func createPodList(t *testing.T, helper etcdHelper, list *api.PodList) error {
	for i := range list.Items {
		returnedObj := &api.Pod{}
		err := createObj(t, helper, list.Items[i].Name, &list.Items[i], returnedObj, 0)
		if err != nil {
			return err
		}
		list.Items[i] = *returnedObj
	}
	return nil
}

func TestList(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := etcdtest.AddPrefix("/some/key")
	helper := newEtcdHelper(server.Client, testapi.Default.Codec(), key)

	list := api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "bar"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "baz"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
		},
	}

	createPodList(t, helper, &list)
	var got api.PodList
	// TODO: a sorted filter function could be applied such implied
	// ordering on the returned list doesn't matter.
	err := helper.List(context.TODO(), key, "", storage.Everything, &got)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}

	if e, a := list.Items, got.Items; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

func TestListFiltered(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := etcdtest.AddPrefix("/some/key")
	helper := newEtcdHelper(server.Client, testapi.Default.Codec(), key)

	list := api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "bar"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "baz"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
		},
	}

	createPodList(t, helper, &list)
	filterFunc := func(obj runtime.Object) bool {
		pod := obj.(*api.Pod)
		return pod.Name == "bar"
	}
	filter := storage.NewSimpleFilter(filterFunc, storage.NoTriggerFunc)

	var got api.PodList
	err := helper.List(context.TODO(), key, "", filter, &got)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	// Check to make certain that the filter function only returns "bar"
	if e, a := list.Items[0], got.Items[0]; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

// TestListAcrossDirectories ensures that the client excludes directories and flattens tree-response - simulates cross-namespace query
func TestListAcrossDirectories(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	rootkey := etcdtest.AddPrefix("/some/key")
	key1 := etcdtest.AddPrefix("/some/key/directory1")
	key2 := etcdtest.AddPrefix("/some/key/directory2")

	roothelper := newEtcdHelper(server.Client, testapi.Default.Codec(), rootkey)
	helper1 := newEtcdHelper(server.Client, testapi.Default.Codec(), key1)
	helper2 := newEtcdHelper(server.Client, testapi.Default.Codec(), key2)

	list := api.PodList{
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "baz"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "foo"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "bar"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
		},
	}

	returnedObj := &api.Pod{}
	// create the 1st 2 elements in one directory
	createObj(t, helper1, list.Items[0].Name, &list.Items[0], returnedObj, 0)
	list.Items[0] = *returnedObj
	createObj(t, helper1, list.Items[1].Name, &list.Items[1], returnedObj, 0)
	list.Items[1] = *returnedObj
	// create the last element in the other directory
	createObj(t, helper2, list.Items[2].Name, &list.Items[2], returnedObj, 0)
	list.Items[2] = *returnedObj

	var got api.PodList
	err := roothelper.List(context.TODO(), rootkey, "", storage.Everything, &got)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if e, a := list.Items, got.Items; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

func TestGet(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := etcdtest.AddPrefix("/some/key")
	helper := newEtcdHelper(server.Client, testapi.Default.Codec(), key)
	expect := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec:       apitesting.DeepEqualSafePodSpec(),
	}
	var got api.Pod
	if err := helper.Create(context.TODO(), key, &expect, &got, 0); err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect = got
	if err := helper.Get(context.TODO(), key, &got, false); err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(got, expect) {
		t.Errorf("Wanted %#v, got %#v", expect, got)
	}
}

func TestGetNotFoundErr(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := etcdtest.AddPrefix("/some/key")
	boguskey := etcdtest.AddPrefix("/some/boguskey")
	helper := newEtcdHelper(server.Client, testapi.Default.Codec(), key)

	var got api.Pod
	err := helper.Get(context.TODO(), boguskey, &got, false)
	if !storage.IsNotFound(err) {
		t.Errorf("Unexpected reponse on key=%v, err=%v", key, err)
	}
}

func TestCreate(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	helper := newEtcdHelper(server.Client, testapi.Default.Codec(), etcdtest.PathPrefix())
	returnedObj := &api.Pod{}
	err := helper.Create(context.TODO(), "/some/key", obj, returnedObj, 5)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	_, err = runtime.Encode(testapi.Default.Codec(), obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	err = helper.Get(context.TODO(), "/some/key", returnedObj, false)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	_, err = runtime.Encode(testapi.Default.Codec(), returnedObj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if obj.Name != returnedObj.Name {
		t.Errorf("Wanted %v, got %v", obj.Name, returnedObj.Name)
	}
}

func TestCreateNilOutParam(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	helper := newEtcdHelper(server.Client, testapi.Default.Codec(), etcdtest.PathPrefix())
	err := helper.Create(context.TODO(), "/some/key", obj, nil, 5)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
}

func TestGuaranteedUpdate(t *testing.T) {
	_, codec := testScheme(t)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := etcdtest.AddPrefix("/some/key")
	helper := newEtcdHelper(server.Client, codec, key)

	obj := &storagetesting.TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}
	err := helper.GuaranteedUpdate(context.TODO(), key, &storagetesting.TestResource{}, true, nil, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		return obj, nil
	}))
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}

	// Update an existing node.
	callbackCalled := false
	objUpdate := &storagetesting.TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 2}
	err = helper.GuaranteedUpdate(context.TODO(), key, &storagetesting.TestResource{}, true, nil, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		callbackCalled = true

		if in.(*storagetesting.TestResource).Value != 1 {
			t.Errorf("Callback input was not current set value")
		}

		return objUpdate, nil
	}))
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}

	objCheck := &storagetesting.TestResource{}
	err = helper.Get(context.TODO(), key, objCheck, false)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if objCheck.Value != 2 {
		t.Errorf("Value should have been 2 but got %v", objCheck.Value)
	}

	if !callbackCalled {
		t.Errorf("tryUpdate callback should have been called.")
	}
}

func TestGuaranteedUpdateNoChange(t *testing.T) {
	_, codec := testScheme(t)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := etcdtest.AddPrefix("/some/key")
	helper := newEtcdHelper(server.Client, codec, key)

	obj := &storagetesting.TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}
	err := helper.GuaranteedUpdate(context.TODO(), key, &storagetesting.TestResource{}, true, nil, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		return obj, nil
	}))
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}

	// Update an existing node with the same data
	callbackCalled := false
	objUpdate := &storagetesting.TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}
	err = helper.GuaranteedUpdate(context.TODO(), key, &storagetesting.TestResource{}, true, nil, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		callbackCalled = true
		return objUpdate, nil
	}))
	if err != nil {
		t.Fatalf("Unexpected error %#v", err)
	}
	if !callbackCalled {
		t.Errorf("tryUpdate callback should have been called.")
	}
}

func TestGuaranteedUpdateKeyNotFound(t *testing.T) {
	_, codec := testScheme(t)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := etcdtest.AddPrefix("/some/key")
	helper := newEtcdHelper(server.Client, codec, key)

	// Create a new node.
	obj := &storagetesting.TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}

	f := storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		return obj, nil
	})

	ignoreNotFound := false
	err := helper.GuaranteedUpdate(context.TODO(), key, &storagetesting.TestResource{}, ignoreNotFound, nil, f)
	if err == nil {
		t.Errorf("Expected error for key not found.")
	}

	ignoreNotFound = true
	err = helper.GuaranteedUpdate(context.TODO(), key, &storagetesting.TestResource{}, ignoreNotFound, nil, f)
	if err != nil {
		t.Errorf("Unexpected error %v.", err)
	}
}

func TestGuaranteedUpdate_CreateCollision(t *testing.T) {
	_, codec := testScheme(t)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := etcdtest.AddPrefix("/some/key")
	helper := newEtcdHelper(server.Client, codec, etcdtest.PathPrefix())

	const concurrency = 10
	var wgDone sync.WaitGroup
	var wgForceCollision sync.WaitGroup
	wgDone.Add(concurrency)
	wgForceCollision.Add(concurrency)

	for i := 0; i < concurrency; i++ {
		// Increment storagetesting.TestResource.Value by 1
		go func() {
			defer wgDone.Done()

			firstCall := true
			err := helper.GuaranteedUpdate(context.TODO(), key, &storagetesting.TestResource{}, true, nil, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
				defer func() { firstCall = false }()

				if firstCall {
					// Force collision by joining all concurrent GuaranteedUpdate operations here.
					wgForceCollision.Done()
					wgForceCollision.Wait()
				}

				currValue := in.(*storagetesting.TestResource).Value
				obj := &storagetesting.TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: currValue + 1}
				return obj, nil
			}))
			if err != nil {
				t.Errorf("Unexpected error %#v", err)
			}
		}()
	}
	wgDone.Wait()

	stored := &storagetesting.TestResource{}
	err := helper.Get(context.TODO(), key, stored, false)
	if err != nil {
		t.Errorf("Unexpected error %#v", stored)
	}
	if stored.Value != concurrency {
		t.Errorf("Some of the writes were lost. Stored value: %d", stored.Value)
	}
}

func TestGuaranteedUpdateUIDMismatch(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	prefix := path.Join("/", etcdtest.PathPrefix())
	helper := newEtcdHelper(server.Client, testapi.Default.Codec(), prefix)

	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo", UID: "A"}}
	podPtr := &api.Pod{}
	err := helper.Create(context.TODO(), "/some/key", obj, podPtr, 0)
	if err != nil {
		t.Fatalf("Unexpected error %#v", err)
	}
	err = helper.GuaranteedUpdate(context.TODO(), "/some/key", podPtr, true, storage.NewUIDPreconditions("B"), storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		return obj, nil
	}))
	if !storage.IsInvalidObj(err) {
		t.Fatalf("Expect a Test Failed (write conflict) error, got: %v", err)
	}
}

func TestPrefixEtcdKey(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	prefix := path.Join("/", etcdtest.PathPrefix())
	helper := newEtcdHelper(server.Client, testapi.Default.Codec(), prefix)

	baseKey := "/some/key"

	// Verify prefix is added
	keyBefore := baseKey
	keyAfter := helper.prefixEtcdKey(keyBefore)

	assert.Equal(t, keyAfter, path.Join(prefix, baseKey), "Prefix incorrectly added by EtcdHelper")

	// Verify prefix is not added
	keyBefore = path.Join(prefix, baseKey)
	keyAfter = helper.prefixEtcdKey(keyBefore)

	assert.Equal(t, keyBefore, keyAfter, "Prefix incorrectly added by EtcdHelper")
}

func TestDeleteUIDMismatch(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	prefix := path.Join("/", etcdtest.PathPrefix())
	helper := newEtcdHelper(server.Client, testapi.Default.Codec(), prefix)

	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo", UID: "A"}}
	podPtr := &api.Pod{}
	err := helper.Create(context.TODO(), "/some/key", obj, podPtr, 0)
	if err != nil {
		t.Fatalf("Unexpected error %#v", err)
	}
	err = helper.Delete(context.TODO(), "/some/key", obj, storage.NewUIDPreconditions("B"))
	if !storage.IsInvalidObj(err) {
		t.Fatalf("Expect a Test Failed (write conflict) error, got: %v", err)
	}
}

type getFunc func(ctx context.Context, key string, opts *etcd.GetOptions) (*etcd.Response, error)

type fakeDeleteKeysAPI struct {
	etcd.KeysAPI
	fakeGetFunc getFunc
	getCount    int
	// The fakeGetFunc will be called fakeGetCap times before the KeysAPI's Get will be called.
	fakeGetCap int
}

func (f *fakeDeleteKeysAPI) Get(ctx context.Context, key string, opts *etcd.GetOptions) (*etcd.Response, error) {
	f.getCount++
	if f.getCount < f.fakeGetCap {
		return f.fakeGetFunc(ctx, key, opts)
	}
	return f.KeysAPI.Get(ctx, key, opts)
}

// This is to emulate the case where another party updates the object when
// etcdHelper.Delete has verified the preconditions, but hasn't carried out the
// deletion yet. Etcd will fail the deletion and report the conflict. etcdHelper
// should retry until there is no conflict.
func TestDeleteWithRetry(t *testing.T) {
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	prefix := path.Join("/", etcdtest.PathPrefix())

	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo", UID: "A"}}
	// fakeGet returns a large ModifiedIndex to emulate the case that another
	// party has updated the object.
	fakeGet := func(ctx context.Context, key string, opts *etcd.GetOptions) (*etcd.Response, error) {
		data, _ := runtime.Encode(testapi.Default.Codec(), obj)
		return &etcd.Response{Node: &etcd.Node{Value: string(data), ModifiedIndex: 99}}, nil
	}
	expectedRetries := 3
	helper := newEtcdHelper(server.Client, testapi.Default.Codec(), prefix)
	fake := &fakeDeleteKeysAPI{KeysAPI: helper.etcdKeysAPI, fakeGetCap: expectedRetries, fakeGetFunc: fakeGet}
	helper.etcdKeysAPI = fake

	returnedObj := &api.Pod{}
	err := helper.Create(context.TODO(), "/some/key", obj, returnedObj, 0)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}

	err = helper.Delete(context.TODO(), "/some/key", obj, storage.NewUIDPreconditions("A"))
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if fake.getCount != expectedRetries {
		t.Errorf("Expect %d retries, got %d", expectedRetries, fake.getCount)
	}
	err = helper.Get(context.TODO(), "/some/key", obj, false)
	if !storage.IsNotFound(err) {
		t.Errorf("Expect an NotFound error, got %v", err)
	}
}
