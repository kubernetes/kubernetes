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
	"fmt"
	"path"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	etcd "github.com/coreos/etcd/client"
	"golang.org/x/net/context"
	apitesting "k8s.io/apimachinery/pkg/api/testing"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/conversion"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	storagetests "k8s.io/apiserver/pkg/storage/tests"
)

// prefixTransformer adds and verifies that all data has the correct prefix on its way in and out.
type prefixTransformer struct {
	prefix string
	stale  bool
	err    error
}

func (p prefixTransformer) TransformStringFromStorage(s string) (string, bool, error) {
	if !strings.HasPrefix(s, p.prefix) {
		return "", false, fmt.Errorf("value does not have expected prefix: %s", s)
	}
	return strings.TrimPrefix(s, p.prefix), p.stale, p.err
}
func (p prefixTransformer) TransformStringToStorage(s string) (string, error) {
	if len(s) > 0 {
		return p.prefix + s, p.err
	}
	return s, p.err
}

func defaultPrefix(s string) string {
	return "test!" + s
}

func defaultPrefixValue(value []byte) string {
	return defaultPrefix(string(value))
}

func testScheme(t *testing.T) (*runtime.Scheme, serializer.CodecFactory) {
	scheme := runtime.NewScheme()
	scheme.Log(t)
	scheme.AddKnownTypes(schema.GroupVersion{Version: runtime.APIVersionInternal}, &storagetesting.TestResource{})
	example.AddToScheme(scheme)
	examplev1.AddToScheme(scheme)
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
	codecs := serializer.NewCodecFactory(scheme)
	return scheme, codecs
}

func newEtcdHelper(client etcd.Client, scheme *runtime.Scheme, codec runtime.Codec, prefix string) etcdHelper {
	return *NewEtcdStorage(client, codec, prefix, false, etcdtest.DeserializationCacheSize, scheme, prefixTransformer{prefix: "test!"}).(*etcdHelper)
}

// Returns an encoded version of example.Pod with the given name.
func getEncodedPod(name string, codec runtime.Codec) string {
	pod, _ := runtime.Encode(codec, &examplev1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: name},
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

func createPodList(t *testing.T, helper etcdHelper, list *example.PodList) error {
	for i := range list.Items {
		returnedObj := &example.Pod{}
		err := createObj(t, helper, list.Items[i].Name, &list.Items[i], returnedObj, 0)
		if err != nil {
			return err
		}
		list.Items[i] = *returnedObj
	}
	return nil
}

func TestList(t *testing.T) {
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	helper := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix())

	list := example.PodList{
		Items: []example.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec:       storagetests.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "baz"},
				Spec:       storagetests.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec:       storagetests.DeepEqualSafePodSpec(),
			},
		},
	}

	createPodList(t, helper, &list)
	var got example.PodList
	// TODO: a sorted filter function could be applied such implied
	// ordering on the returned list doesn't matter.
	err := helper.List(context.TODO(), "/", "", storage.Everything, &got)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}

	if e, a := list.Items, got.Items; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

func TestTransformationFailure(t *testing.T) {
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	helper := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix())

	pods := []example.Pod{
		{
			ObjectMeta: metav1.ObjectMeta{Name: "bar"},
			Spec:       storagetests.DeepEqualSafePodSpec(),
		},
		{
			ObjectMeta: metav1.ObjectMeta{Name: "baz"},
			Spec:       storagetests.DeepEqualSafePodSpec(),
		},
	}
	createPodList(t, helper, &example.PodList{Items: pods[:1]})

	// create a second resource with an invalid prefix
	oldTransformer := helper.transformer
	helper.transformer = prefixTransformer{prefix: "otherprefix!"}
	createPodList(t, helper, &example.PodList{Items: pods[1:]})
	helper.transformer = oldTransformer

	// only the first item is returned, and no error
	var got example.PodList
	if err := helper.List(context.TODO(), "/", "", storage.Everything, &got); err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if e, a := pods[:1], got.Items; !reflect.DeepEqual(e, a) {
		t.Errorf("Unexpected: %s", diff.ObjectReflectDiff(e, a))
	}

	// Get should fail
	if err := helper.Get(context.TODO(), "/baz", "", &example.Pod{}, false); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	// GuaranteedUpdate should return an error
	if err := helper.GuaranteedUpdate(context.TODO(), "/baz", &example.Pod{}, false, nil, func(input runtime.Object, res storage.ResponseMeta) (output runtime.Object, ttl *uint64, err error) {
		return input, nil, nil
	}, &pods[1]); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	// Delete succeeds but reports an error because we cannot access the body
	if err := helper.Delete(context.TODO(), "/baz", &example.Pod{}, nil); !storage.IsInternalError(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	if err := helper.Get(context.TODO(), "/baz", "", &example.Pod{}, false); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestListFiltered(t *testing.T) {
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	helper := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix())

	list := example.PodList{
		Items: []example.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec:       storagetests.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "baz"},
				Spec:       storagetests.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec:       storagetests.DeepEqualSafePodSpec(),
			},
		},
	}

	createPodList(t, helper, &list)
	// List only "bar" pod
	p := storage.SelectionPredicate{
		Label: labels.Everything(),
		Field: fields.SelectorFromSet(fields.Set{"metadata.name": "bar"}),
		GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
			pod := obj.(*example.Pod)
			return labels.Set(pod.Labels), fields.Set{"metadata.name": pod.Name}, nil
		},
	}
	var got example.PodList
	err := helper.List(context.TODO(), "/", "", p, &got)
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
	scheme, codecs := testScheme(t)
	server := etcdtesting.NewEtcdTestClientServer(t)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	defer server.Terminate(t)

	roothelper := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix())
	helper1 := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix()+"/dir1")
	helper2 := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix()+"/dir2")

	list := example.PodList{
		Items: []example.Pod{
			{
				ObjectMeta: metav1.ObjectMeta{Name: "baz"},
				Spec:       storagetests.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "foo"},
				Spec:       storagetests.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: metav1.ObjectMeta{Name: "bar"},
				Spec:       storagetests.DeepEqualSafePodSpec(),
			},
		},
	}

	returnedObj := &example.Pod{}
	// create the 1st 2 elements in one directory
	createObj(t, helper1, list.Items[0].Name, &list.Items[0], returnedObj, 0)
	list.Items[0] = *returnedObj
	createObj(t, helper1, list.Items[1].Name, &list.Items[1], returnedObj, 0)
	list.Items[1] = *returnedObj
	// create the last element in the other directory
	createObj(t, helper2, list.Items[2].Name, &list.Items[2], returnedObj, 0)
	list.Items[2] = *returnedObj

	var got example.PodList
	err := roothelper.List(context.TODO(), "/", "", storage.Everything, &got)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if e, a := list.Items, got.Items; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

func TestGet(t *testing.T) {
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := "/some/key"
	helper := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix())
	expect := example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       storagetests.DeepEqualSafePodSpec(),
	}
	var got example.Pod
	if err := helper.Create(context.TODO(), key, &expect, &got, 0); err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect = got
	if err := helper.Get(context.TODO(), key, "", &got, false); err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(got, expect) {
		t.Errorf("Wanted %#v, got %#v", expect, got)
	}
}

func TestGetNotFoundErr(t *testing.T) {
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, schema.GroupVersion{Version: "v1"})
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	boguskey := "/some/boguskey"
	helper := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix())

	var got example.Pod
	err := helper.Get(context.TODO(), boguskey, "", &got, false)
	if !storage.IsNotFound(err) {
		t.Errorf("Unexpected reponse on key=%v, err=%v", boguskey, err)
	}
}

func TestCreate(t *testing.T) {
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	helper := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix())
	returnedObj := &example.Pod{}
	err := helper.Create(context.TODO(), "/some/key", obj, returnedObj, 5)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	_, err = runtime.Encode(codec, obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	err = helper.Get(context.TODO(), "/some/key", "", returnedObj, false)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	_, err = runtime.Encode(codec, returnedObj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if obj.Name != returnedObj.Name {
		t.Errorf("Wanted %v, got %v", obj.Name, returnedObj.Name)
	}
}

func TestCreateNilOutParam(t *testing.T) {
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	helper := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix())
	err := helper.Create(context.TODO(), "/some/key", obj, nil, 5)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
}

func TestGuaranteedUpdate(t *testing.T) {
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, schema.GroupVersion{Version: runtime.APIVersionInternal})
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := "/some/key"
	helper := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix())

	obj := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, Value: 1}
	err := helper.GuaranteedUpdate(context.TODO(), key, &storagetesting.TestResource{}, true, nil, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		return obj, nil
	}))
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}

	// Update an existing node.
	callbackCalled := false
	objUpdate := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, Value: 2}
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
	err = helper.Get(context.TODO(), key, "", objCheck, false)
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
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, schema.GroupVersion{Version: runtime.APIVersionInternal})
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := "/some/key"
	helper := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix())

	obj := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, Value: 1}
	original := &storagetesting.TestResource{}
	err := helper.GuaranteedUpdate(context.TODO(), key, original, true, nil, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		return obj, nil
	}))
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}

	// Update an existing node with the same data
	callbackCalled := false
	objUpdate := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: original.ResourceVersion}, Value: 1}
	result := &storagetesting.TestResource{}
	err = helper.GuaranteedUpdate(context.TODO(), key, result, true, nil, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		callbackCalled = true
		return objUpdate, nil
	}))
	if err != nil {
		t.Fatalf("Unexpected error %#v", err)
	}
	if !callbackCalled {
		t.Errorf("tryUpdate callback should have been called.")
	}
	if result.ResourceVersion != original.ResourceVersion {
		t.Fatalf("updated the object resource version")
	}

	// Update an existing node with the same data but return stale
	helper.transformer = prefixTransformer{prefix: "test!", stale: true}
	callbackCalled = false
	result = &storagetesting.TestResource{}
	objUpdate = &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, Value: 1}
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
	if result.ResourceVersion == original.ResourceVersion {
		t.Errorf("did not update the object resource version")
	}
}

func TestGuaranteedUpdateKeyNotFound(t *testing.T) {
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, schema.GroupVersion{Version: runtime.APIVersionInternal})
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := "/some/key"
	helper := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix())

	// Create a new node.
	obj := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, Value: 1}

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
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, schema.GroupVersion{Version: runtime.APIVersionInternal})
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	key := "/some/key"
	helper := newEtcdHelper(server.Client, scheme, codec, etcdtest.PathPrefix())

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
				obj := &storagetesting.TestResource{ObjectMeta: metav1.ObjectMeta{Name: "foo"}, Value: currValue + 1}
				return obj, nil
			}))
			if err != nil {
				t.Errorf("Unexpected error %#v", err)
			}
		}()
	}
	wgDone.Wait()

	stored := &storagetesting.TestResource{}
	err := helper.Get(context.TODO(), key, "", stored, false)
	if err != nil {
		t.Errorf("Unexpected error %#v", stored)
	}
	if stored.Value != concurrency {
		t.Errorf("Some of the writes were lost. Stored value: %d", stored.Value)
	}
}

func TestGuaranteedUpdateUIDMismatch(t *testing.T) {
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	prefix := path.Join("/", etcdtest.PathPrefix())
	helper := newEtcdHelper(server.Client, scheme, codec, prefix)

	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "A"}}
	podPtr := &example.Pod{}
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

func TestDeleteUIDMismatch(t *testing.T) {
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	prefix := path.Join("/", etcdtest.PathPrefix())
	helper := newEtcdHelper(server.Client, scheme, codec, prefix)

	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "A"}}
	podPtr := &example.Pod{}
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
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)
	prefix := path.Join("/", etcdtest.PathPrefix())

	obj := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "A"}}
	// fakeGet returns a large ModifiedIndex to emulate the case that another
	// party has updated the object.
	fakeGet := func(ctx context.Context, key string, opts *etcd.GetOptions) (*etcd.Response, error) {
		data, _ := runtime.Encode(codec, obj)
		return &etcd.Response{Node: &etcd.Node{Value: defaultPrefixValue(data), ModifiedIndex: 99}}, nil
	}
	expectedRetries := 3
	helper := newEtcdHelper(server.Client, scheme, codec, prefix)
	fake := &fakeDeleteKeysAPI{KeysAPI: helper.etcdKeysAPI, fakeGetCap: expectedRetries, fakeGetFunc: fakeGet}
	helper.etcdKeysAPI = fake

	returnedObj := &example.Pod{}
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
	err = helper.Get(context.TODO(), "/some/key", "", obj, false)
	if !storage.IsNotFound(err) {
		t.Errorf("Expect an NotFound error, got %v", err)
	}
}

func TestPrefix(t *testing.T) {
	scheme, codecs := testScheme(t)
	codec := apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion)
	server := etcdtesting.NewEtcdTestClientServer(t)
	defer server.Terminate(t)

	testcases := map[string]string{
		"custom/prefix":     "/custom/prefix",
		"/custom//prefix//": "/custom/prefix",
		"/registry":         "/registry",
	}
	for configuredPrefix, effectivePrefix := range testcases {
		helper := newEtcdHelper(server.Client, scheme, codec, configuredPrefix)
		if helper.pathPrefix != effectivePrefix {
			t.Errorf("configured prefix of %s, expected effective prefix of %s, got %s", configuredPrefix, effectivePrefix, helper.pathPrefix)
		}
	}
}
