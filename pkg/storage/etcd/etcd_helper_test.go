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

package etcd

import (
	"errors"
	"fmt"
	"math/rand"
	"net"
	"net/http"
	"net/http/httptest"
	"path"
	"reflect"
	"strconv"
	"sync"
	"testing"
	"time"

	"github.com/coreos/go-etcd/etcd"
	"github.com/stretchr/testify/assert"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	apitesting "k8s.io/kubernetes/pkg/api/testing"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/storage"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
)

const validEtcdVersion = "etcd 2.0.9"

type TestResource struct {
	unversioned.TypeMeta `json:",inline"`
	api.ObjectMeta       `json:"metadata"`
	Value                int `json:"value"`
}

func (*TestResource) IsAnAPIObject() {}

var scheme *runtime.Scheme
var codec runtime.Codec

func init() {
	scheme = runtime.NewScheme()
	scheme.AddKnownTypes("", &TestResource{})
	scheme.AddKnownTypes(testapi.Default.Version(), &TestResource{})
	codec = runtime.CodecFor(scheme, testapi.Default.Version())
	scheme.AddConversionFuncs(
		func(in *TestResource, out *TestResource, s conversion.Scope) error {
			*out = *in
			return nil
		},
	)
}

func newEtcdHelper(client tools.EtcdClient, codec runtime.Codec, prefix string) etcdHelper {
	return *NewEtcdStorage(client, codec, prefix).(*etcdHelper)
}

func TestIsEtcdNotFound(t *testing.T) {
	try := func(err error, isNotFound bool) {
		if IsEtcdNotFound(err) != isNotFound {
			t.Errorf("Expected %#v to return %v, but it did not", err, isNotFound)
		}
	}
	try(tools.EtcdErrorNotFound, true)
	try(&etcd.EtcdError{ErrorCode: 101}, false)
	try(nil, false)
	try(fmt.Errorf("some other kind of error"), false)
}

// Returns an encoded version of api.Pod with the given name.
func getEncodedPod(name string) string {
	pod, _ := testapi.Default.Codec().Encode(&api.Pod{
		ObjectMeta: api.ObjectMeta{Name: name},
	})
	return string(pod)
}

func TestList(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	key := etcdtest.AddPrefix("/some/key")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			EtcdIndex: 10,
			Node: &etcd.Node{
				Dir: true,
				Nodes: []*etcd.Node{
					{
						Key:           "/foo",
						Value:         getEncodedPod("foo"),
						Dir:           false,
						ModifiedIndex: 1,
					},
					{
						Key:           "/bar",
						Value:         getEncodedPod("bar"),
						Dir:           false,
						ModifiedIndex: 2,
					},
					{
						Key:           "/baz",
						Value:         getEncodedPod("baz"),
						Dir:           false,
						ModifiedIndex: 3,
					},
				},
			},
		},
	}
	expect := api.PodList{
		ListMeta: unversioned.ListMeta{ResourceVersion: "10"},
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "2"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "baz", ResourceVersion: "3"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
		},
	}

	var got api.PodList
	err := helper.List("/some/key", storage.Everything, &got)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

func TestListFiltered(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	key := etcdtest.AddPrefix("/some/key")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			EtcdIndex: 10,
			Node: &etcd.Node{
				Dir: true,
				Nodes: []*etcd.Node{
					{
						Key:           "/foo",
						Value:         getEncodedPod("foo"),
						Dir:           false,
						ModifiedIndex: 1,
					},
					{
						Key:           "/bar",
						Value:         getEncodedPod("bar"),
						Dir:           false,
						ModifiedIndex: 2,
					},
					{
						Key:           "/baz",
						Value:         getEncodedPod("baz"),
						Dir:           false,
						ModifiedIndex: 3,
					},
				},
			},
		},
	}
	expect := api.PodList{
		ListMeta: unversioned.ListMeta{ResourceVersion: "10"},
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "2"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
		},
	}

	filter := func(obj runtime.Object) bool {
		pod := obj.(*api.Pod)
		return pod.Name == "bar"
	}

	var got api.PodList
	err := helper.List("/some/key", filter, &got)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

// TestListAcrossDirectories ensures that the client excludes directories and flattens tree-response - simulates cross-namespace query
func TestListAcrossDirectories(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	key := etcdtest.AddPrefix("/some/key")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			EtcdIndex: 10,
			Node: &etcd.Node{
				Dir: true,
				Nodes: []*etcd.Node{
					{
						Key:   "/directory1",
						Value: `{"name": "directory1"}`,
						Dir:   true,
						Nodes: []*etcd.Node{
							{
								Key:           "/foo",
								Value:         getEncodedPod("foo"),
								Dir:           false,
								ModifiedIndex: 1,
							},
							{
								Key:           "/baz",
								Value:         getEncodedPod("baz"),
								Dir:           false,
								ModifiedIndex: 3,
							},
						},
					},
					{
						Key:   "/directory2",
						Value: `{"name": "directory2"}`,
						Dir:   true,
						Nodes: []*etcd.Node{
							{
								Key:           "/bar",
								Value:         getEncodedPod("bar"),
								ModifiedIndex: 2,
							},
						},
					},
				},
			},
		},
	}
	expect := api.PodList{
		ListMeta: unversioned.ListMeta{ResourceVersion: "10"},
		Items: []api.Pod{
			// We expect list to be sorted by directory (e.g. namespace) first, then by name.
			{
				ObjectMeta: api.ObjectMeta{Name: "baz", ResourceVersion: "3"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "2"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
		},
	}

	var got api.PodList
	err := helper.List("/some/key", storage.Everything, &got)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

func TestListExcludesDirectories(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	key := etcdtest.AddPrefix("/some/key")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			EtcdIndex: 10,
			Node: &etcd.Node{
				Dir: true,
				Nodes: []*etcd.Node{
					{
						Key:           "/foo",
						Value:         getEncodedPod("foo"),
						ModifiedIndex: 1,
					},
					{
						Key:           "/bar",
						Value:         getEncodedPod("bar"),
						ModifiedIndex: 2,
					},
					{
						Key:           "/baz",
						Value:         getEncodedPod("baz"),
						ModifiedIndex: 3,
					},
					{
						Key:   "/directory",
						Value: `{"name": "directory"}`,
						Dir:   true,
					},
				},
			},
		},
	}
	expect := api.PodList{
		ListMeta: unversioned.ListMeta{ResourceVersion: "10"},
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "2"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "baz", ResourceVersion: "3"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
				Spec:       apitesting.DeepEqualSafePodSpec(),
			},
		},
	}

	var got api.PodList
	err := helper.List("/some/key", storage.Everything, &got)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

func TestGet(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	key := etcdtest.AddPrefix("/some/key")
	expect := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec:       apitesting.DeepEqualSafePodSpec(),
	}
	fakeClient.Set(key, runtime.EncodeOrDie(testapi.Default.Codec(), &expect), 0)
	var got api.Pod
	err := helper.Get("/some/key", &got, false)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(got, expect) {
		t.Errorf("Wanted %#v, got %#v", expect, got)
	}
}

func TestGetNotFoundErr(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	key1 := etcdtest.AddPrefix("/some/key")
	fakeClient.Data[key1] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			ErrorCode: 100,
		},
	}
	key2 := etcdtest.AddPrefix("/some/key2")
	fakeClient.Data[key2] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
	}
	key3 := etcdtest.AddPrefix("/some/key3")
	fakeClient.Data[key3] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: "",
			},
		},
	}
	try := func(key string) {
		var got api.Pod
		err := helper.Get(key, &got, false)
		if err == nil {
			t.Errorf("%s: wanted error but didn't get one", key)
		}
		err = helper.Get(key, &got, true)
		if err != nil {
			t.Errorf("%s: didn't want error but got %#v", key, err)
		}
	}

	try("/some/key")
	try("/some/key2")
	try("/some/key3")
}

func TestCreate(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	fakeClient := tools.NewFakeEtcdClient(t)
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	returnedObj := &api.Pod{}
	err := helper.Create("/some/key", obj, returnedObj, 5)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err := testapi.Default.Codec().Encode(obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	key := etcdtest.AddPrefix("/some/key")
	node := fakeClient.Data[key].R.Node
	if e, a := string(data), node.Value; e != a {
		t.Errorf("Wanted %v, got %v", e, a)
	}
	if e, a := uint64(5), fakeClient.LastSetTTL; e != a {
		t.Errorf("Wanted %v, got %v", e, a)
	}
	if obj.ResourceVersion != returnedObj.ResourceVersion || obj.Name != returnedObj.Name {
		t.Errorf("If set was successful but returned object did not have correct resource version")
	}
}

func TestCreateNilOutParam(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	fakeClient := tools.NewFakeEtcdClient(t)
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	err := helper.Create("/some/key", obj, nil, 5)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
}

func TestSet(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	fakeClient := tools.NewFakeEtcdClient(t)
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	returnedObj := &api.Pod{}
	err := helper.Set("/some/key", obj, returnedObj, 5)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err := testapi.Default.Codec().Encode(obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect := string(data)
	key := etcdtest.AddPrefix("/some/key")
	got := fakeClient.Data[key].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}
	if e, a := uint64(5), fakeClient.LastSetTTL; e != a {
		t.Errorf("Wanted %v, got %v", e, a)
	}
	if obj.ResourceVersion != returnedObj.ResourceVersion || obj.Name != returnedObj.Name {
		t.Errorf("If set was successful but returned object did not have correct resource version")
	}
}

func TestSetFailCAS(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"}}
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.CasErr = fakeClient.NewError(123)
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	err := helper.Set("/some/key", obj, nil, 5)
	if err == nil {
		t.Errorf("Expecting error.")
	}
}

func TestSetWithVersion(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"}}
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	key := etcdtest.AddPrefix("/some/key")
	fakeClient.Data[key] = tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Default.Codec(), obj),
				ModifiedIndex: 1,
			},
		},
	}

	returnedObj := &api.Pod{}
	err := helper.Set("/some/key", obj, returnedObj, 7)
	if err != nil {
		t.Fatalf("Unexpected error %#v", err)
	}
	data, err := testapi.Default.Codec().Encode(obj)
	if err != nil {
		t.Fatalf("Unexpected error %#v", err)
	}
	expect := string(data)
	got := fakeClient.Data[key].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}
	if e, a := uint64(7), fakeClient.LastSetTTL; e != a {
		t.Errorf("Wanted %v, got %v", e, a)
	}
	if obj.ResourceVersion != returnedObj.ResourceVersion || obj.Name != returnedObj.Name {
		t.Errorf("If set was successful but returned object did not have correct resource version")
	}
}

func TestSetWithoutResourceVersioner(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	fakeClient := tools.NewFakeEtcdClient(t)
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	helper.versioner = nil
	returnedObj := &api.Pod{}
	err := helper.Set("/some/key", obj, returnedObj, 3)
	key := etcdtest.AddPrefix("/some/key")
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err := testapi.Default.Codec().Encode(obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect := string(data)
	got := fakeClient.Data[key].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}
	if e, a := uint64(3), fakeClient.LastSetTTL; e != a {
		t.Errorf("Wanted %v, got %v", e, a)
	}
	if obj.ResourceVersion != returnedObj.ResourceVersion || obj.Name != returnedObj.Name {
		t.Errorf("If set was successful but returned object did not have correct resource version")
	}
}

func TestSetNilOutParam(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	fakeClient := tools.NewFakeEtcdClient(t)
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), etcdtest.PathPrefix())
	helper.versioner = nil
	err := helper.Set("/some/key", obj, nil, 3)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
}

func TestGuaranteedUpdate(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	helper := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())
	key := etcdtest.AddPrefix("/some/key")

	// Create a new node.
	fakeClient.ExpectNotFoundGet(key)
	obj := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}
	err := helper.GuaranteedUpdate("/some/key", &TestResource{}, true, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		return obj, nil
	}))
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err := codec.Encode(obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect := string(data)
	got := fakeClient.Data[key].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}

	// Update an existing node.
	callbackCalled := false
	objUpdate := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 2}
	err = helper.GuaranteedUpdate("/some/key", &TestResource{}, true, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		callbackCalled = true

		if in.(*TestResource).Value != 1 {
			t.Errorf("Callback input was not current set value")
		}

		return objUpdate, nil
	}))
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err = codec.Encode(objUpdate)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect = string(data)
	got = fakeClient.Data[key].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}

	if !callbackCalled {
		t.Errorf("tryUpdate callback should have been called.")
	}
}

func TestGuaranteedUpdateTTL(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	helper := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())
	key := etcdtest.AddPrefix("/some/key")

	// Create a new node.
	fakeClient.ExpectNotFoundGet(key)
	obj := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}
	err := helper.GuaranteedUpdate("/some/key", &TestResource{}, true, func(in runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		if res.TTL != 0 {
			t.Fatalf("unexpected response meta: %#v", res)
		}
		ttl := uint64(10)
		return obj, &ttl, nil
	})
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err := codec.Encode(obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect := string(data)
	got := fakeClient.Data[key].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}
	if fakeClient.Data[key].R.Node.TTL != 10 {
		t.Errorf("expected TTL set: %d", fakeClient.Data[key].R.Node.TTL)
	}

	// Update an existing node.
	callbackCalled := false
	objUpdate := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 2}
	err = helper.GuaranteedUpdate("/some/key", &TestResource{}, true, func(in runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		if res.TTL != 10 {
			t.Fatalf("unexpected response meta: %#v", res)
		}
		callbackCalled = true

		if in.(*TestResource).Value != 1 {
			t.Errorf("Callback input was not current set value")
		}

		return objUpdate, nil, nil
	})

	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err = codec.Encode(objUpdate)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect = string(data)
	got = fakeClient.Data[key].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}
	if fakeClient.Data[key].R.Node.TTL != 10 {
		t.Errorf("expected TTL remained set: %d", fakeClient.Data[key].R.Node.TTL)
	}

	// Update an existing node and change ttl
	callbackCalled = false
	objUpdate = &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 3}
	err = helper.GuaranteedUpdate("/some/key", &TestResource{}, true, func(in runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		if res.TTL != 10 {
			t.Fatalf("unexpected response meta: %#v", res)
		}
		callbackCalled = true

		if in.(*TestResource).Value != 2 {
			t.Errorf("Callback input was not current set value")
		}

		newTTL := uint64(20)
		return objUpdate, &newTTL, nil
	})
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err = codec.Encode(objUpdate)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect = string(data)
	got = fakeClient.Data[key].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}
	if fakeClient.Data[key].R.Node.TTL != 20 {
		t.Errorf("expected TTL changed: %d", fakeClient.Data[key].R.Node.TTL)
	}

	if !callbackCalled {
		t.Errorf("tryUpdate callback should have been called.")
	}
}

func TestGuaranteedUpdateNoChange(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	helper := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())
	key := etcdtest.AddPrefix("/some/key")

	// Create a new node.
	fakeClient.ExpectNotFoundGet(key)
	obj := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}
	err := helper.GuaranteedUpdate("/some/key", &TestResource{}, true, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		return obj, nil
	}))
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}

	// Update an existing node with the same data
	callbackCalled := false
	objUpdate := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}
	err = helper.GuaranteedUpdate("/some/key", &TestResource{}, true, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		fakeClient.Err = errors.New("should not be called")
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
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	helper := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())
	key := etcdtest.AddPrefix("/some/key")

	// Create a new node.
	fakeClient.ExpectNotFoundGet(key)
	obj := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}

	f := storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
		return obj, nil
	})

	ignoreNotFound := false
	err := helper.GuaranteedUpdate("/some/key", &TestResource{}, ignoreNotFound, f)
	if err == nil {
		t.Errorf("Expected error for key not found.")
	}

	ignoreNotFound = true
	err = helper.GuaranteedUpdate("/some/key", &TestResource{}, ignoreNotFound, f)
	if err != nil {
		t.Errorf("Unexpected error %v.", err)
	}
}

func TestGuaranteedUpdate_CreateCollision(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	helper := newEtcdHelper(fakeClient, codec, etcdtest.PathPrefix())
	key := etcdtest.AddPrefix("/some/key")

	fakeClient.ExpectNotFoundGet(key)

	const concurrency = 10
	var wgDone sync.WaitGroup
	var wgForceCollision sync.WaitGroup
	wgDone.Add(concurrency)
	wgForceCollision.Add(concurrency)

	for i := 0; i < concurrency; i++ {
		// Increment TestResource.Value by 1
		go func() {
			defer wgDone.Done()

			firstCall := true
			err := helper.GuaranteedUpdate("/some/key", &TestResource{}, true, storage.SimpleUpdate(func(in runtime.Object) (runtime.Object, error) {
				defer func() { firstCall = false }()

				if firstCall {
					// Force collision by joining all concurrent GuaranteedUpdate operations here.
					wgForceCollision.Done()
					wgForceCollision.Wait()
				}

				currValue := in.(*TestResource).Value
				obj := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: currValue + 1}
				return obj, nil
			}))
			if err != nil {
				t.Errorf("Unexpected error %#v", err)
			}
		}()
	}
	wgDone.Wait()

	// Check that stored TestResource has received all updates.
	body := fakeClient.Data[key].R.Node.Value
	stored := &TestResource{}
	if err := codec.DecodeInto([]byte(body), stored); err != nil {
		t.Errorf("Error decoding stored value: %v", body)
	}
	if stored.Value != concurrency {
		t.Errorf("Some of the writes were lost. Stored value: %d", stored.Value)
	}
}

func TestGetEtcdVersion_ValidVersion(t *testing.T) {
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		fmt.Fprint(w, validEtcdVersion)
	}))
	defer testServer.Close()

	var version string
	var err error
	if version, err = GetEtcdVersion(testServer.URL); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	assert.Equal(t, validEtcdVersion, version, "Unexpected version")
	assert.Nil(t, err)
}

func TestGetEtcdVersion_ErrorStatus(t *testing.T) {
	testServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusServiceUnavailable)
	}))
	defer testServer.Close()

	_, err := GetEtcdVersion(testServer.URL)
	assert.NotNil(t, err)
}

func TestGetEtcdVersion_NotListening(t *testing.T) {
	portIsOpen := func(port int) bool {
		conn, err := net.DialTimeout("tcp", "127.0.0.1:"+strconv.Itoa(port), 1*time.Second)
		if err == nil {
			conn.Close()
			return true
		}
		return false
	}

	port := rand.Intn((1 << 16) - 1)
	for tried := 0; portIsOpen(port); tried++ {
		if tried >= 10 {
			t.Fatal("Couldn't find a closed TCP port to continue testing")
		}
		port++
	}

	_, err := GetEtcdVersion("http://127.0.0.1:" + strconv.Itoa(port))
	assert.NotNil(t, err)
}

func TestPrefixEtcdKey(t *testing.T) {
	fakeClient := tools.NewFakeEtcdClient(t)
	prefix := path.Join("/", etcdtest.PathPrefix())
	helper := newEtcdHelper(fakeClient, testapi.Default.Codec(), prefix)

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

func TestEtcdHealthCheck(t *testing.T) {
	tests := []struct {
		data      string
		expectErr bool
	}{
		{
			data:      "{\"health\": \"true\"}",
			expectErr: false,
		},
		{
			data:      "{\"health\": \"false\"}",
			expectErr: true,
		},
		{
			data:      "invalid json",
			expectErr: true,
		},
	}
	for _, test := range tests {
		err := EtcdHealthCheck([]byte(test.data))
		if err != nil && !test.expectErr {
			t.Errorf("unexpected error: %v", err)
		}
		if err == nil && test.expectErr {
			t.Error("unexpected non-error")
		}
	}
}
