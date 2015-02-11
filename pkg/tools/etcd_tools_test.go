/*
Copyright 2014 Google Inc. All rights reserved.

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

package tools

import (
	"errors"
	"fmt"
	"reflect"
	"sync"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/meta"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/coreos/go-etcd/etcd"
)

type TestResource struct {
	api.TypeMeta   `json:",inline"`
	api.ObjectMeta `json:"metadata"`
	Value          int `json:"value"`
}

func (*TestResource) IsAnAPIObject() {}

var scheme *runtime.Scheme
var codec runtime.Codec
var versioner = RuntimeVersionAdapter{meta.NewAccessor()}

func init() {
	scheme = runtime.NewScheme()
	scheme.AddKnownTypes("", &TestResource{})
	scheme.AddKnownTypes("v1beta1", &TestResource{})
	codec = runtime.CodecFor(scheme, "v1beta1")
	scheme.AddConversionFuncs(
		func(in *TestResource, out *TestResource, s conversion.Scope) error {
			*out = *in
			return nil
		},
	)
}

func TestIsEtcdNotFound(t *testing.T) {
	try := func(err error, isNotFound bool) {
		if IsEtcdNotFound(err) != isNotFound {
			t.Errorf("Expected %#v to return %v, but it did not", err, isNotFound)
		}
	}
	try(EtcdErrorNotFound, true)
	try(&etcd.EtcdError{ErrorCode: 101}, false)
	try(nil, false)
	try(fmt.Errorf("some other kind of error"), false)
}

func TestExtractToList(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
		R: &etcd.Response{
			EtcdIndex: 10,
			Node: &etcd.Node{
				Dir: true,
				Nodes: []*etcd.Node{
					{
						Key:           "/foo",
						Value:         `{"id":"foo","kind":"Pod","apiVersion":"v1beta1"}`,
						Dir:           false,
						ModifiedIndex: 1,
					},
					{
						Key:           "/bar",
						Value:         `{"id":"bar","kind":"Pod","apiVersion":"v1beta1"}`,
						Dir:           false,
						ModifiedIndex: 2,
					},
					{
						Key:           "/baz",
						Value:         `{"id":"baz","kind":"Pod","apiVersion":"v1beta1"}`,
						Dir:           false,
						ModifiedIndex: 3,
					},
				},
			},
		},
	}
	expect := api.PodList{
		ListMeta: api.ListMeta{ResourceVersion: "10"},
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "2"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "baz", ResourceVersion: "3"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
	}

	var got api.PodList
	helper := EtcdHelper{fakeClient, testapi.Codec(), versioner}
	err := helper.ExtractToList("/some/key", &got)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

// TestExtractToListAcrossDirectories ensures that the client excludes directories and flattens tree-response - simulates cross-namespace query
func TestExtractToListAcrossDirectories(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
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
								Value:         `{"id":"foo","kind":"Pod","apiVersion":"v1beta1"}`,
								Dir:           false,
								ModifiedIndex: 1,
							},
							{
								Key:           "/baz",
								Value:         `{"id":"baz","kind":"Pod","apiVersion":"v1beta1"}`,
								Dir:           false,
								ModifiedIndex: 1,
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
								Value:         `{"id":"bar","kind":"Pod","apiVersion":"v1beta1"}`,
								ModifiedIndex: 2,
							},
						},
					},
				},
			},
		},
	}
	expect := api.PodList{
		ListMeta: api.ListMeta{ResourceVersion: "10"},
		Items: []api.Pod{
			// We expect list to be sorted by directory (e.g. namespace) first, then by name.
			{
				ObjectMeta: api.ObjectMeta{Name: "baz", ResourceVersion: "1"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "2"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
	}

	var got api.PodList
	helper := EtcdHelper{fakeClient, testapi.Codec(), versioner}
	err := helper.ExtractToList("/some/key", &got)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

func TestExtractToListExcludesDirectories(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
		R: &etcd.Response{
			EtcdIndex: 10,
			Node: &etcd.Node{
				Dir: true,
				Nodes: []*etcd.Node{
					{
						Key:           "/foo",
						Value:         `{"id":"foo","kind":"Pod","apiVersion":"v1beta1"}`,
						ModifiedIndex: 1,
					},
					{
						Key:           "/bar",
						Value:         `{"id":"bar","kind":"Pod","apiVersion":"v1beta1"}`,
						ModifiedIndex: 2,
					},
					{
						Key:           "/baz",
						Value:         `{"id":"baz","kind":"Pod","apiVersion":"v1beta1"}`,
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
		ListMeta: api.ListMeta{ResourceVersion: "10"},
		Items: []api.Pod{
			{
				ObjectMeta: api.ObjectMeta{Name: "bar", ResourceVersion: "2"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "baz", ResourceVersion: "3"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
			{
				ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
				Spec: api.PodSpec{
					RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
					DNSPolicy:     api.DNSClusterFirst,
				},
			},
		},
	}

	var got api.PodList
	helper := EtcdHelper{fakeClient, testapi.Codec(), versioner}
	err := helper.ExtractToList("/some/key", &got)
	if err != nil {
		t.Errorf("Unexpected error %v", err)
	}
	if e, a := expect, got; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}
}

func TestExtractObj(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	expect := api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicy{Always: &api.RestartPolicyAlways{}},
			DNSPolicy:     api.DNSClusterFirst,
		},
	}
	fakeClient.Set("/some/key", runtime.EncodeOrDie(testapi.Codec(), &expect), 0)
	helper := EtcdHelper{fakeClient, testapi.Codec(), versioner}
	var got api.Pod
	err := helper.ExtractObj("/some/key", &got, false)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(got, expect) {
		t.Errorf("Wanted %#v, got %#v", expect, got)
	}
}

func TestExtractObjNotFoundErr(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
		E: &etcd.EtcdError{
			ErrorCode: 100,
		},
	}
	fakeClient.Data["/some/key2"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: nil,
		},
	}
	fakeClient.Data["/some/key3"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value: "",
			},
		},
	}
	helper := EtcdHelper{fakeClient, codec, versioner}
	try := func(key string) {
		var got api.Pod
		err := helper.ExtractObj(key, &got, false)
		if err == nil {
			t.Errorf("%s: wanted error but didn't get one", key)
		}
		err = helper.ExtractObj(key, &got, true)
		if err != nil {
			t.Errorf("%s: didn't want error but got %#v", key, err)
		}
	}

	try("/some/key")
	try("/some/key2")
	try("/some/key3")
}

func TestCreateObj(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	fakeClient := NewFakeEtcdClient(t)
	helper := EtcdHelper{fakeClient, testapi.Codec(), versioner}
	err := helper.CreateObj("/some/key", obj, 5)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err := testapi.Codec().Encode(obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	node := fakeClient.Data["/some/key"].R.Node
	if e, a := string(data), node.Value; e != a {
		t.Errorf("Wanted %v, got %v", e, a)
	}
	if e, a := uint64(5), fakeClient.LastSetTTL; e != a {
		t.Errorf("Wanted %v, got %v", e, a)
	}
}

func TestSetObj(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	fakeClient := NewFakeEtcdClient(t)
	helper := EtcdHelper{fakeClient, testapi.Codec(), versioner}
	err := helper.SetObj("/some/key", obj, 5)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err := testapi.Codec().Encode(obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect := string(data)
	got := fakeClient.Data["/some/key"].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}
	if e, a := uint64(5), fakeClient.LastSetTTL; e != a {
		t.Errorf("Wanted %v, got %v", e, a)
	}

}

func TestSetObjWithVersion(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"}}
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), obj),
				ModifiedIndex: 1,
			},
		},
	}

	helper := EtcdHelper{fakeClient, testapi.Codec(), versioner}
	err := helper.SetObj("/some/key", obj, 7)
	if err != nil {
		t.Fatalf("Unexpected error %#v", err)
	}
	data, err := testapi.Codec().Encode(obj)
	if err != nil {
		t.Fatalf("Unexpected error %#v", err)
	}
	expect := string(data)
	got := fakeClient.Data["/some/key"].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}
	if e, a := uint64(7), fakeClient.LastSetTTL; e != a {
		t.Errorf("Wanted %v, got %v", e, a)
	}
}

func TestSetObjWithoutResourceVersioner(t *testing.T) {
	obj := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	fakeClient := NewFakeEtcdClient(t)
	helper := EtcdHelper{fakeClient, testapi.Codec(), nil}
	err := helper.SetObj("/some/key", obj, 3)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err := testapi.Codec().Encode(obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect := string(data)
	got := fakeClient.Data["/some/key"].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}
	if e, a := uint64(3), fakeClient.LastSetTTL; e != a {
		t.Errorf("Wanted %v, got %v", e, a)
	}
}

func TestAtomicUpdate(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	helper := EtcdHelper{fakeClient, codec, versioner}

	// Create a new node.
	fakeClient.ExpectNotFoundGet("/some/key")
	obj := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}
	err := helper.AtomicUpdate("/some/key", &TestResource{}, true, func(in runtime.Object) (runtime.Object, error) {
		return obj, nil
	})
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err := codec.Encode(obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect := string(data)
	got := fakeClient.Data["/some/key"].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}

	// Update an existing node.
	callbackCalled := false
	objUpdate := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 2}
	err = helper.AtomicUpdate("/some/key", &TestResource{}, true, func(in runtime.Object) (runtime.Object, error) {
		callbackCalled = true

		if in.(*TestResource).Value != 1 {
			t.Errorf("Callback input was not current set value")
		}

		return objUpdate, nil
	})
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	data, err = codec.Encode(objUpdate)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect = string(data)
	got = fakeClient.Data["/some/key"].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}

	if !callbackCalled {
		t.Errorf("tryUpdate callback should have been called.")
	}
}

func TestAtomicUpdateNoChange(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	helper := EtcdHelper{fakeClient, codec, versioner}

	// Create a new node.
	fakeClient.ExpectNotFoundGet("/some/key")
	obj := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}
	err := helper.AtomicUpdate("/some/key", &TestResource{}, true, func(in runtime.Object) (runtime.Object, error) {
		return obj, nil
	})
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}

	// Update an existing node with the same data
	callbackCalled := false
	objUpdate := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}
	fakeClient.Err = errors.New("should not be called")
	err = helper.AtomicUpdate("/some/key", &TestResource{}, true, func(in runtime.Object) (runtime.Object, error) {
		callbackCalled = true
		return objUpdate, nil
	})
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !callbackCalled {
		t.Errorf("tryUpdate callback should have been called.")
	}
}

func TestAtomicUpdateKeyNotFound(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	helper := EtcdHelper{fakeClient, codec, versioner}

	// Create a new node.
	fakeClient.ExpectNotFoundGet("/some/key")
	obj := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: 1}

	f := func(in runtime.Object) (runtime.Object, error) {
		return obj, nil
	}

	ignoreNotFound := false
	err := helper.AtomicUpdate("/some/key", &TestResource{}, ignoreNotFound, f)
	if err == nil {
		t.Errorf("Expected error for key not found.")
	}

	ignoreNotFound = true
	err = helper.AtomicUpdate("/some/key", &TestResource{}, ignoreNotFound, f)
	if err != nil {
		t.Errorf("Unexpected error %v.", err)
	}
}

func TestAtomicUpdate_CreateCollision(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	helper := EtcdHelper{fakeClient, codec, versioner}

	fakeClient.ExpectNotFoundGet("/some/key")

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
			err := helper.AtomicUpdate("/some/key", &TestResource{}, true, func(in runtime.Object) (runtime.Object, error) {
				defer func() { firstCall = false }()

				if firstCall {
					// Force collision by joining all concurrent AtomicUpdate operations here.
					wgForceCollision.Done()
					wgForceCollision.Wait()
				}

				currValue := in.(*TestResource).Value
				obj := &TestResource{ObjectMeta: api.ObjectMeta{Name: "foo"}, Value: currValue + 1}
				return obj, nil
			})
			if err != nil {
				t.Errorf("Unexpected error %#v", err)
			}
		}()
	}
	wgDone.Wait()

	// Check that stored TestResource has received all updates.
	body := fakeClient.Data["/some/key"].R.Node.Value
	stored := &TestResource{}
	if err := codec.DecodeInto([]byte(body), stored); err != nil {
		t.Errorf("Error decoding stored value: %v", body)
	}
	if stored.Value != concurrency {
		t.Errorf("Some of the writes were lost. Stored value: %d", stored.Value)
	}
}
