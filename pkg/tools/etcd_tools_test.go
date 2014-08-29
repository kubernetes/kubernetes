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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/coreos/go-etcd/etcd"
)

type fakeClientGetSet struct {
	get func(key string, sort, recursive bool) (*etcd.Response, error)
	set func(key, value string, ttl uint64) (*etcd.Response, error)
}

type TestResource struct {
	api.JSONBase `json:",inline" yaml:",inline"`
	Value        int `json:"value" yaml:"value,omitempty"`
}

var scheme *conversion.Scheme
var codec = api.Codec
var versioner = api.ResourceVersioner

func init() {
	scheme = conversion.NewScheme()
	scheme.ExternalVersion = "v1beta1"
	scheme.AddKnownTypes("", TestResource{})
	scheme.AddKnownTypes("v1beta1", TestResource{})
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

func TestExtractList(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
		R: &etcd.Response{
			EtcdIndex: 10,
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value:         `{"id":"foo"}`,
						ModifiedIndex: 1,
					},
					{
						Value:         `{"id":"bar"}`,
						ModifiedIndex: 2,
					},
					{
						Value:         `{"id":"baz"}`,
						ModifiedIndex: 3,
					},
				},
			},
		},
	}
	expect := []api.Pod{
		{JSONBase: api.JSONBase{ID: "foo", ResourceVersion: 1}},
		{JSONBase: api.JSONBase{ID: "bar", ResourceVersion: 2}},
		{JSONBase: api.JSONBase{ID: "baz", ResourceVersion: 3}},
	}

	var got []api.Pod
	helper := EtcdHelper{fakeClient, codec, versioner}
	resourceVersion := uint64(0)
	err := helper.ExtractList("/some/key", &got, &resourceVersion)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if resourceVersion != 10 {
		t.Errorf("Unexpected resource version %d", resourceVersion)
	}

	for i := 0; i < len(expect); i++ {
		if !reflect.DeepEqual(got[i], expect[i]) {
			t.Errorf("\nWanted:\n%#v\nGot:\n%#v\n", expect[i], got[i])
		}
	}
}

func TestExtractObj(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	expect := api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	fakeClient.Set("/some/key", util.EncodeJSON(expect), 0)
	helper := EtcdHelper{fakeClient, codec, versioner}
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

func TestSetObj(t *testing.T) {
	obj := api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	fakeClient := NewFakeEtcdClient(t)
	helper := EtcdHelper{fakeClient, codec, versioner}
	err := helper.SetObj("/some/key", obj)
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
}

func TestSetObjWithVersion(t *testing.T) {
	obj := api.Pod{JSONBase: api.JSONBase{ID: "foo", ResourceVersion: 1}}
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         api.EncodeOrDie(obj),
				ModifiedIndex: 1,
			},
		},
	}

	helper := EtcdHelper{fakeClient, codec, versioner}
	err := helper.SetObj("/some/key", obj)
	if err != nil {
		t.Fatalf("Unexpected error %#v", err)
	}
	data, err := codec.Encode(obj)
	if err != nil {
		t.Fatalf("Unexpected error %#v", err)
	}
	expect := string(data)
	got := fakeClient.Data["/some/key"].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}
}

func TestSetObjWithoutResourceVersioner(t *testing.T) {
	obj := api.Pod{JSONBase: api.JSONBase{ID: "foo"}}
	fakeClient := NewFakeEtcdClient(t)
	helper := EtcdHelper{fakeClient, codec, nil}
	err := helper.SetObj("/some/key", obj)
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
}

func TestAtomicUpdate(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	codec := scheme
	helper := EtcdHelper{fakeClient, codec, api.NewJSONBaseResourceVersioner()}

	// Create a new node.
	fakeClient.ExpectNotFoundGet("/some/key")
	obj := &TestResource{JSONBase: api.JSONBase{ID: "foo"}, Value: 1}
	err := helper.AtomicUpdate("/some/key", &TestResource{}, func(in interface{}) (interface{}, error) {
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
	objUpdate := &TestResource{JSONBase: api.JSONBase{ID: "foo"}, Value: 2}
	err = helper.AtomicUpdate("/some/key", &TestResource{}, func(in interface{}) (interface{}, error) {
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
	helper := EtcdHelper{fakeClient, scheme, api.NewJSONBaseResourceVersioner()}

	// Create a new node.
	fakeClient.ExpectNotFoundGet("/some/key")
	obj := &TestResource{JSONBase: api.JSONBase{ID: "foo"}, Value: 1}
	err := helper.AtomicUpdate("/some/key", &TestResource{}, func(in interface{}) (interface{}, error) {
		return obj, nil
	})
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}

	// Update an existing node with the same data
	callbackCalled := false
	objUpdate := &TestResource{JSONBase: api.JSONBase{ID: "foo"}, Value: 1}
	fakeClient.Err = errors.New("should not be called")
	err = helper.AtomicUpdate("/some/key", &TestResource{}, func(in interface{}) (interface{}, error) {
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

func TestAtomicUpdate_CreateCollision(t *testing.T) {
	fakeClient := NewFakeEtcdClient(t)
	fakeClient.TestIndex = true
	codec := scheme
	helper := EtcdHelper{fakeClient, codec, api.NewJSONBaseResourceVersioner()}

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
			err := helper.AtomicUpdate("/some/key", &TestResource{}, func(in interface{}) (interface{}, error) {
				defer func() { firstCall = false }()

				if firstCall {
					// Force collision by joining all concurrent AtomicUpdate operations here.
					wgForceCollision.Done()
					wgForceCollision.Wait()
				}

				currValue := in.(*TestResource).Value
				obj := TestResource{JSONBase: api.JSONBase{ID: "foo"}, Value: currValue + 1}
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
