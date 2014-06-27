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

package util

import (
	"fmt"
	"reflect"
	"testing"

	"github.com/coreos/go-etcd/etcd"
)

type fakeEtcdGetSet struct {
	get func(key string, sort, recursive bool) (*etcd.Response, error)
	set func(key, value string, ttl uint64) (*etcd.Response, error)
}

func TestIsNotFoundErr(t *testing.T) {
	try := func(err error, isNotFound bool) {
		if IsEtcdNotFound(err) != isNotFound {
			t.Errorf("Expected %#v to return %v, but it did not", err, isNotFound)
		}
	}
	try(&etcd.EtcdError{ErrorCode: 100}, true)
	try(&etcd.EtcdError{ErrorCode: 101}, false)
	try(nil, false)
	try(fmt.Errorf("some other kind of error"), false)
}

type testMarshalType struct {
	ID string `json:"id"`
}

func TestExtractList(t *testing.T) {
	fakeClient := MakeFakeEtcdClient(t)
	fakeClient.Data["/some/key"] = EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Nodes: []*etcd.Node{
					{
						Value: `{"id":"foo"}`,
					},
					{
						Value: `{"id":"bar"}`,
					},
					{
						Value: `{"id":"baz"}`,
					},
				},
			},
		},
	}
	expect := []testMarshalType{
		{"foo"},
		{"bar"},
		{"baz"},
	}
	var got []testMarshalType
	helper := EtcdHelper{fakeClient}
	err := helper.ExtractList("/some/key", &got)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(got, expect) {
		t.Errorf("Wanted %#v, got %#v", expect, got)
	}
}

func TestExtractObj(t *testing.T) {
	fakeClient := MakeFakeEtcdClient(t)
	expect := testMarshalType{ID: "foo"}
	fakeClient.Set("/some/key", MakeJSONString(expect), 0)
	helper := EtcdHelper{fakeClient}
	var got testMarshalType
	err, _ := helper.ExtractObj("/some/key", &got, false)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	if !reflect.DeepEqual(got, expect) {
		t.Errorf("Wanted %#v, got %#v", expect, got)
	}
}

func TestExtractObjNotFoundErr(t *testing.T) {
	fakeClient := MakeFakeEtcdClient(t)
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
	helper := EtcdHelper{fakeClient}
	try := func(key string) {
		var got testMarshalType
		err, _ := helper.ExtractObj(key, &got, false)
		if err == nil {
			t.Errorf("%s: wanted error but didn't get one", key)
		}
		err, _ = helper.ExtractObj(key, &got, true)
		if err != nil {
			t.Errorf("%s: didn't want error but got %#v", key, err)
		}
	}

	try("/some/key")
	try("/some/key2")
	try("/some/key3")
}

func TestSetObj(t *testing.T) {
	obj := testMarshalType{ID: "foo"}
	fakeClient := MakeFakeEtcdClient(t)
	helper := EtcdHelper{fakeClient}
	err := helper.SetObj("/some/key", obj)
	if err != nil {
		t.Errorf("Unexpected error %#v", err)
	}
	expect := MakeJSONString(obj)
	got := fakeClient.Data["/some/key"].R.Node.Value
	if expect != got {
		t.Errorf("Wanted %v, got %v", expect, got)
	}
}
