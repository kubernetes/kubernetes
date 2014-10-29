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

package etcd

import (
	"fmt"
	"path"
	"reflect"
	"testing"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/errors"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/testapi"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/registry/generic"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/tools"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	"github.com/coreos/go-etcd/etcd"
)

func NewTestGenericEtcdRegistry(t *testing.T) (*tools.FakeEtcdClient, *Etcd) {
	f := tools.NewFakeEtcdClient(t)
	f.TestIndex = true
	h := tools.EtcdHelper{f, testapi.Codec(), tools.RuntimeVersionAdapter{testapi.MetadataAccessor()}}
	return f, &Etcd{
		NewFunc:      func() runtime.Object { return &api.Pod{} },
		NewListFunc:  func() runtime.Object { return &api.PodList{} },
		EndpointName: "pods",
		KeyRoot:      "/registry/pods",
		KeyFunc: func(id string) string {
			return path.Join("/registry/pods", id)
		},
		Helper: h,
	}
}

// SetMatcher is a matcher that matches any pod with id in the set.
// Makes testing simpler.
type SetMatcher struct {
	util.StringSet
}

func (sm SetMatcher) Matches(obj runtime.Object) (bool, error) {
	pod, ok := obj.(*api.Pod)
	if !ok {
		return false, fmt.Errorf("wrong object")
	}
	return sm.Has(pod.Name), nil
}

// EverythingMatcher matches everything
type EverythingMatcher struct{}

func (EverythingMatcher) Matches(obj runtime.Object) (bool, error) {
	return true, nil
}

func TestEtcdList(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta:   api.ObjectMeta{Name: "foo"},
		DesiredState: api.PodState{Host: "machine"},
	}
	podB := &api.Pod{
		ObjectMeta:   api.ObjectMeta{Name: "bar"},
		DesiredState: api.PodState{Host: "machine"},
	}

	normalListResp := &etcd.Response{
		Node: &etcd.Node{
			Nodes: []*etcd.Node{
				{Value: runtime.EncodeOrDie(testapi.Codec(), podA)},
				{Value: runtime.EncodeOrDie(testapi.Codec(), podB)},
			},
		},
	}

	table := map[string]struct {
		in      tools.EtcdResponseWithError
		m       generic.Matcher
		out     runtime.Object
		succeed bool
	}{
		"empty": {
			in: tools.EtcdResponseWithError{
				R: &etcd.Response{
					Node: &etcd.Node{
						Nodes: []*etcd.Node{},
					},
				},
				E: nil,
			},
			m:       EverythingMatcher{},
			out:     &api.PodList{Items: []api.Pod{}},
			succeed: true,
		},
		"notFound": {
			in: tools.EtcdResponseWithError{
				R: &etcd.Response{},
				E: tools.EtcdErrorNotFound,
			},
			m:       EverythingMatcher{},
			out:     &api.PodList{Items: []api.Pod{}},
			succeed: true,
		},
		"normal": {
			in: tools.EtcdResponseWithError{
				R: normalListResp,
				E: nil,
			},
			m:       EverythingMatcher{},
			out:     &api.PodList{Items: []api.Pod{*podA, *podB}},
			succeed: true,
		},
		"normalFiltered": {
			in: tools.EtcdResponseWithError{
				R: normalListResp,
				E: nil,
			},
			m:       SetMatcher{util.NewStringSet("foo")},
			out:     &api.PodList{Items: []api.Pod{*podA}},
			succeed: true,
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestGenericEtcdRegistry(t)
		fakeClient.Data[registry.KeyRoot] = item.in
		list, err := registry.List(api.NewContext(), item.m)
		if e, a := item.succeed, err == nil; e != a {
			t.Errorf("%v: expected %v, got %v", name, e, a)
			continue
		}

		if e, a := item.out, list; !reflect.DeepEqual(e, a) {
			t.Errorf("%v: Expected %#v, got %#v", name, e, a)
		}
	}
}

func TestEtcdCreate(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta:   api.ObjectMeta{Name: "foo"},
		DesiredState: api.PodState{Host: "machine"},
	}
	podB := &api.Pod{
		ObjectMeta:   api.ObjectMeta{Name: "foo"},
		DesiredState: api.PodState{Host: "machine2"},
	}

	nodeWithPodA := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), podA),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
		E: nil,
	}

	emptyNode := tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}

	path := "/registry/pods/foo"
	key := "foo"

	table := map[string]struct {
		existing tools.EtcdResponseWithError
		expect   tools.EtcdResponseWithError
		toCreate runtime.Object
		errOK    func(error) bool
	}{
		"normal": {
			existing: emptyNode,
			expect:   nodeWithPodA,
			toCreate: podA,
			errOK:    func(err error) bool { return err == nil },
		},
		"preExisting": {
			existing: nodeWithPodA,
			expect:   nodeWithPodA,
			toCreate: podB,
			errOK:    errors.IsAlreadyExists,
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestGenericEtcdRegistry(t)
		fakeClient.Data[path] = item.existing
		err := registry.Create(api.NewContext(), key, item.toCreate)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		if e, a := item.expect, fakeClient.Data[path]; !reflect.DeepEqual(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
		}
	}
}

func TestEtcdUpdate(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta:   api.ObjectMeta{Name: "foo"},
		DesiredState: api.PodState{Host: "machine"},
	}
	podB := &api.Pod{
		ObjectMeta:   api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		DesiredState: api.PodState{Host: "machine2"},
	}

	nodeWithPodA := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), podA),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
		E: nil,
	}

	nodeWithPodB := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), podB),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
		E: nil,
	}

	emptyNode := tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}

	path := "/registry/pods/foo"
	key := "foo"

	table := map[string]struct {
		existing tools.EtcdResponseWithError
		expect   tools.EtcdResponseWithError
		toUpdate runtime.Object
		errOK    func(error) bool
	}{
		"normal": {
			existing: nodeWithPodA,
			expect:   nodeWithPodB,
			toUpdate: podB,
			errOK:    func(err error) bool { return err == nil },
		},
		"notExisting": {
			existing: emptyNode,
			expect:   nodeWithPodA,
			toUpdate: podA,
			// TODO: Should updating a non-existing thing fail?
			errOK: func(err error) bool { return err == nil },
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestGenericEtcdRegistry(t)
		fakeClient.Data[path] = item.existing
		err := registry.Update(api.NewContext(), key, item.toUpdate)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		if e, a := item.expect, fakeClient.Data[path]; !reflect.DeepEqual(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
		}
	}
}

func TestEtcdGet(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta:   api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		DesiredState: api.PodState{Host: "machine"},
	}

	nodeWithPodA := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), podA),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
		E: nil,
	}

	emptyNode := tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}

	path := "/registry/pods/foo"
	key := "foo"

	table := map[string]struct {
		existing tools.EtcdResponseWithError
		expect   runtime.Object
		errOK    func(error) bool
	}{
		"normal": {
			existing: nodeWithPodA,
			expect:   podA,
			errOK:    func(err error) bool { return err == nil },
		},
		"notExisting": {
			existing: emptyNode,
			expect:   nil,
			errOK:    errors.IsNotFound,
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestGenericEtcdRegistry(t)
		fakeClient.Data[path] = item.existing
		got, err := registry.Get(api.NewContext(), key)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		if e, a := item.expect, got; !reflect.DeepEqual(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
		}
	}
}

func TestEtcdDelete(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta:   api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		DesiredState: api.PodState{Host: "machine"},
	}

	nodeWithPodA := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Codec(), podA),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
		E: nil,
	}

	emptyNode := tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}

	path := "/registry/pods/foo"
	key := "foo"

	table := map[string]struct {
		existing tools.EtcdResponseWithError
		expect   tools.EtcdResponseWithError
		errOK    func(error) bool
	}{
		"normal": {
			existing: nodeWithPodA,
			expect:   emptyNode,
			errOK:    func(err error) bool { return err == nil },
		},
		"notExisting": {
			existing: emptyNode,
			expect:   emptyNode,
			errOK:    func(err error) bool { return err == nil },
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestGenericEtcdRegistry(t)
		fakeClient.Data[path] = item.existing
		err := registry.Delete(api.NewContext(), key)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		if e, a := item.expect, fakeClient.Data[path]; !reflect.DeepEqual(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
		}
	}
}

func TestEtcdWatch(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta:   api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		DesiredState: api.PodState{Host: "machine"},
	}
	respWithPodA := &etcd.Response{
		Node: &etcd.Node{
			Value:         runtime.EncodeOrDie(testapi.Codec(), podA),
			ModifiedIndex: 1,
			CreatedIndex:  1,
		},
		Action: "create",
	}

	fakeClient, registry := NewTestGenericEtcdRegistry(t)
	wi, err := registry.Watch(api.NewContext(), EverythingMatcher{}, 1)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	fakeClient.WaitForWatchCompletion()

	go func() {
		fakeClient.WatchResponse <- respWithPodA
	}()

	got, open := <-wi.ResultChan()
	if !open {
		t.Fatalf("unexpected channel close")
	}

	if e, a := podA, got.Object; !reflect.DeepEqual(e, a) {
		t.Errorf("difference: %s", util.ObjectDiff(e, a))
	}
}
