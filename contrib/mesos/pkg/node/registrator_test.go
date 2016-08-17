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

package node

import (
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/unversioned"
	unversionedcore "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/unversioned/fake"
	"k8s.io/kubernetes/pkg/client/testing/core"
	"k8s.io/kubernetes/pkg/runtime"
)

type fakeNodes struct {
	*fake.FakeNodes
}

func (f *fakeNodes) Nodes() unversionedcore.NodeInterface {
	return f
}

func calledOnce(h bool, ret runtime.Object, err error) (<-chan struct{}, func(core.Action) (bool, runtime.Object, error)) {
	ch := make(chan struct{})
	return ch, func(_ core.Action) (bool, runtime.Object, error) {
		select {
		case <-ch:
			panic("called more than once")
		default:
			close(ch)
		}
		return h, ret, err
	}
}

func TestRegister_withUnknownNode(t *testing.T) {
	fc := &core.Fake{}
	nodes := &fakeNodes{&fake.FakeNodes{Fake: &fake.FakeCore{Fake: fc}}}
	createCalled, createOnce := calledOnce(true, nil, nil)
	fc.AddReactor("create", "nodes", createOnce)

	lookup := func(hostName string) *api.Node {
		select {
		case <-createCalled:
			return &api.Node{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		default:
			return nil
		}
	}

	r := NewRegistrator(nodes, lookup)
	ch := make(chan struct{})
	defer close(ch)
	r.Run(ch)

	t.Logf("registering node foo")
	ok, err := r.Register("foo", nil)
	if !ok {
		t.Fatalf("registration failed without error")
	} else if err != nil {
		t.Fatalf("registration failed with error %v", err)
	}

	// wait for node creation
	t.Logf("awaiting node creation")
	<-createCalled
}

func TestRegister_withKnownNode(t *testing.T) {
	fc := &core.Fake{}
	nodes := &fakeNodes{&fake.FakeNodes{Fake: &fake.FakeCore{Fake: fc}}}
	updateCalled, updateOnce := calledOnce(true, nil, nil)
	fc.AddReactor("update", "nodes", updateOnce)

	lookup := func(hostName string) *api.Node {
		select {
		case <-updateCalled:
			return &api.Node{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		default:
			// this node needs an update because it has labels: the updated version doesn't
			return &api.Node{ObjectMeta: api.ObjectMeta{Name: "foo", Labels: map[string]string{"a": "b"}}}
		}
	}

	r := NewRegistrator(nodes, lookup)
	ch := make(chan struct{})
	defer close(ch)
	r.Run(ch)

	t.Logf("registering node foo")
	ok, err := r.Register("foo", nil)
	if !ok {
		t.Fatalf("registration failed without error")
	} else if err != nil {
		t.Fatalf("registration failed with error %v", err)
	}

	// wait for node update
	t.Logf("awaiting node update")
	<-updateCalled
}

func TestRegister_withSemiKnownNode(t *testing.T) {
	// semi-known because the lookup func doesn't see the a very newly created node
	// but our apiserver "create" call returns an already-exists error. in this case
	// CreateOrUpdate should proceed to attempt an update.

	fc := &core.Fake{}
	nodes := &fakeNodes{&fake.FakeNodes{Fake: &fake.FakeCore{Fake: fc}}}

	createCalled, createOnce := calledOnce(true, nil, errors.NewAlreadyExists(unversioned.GroupResource{Group: "", Resource: ""}, "nodes"))
	fc.AddReactor("create", "nodes", createOnce)

	updateCalled, updateOnce := calledOnce(true, nil, nil)
	fc.AddReactor("update", "nodes", updateOnce)

	lookup := func(hostName string) *api.Node {
		select {
		case <-updateCalled:
			return &api.Node{ObjectMeta: api.ObjectMeta{Name: "foo"}}
		default:
			// this makes the node semi-known: apiserver knows it but the store/cache doesn't
			return nil
		}
	}

	r := NewRegistrator(nodes, lookup)
	ch := make(chan struct{})
	defer close(ch)
	r.Run(ch)

	t.Logf("registering node foo")
	ok, err := r.Register("foo", nil)
	if !ok {
		t.Fatalf("registration failed without error")
	} else if err != nil {
		t.Fatalf("registration failed with error %v", err)
	}

	// wait for node update
	t.Logf("awaiting node update")
	<-createCalled
	<-updateCalled
}
