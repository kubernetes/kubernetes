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
	"fmt"
	"path"
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/tools"
	"k8s.io/kubernetes/pkg/tools/etcdtest"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/util/fielderrors"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/coreos/go-etcd/etcd"
)

type testRESTStrategy struct {
	runtime.ObjectTyper
	api.NameGenerator
	namespaceScoped          bool
	allowCreateOnUpdate      bool
	allowUnconditionalUpdate bool
}

func (t *testRESTStrategy) NamespaceScoped() bool          { return t.namespaceScoped }
func (t *testRESTStrategy) AllowCreateOnUpdate() bool      { return t.allowCreateOnUpdate }
func (t *testRESTStrategy) AllowUnconditionalUpdate() bool { return t.allowUnconditionalUpdate }

func (t *testRESTStrategy) PrepareForCreate(obj runtime.Object)      {}
func (t *testRESTStrategy) PrepareForUpdate(obj, old runtime.Object) {}
func (t *testRESTStrategy) Validate(ctx api.Context, obj runtime.Object) fielderrors.ValidationErrorList {
	return nil
}
func (t *testRESTStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) fielderrors.ValidationErrorList {
	return nil
}

func hasCreated(t *testing.T, pod *api.Pod) func(runtime.Object) bool {
	return func(obj runtime.Object) bool {
		actualPod := obj.(*api.Pod)
		if !api.Semantic.DeepDerivative(pod.Status, actualPod.Status) {
			t.Errorf("not a deep derivative %#v", actualPod)
			return false
		}
		return api.HasObjectMetaSystemFieldValues(&actualPod.ObjectMeta)
	}
}

func NewTestGenericEtcdRegistry(t *testing.T) (*tools.FakeEtcdClient, *Etcd) {
	f := tools.NewFakeEtcdClient(t)
	f.TestIndex = true
	s := etcdstorage.NewEtcdStorage(f, testapi.Default.Codec(), etcdtest.PathPrefix())
	strategy := &testRESTStrategy{api.Scheme, api.SimpleNameGenerator, true, false, true}
	podPrefix := "/pods"
	return f, &Etcd{
		NewFunc:        func() runtime.Object { return &api.Pod{} },
		NewListFunc:    func() runtime.Object { return &api.PodList{} },
		EndpointName:   "pods",
		CreateStrategy: strategy,
		UpdateStrategy: strategy,
		KeyRootFunc: func(ctx api.Context) string {
			return podPrefix
		},
		KeyFunc: func(ctx api.Context, id string) (string, error) {
			return path.Join(podPrefix, id), nil
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) { return obj.(*api.Pod).Name, nil },
		Storage:        s,
	}
}

// setMatcher is a matcher that matches any pod with id in the set.
// Makes testing simpler.
type setMatcher struct {
	sets.String
}

func (sm setMatcher) Matches(obj runtime.Object) (bool, error) {
	pod, ok := obj.(*api.Pod)
	if !ok {
		return false, fmt.Errorf("wrong object")
	}
	return sm.Has(pod.Name), nil
}

func (sm setMatcher) MatchesSingle() (string, bool) {
	if sm.Len() == 1 {
		// Since pod name is its key, we can optimize this case.
		return sm.List()[0], true
	}
	return "", false
}

// everythingMatcher matches everything
type everythingMatcher struct{}

func (everythingMatcher) Matches(obj runtime.Object) (bool, error) {
	return true, nil
}

func (everythingMatcher) MatchesSingle() (string, bool) {
	return "", false
}

func TestEtcdList(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}
	podB := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "bar"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}

	singleElemListResp := &etcd.Response{
		Node: &etcd.Node{
			Value: runtime.EncodeOrDie(testapi.Default.Codec(), podA),
		},
	}
	normalListResp := &etcd.Response{
		Node: &etcd.Node{
			Nodes: []*etcd.Node{
				{Value: runtime.EncodeOrDie(testapi.Default.Codec(), podA)},
				{Value: runtime.EncodeOrDie(testapi.Default.Codec(), podB)},
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
			m:       everythingMatcher{},
			out:     &api.PodList{Items: []api.Pod{}},
			succeed: true,
		},
		"notFound": {
			in: tools.EtcdResponseWithError{
				R: &etcd.Response{},
				E: tools.EtcdErrorNotFound,
			},
			m:       everythingMatcher{},
			out:     &api.PodList{Items: []api.Pod{}},
			succeed: true,
		},
		"normal": {
			in: tools.EtcdResponseWithError{
				R: normalListResp,
				E: nil,
			},
			m:       everythingMatcher{},
			out:     &api.PodList{Items: []api.Pod{*podA, *podB}},
			succeed: true,
		},
		"normalFiltered": {
			in: tools.EtcdResponseWithError{
				R: singleElemListResp,
				E: nil,
			},
			m:       setMatcher{sets.NewString("foo")},
			out:     &api.PodList{Items: []api.Pod{*podA}},
			succeed: true,
		},
		"normalFilteredMatchMultiple": {
			in: tools.EtcdResponseWithError{
				R: normalListResp,
				E: nil,
			},
			m:       setMatcher{sets.NewString("foo", "makeMatchSingleReturnFalse")},
			out:     &api.PodList{Items: []api.Pod{*podA}},
			succeed: true,
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestGenericEtcdRegistry(t)
		if name, ok := item.m.MatchesSingle(); ok {
			key, err := registry.KeyFunc(api.NewContext(), name)
			if err != nil {
				t.Errorf("Couldn't create key for %v", name)
				continue
			}
			key = etcdtest.AddPrefix(key)
			fakeClient.Data[key] = item.in
		} else {
			key, _ := registry.KeyFunc(api.NewContext(), name)
			key = etcdtest.AddPrefix(key)
			fakeClient.Data[key] = item.in
		}
		list, err := registry.ListPredicate(api.NewContext(), item.m)
		if e, a := item.succeed, err == nil; e != a {
			t.Errorf("%v: expected %v, got %v", name, e, a)
			continue
		}
		if e, a := item.out, list; !api.Semantic.DeepDerivative(e, a) {
			t.Errorf("%v: Expected %#v, got %#v", name, e, a)
		}
	}
}

func TestEtcdCreate(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault},
		Spec:       api.PodSpec{NodeName: "machine"},
	}
	podB := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault},
		Spec:       api.PodSpec{NodeName: "machine2"},
	}

	nodeWithPodA := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Default.Codec(), podA),
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

	table := map[string]struct {
		existing tools.EtcdResponseWithError
		expect   tools.EtcdResponseWithError
		toCreate runtime.Object
		objOK    func(obj runtime.Object) bool
		errOK    func(error) bool
	}{
		"normal": {
			existing: emptyNode,
			toCreate: podA,
			objOK:    hasCreated(t, podA),
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
		path := etcdtest.AddPrefix("pods/foo")
		fakeClient.Data[path] = item.existing
		obj, err := registry.Create(api.NewDefaultContext(), item.toCreate)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		actual := fakeClient.Data[path]
		if item.objOK != nil {
			if !item.objOK(obj) {
				t.Errorf("%v: unexpected returned: %v", name, obj)
			}
			actualObj, err := api.Scheme.Decode([]byte(actual.R.Node.Value))
			if err != nil {
				t.Errorf("unable to decode stored value for %#v", actual)
				continue
			}
			if !item.objOK(actualObj) {
				t.Errorf("%v: unexpected response: %v", name, actual)
			}
		} else {
			if e, a := item.expect, actual; !api.Semantic.DeepDerivative(e, a) {
				t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
			}
		}
	}
}

func TestEtcdUpdate(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault},
		Spec:       api.PodSpec{NodeName: "machine"},
	}
	podB := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault, ResourceVersion: "1"},
		Spec:       api.PodSpec{NodeName: "machine2"},
	}
	podAWithResourceVersion := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: api.NamespaceDefault, ResourceVersion: "3"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}
	nodeWithPodA := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Default.Codec(), podA),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
		E: nil,
	}

	newerNodeWithPodA := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Default.Codec(), podA),
				ModifiedIndex: 2,
				CreatedIndex:  1,
			},
		},
		E: nil,
	}

	nodeWithPodB := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Default.Codec(), podB),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
		},
		E: nil,
	}

	nodeWithPodAWithResourceVersion := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Default.Codec(), podAWithResourceVersion),
				ModifiedIndex: 3,
				CreatedIndex:  1,
			},
		},
		E: nil,
	}
	emptyNode := tools.EtcdResponseWithError{
		R: &etcd.Response{},
		E: tools.EtcdErrorNotFound,
	}

	table := map[string]struct {
		existing                 tools.EtcdResponseWithError
		expect                   tools.EtcdResponseWithError
		toUpdate                 runtime.Object
		allowCreate              bool
		allowUnconditionalUpdate bool
		objOK                    func(obj runtime.Object) bool
		errOK                    func(error) bool
	}{
		"normal": {
			existing: nodeWithPodA,
			expect:   nodeWithPodB,
			toUpdate: podB,
			errOK:    func(err error) bool { return err == nil },
		},
		"notExisting": {
			existing: emptyNode,
			expect:   emptyNode,
			toUpdate: podA,
			errOK:    func(err error) bool { return errors.IsNotFound(err) },
		},
		"createIfNotFound": {
			existing:    emptyNode,
			toUpdate:    podA,
			allowCreate: true,
			objOK:       hasCreated(t, podA),
			errOK:       func(err error) bool { return err == nil },
		},
		"outOfDate": {
			existing: newerNodeWithPodA,
			expect:   newerNodeWithPodA,
			toUpdate: podB,
			errOK:    func(err error) bool { return errors.IsConflict(err) },
		},
		"unconditionalUpdate": {
			existing:                 nodeWithPodAWithResourceVersion,
			allowUnconditionalUpdate: true,
			toUpdate:                 podA,
			objOK:                    func(obj runtime.Object) bool { return true },
			errOK:                    func(err error) bool { return err == nil },
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestGenericEtcdRegistry(t)
		registry.UpdateStrategy.(*testRESTStrategy).allowCreateOnUpdate = item.allowCreate
		registry.UpdateStrategy.(*testRESTStrategy).allowUnconditionalUpdate = item.allowUnconditionalUpdate
		path := etcdtest.AddPrefix("pods/foo")
		fakeClient.Data[path] = item.existing
		obj, _, err := registry.Update(api.NewDefaultContext(), item.toUpdate)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		actual := fakeClient.Data[path]
		if item.objOK != nil {
			if !item.objOK(obj) {
				t.Errorf("%v: unexpected returned: %#v", name, obj)
			}
			actualObj, err := api.Scheme.Decode([]byte(actual.R.Node.Value))
			if err != nil {
				t.Errorf("unable to decode stored value for %#v", actual)
				continue
			}
			if !item.objOK(actualObj) {
				t.Errorf("%v: unexpected response: %#v", name, actual)
			}
		} else {
			if e, a := item.expect, actual; !api.Semantic.DeepDerivative(e, a) {
				t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
			}
		}
	}
}

func TestEtcdGet(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}

	nodeWithPodA := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Default.Codec(), podA),
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
		path := etcdtest.AddPrefix("pods/foo")
		fakeClient.Data[path] = item.existing
		got, err := registry.Get(api.NewContext(), key)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v", name, err)
		}

		if e, a := item.expect, got; !api.Semantic.DeepDerivative(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
		}
	}
}

func TestEtcdDelete(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", ResourceVersion: "1"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}

	nodeWithPodA := tools.EtcdResponseWithError{
		R: &etcd.Response{
			Node: &etcd.Node{
				Value:         runtime.EncodeOrDie(testapi.Default.Codec(), podA),
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
			errOK:    func(err error) bool { return errors.IsNotFound(err) },
		},
	}

	for name, item := range table {
		fakeClient, registry := NewTestGenericEtcdRegistry(t)
		path := etcdtest.AddPrefix("pods/foo")
		fakeClient.Data[path] = item.existing
		obj, err := registry.Delete(api.NewContext(), key, nil)
		if !item.errOK(err) {
			t.Errorf("%v: unexpected error: %v (%#v)", name, err, obj)
		}

		if item.expect.E != nil {
			item.expect.E.(*etcd.EtcdError).Index = fakeClient.ChangeIndex
		}
		if e, a := item.expect, fakeClient.Data[path]; !api.Semantic.DeepDerivative(e, a) {
			t.Errorf("%v:\n%s", name, util.ObjectDiff(e, a))
		}
	}
}

func TestEtcdWatch(t *testing.T) {
	table := map[string]generic.Matcher{
		"single": setMatcher{sets.NewString("foo")},
		"multi":  setMatcher{sets.NewString("foo", "bar")},
	}

	for name, m := range table {
		podA := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:            "foo",
				Namespace:       api.NamespaceDefault,
				ResourceVersion: "1",
			},
			Spec: api.PodSpec{NodeName: "machine"},
		}
		respWithPodA := &etcd.Response{
			Node: &etcd.Node{
				Key:           "/registry/pods/default/foo",
				Value:         runtime.EncodeOrDie(testapi.Default.Codec(), podA),
				ModifiedIndex: 1,
				CreatedIndex:  1,
			},
			Action: "create",
		}

		fakeClient, registry := NewTestGenericEtcdRegistry(t)
		wi, err := registry.WatchPredicate(api.NewContext(), m, "1")
		if err != nil {
			t.Errorf("%v: unexpected error: %v", name, err)
			continue
		}
		fakeClient.WaitForWatchCompletion()

		go func() {
			fakeClient.WatchResponse <- respWithPodA
		}()

		got, open := <-wi.ResultChan()
		if !open {
			t.Errorf("%v: unexpected channel close", name)
			continue
		}

		if e, a := podA, got.Object; !api.Semantic.DeepDerivative(e, a) {
			t.Errorf("%v: difference: %s", name, util.ObjectDiff(e, a))
		}
	}
}
