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
	"reflect"
	"strconv"
	"testing"

	"sync"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/fields"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/runtime"
	etcdstorage "k8s.io/kubernetes/pkg/storage/etcd"
	"k8s.io/kubernetes/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/kubernetes/pkg/storage/etcd/testing"
	storagetesting "k8s.io/kubernetes/pkg/storage/testing"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/util/validation/field"
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

func (t *testRESTStrategy) PrepareForCreate(obj runtime.Object) {
	metaObj, err := meta.Accessor(obj)
	if err != nil {
		panic(err.Error())
	}
	labels := metaObj.GetLabels()
	if labels == nil {
		labels = map[string]string{}
	}
	labels["prepare_create"] = "true"
	metaObj.SetLabels(labels)
}

func (t *testRESTStrategy) PrepareForUpdate(obj, old runtime.Object) {}
func (t *testRESTStrategy) Validate(ctx api.Context, obj runtime.Object) field.ErrorList {
	return nil
}
func (t *testRESTStrategy) ValidateUpdate(ctx api.Context, obj, old runtime.Object) field.ErrorList {
	return nil
}
func (t *testRESTStrategy) Canonicalize(obj runtime.Object) {}

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

func NewTestGenericEtcdRegistry(t *testing.T) (*etcdtesting.EtcdTestServer, *Etcd) {
	podPrefix := "/pods"
	server := etcdtesting.NewEtcdTestClientServer(t)
	s := etcdstorage.NewEtcdStorage(server.Client, testapi.Default.Codec(), etcdtest.PathPrefix(), false)
	strategy := &testRESTStrategy{api.Scheme, api.SimpleNameGenerator, true, false, true}

	return server, &Etcd{
		NewFunc:           func() runtime.Object { return &api.Pod{} },
		NewListFunc:       func() runtime.Object { return &api.PodList{} },
		QualifiedResource: api.Resource("pods"),
		CreateStrategy:    strategy,
		UpdateStrategy:    strategy,
		KeyRootFunc: func(ctx api.Context) string {
			return podPrefix
		},
		KeyFunc: func(ctx api.Context, id string) (string, error) {
			if _, ok := api.NamespaceFrom(ctx); !ok {
				return "", fmt.Errorf("namespace is required")
			}
			return path.Join(podPrefix, id), nil
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) { return obj.(*api.Pod).Name, nil },
		PredicateFunc: func(label labels.Selector, field fields.Selector) generic.Matcher {
			return &generic.SelectionPredicate{
				Label: label,
				Field: field,
				GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
					pod, ok := obj.(*api.Pod)
					if !ok {
						return nil, nil, fmt.Errorf("not a pod")
					}
					return labels.Set(pod.ObjectMeta.Labels), generic.ObjectMetaFieldsSet(pod.ObjectMeta, true), nil
				},
			}
		},
		Storage: s,
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
		ObjectMeta: api.ObjectMeta{Namespace: "test", Name: "bar"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}
	podB := &api.Pod{
		ObjectMeta: api.ObjectMeta{Namespace: "test", Name: "foo"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}

	testContext := api.WithNamespace(api.NewContext(), "test")
	noNamespaceContext := api.NewContext()

	table := map[string]struct {
		in      *api.PodList
		m       generic.Matcher
		out     runtime.Object
		context api.Context
	}{
		"notFound": {
			in:  nil,
			m:   everythingMatcher{},
			out: &api.PodList{Items: []api.Pod{}},
		},
		"normal": {
			in:  &api.PodList{Items: []api.Pod{*podA, *podB}},
			m:   everythingMatcher{},
			out: &api.PodList{Items: []api.Pod{*podA, *podB}},
		},
		"normalFiltered": {
			in:  &api.PodList{Items: []api.Pod{*podA, *podB}},
			m:   setMatcher{sets.NewString("foo")},
			out: &api.PodList{Items: []api.Pod{*podB}},
		},
		"normalFilteredNoNamespace": {
			in:      &api.PodList{Items: []api.Pod{*podA, *podB}},
			m:       setMatcher{sets.NewString("foo")},
			out:     &api.PodList{Items: []api.Pod{*podB}},
			context: noNamespaceContext,
		},
		"normalFilteredMatchMultiple": {
			in:  &api.PodList{Items: []api.Pod{*podA, *podB}},
			m:   setMatcher{sets.NewString("foo", "makeMatchSingleReturnFalse")},
			out: &api.PodList{Items: []api.Pod{*podB}},
		},
	}

	for name, item := range table {
		ctx := testContext
		if item.context != nil {
			ctx = item.context
		}
		server, registry := NewTestGenericEtcdRegistry(t)

		if item.in != nil {
			if err := storagetesting.CreateList("/pods", registry.Storage, item.in); err != nil {
				t.Errorf("Unexpected error %v", err)
			}
		}

		list, err := registry.ListPredicate(ctx, item.m, nil)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
			continue
		}

		// DeepDerivative e,a is needed here b/c the storage layer sets ResourceVersion
		if e, a := item.out, list; !api.Semantic.DeepDerivative(e, a) {
			t.Errorf("%v: Expected %#v, got %#v", name, e, a)
		}
		server.Terminate(t)
	}
}

func TestEtcdCreate(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}
	podB := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec:       api.PodSpec{NodeName: "machine2"},
	}

	testContext := api.WithNamespace(api.NewContext(), "test")
	server, registry := NewTestGenericEtcdRegistry(t)
	defer server.Terminate(t)

	// create the object
	objA, err := registry.Create(testContext, podA)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// get the object
	checkobj, err := registry.Get(testContext, podA.Name)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// verify objects are equal
	if e, a := objA, checkobj; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}

	// now try to create the second pod
	_, err = registry.Create(testContext, podB)
	if !errors.IsAlreadyExists(err) {
		t.Errorf("Unexpected error: %v", err)
	}
}

func updateAndVerify(t *testing.T, ctx api.Context, registry *Etcd, pod *api.Pod) bool {
	obj, _, err := registry.Update(ctx, pod)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
		return false
	}
	checkObj, err := registry.Get(ctx, pod.Name)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
		return false
	}
	if e, a := obj, checkObj; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
		return false
	}
	return true
}

func TestEtcdUpdate(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}
	podB := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec:       api.PodSpec{NodeName: "machine2"},
	}
	podAWithResourceVersion := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "7"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}

	testContext := api.WithNamespace(api.NewContext(), "test")
	server, registry := NewTestGenericEtcdRegistry(t)
	defer server.Terminate(t)

	// Test1 try to update a non-existing node
	_, _, err := registry.Update(testContext, podA)
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	// Test2 createIfNotFound and verify
	registry.UpdateStrategy.(*testRESTStrategy).allowCreateOnUpdate = true
	if !updateAndVerify(t, testContext, registry, podA) {
		t.Errorf("Unexpected error updating podA")
	}
	registry.UpdateStrategy.(*testRESTStrategy).allowCreateOnUpdate = false

	// Test3 outofDate
	_, _, err = registry.Update(testContext, podAWithResourceVersion)
	if !errors.IsConflict(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	// Test4 normal update and verify
	if !updateAndVerify(t, testContext, registry, podB) {
		t.Errorf("Unexpected error updating podB")
	}

	// Test5 unconditional update
	// NOTE: The logic for unconditional updates doesn't make sense to me, and imho should be removed.
	// doUnconditionalUpdate := resourceVersion == 0 && e.UpdateStrategy.AllowUnconditionalUpdate()
	// ^^ That condition can *never be true due to the creation of root objects.
	//
	// registry.UpdateStrategy.(*testRESTStrategy).allowUnconditionalUpdate = true
	// updateAndVerify(t, testContext, registry, podAWithResourceVersion)

}

func TestNoOpUpdates(t *testing.T) {
	server, registry := NewTestGenericEtcdRegistry(t)
	defer server.Terminate(t)

	newPod := func() *api.Pod {
		return &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Namespace: api.NamespaceDefault,
				Name:      "foo",
				Labels:    map[string]string{"prepare_create": "true"},
			},
			Spec: api.PodSpec{NodeName: "machine"},
		}
	}

	var err error
	var createResult runtime.Object
	if createResult, err = registry.Create(api.NewDefaultContext(), newPod()); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	createdPod, err := registry.Get(api.NewDefaultContext(), "foo")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	var updateResult runtime.Object
	if updateResult, _, err = registry.Update(api.NewDefaultContext(), newPod()); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Check whether we do not return empty result on no-op update.
	if !reflect.DeepEqual(createResult, updateResult) {
		t.Errorf("no-op update should return a correct value, got: %#v", updateResult)
	}

	updatedPod, err := registry.Get(api.NewDefaultContext(), "foo")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	createdMeta, err := meta.Accessor(createdPod)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	updatedMeta, err := meta.Accessor(updatedPod)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	if createdMeta.GetResourceVersion() != updatedMeta.GetResourceVersion() {
		t.Errorf("no-op update should be ignored and not written to etcd")
	}
}

// TODO: Add a test to check no-op update if we have object with ResourceVersion
// already stored in etcd. Currently there is no easy way to store object with
// ResourceVersion in etcd.

type testPodExport struct{}

func (t testPodExport) Export(obj runtime.Object, exact bool) error {
	pod := obj.(*api.Pod)
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	pod.Labels["exported"] = "true"
	pod.Labels["exact"] = strconv.FormatBool(exact)

	return nil
}

func TestEtcdCustomExport(t *testing.T) {
	podA := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Namespace: "test",
			Name:      "foo",
			Labels:    map[string]string{},
		},
		Spec: api.PodSpec{NodeName: "machine"},
	}

	server, registry := NewTestGenericEtcdRegistry(t)
	defer server.Terminate(t)

	registry.ExportStrategy = testPodExport{}

	testContext := api.WithNamespace(api.NewContext(), "test")
	registry.UpdateStrategy.(*testRESTStrategy).allowCreateOnUpdate = true
	if !updateAndVerify(t, testContext, registry, &podA) {
		t.Errorf("Unexpected error updating podA")
	}

	obj, err := registry.Export(testContext, podA.Name, unversioned.ExportOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	exportedPod := obj.(*api.Pod)
	if exportedPod.Labels["exported"] != "true" {
		t.Errorf("expected: exported->true, found: %s", exportedPod.Labels["exported"])
	}
	if exportedPod.Labels["exact"] != "false" {
		t.Errorf("expected: exact->false, found: %s", exportedPod.Labels["exact"])
	}
	delete(exportedPod.Labels, "exported")
	delete(exportedPod.Labels, "exact")
	exportObjectMeta(&podA.ObjectMeta, false)
	podA.Spec = exportedPod.Spec
	if !reflect.DeepEqual(&podA, exportedPod) {
		t.Errorf("expected:\n%v\nsaw:\n%v\n", &podA, exportedPod)
	}
}

func TestEtcdBasicExport(t *testing.T) {
	podA := api.Pod{
		ObjectMeta: api.ObjectMeta{
			Namespace: "test",
			Name:      "foo",
			Labels:    map[string]string{},
		},
		Spec:   api.PodSpec{NodeName: "machine"},
		Status: api.PodStatus{HostIP: "1.2.3.4"},
	}

	server, registry := NewTestGenericEtcdRegistry(t)
	defer server.Terminate(t)

	testContext := api.WithNamespace(api.NewContext(), "test")
	registry.UpdateStrategy.(*testRESTStrategy).allowCreateOnUpdate = true
	if !updateAndVerify(t, testContext, registry, &podA) {
		t.Errorf("Unexpected error updating podA")
	}

	obj, err := registry.Export(testContext, podA.Name, unversioned.ExportOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	exportedPod := obj.(*api.Pod)
	if exportedPod.Labels["prepare_create"] != "true" {
		t.Errorf("expected: prepare_create->true, found: %s", exportedPod.Labels["prepare_create"])
	}
	exportObjectMeta(&podA.ObjectMeta, false)
	podA.Spec = exportedPod.Spec
	if !reflect.DeepEqual(&podA, exportedPod) {
		t.Errorf("expected:\n%v\nsaw:\n%v\n", &podA, exportedPod)
	}
}

func TestEtcdGet(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta: api.ObjectMeta{Namespace: "test", Name: "foo"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}

	testContext := api.WithNamespace(api.NewContext(), "test")
	server, registry := NewTestGenericEtcdRegistry(t)
	defer server.Terminate(t)

	_, err := registry.Get(testContext, podA.Name)
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	registry.UpdateStrategy.(*testRESTStrategy).allowCreateOnUpdate = true
	if !updateAndVerify(t, testContext, registry, podA) {
		t.Errorf("Unexpected error updating podA")
	}
}

func TestEtcdDelete(t *testing.T) {
	podA := &api.Pod{
		ObjectMeta: api.ObjectMeta{Name: "foo"},
		Spec:       api.PodSpec{NodeName: "machine"},
	}

	testContext := api.WithNamespace(api.NewContext(), "test")
	server, registry := NewTestGenericEtcdRegistry(t)
	defer server.Terminate(t)

	// test failure condition
	_, err := registry.Delete(testContext, podA.Name, nil)
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	// create pod
	_, err = registry.Create(testContext, podA)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// delete object
	_, err = registry.Delete(testContext, podA.Name, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// try to get a item which should be deleted
	_, err = registry.Get(testContext, podA.Name)
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestEtcdDeleteCollection(t *testing.T) {
	podA := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	podB := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "bar"}}

	testContext := api.WithNamespace(api.NewContext(), "test")
	server, registry := NewTestGenericEtcdRegistry(t)
	defer server.Terminate(t)

	if _, err := registry.Create(testContext, podA); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if _, err := registry.Create(testContext, podB); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Delete all pods.
	deleted, err := registry.DeleteCollection(testContext, nil, &api.ListOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	deletedPods := deleted.(*api.PodList)
	if len(deletedPods.Items) != 2 {
		t.Errorf("Unexpected number of pods deleted: %d, expected: 2", len(deletedPods.Items))
	}

	if _, err := registry.Get(testContext, podA.Name); !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	if _, err := registry.Get(testContext, podB.Name); !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestEtcdDeleteCollectionNotFound(t *testing.T) {
	server, registry := NewTestGenericEtcdRegistry(t)
	defer server.Terminate(t)

	testContext := api.WithNamespace(api.NewContext(), "test")

	podA := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}
	podB := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "bar"}}

	for i := 0; i < 10; i++ {
		// Setup
		if _, err := registry.Create(testContext, podA); err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if _, err := registry.Create(testContext, podB); err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		// Kick off multiple delete collection calls to test notfound behavior
		wg := &sync.WaitGroup{}
		for j := 0; j < 2; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, err := registry.DeleteCollection(testContext, nil, &api.ListOptions{})
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
			}()
		}
		wg.Wait()

		if _, err := registry.Get(testContext, podA.Name); !errors.IsNotFound(err) {
			t.Errorf("Unexpected error: %v", err)
		}
		if _, err := registry.Get(testContext, podB.Name); !errors.IsNotFound(err) {
			t.Errorf("Unexpected error: %v", err)
		}
	}
}

// Test whether objects deleted with DeleteCollection are correctly delivered
// to watchers.
func TestEtcdDeleteCollectionWithWatch(t *testing.T) {
	podA := &api.Pod{ObjectMeta: api.ObjectMeta{Name: "foo"}}

	testContext := api.WithNamespace(api.NewContext(), "test")
	server, registry := NewTestGenericEtcdRegistry(t)
	defer server.Terminate(t)

	objCreated, err := registry.Create(testContext, podA)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	podCreated := objCreated.(*api.Pod)

	watcher, err := registry.WatchPredicate(testContext, setMatcher{sets.NewString("foo")}, podCreated.ResourceVersion)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watcher.Stop()

	if _, err := registry.DeleteCollection(testContext, nil, &api.ListOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	got, open := <-watcher.ResultChan()
	if !open {
		t.Errorf("Unexpected channel close")
	} else {
		if got.Type != "DELETED" {
			t.Errorf("Unexpected event type: %s", got.Type)
		}
		gotObject := got.Object.(*api.Pod)
		gotObject.ResourceVersion = podCreated.ResourceVersion
		if e, a := podCreated, gotObject; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected: %#v, got: %#v", e, a)
		}
	}
}

func TestEtcdWatch(t *testing.T) {
	testContext := api.WithNamespace(api.NewContext(), "test")
	noNamespaceContext := api.NewContext()

	table := map[string]struct {
		generic.Matcher
		context api.Context
	}{
		"single": {
			Matcher: setMatcher{sets.NewString("foo")},
		},
		"multi": {
			Matcher: setMatcher{sets.NewString("foo", "bar")},
		},
		"singleNoNamespace": {
			Matcher: setMatcher{sets.NewString("foo")},
			context: noNamespaceContext,
		},
	}

	for name, m := range table {
		ctx := testContext
		if m.context != nil {
			ctx = m.context
		}
		podA := &api.Pod{
			ObjectMeta: api.ObjectMeta{
				Name:      "foo",
				Namespace: "test",
			},
			Spec: api.PodSpec{NodeName: "machine"},
		}

		server, registry := NewTestGenericEtcdRegistry(t)
		wi, err := registry.WatchPredicate(ctx, m, "0")
		if err != nil {
			t.Errorf("%v: unexpected error: %v", name, err)
		} else {
			obj, err := registry.Create(testContext, podA)
			if err != nil {
				got, open := <-wi.ResultChan()
				if !open {
					t.Errorf("%v: unexpected channel close", name)
				} else {
					if e, a := obj, got.Object; !reflect.DeepEqual(e, a) {
						t.Errorf("Expected %#v, got %#v", e, a)
					}
				}
			}
			wi.Stop()
		}

		server.Terminate(t)
	}
}
