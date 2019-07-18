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

package registry

import (
	"context"
	"fmt"
	"path"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	apitesting "k8s.io/apimachinery/pkg/api/apitesting"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metainternalversion "k8s.io/apimachinery/pkg/apis/meta/internalversion"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/selection"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/registry/rest"
	"k8s.io/apiserver/pkg/storage"
	cacherstorage "k8s.io/apiserver/pkg/storage/cacher"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
)

var scheme = runtime.NewScheme()
var codecs = serializer.NewCodecFactory(scheme)

const validInitializerName = "test.k8s.io"

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
}

type testGracefulStrategy struct {
	testRESTStrategy
}

func (t testGracefulStrategy) CheckGracefulDelete(ctx context.Context, obj runtime.Object, options *metav1.DeleteOptions) bool {
	return true
}

var _ rest.RESTGracefulDeleteStrategy = testGracefulStrategy{}

type testOrphanDeleteStrategy struct {
	*testRESTStrategy
}

func (t *testOrphanDeleteStrategy) DefaultGarbageCollectionPolicy(ctx context.Context) rest.GarbageCollectionPolicy {
	return rest.OrphanDependents
}

type testRESTStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
	namespaceScoped          bool
	allowCreateOnUpdate      bool
	allowUnconditionalUpdate bool
}

func (t *testRESTStrategy) NamespaceScoped() bool          { return t.namespaceScoped }
func (t *testRESTStrategy) AllowCreateOnUpdate() bool      { return t.allowCreateOnUpdate }
func (t *testRESTStrategy) AllowUnconditionalUpdate() bool { return t.allowUnconditionalUpdate }

func (t *testRESTStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
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

func (t *testRESTStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {}
func (t *testRESTStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	return nil
}
func (t *testRESTStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return nil
}
func (t *testRESTStrategy) Canonicalize(obj runtime.Object) {}

func NewTestGenericStoreRegistry(t *testing.T) (factory.DestroyFunc, *Store) {
	return newTestGenericStoreRegistry(t, scheme, false)
}

func getPodAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	pod := obj.(*example.Pod)
	return labels.Set{"name": pod.ObjectMeta.Name}, nil, nil
}

// matchPodName returns selection predicate that matches any pod with name in the set.
// Makes testing simpler.
func matchPodName(names ...string) storage.SelectionPredicate {
	// Note: even if pod name is a field, we have to use labels,
	// because field selector doesn't support "IN" operator.
	l, err := labels.NewRequirement("name", selection.In, names)
	if err != nil {
		panic("Labels requirement must validate successfully")
	}
	return storage.SelectionPredicate{
		Label:    labels.Everything().Add(*l),
		Field:    fields.Everything(),
		GetAttrs: getPodAttrs,
	}
}

func matchEverything() storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label: labels.Everything(),
		Field: fields.Everything(),
		GetAttrs: func(obj runtime.Object) (label labels.Set, field fields.Set, err error) {
			return nil, nil, nil
		},
	}
}

func TestStoreList(t *testing.T) {
	podA := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "bar"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}
	podB := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	noNamespaceContext := genericapirequest.NewContext()

	table := map[string]struct {
		in      *example.PodList
		m       storage.SelectionPredicate
		out     runtime.Object
		context context.Context
	}{
		"notFound": {
			in:  nil,
			m:   matchEverything(),
			out: &example.PodList{Items: []example.Pod{}},
		},
		"normal": {
			in:  &example.PodList{Items: []example.Pod{*podA, *podB}},
			m:   matchEverything(),
			out: &example.PodList{Items: []example.Pod{*podA, *podB}},
		},
		"normalFiltered": {
			in:  &example.PodList{Items: []example.Pod{*podA, *podB}},
			m:   matchPodName("foo"),
			out: &example.PodList{Items: []example.Pod{*podB}},
		},
		"normalFilteredNoNamespace": {
			in:      &example.PodList{Items: []example.Pod{*podA, *podB}},
			m:       matchPodName("foo"),
			out:     &example.PodList{Items: []example.Pod{*podB}},
			context: noNamespaceContext,
		},
		"normalFilteredMatchMultiple": {
			in:  &example.PodList{Items: []example.Pod{*podA, *podB}},
			m:   matchPodName("foo", "makeMatchSingleReturnFalse"),
			out: &example.PodList{Items: []example.Pod{*podB}},
		},
	}

	for name, item := range table {
		ctx := testContext
		if item.context != nil {
			ctx = item.context
		}
		destroyFunc, registry := NewTestGenericStoreRegistry(t)

		if item.in != nil {
			if err := storagetesting.CreateList("/pods", registry.Storage.Storage, item.in); err != nil {
				t.Errorf("Unexpected error %v", err)
			}
		}

		list, err := registry.ListPredicate(ctx, item.m, nil)
		if err != nil {
			t.Errorf("Unexpected error %v", err)
			continue
		}

		// DeepDerivative e,a is needed here b/c the storage layer sets ResourceVersion
		if e, a := item.out, list; !apiequality.Semantic.DeepDerivative(e, a) {
			t.Errorf("%v: Expected %#v, got %#v", name, e, a)
		}
		destroyFunc()
	}
}

// TestStoreListResourceVersion tests that if List with ResourceVersion > 0, it will wait until
// the results are as fresh as given version.
func TestStoreListResourceVersion(t *testing.T) {
	fooPod := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}
	barPod := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "bar"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")

	destroyFunc, registry := newTestGenericStoreRegistry(t, scheme, true)
	defer destroyFunc()

	obj, err := registry.Create(ctx, fooPod, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	versioner := etcd3.APIObjectVersioner{}
	rev, err := versioner.ObjectResourceVersion(obj)
	if err != nil {
		t.Fatal(err)
	}

	waitListCh := make(chan runtime.Object, 1)
	go func(listRev uint64) {
		option := &metainternalversion.ListOptions{ResourceVersion: strconv.FormatUint(listRev, 10)}
		// It will wait until we create the second pod.
		l, err := registry.List(ctx, option)
		if err != nil {
			close(waitListCh)
			t.Fatal(err)
			return
		}
		waitListCh <- l
	}(rev + 1)

	select {
	case <-time.After(500 * time.Millisecond):
	case l := <-waitListCh:
		t.Fatalf("expected waiting, but get %#v", l)
	}

	if _, err := registry.Create(ctx, barPod, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
		t.Fatal(err)
	}

	select {
	case <-time.After(wait.ForeverTestTimeout):
		t.Fatalf("timeout after %v", wait.ForeverTestTimeout)
	case l, ok := <-waitListCh:
		if !ok {
			return
		}
		pl := l.(*example.PodList).Items
		if len(pl) != 2 {
			t.Errorf("Expected get 2 items, but got %d", len(pl))
		}
	}
}

func TestStoreCreate(t *testing.T) {
	gracefulPeriod := int64(50)
	podA := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}
	podB := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec:       example.PodSpec{NodeName: "machine2"},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()
	// re-define delete strategy to have graceful delete capability
	defaultDeleteStrategy := testRESTStrategy{scheme, names.SimpleNameGenerator, true, false, true}
	registry.DeleteStrategy = testGracefulStrategy{defaultDeleteStrategy}
	registry.Decorator = func(obj runtime.Object) error {
		pod := obj.(*example.Pod)
		pod.Status.Phase = example.PodPhase("Testing")
		return nil
	}

	// create the object with denying admission
	objA, err := registry.Create(testContext, podA, denyCreateValidation, &metav1.CreateOptions{})
	if err == nil {
		t.Errorf("Expected admission error: %v", err)
	}

	// create the object
	objA, err = registry.Create(testContext, podA, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// verify the decorator was called
	if objA.(*example.Pod).Status.Phase != example.PodPhase("Testing") {
		t.Errorf("Decorator was not called: %#v", objA)
	}

	// get the object
	checkobj, err := registry.Get(testContext, podA.Name, &metav1.GetOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// verify objects are equal
	if e, a := objA, checkobj; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected %#v, got %#v", e, a)
	}

	// now try to create the second pod
	_, err = registry.Create(testContext, podB, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if !errors.IsAlreadyExists(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	// verify graceful delete capability is defined
	_, ok := registry.DeleteStrategy.(rest.RESTGracefulDeleteStrategy)
	if !ok {
		t.Fatalf("No graceful capability set.")
	}

	// now delete pod with graceful period set
	delOpts := &metav1.DeleteOptions{GracePeriodSeconds: &gracefulPeriod}
	_, _, err = registry.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, delOpts)
	if err != nil {
		t.Fatalf("Failed to delete pod gracefully. Unexpected error: %v", err)
	}

	// try to create before graceful deletion period is over
	_, err = registry.Create(testContext, podA, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err == nil || !errors.IsAlreadyExists(err) {
		t.Fatalf("Expected 'already exists' error from storage, but got %v", err)
	}

	// check the 'alredy exists' msg was edited
	msg := &err.(*errors.StatusError).ErrStatus.Message
	if !strings.Contains(*msg, "object is being deleted:") {
		t.Errorf("Unexpected error without the 'object is being deleted:' in message: %v", err)
	}
}

func isQualifiedResource(err error, kind, group string) bool {
	if err.(errors.APIStatus).Status().Details.Kind != kind || err.(errors.APIStatus).Status().Details.Group != group {
		return false
	}
	return true
}

func updateAndVerify(t *testing.T, ctx context.Context, registry *Store, pod *example.Pod) bool {
	obj, _, err := registry.Update(ctx, pod.Name, rest.DefaultUpdatedObjectInfo(pod), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
		return false
	}
	checkObj, err := registry.Get(ctx, pod.Name, &metav1.GetOptions{})
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

func TestStoreUpdate(t *testing.T) {
	podA := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}
	podB := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec:       example.PodSpec{NodeName: "machine2"},
	}
	podAWithResourceVersion := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", ResourceVersion: "7"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	// try to update a non-existing node with denying admission, should still return NotFound
	_, _, err := registry.Update(testContext, podA.Name, rest.DefaultUpdatedObjectInfo(podA), denyCreateValidation, denyUpdateValidation, false, &metav1.UpdateOptions{})
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	// try to update a non-existing node
	_, _, err = registry.Update(testContext, podA.Name, rest.DefaultUpdatedObjectInfo(podA), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	// allow creation
	registry.UpdateStrategy.(*testRESTStrategy).allowCreateOnUpdate = true

	// createIfNotFound with denying create admission
	_, _, err = registry.Update(testContext, podA.Name, rest.DefaultUpdatedObjectInfo(podA), denyCreateValidation, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err == nil {
		t.Errorf("expected admission error on create")
	}

	// createIfNotFound and verify
	if !updateAndVerify(t, testContext, registry, podA) {
		t.Errorf("Unexpected error updating podA")
	}

	// forbid creation again
	registry.UpdateStrategy.(*testRESTStrategy).allowCreateOnUpdate = false

	// outofDate
	_, _, err = registry.Update(testContext, podAWithResourceVersion.Name, rest.DefaultUpdatedObjectInfo(podAWithResourceVersion), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if !errors.IsConflict(err) {
		t.Errorf("Unexpected error updating podAWithResourceVersion: %v", err)
	}

	// try to update with denying admission
	_, _, err = registry.Update(testContext, podA.Name, rest.DefaultUpdatedObjectInfo(podA), rest.ValidateAllObjectFunc, denyUpdateValidation, false, &metav1.UpdateOptions{})
	if err == nil {
		t.Errorf("expected admission error on update")
	}

	// normal update and verify
	if !updateAndVerify(t, testContext, registry, podB) {
		t.Errorf("Unexpected error updating podB")
	}

	// unconditional update
	// NOTE: The logic for unconditional updates doesn't make sense to me, and imho should be removed.
	// doUnconditionalUpdate := resourceVersion == 0 && e.UpdateStrategy.AllowUnconditionalUpdate()
	// ^^ That condition can *never be true due to the creation of root objects.
	//
	// registry.UpdateStrategy.(*testRESTStrategy).allowUnconditionalUpdate = true
	// updateAndVerify(t, testContext, registry, podAWithResourceVersion)

}

func TestNoOpUpdates(t *testing.T) {
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	newPod := func() *example.Pod {
		return &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Namespace: metav1.NamespaceDefault,
				Name:      "foo",
				Labels:    map[string]string{"prepare_create": "true"},
			},
			Spec: example.PodSpec{NodeName: "machine"},
		}
	}

	var err error
	var createResult runtime.Object
	if createResult, err = registry.Create(genericapirequest.NewDefaultContext(), newPod(), rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	createdPod, err := registry.Get(genericapirequest.NewDefaultContext(), "foo", &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	var updateResult runtime.Object
	p := newPod()
	if updateResult, _, err = registry.Update(genericapirequest.NewDefaultContext(), p.Name, rest.DefaultUpdatedObjectInfo(p), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Check whether we do not return empty result on no-op update.
	if !reflect.DeepEqual(createResult, updateResult) {
		t.Errorf("no-op update should return a correct value, got: %#v", updateResult)
	}

	updatedPod, err := registry.Get(genericapirequest.NewDefaultContext(), "foo", &metav1.GetOptions{})
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

func (t testPodExport) Export(ctx context.Context, obj runtime.Object, exact bool) error {
	pod := obj.(*example.Pod)
	if pod.Labels == nil {
		pod.Labels = map[string]string{}
	}
	pod.Labels["exported"] = "true"
	pod.Labels["exact"] = strconv.FormatBool(exact)

	return nil
}

func TestStoreCustomExport(t *testing.T) {
	podA := example.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test",
			Name:      "foo",
			Labels:    map[string]string{},
		},
		Spec: example.PodSpec{NodeName: "machine"},
	}

	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	registry.ExportStrategy = testPodExport{}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	registry.UpdateStrategy.(*testRESTStrategy).allowCreateOnUpdate = true
	if !updateAndVerify(t, testContext, registry, &podA) {
		t.Errorf("Unexpected error updating podA")
	}

	obj, err := registry.Export(testContext, podA.Name, metav1.ExportOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	exportedPod := obj.(*example.Pod)
	if exportedPod.Labels["exported"] != "true" {
		t.Errorf("expected: exported->true, found: %s", exportedPod.Labels["exported"])
	}
	if exportedPod.Labels["exact"] != "false" {
		t.Errorf("expected: exact->false, found: %s", exportedPod.Labels["exact"])
	}
	if exportedPod.Labels["prepare_create"] != "true" {
		t.Errorf("expected: prepare_create->true, found: %s", exportedPod.Labels["prepare_create"])
	}
	delete(exportedPod.Labels, "exported")
	delete(exportedPod.Labels, "exact")
	delete(exportedPod.Labels, "prepare_create")
	exportObjectMeta(&podA.ObjectMeta, false)
	podA.Spec = exportedPod.Spec
	if !reflect.DeepEqual(&podA, exportedPod) {
		t.Errorf("expected:\n%v\nsaw:\n%v\n", &podA, exportedPod)
	}
}

func TestStoreBasicExport(t *testing.T) {
	podA := example.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Namespace: "test",
			Name:      "foo",
			Labels:    map[string]string{},
		},
		Spec:   example.PodSpec{NodeName: "machine"},
		Status: example.PodStatus{HostIP: "1.2.3.4"},
	}

	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	registry.UpdateStrategy.(*testRESTStrategy).allowCreateOnUpdate = true
	if !updateAndVerify(t, testContext, registry, &podA) {
		t.Errorf("Unexpected error updating podA")
	}

	obj, err := registry.Export(testContext, podA.Name, metav1.ExportOptions{})
	if err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	exportedPod := obj.(*example.Pod)
	if exportedPod.Labels["prepare_create"] != "true" {
		t.Errorf("expected: prepare_create->true, found: %s", exportedPod.Labels["prepare_create"])
	}
	delete(exportedPod.Labels, "prepare_create")
	exportObjectMeta(&podA.ObjectMeta, false)
	podA.Spec = exportedPod.Spec
	if !reflect.DeepEqual(&podA, exportedPod) {
		t.Errorf("expected:\n%v\nsaw:\n%v\n", &podA, exportedPod)
	}
}

func TestStoreGet(t *testing.T) {
	podA := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "test", Name: "foo"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	_, err := registry.Get(testContext, podA.Name, &metav1.GetOptions{})
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	registry.UpdateStrategy.(*testRESTStrategy).allowCreateOnUpdate = true
	if !updateAndVerify(t, testContext, registry, podA) {
		t.Errorf("Unexpected error updating podA")
	}
}

func TestStoreDelete(t *testing.T) {
	podA := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	// test failure condition
	_, _, err := registry.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, nil)
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}

	// create pod
	_, err = registry.Create(testContext, podA, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// delete object
	_, wasDeleted, err := registry.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, nil)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if !wasDeleted {
		t.Errorf("unexpected, pod %s should have been deleted immediately", podA.Name)
	}

	// try to get a item which should be deleted
	_, err = registry.Get(testContext, podA.Name, &metav1.GetOptions{})
	if !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}
}

// TestGracefulStoreCanDeleteIfExistingGracePeriodZero tests recovery from
// race condition where the graceful delete is unable to complete
// in prior operation, but the pod remains with deletion timestamp
// and grace period set to 0.
func TestGracefulStoreCanDeleteIfExistingGracePeriodZero(t *testing.T) {
	deletionTimestamp := metav1.NewTime(time.Now())
	deletionGracePeriodSeconds := int64(0)
	initialGeneration := int64(1)
	pod := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:                       "foo",
			Generation:                 initialGeneration,
			DeletionGracePeriodSeconds: &deletionGracePeriodSeconds,
			DeletionTimestamp:          &deletionTimestamp,
		},
		Spec: example.PodSpec{NodeName: "machine"},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	registry.EnableGarbageCollection = false
	defaultDeleteStrategy := testRESTStrategy{scheme, names.SimpleNameGenerator, true, false, true}
	registry.DeleteStrategy = testGracefulStrategy{defaultDeleteStrategy}
	defer destroyFunc()

	graceful, gracefulPending, err := rest.BeforeDelete(registry.DeleteStrategy, testContext, pod, metav1.NewDeleteOptions(0))
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if graceful {
		t.Fatalf("graceful should be false if object has DeletionTimestamp and DeletionGracePeriodSeconds is 0")
	}
	if gracefulPending {
		t.Fatalf("gracefulPending should be false if object has DeletionTimestamp and DeletionGracePeriodSeconds is 0")
	}
}

func TestGracefulStoreHandleFinalizers(t *testing.T) {
	initialGeneration := int64(1)
	podWithFinalizer := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Finalizers: []string{"foo.com/x"}, Generation: initialGeneration},
		Spec:       example.PodSpec{NodeName: "machine"},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defaultDeleteStrategy := testRESTStrategy{scheme, names.SimpleNameGenerator, true, false, true}
	registry.DeleteStrategy = testGracefulStrategy{defaultDeleteStrategy}
	defer destroyFunc()

	gcStates := []bool{true, false}
	for _, gcEnabled := range gcStates {
		t.Logf("garbage collection enabled: %t", gcEnabled)
		registry.EnableGarbageCollection = gcEnabled

		// create pod
		_, err := registry.Create(testContext, podWithFinalizer, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		// delete the pod with grace period=0, the pod should still exist because it has a finalizer
		_, wasDeleted, err := registry.Delete(testContext, podWithFinalizer.Name, rest.ValidateAllObjectFunc, metav1.NewDeleteOptions(0))
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if wasDeleted {
			t.Errorf("unexpected, pod %s should not have been deleted immediately", podWithFinalizer.Name)
		}
		_, err = registry.Get(testContext, podWithFinalizer.Name, &metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		updatedPodWithFinalizer := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Finalizers: []string{"foo.com/x"}, ResourceVersion: podWithFinalizer.ObjectMeta.ResourceVersion},
			Spec:       example.PodSpec{NodeName: "machine"},
		}
		_, _, err = registry.Update(testContext, updatedPodWithFinalizer.ObjectMeta.Name, rest.DefaultUpdatedObjectInfo(updatedPodWithFinalizer), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// the object should still exist, because it still has a finalizer
		_, err = registry.Get(testContext, podWithFinalizer.Name, &metav1.GetOptions{})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		podWithNoFinalizer := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: podWithFinalizer.ObjectMeta.ResourceVersion},
			Spec:       example.PodSpec{NodeName: "anothermachine"},
		}
		_, _, err = registry.Update(testContext, podWithFinalizer.ObjectMeta.Name, rest.DefaultUpdatedObjectInfo(podWithNoFinalizer), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		// the pod should be removed, because its finalizer is removed
		_, err = registry.Get(testContext, podWithFinalizer.Name, &metav1.GetOptions{})
		if !errors.IsNotFound(err) {
			t.Fatalf("Unexpected error: %v", err)
		}
	}
}

func TestNonGracefulStoreHandleFinalizers(t *testing.T) {
	initialGeneration := int64(1)
	podWithFinalizer := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Finalizers: []string{"foo.com/x"}, Generation: initialGeneration},
		Spec:       example.PodSpec{NodeName: "machine"},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)

	defer destroyFunc()
	gcStates := []bool{true, false}
	for _, gcEnabled := range gcStates {
		t.Logf("garbage collection enabled: %t", gcEnabled)
		registry.EnableGarbageCollection = gcEnabled

		// create pod
		_, err := registry.Create(testContext, podWithFinalizer, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		// delete object with nil delete options doesn't delete the object
		_, wasDeleted, err := registry.Delete(testContext, podWithFinalizer.Name, rest.ValidateAllObjectFunc, nil)
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if wasDeleted {
			t.Errorf("unexpected, pod %s should not have been deleted immediately", podWithFinalizer.Name)
		}

		// the object should still exist
		obj, err := registry.Get(testContext, podWithFinalizer.Name, &metav1.GetOptions{})
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		podWithFinalizer, ok := obj.(*example.Pod)
		if !ok {
			t.Errorf("Unexpected object: %#v", obj)
		}
		if podWithFinalizer.ObjectMeta.DeletionTimestamp == nil {
			t.Errorf("Expect the object to have DeletionTimestamp set, but got %#v", podWithFinalizer.ObjectMeta)
		}
		if podWithFinalizer.ObjectMeta.DeletionGracePeriodSeconds == nil || *podWithFinalizer.ObjectMeta.DeletionGracePeriodSeconds != 0 {
			t.Errorf("Expect the object to have 0 DeletionGracePeriodSecond, but got %#v", podWithFinalizer.ObjectMeta)
		}
		if podWithFinalizer.Generation <= initialGeneration {
			t.Errorf("Deletion didn't increase Generation.")
		}

		updatedPodWithFinalizer := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", Finalizers: []string{"foo.com/x"}, ResourceVersion: podWithFinalizer.ObjectMeta.ResourceVersion},
			Spec:       example.PodSpec{NodeName: "machine"},
		}
		_, _, err = registry.Update(testContext, updatedPodWithFinalizer.ObjectMeta.Name, rest.DefaultUpdatedObjectInfo(updatedPodWithFinalizer), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		// the object should still exist, because it still has a finalizer
		obj, err = registry.Get(testContext, podWithFinalizer.Name, &metav1.GetOptions{})
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		podWithFinalizer, ok = obj.(*example.Pod)
		if !ok {
			t.Errorf("Unexpected object: %#v", obj)
		}

		podWithNoFinalizer := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: "foo", ResourceVersion: podWithFinalizer.ObjectMeta.ResourceVersion},
			Spec:       example.PodSpec{NodeName: "anothermachine"},
		}
		_, _, err = registry.Update(testContext, podWithFinalizer.ObjectMeta.Name, rest.DefaultUpdatedObjectInfo(podWithNoFinalizer), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		// the pod should be removed, because its finalizer is removed
		_, err = registry.Get(testContext, podWithFinalizer.Name, &metav1.GetOptions{})
		if !errors.IsNotFound(err) {
			t.Errorf("Unexpected error: %v", err)
		}
	}
}

func TestStoreDeleteWithOrphanDependents(t *testing.T) {
	initialGeneration := int64(1)
	podWithOrphanFinalizer := func(name string) *example.Pod {
		return &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Finalizers: []string{"foo.com/x", metav1.FinalizerOrphanDependents, "bar.com/y"}, Generation: initialGeneration},
			Spec:       example.PodSpec{NodeName: "machine"},
		}
	}
	podWithOtherFinalizers := func(name string) *example.Pod {
		return &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Finalizers: []string{"foo.com/x", "bar.com/y"}, Generation: initialGeneration},
			Spec:       example.PodSpec{NodeName: "machine"},
		}
	}
	podWithNoFinalizer := func(name string) *example.Pod {
		return &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Generation: initialGeneration},
			Spec:       example.PodSpec{NodeName: "machine"},
		}
	}
	podWithOnlyOrphanFinalizer := func(name string) *example.Pod {
		return &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: name, Finalizers: []string{metav1.FinalizerOrphanDependents}, Generation: initialGeneration},
			Spec:       example.PodSpec{NodeName: "machine"},
		}
	}
	trueVar, falseVar := true, false
	orphanOptions := &metav1.DeleteOptions{OrphanDependents: &trueVar}
	nonOrphanOptions := &metav1.DeleteOptions{OrphanDependents: &falseVar}
	nilOrphanOptions := &metav1.DeleteOptions{}

	// defaultDeleteStrategy doesn't implement rest.GarbageCollectionDeleteStrategy.
	defaultDeleteStrategy := &testRESTStrategy{scheme, names.SimpleNameGenerator, true, false, true}
	// orphanDeleteStrategy indicates the default garbage collection policy is
	// to orphan dependentes.
	orphanDeleteStrategy := &testOrphanDeleteStrategy{defaultDeleteStrategy}

	testcases := []struct {
		pod               *example.Pod
		options           *metav1.DeleteOptions
		strategy          rest.RESTDeleteStrategy
		expectNotFound    bool
		updatedFinalizers []string
	}{
		// cases run with DeleteOptions.OrphanDedependents=true
		{
			podWithOrphanFinalizer("pod1"),
			orphanOptions,
			defaultDeleteStrategy,
			false,
			[]string{"foo.com/x", metav1.FinalizerOrphanDependents, "bar.com/y"},
		},
		{
			podWithOtherFinalizers("pod2"),
			orphanOptions,
			defaultDeleteStrategy,
			false,
			[]string{"foo.com/x", "bar.com/y", metav1.FinalizerOrphanDependents},
		},
		{
			podWithNoFinalizer("pod3"),
			orphanOptions,
			defaultDeleteStrategy,
			false,
			[]string{metav1.FinalizerOrphanDependents},
		},
		{
			podWithOnlyOrphanFinalizer("pod4"),
			orphanOptions,
			defaultDeleteStrategy,
			false,
			[]string{metav1.FinalizerOrphanDependents},
		},
		// cases run with DeleteOptions.OrphanDedependents=false
		// these cases all have oprhanDeleteStrategy, which should be ignored
		// because DeleteOptions has the highest priority.
		{
			podWithOrphanFinalizer("pod5"),
			nonOrphanOptions,
			orphanDeleteStrategy,
			false,
			[]string{"foo.com/x", "bar.com/y"},
		},
		{
			podWithOtherFinalizers("pod6"),
			nonOrphanOptions,
			orphanDeleteStrategy,
			false,
			[]string{"foo.com/x", "bar.com/y"},
		},
		{
			podWithNoFinalizer("pod7"),
			nonOrphanOptions,
			orphanDeleteStrategy,
			true,
			[]string{},
		},
		{
			podWithOnlyOrphanFinalizer("pod8"),
			nonOrphanOptions,
			orphanDeleteStrategy,
			true,
			[]string{},
		},
		// cases run with nil DeleteOptions.OrphanDependents. If the object
		// already has the orphan finalizer, then the DeleteStrategy should be
		// ignored. Otherwise the DeleteStrategy decides whether to add the
		// orphan finalizer.
		{
			podWithOrphanFinalizer("pod9"),
			nilOrphanOptions,
			defaultDeleteStrategy,
			false,
			[]string{"foo.com/x", metav1.FinalizerOrphanDependents, "bar.com/y"},
		},
		{
			podWithOrphanFinalizer("pod10"),
			nilOrphanOptions,
			orphanDeleteStrategy,
			false,
			[]string{"foo.com/x", metav1.FinalizerOrphanDependents, "bar.com/y"},
		},
		{
			podWithOtherFinalizers("pod11"),
			nilOrphanOptions,
			defaultDeleteStrategy,
			false,
			[]string{"foo.com/x", "bar.com/y"},
		},
		{
			podWithOtherFinalizers("pod12"),
			nilOrphanOptions,
			orphanDeleteStrategy,
			false,
			[]string{"foo.com/x", "bar.com/y", metav1.FinalizerOrphanDependents},
		},
		{
			podWithNoFinalizer("pod13"),
			nilOrphanOptions,
			defaultDeleteStrategy,
			true,
			[]string{},
		},
		{
			podWithNoFinalizer("pod14"),
			nilOrphanOptions,
			orphanDeleteStrategy,
			false,
			[]string{metav1.FinalizerOrphanDependents},
		},
		{
			podWithOnlyOrphanFinalizer("pod15"),
			nilOrphanOptions,
			defaultDeleteStrategy,
			false,
			[]string{metav1.FinalizerOrphanDependents},
		},
		{
			podWithOnlyOrphanFinalizer("pod16"),
			nilOrphanOptions,
			orphanDeleteStrategy,
			false,
			[]string{metav1.FinalizerOrphanDependents},
		},

		// cases run with nil DeleteOptions should have exact same behavior.
		// They should be exactly the same as above cases where
		// DeleteOptions.OrphanDependents is nil.
		{
			podWithOrphanFinalizer("pod17"),
			nil,
			defaultDeleteStrategy,
			false,
			[]string{"foo.com/x", metav1.FinalizerOrphanDependents, "bar.com/y"},
		},
		{
			podWithOrphanFinalizer("pod18"),
			nil,
			orphanDeleteStrategy,
			false,
			[]string{"foo.com/x", metav1.FinalizerOrphanDependents, "bar.com/y"},
		},
		{
			podWithOtherFinalizers("pod19"),
			nil,
			defaultDeleteStrategy,
			false,
			[]string{"foo.com/x", "bar.com/y"},
		},
		{
			podWithOtherFinalizers("pod20"),
			nil,
			orphanDeleteStrategy,
			false,
			[]string{"foo.com/x", "bar.com/y", metav1.FinalizerOrphanDependents},
		},
		{
			podWithNoFinalizer("pod21"),
			nil,
			defaultDeleteStrategy,
			true,
			[]string{},
		},
		{
			podWithNoFinalizer("pod22"),
			nil,
			orphanDeleteStrategy,
			false,
			[]string{metav1.FinalizerOrphanDependents},
		},
		{
			podWithOnlyOrphanFinalizer("pod23"),
			nil,
			defaultDeleteStrategy,
			false,
			[]string{metav1.FinalizerOrphanDependents},
		},
		{
			podWithOnlyOrphanFinalizer("pod24"),
			nil,
			orphanDeleteStrategy,
			false,
			[]string{metav1.FinalizerOrphanDependents},
		},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	registry.EnableGarbageCollection = true
	defer destroyFunc()

	for _, tc := range testcases {
		registry.DeleteStrategy = tc.strategy
		// create pod
		_, err := registry.Create(testContext, tc.pod, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		_, _, err = registry.Delete(testContext, tc.pod.Name, rest.ValidateAllObjectFunc, tc.options)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		obj, err := registry.Get(testContext, tc.pod.Name, &metav1.GetOptions{})
		if tc.expectNotFound && (err == nil || !errors.IsNotFound(err)) {
			t.Fatalf("Unexpected error: %v", err)
		}
		if !tc.expectNotFound && err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if !tc.expectNotFound {
			pod, ok := obj.(*example.Pod)
			if !ok {
				t.Fatalf("Expect the object to be a pod, but got %#v", obj)
			}
			if pod.ObjectMeta.DeletionTimestamp == nil {
				t.Errorf("%v: Expect the object to have DeletionTimestamp set, but got %#v", pod.Name, pod.ObjectMeta)
			}
			if pod.ObjectMeta.DeletionGracePeriodSeconds == nil || *pod.ObjectMeta.DeletionGracePeriodSeconds != 0 {
				t.Errorf("%v: Expect the object to have 0 DeletionGracePeriodSecond, but got %#v", pod.Name, pod.ObjectMeta)
			}
			if pod.Generation <= initialGeneration {
				t.Errorf("%v: Deletion didn't increase Generation.", pod.Name)
			}
			if e, a := tc.updatedFinalizers, pod.ObjectMeta.Finalizers; !reflect.DeepEqual(e, a) {
				t.Errorf("%v: Expect object %s to have finalizers %v, got %v", pod.Name, pod.ObjectMeta.Name, e, a)
			}
		}
	}
}

// Test the DeleteOptions.PropagationPolicy is handled correctly
func TestStoreDeletionPropagation(t *testing.T) {
	initialGeneration := int64(1)

	// defaultDeleteStrategy doesn't implement rest.GarbageCollectionDeleteStrategy.
	defaultDeleteStrategy := &testRESTStrategy{scheme, names.SimpleNameGenerator, true, false, true}
	// orphanDeleteStrategy indicates the default garbage collection policy is
	// to orphan dependentes.
	orphanDeleteStrategy := &testOrphanDeleteStrategy{defaultDeleteStrategy}

	foregroundPolicy := metav1.DeletePropagationForeground
	backgroundPolicy := metav1.DeletePropagationBackground
	orphanPolicy := metav1.DeletePropagationOrphan

	testcases := map[string]struct {
		options  *metav1.DeleteOptions
		strategy rest.RESTDeleteStrategy
		// finalizers that are already set in the object
		existingFinalizers []string
		expectedNotFound   bool
		expectedFinalizers []string
	}{
		"no existing finalizers, PropagationPolicy=Foreground, defaultDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &foregroundPolicy},
			strategy:           defaultDeleteStrategy,
			expectedFinalizers: []string{metav1.FinalizerDeleteDependents},
		},
		"no existing finalizers, PropagationPolicy=Foreground, orphanDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &foregroundPolicy},
			strategy:           orphanDeleteStrategy,
			expectedFinalizers: []string{metav1.FinalizerDeleteDependents},
		},
		"no existing finalizers, PropagationPolicy=Background, defaultDeleteStrategy": {
			options:          &metav1.DeleteOptions{PropagationPolicy: &backgroundPolicy},
			strategy:         defaultDeleteStrategy,
			expectedNotFound: true,
		},
		"no existing finalizers, PropagationPolicy=Background, orphanDeleteStrategy": {
			options:          &metav1.DeleteOptions{PropagationPolicy: &backgroundPolicy},
			strategy:         orphanDeleteStrategy,
			expectedNotFound: true,
		},
		"no existing finalizers, PropagationPolicy=OrphanDependents, defaultDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &orphanPolicy},
			strategy:           defaultDeleteStrategy,
			expectedFinalizers: []string{metav1.FinalizerOrphanDependents},
		},
		"no existing finalizers, PropagationPolicy=OrphanDependents, orphanDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &orphanPolicy},
			strategy:           orphanDeleteStrategy,
			expectedFinalizers: []string{metav1.FinalizerOrphanDependents},
		},
		"no existing finalizers, PropagationPolicy=Default, defaultDeleteStrategy": {
			options:          &metav1.DeleteOptions{PropagationPolicy: nil},
			strategy:         defaultDeleteStrategy,
			expectedNotFound: true,
		},
		"no existing finalizers, PropagationPolicy=Default, orphanDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: nil},
			strategy:           orphanDeleteStrategy,
			expectedFinalizers: []string{metav1.FinalizerOrphanDependents},
		},

		// all cases in the following block have "existing orphan finalizer"
		"existing orphan finalizer, PropagationPolicy=Foreground, defaultDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &foregroundPolicy},
			strategy:           defaultDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerOrphanDependents},
			expectedFinalizers: []string{metav1.FinalizerDeleteDependents},
		},
		"existing orphan finalizer, PropagationPolicy=Foreground, orphanDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &foregroundPolicy},
			strategy:           orphanDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerOrphanDependents},
			expectedFinalizers: []string{metav1.FinalizerDeleteDependents},
		},
		"existing orphan finalizer, PropagationPolicy=Background, defaultDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &backgroundPolicy},
			strategy:           defaultDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerOrphanDependents},
			expectedNotFound:   true,
		},
		"existing orphan finalizer, PropagationPolicy=Background, orphanDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &backgroundPolicy},
			strategy:           orphanDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerOrphanDependents},
			expectedNotFound:   true,
		},
		"existing orphan finalizer, PropagationPolicy=OrphanDependents, defaultDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &orphanPolicy},
			strategy:           defaultDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerOrphanDependents},
			expectedFinalizers: []string{metav1.FinalizerOrphanDependents},
		},
		"existing orphan finalizer, PropagationPolicy=OrphanDependents, orphanDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &orphanPolicy},
			strategy:           orphanDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerOrphanDependents},
			expectedFinalizers: []string{metav1.FinalizerOrphanDependents},
		},
		"existing orphan finalizer, PropagationPolicy=Default, defaultDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: nil},
			strategy:           defaultDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerOrphanDependents},
			expectedFinalizers: []string{metav1.FinalizerOrphanDependents},
		},
		"existing orphan finalizer, PropagationPolicy=Default, orphanDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: nil},
			strategy:           orphanDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerOrphanDependents},
			expectedFinalizers: []string{metav1.FinalizerOrphanDependents},
		},

		// all cases in the following block have "existing deleteDependents finalizer"
		"existing deleteDependents finalizer, PropagationPolicy=Foreground, defaultDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &foregroundPolicy},
			strategy:           defaultDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerDeleteDependents},
			expectedFinalizers: []string{metav1.FinalizerDeleteDependents},
		},
		"existing deleteDependents finalizer, PropagationPolicy=Foreground, orphanDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &foregroundPolicy},
			strategy:           orphanDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerDeleteDependents},
			expectedFinalizers: []string{metav1.FinalizerDeleteDependents},
		},
		"existing deleteDependents finalizer, PropagationPolicy=Background, defaultDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &backgroundPolicy},
			strategy:           defaultDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerDeleteDependents},
			expectedNotFound:   true,
		},
		"existing deleteDependents finalizer, PropagationPolicy=Background, orphanDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &backgroundPolicy},
			strategy:           orphanDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerDeleteDependents},
			expectedNotFound:   true,
		},
		"existing deleteDependents finalizer, PropagationPolicy=OrphanDependents, defaultDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &orphanPolicy},
			strategy:           defaultDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerDeleteDependents},
			expectedFinalizers: []string{metav1.FinalizerOrphanDependents},
		},
		"existing deleteDependents finalizer, PropagationPolicy=OrphanDependents, orphanDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: &orphanPolicy},
			strategy:           orphanDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerDeleteDependents},
			expectedFinalizers: []string{metav1.FinalizerOrphanDependents},
		},
		"existing deleteDependents finalizer, PropagationPolicy=Default, defaultDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: nil},
			strategy:           defaultDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerDeleteDependents},
			expectedFinalizers: []string{metav1.FinalizerDeleteDependents},
		},
		"existing deleteDependents finalizer, PropagationPolicy=Default, orphanDeleteStrategy": {
			options:            &metav1.DeleteOptions{PropagationPolicy: nil},
			strategy:           orphanDeleteStrategy,
			existingFinalizers: []string{metav1.FinalizerDeleteDependents},
			expectedFinalizers: []string{metav1.FinalizerDeleteDependents},
		},
	}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	registry.EnableGarbageCollection = true
	defer destroyFunc()

	createPod := func(i int, finalizers []string) *example.Pod {
		return &example.Pod{
			ObjectMeta: metav1.ObjectMeta{Name: fmt.Sprintf("pod-%d", i), Finalizers: finalizers, Generation: initialGeneration},
			Spec:       example.PodSpec{NodeName: "machine"},
		}
	}

	i := 0
	for title, tc := range testcases {
		t.Logf("case title: %s", title)
		registry.DeleteStrategy = tc.strategy
		i++
		pod := createPod(i, tc.existingFinalizers)
		// create pod
		_, err := registry.Create(testContext, pod, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		_, _, err = registry.Delete(testContext, pod.Name, rest.ValidateAllObjectFunc, tc.options)
		obj, err := registry.Get(testContext, pod.Name, &metav1.GetOptions{})
		if tc.expectedNotFound {
			if err == nil || !errors.IsNotFound(err) {
				t.Fatalf("Unexpected error: %v", err)
			}
			continue
		}
		if !tc.expectedNotFound && err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		if !tc.expectedNotFound {
			pod, ok := obj.(*example.Pod)
			if !ok {
				t.Fatalf("Expect the object to be a pod, but got %#v", obj)
			}
			if e, a := tc.expectedFinalizers, pod.ObjectMeta.Finalizers; !reflect.DeepEqual(e, a) {
				t.Errorf("%v: Expect object %s to have finalizers %v, got %v", pod.Name, pod.ObjectMeta.Name, e, a)
			}
			if pod.ObjectMeta.DeletionTimestamp == nil {
				t.Errorf("%v: Expect the object to have DeletionTimestamp set, but got %#v", pod.Name, pod.ObjectMeta)
			}
			if pod.ObjectMeta.DeletionGracePeriodSeconds == nil || *pod.ObjectMeta.DeletionGracePeriodSeconds != 0 {
				t.Errorf("%v: Expect the object to have 0 DeletionGracePeriodSecond, but got %#v", pod.Name, pod.ObjectMeta)
			}
			if pod.Generation <= initialGeneration {
				t.Errorf("%v: Deletion didn't increase Generation.", pod.Name)
			}
		}
	}
}

func TestStoreDeleteCollection(t *testing.T) {
	podA := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	podB := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	if _, err := registry.Create(testContext, podA, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if _, err := registry.Create(testContext, podB, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Delete all pods.
	deleted, err := registry.DeleteCollection(testContext, rest.ValidateAllObjectFunc, nil, &metainternalversion.ListOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	deletedPods := deleted.(*example.PodList)
	if len(deletedPods.Items) != 2 {
		t.Errorf("Unexpected number of pods deleted: %d, expected: 3", len(deletedPods.Items))
	}

	if _, err := registry.Get(testContext, podA.Name, &metav1.GetOptions{}); !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	if _, err := registry.Get(testContext, podB.Name, &metav1.GetOptions{}); !errors.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestStoreDeleteCollectionNotFound(t *testing.T) {
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")

	podA := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}
	podB := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "bar"}}

	for i := 0; i < 10; i++ {
		// Setup
		if _, err := registry.Create(testContext, podA, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if _, err := registry.Create(testContext, podB, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
			t.Errorf("Unexpected error: %v", err)
		}

		// Kick off multiple delete collection calls to test notfound behavior
		wg := &sync.WaitGroup{}
		for j := 0; j < 2; j++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, err := registry.DeleteCollection(testContext, rest.ValidateAllObjectFunc, nil, &metainternalversion.ListOptions{})
				if err != nil {
					t.Fatalf("Unexpected error: %v", err)
				}
			}()
		}
		wg.Wait()

		if _, err := registry.Get(testContext, podA.Name, &metav1.GetOptions{}); !errors.IsNotFound(err) {
			t.Errorf("Unexpected error: %v", err)
		}
		if _, err := registry.Get(testContext, podB.Name, &metav1.GetOptions{}); !errors.IsNotFound(err) {
			t.Errorf("Unexpected error: %v", err)
		}
	}
}

// Test whether objects deleted with DeleteCollection are correctly delivered
// to watchers.
func TestStoreDeleteCollectionWithWatch(t *testing.T) {
	podA := &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: "foo"}}

	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	objCreated, err := registry.Create(testContext, podA, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	podCreated := objCreated.(*example.Pod)

	watcher, err := registry.WatchPredicate(testContext, matchPodName("foo"), podCreated.ResourceVersion)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watcher.Stop()

	if _, err := registry.DeleteCollection(testContext, rest.ValidateAllObjectFunc, nil, &metainternalversion.ListOptions{}); err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	got, open := <-watcher.ResultChan()
	if !open {
		t.Errorf("Unexpected channel close")
	} else {
		if got.Type != "DELETED" {
			t.Errorf("Unexpected event type: %s", got.Type)
		}
		gotObject := got.Object.(*example.Pod)
		gotObject.ResourceVersion = podCreated.ResourceVersion
		if e, a := podCreated, gotObject; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected: %#v, got: %#v", e, a)
		}
	}
}

func TestStoreWatch(t *testing.T) {
	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	noNamespaceContext := genericapirequest.NewContext()

	table := map[string]struct {
		selectPred storage.SelectionPredicate
		context    context.Context
	}{
		"single": {
			selectPred: matchPodName("foo"),
		},
		"multi": {
			selectPred: matchPodName("foo", "bar"),
		},
		"singleNoNamespace": {
			selectPred: matchPodName("foo"),
			context:    noNamespaceContext,
		},
	}

	for name, m := range table {
		ctx := testContext
		if m.context != nil {
			ctx = m.context
		}
		podA := &example.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo",
				Namespace: "test",
			},
			Spec: example.PodSpec{NodeName: "machine"},
		}

		destroyFunc, registry := NewTestGenericStoreRegistry(t)
		wi, err := registry.WatchPredicate(ctx, m.selectPred, "0")
		if err != nil {
			t.Errorf("%v: unexpected error: %v", name, err)
		} else {
			obj, err := registry.Create(testContext, podA, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
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
		destroyFunc()
	}
}

func newTestGenericStoreRegistry(t *testing.T, scheme *runtime.Scheme, hasCacheEnabled bool) (factory.DestroyFunc, *Store) {
	podPrefix := "/pods"
	server, sc := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	strategy := &testRESTStrategy{scheme, names.SimpleNameGenerator, true, false, true}

	sc.Codec = apitesting.TestStorageCodec(codecs, examplev1.SchemeGroupVersion)
	s, dFunc, err := factory.Create(*sc)
	if err != nil {
		t.Fatalf("Error creating storage: %v", err)
	}
	destroyFunc := func() {
		dFunc()
		server.Terminate(t)
	}
	if hasCacheEnabled {
		config := cacherstorage.Config{
			CacheCapacity:  10,
			Storage:        s,
			Versioner:      etcd3.APIObjectVersioner{},
			ResourcePrefix: podPrefix,
			KeyFunc:        func(obj runtime.Object) (string, error) { return storage.NoNamespaceKeyFunc(podPrefix, obj) },
			GetAttrsFunc:   getPodAttrs,
			NewFunc:        func() runtime.Object { return &example.Pod{} },
			NewListFunc:    func() runtime.Object { return &example.PodList{} },
			Codec:          sc.Codec,
		}
		cacher, err := cacherstorage.NewCacherFromConfig(config)
		if err != nil {
			t.Fatalf("Couldn't create cacher: %v", err)
		}
		d := destroyFunc
		s = cacher
		destroyFunc = func() {
			cacher.Stop()
			d()
		}
	}

	return destroyFunc, &Store{
		NewFunc:                  func() runtime.Object { return &example.Pod{} },
		NewListFunc:              func() runtime.Object { return &example.PodList{} },
		DefaultQualifiedResource: example.Resource("pods"),
		CreateStrategy:           strategy,
		UpdateStrategy:           strategy,
		DeleteStrategy:           strategy,
		KeyRootFunc: func(ctx context.Context) string {
			return podPrefix
		},
		KeyFunc: func(ctx context.Context, id string) (string, error) {
			if _, ok := genericapirequest.NamespaceFrom(ctx); !ok {
				return "", fmt.Errorf("namespace is required")
			}
			return path.Join(podPrefix, id), nil
		},
		ObjectNameFunc: func(obj runtime.Object) (string, error) { return obj.(*example.Pod).Name, nil },
		PredicateFunc: func(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
			return storage.SelectionPredicate{
				Label: label,
				Field: field,
				GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
					pod, ok := obj.(*example.Pod)
					if !ok {
						return nil, nil, fmt.Errorf("not a pod")
					}
					return labels.Set(pod.ObjectMeta.Labels), generic.ObjectMetaFieldsSet(&pod.ObjectMeta, true), nil
				},
			}
		},
		Storage: DryRunnableStorage{Storage: s},
	}
}

func TestFinalizeDelete(t *testing.T) {
	// Verify that it returns the expected Status.
	destroyFunc, s := NewTestGenericStoreRegistry(t)
	defer destroyFunc()
	obj := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", UID: "random-uid"},
	}
	result, err := s.finalizeDelete(genericapirequest.NewContext(), obj, false)
	if err != nil {
		t.Fatalf("unexpected err: %s", err)
	}
	returnedObj := result.(*metav1.Status)

	expectedObj := &metav1.Status{
		Status: metav1.StatusSuccess,
		Details: &metav1.StatusDetails{
			Name:  "foo",
			Group: s.DefaultQualifiedResource.Group,
			Kind:  s.DefaultQualifiedResource.Resource,
			UID:   "random-uid",
		},
	}
	if !apiequality.Semantic.DeepEqual(expectedObj, returnedObj) {
		t.Errorf("unexpected obj. expected %#v, got %#v", expectedObj, returnedObj)
	}
}

func fakeRequestInfo(resource, apiGroup string) *genericapirequest.RequestInfo {
	return &genericapirequest.RequestInfo{
		IsResourceRequest: true,
		Path:              "/api/v1/test",
		Verb:              "test",
		APIPrefix:         "api",
		APIGroup:          apiGroup,
		APIVersion:        "v1",
		Namespace:         "",
		Resource:          resource,
		Subresource:       "",
		Name:              "",
		Parts:             []string{"test"},
	}
}

func TestQualifiedResource(t *testing.T) {
	podA := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test"},
		Spec:       example.PodSpec{NodeName: "machine"},
	}

	qualifiedKind := "pod"
	qualifiedGroup := "test"
	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	testContext = genericapirequest.WithRequestInfo(testContext, fakeRequestInfo(qualifiedKind, qualifiedGroup))

	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	// update a non-exist object
	_, _, err := registry.Update(testContext, podA.Name, rest.DefaultUpdatedObjectInfo(podA), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if !errors.IsNotFound(err) {
		t.Fatalf("Unexpected error: %v", err)
	}

	if !isQualifiedResource(err, qualifiedKind, qualifiedGroup) {
		t.Fatalf("Unexpected error: %#v", err)
	}

	// get a non-exist object
	_, err = registry.Get(testContext, podA.Name, &metav1.GetOptions{})

	if !errors.IsNotFound(err) {
		t.Fatalf("Unexpected error: %v", err)
	}

	if !isQualifiedResource(err, qualifiedKind, qualifiedGroup) {
		t.Fatalf("Unexpected error: %#v", err)
	}

	// delete a non-exist object
	_, _, err = registry.Delete(testContext, podA.Name, rest.ValidateAllObjectFunc, nil)

	if !errors.IsNotFound(err) {
		t.Fatalf("Unexpected error: %v", err)
	}

	if !isQualifiedResource(err, qualifiedKind, qualifiedGroup) {
		t.Fatalf("Unexpected error: %#v", err)
	}

	// create a non-exist object
	_, err = registry.Create(testContext, podA, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}

	// create a exist object will fail
	_, err = registry.Create(testContext, podA, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if !errors.IsAlreadyExists(err) {
		t.Fatalf("Unexpected error: %v", err)
	}

	if !isQualifiedResource(err, qualifiedKind, qualifiedGroup) {
		t.Fatalf("Unexpected error: %#v", err)
	}
}

func denyCreateValidation(obj runtime.Object) error {
	return fmt.Errorf("admission denied")
}

func denyUpdateValidation(obj, old runtime.Object) error {
	return fmt.Errorf("admission denied")
}

type fakeStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

func (fakeStrategy) DefaultGarbageCollectionPolicy(ctx context.Context) rest.GarbageCollectionPolicy {
	appsv1beta1 := schema.GroupVersion{Group: "apps", Version: "v1beta1"}
	appsv1beta2 := schema.GroupVersion{Group: "apps", Version: "v1beta2"}
	extensionsv1beta1 := schema.GroupVersion{Group: "extensions", Version: "v1beta1"}
	if requestInfo, found := genericapirequest.RequestInfoFrom(ctx); found {
		groupVersion := schema.GroupVersion{Group: requestInfo.APIGroup, Version: requestInfo.APIVersion}
		switch groupVersion {
		case appsv1beta1, appsv1beta2, extensionsv1beta1:
			// for back compatibility
			return rest.OrphanDependents
		default:
			return rest.DeleteDependents
		}
	}
	return rest.OrphanDependents
}

func TestDeletionFinalizersForGarbageCollection(t *testing.T) {
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	registry.DeleteStrategy = fakeStrategy{}
	registry.EnableGarbageCollection = true

	tests := []struct {
		requestInfo       genericapirequest.RequestInfo
		desiredFinalizers []string
		isNilRequestInfo  bool
		changed           bool
	}{
		{
			genericapirequest.RequestInfo{
				APIGroup:   "extensions",
				APIVersion: "v1beta1",
			},
			[]string{metav1.FinalizerOrphanDependents},
			false,
			true,
		},
		{
			genericapirequest.RequestInfo{
				APIGroup:   "apps",
				APIVersion: "v1beta1",
			},
			[]string{metav1.FinalizerOrphanDependents},
			false,
			true,
		},
		{
			genericapirequest.RequestInfo{
				APIGroup:   "apps",
				APIVersion: "v1beta2",
			},
			[]string{metav1.FinalizerOrphanDependents},
			false,
			true,
		},
		{
			genericapirequest.RequestInfo{
				APIGroup:   "apps",
				APIVersion: "v1",
			},
			[]string{},
			false,
			false,
		},
	}

	for _, test := range tests {
		context := genericapirequest.NewContext()
		if !test.isNilRequestInfo {
			context = genericapirequest.WithRequestInfo(context, &test.requestInfo)
		}
		changed, finalizers := deletionFinalizersForGarbageCollection(context, registry, &example.ReplicaSet{}, &metav1.DeleteOptions{})
		if !changed {
			if test.changed {
				t.Errorf("%s/%s: no new finalizers are added", test.requestInfo.APIGroup, test.requestInfo.APIVersion)
			}
		} else if !reflect.DeepEqual(finalizers, test.desiredFinalizers) {
			t.Errorf("%s/%s: want %#v, got %#v", test.requestInfo.APIGroup, test.requestInfo.APIVersion,
				test.desiredFinalizers, finalizers)
		}
	}
}

func TestMarkAsDeleting(t *testing.T) {
	now := time.Now()
	soon := now.Add(time.Second)
	past := now.Add(-time.Second)

	newTimePointer := func(t time.Time) *metav1.Time {
		metaTime := metav1.NewTime(t)
		return &metaTime
	}
	testcases := []struct {
		name                    string
		deletionTimestamp       *metav1.Time
		expectDeletionTimestamp *metav1.Time
	}{
		{
			name:                    "unset",
			deletionTimestamp:       nil,
			expectDeletionTimestamp: newTimePointer(now),
		},
		{
			name:                    "set to future",
			deletionTimestamp:       newTimePointer(soon),
			expectDeletionTimestamp: newTimePointer(now),
		},
		{
			name:                    "set to past",
			deletionTimestamp:       newTimePointer(past),
			expectDeletionTimestamp: newTimePointer(past),
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			rs := &example.ReplicaSet{}
			rs.DeletionTimestamp = tc.deletionTimestamp
			if err := markAsDeleting(rs, now); err != nil {
				t.Error(err)
			}
			if reflect.DeepEqual(*rs.DeletionTimestamp, tc.expectDeletionTimestamp) {
				t.Errorf("expected %v, got %v", tc.expectDeletionTimestamp, *rs.DeletionTimestamp)
			}
		})
	}
}

type staleGuaranteedUpdateStorage struct {
	storage.Interface
	cachedObj runtime.Object
}

// GuaranteedUpdate overwrites the method with one that always suggests the cachedObj.
func (s *staleGuaranteedUpdateStorage) GuaranteedUpdate(
	ctx context.Context, key string, ptrToType runtime.Object, ignoreNotFound bool,
	preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, _ ...runtime.Object) error {
	return s.Interface.GuaranteedUpdate(ctx, key, ptrToType, ignoreNotFound, preconditions, tryUpdate, s.cachedObj)
}

func TestDeleteWithCachedObject(t *testing.T) {
	podName := "foo"
	podWithFinalizer := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName, Finalizers: []string{"foo.com/x"}},
		Spec:       example.PodSpec{NodeName: "machine"},
	}
	podWithNoFinalizer := &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName},
		Spec:       example.PodSpec{NodeName: "machine"},
	}
	ctx := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := newTestGenericStoreRegistry(t, scheme, false)
	defer destroyFunc()
	// cached object does not have any finalizer.
	registry.Storage.Storage = &staleGuaranteedUpdateStorage{Interface: registry.Storage.Storage, cachedObj: podWithNoFinalizer}
	// created object with pending finalizer.
	_, err := registry.Create(ctx, podWithFinalizer, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatal(err)
	}
	// The object shouldn't be deleted, because the persisted object has pending finalizers.
	_, _, err = registry.Delete(ctx, podName, rest.ValidateAllObjectFunc, nil)
	if err != nil {
		t.Fatal(err)
	}
	// The object should still be there
	_, err = registry.Get(ctx, podName, &metav1.GetOptions{})
	if err != nil {
		t.Fatal(err)
	}
}

// TestRetryDeleteValidation checks if the deleteValidation is called again if
// the GuaranteedUpdate in the Delete handler conflicts with a simultaneous
// Update.
func TestRetryDeleteValidation(t *testing.T) {
	testContext := genericapirequest.WithNamespace(genericapirequest.NewContext(), "test")
	destroyFunc, registry := NewTestGenericStoreRegistry(t)
	defer destroyFunc()

	tests := []struct {
		pod     *example.Pod
		deleted bool
	}{
		{
			pod: &example.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "test", Finalizers: []string{"pending"}},
				Spec:       example.PodSpec{NodeName: "machine"},
			},
			deleted: false,
		},

		{
			pod: &example.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "bar", Namespace: "test"},
				Spec:       example.PodSpec{NodeName: "machine"},
			},
			deleted: true,
		},
	}

	for _, test := range tests {
		ready := make(chan struct{})
		updated := make(chan struct{})
		var readyOnce, updatedOnce sync.Once
		var called int
		deleteValidation := func(runtime.Object) error {
			readyOnce.Do(func() {
				close(ready)
			})
			// wait for the update completes
			<-updated
			called++
			return nil
		}

		if _, err := registry.Create(testContext, test.pod, rest.ValidateAllObjectFunc, &metav1.CreateOptions{}); err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		transformer := func(ctx context.Context, newObj runtime.Object, oldObj runtime.Object) (transformedNewObj runtime.Object, err error) {
			<-ready
			pod, ok := newObj.(*example.Pod)
			if !ok {
				t.Fatalf("unexpected object %v", newObj)
			}
			pod.Labels = map[string]string{
				"modified": "true",
			}
			return pod, nil
		}

		go func() {
			// This update will cause the Delete to retry due to conflict.
			_, _, err := registry.Update(testContext, test.pod.Name, rest.DefaultUpdatedObjectInfo(test.pod, transformer), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
			if err != nil {
				t.Fatal(err)
			}
			updatedOnce.Do(func() {
				close(updated)
			})
		}()

		_, deleted, err := registry.Delete(testContext, test.pod.Name, deleteValidation, &metav1.DeleteOptions{})
		if err != nil {
			t.Fatal(err)
		}
		if a, e := deleted, test.deleted; a != e {
			t.Fatalf("expected deleted to be %v, got %v", e, a)
		}
		if called != 2 {
			t.Fatalf("expected deleteValidation to be called twice")
		}
	}
}
