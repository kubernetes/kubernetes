/*
Copyright 2015 The Kubernetes Authors.

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

package tests

import (
	"context"
	"fmt"
	"reflect"
	goruntime "runtime"
	"strconv"
	"sync"
	"testing"
	"time"

	apitesting "k8s.io/apimachinery/pkg/api/apitesting"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	cacherstorage "k8s.io/apiserver/pkg/storage/cacher"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/value"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

const (
	// watchCacheDefaultCapacity syncs watch cache defaultLowerBoundCapacity.
	watchCacheDefaultCapacity = 100
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	pod, ok := obj.(*example.Pod)
	if !ok {
		return nil, nil, fmt.Errorf("not a pod")
	}
	return labels.Set(pod.ObjectMeta.Labels), PodToSelectableFields(pod), nil
}

// PodToSelectableFields returns a field set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func PodToSelectableFields(pod *example.Pod) fields.Set {
	// The purpose of allocation with a given number of elements is to reduce
	// amount of allocations needed to create the fields.Set. If you add any
	// field here or the number of object-meta related fields changes, this should
	// be adjusted.
	podSpecificFieldsSet := make(fields.Set, 5)
	podSpecificFieldsSet["spec.nodeName"] = pod.Spec.NodeName
	podSpecificFieldsSet["spec.restartPolicy"] = string(pod.Spec.RestartPolicy)
	podSpecificFieldsSet["status.phase"] = string(pod.Status.Phase)
	return AddObjectMetaFieldsSet(podSpecificFieldsSet, &pod.ObjectMeta, true)
}

func AddObjectMetaFieldsSet(source fields.Set, objectMeta *metav1.ObjectMeta, hasNamespaceField bool) fields.Set {
	source["metadata.name"] = objectMeta.Name
	if hasNamespaceField {
		source["metadata.namespace"] = objectMeta.Namespace
	}
	return source
}

func newPod() runtime.Object     { return &example.Pod{} }
func newPodList() runtime.Object { return &example.PodList{} }

func newEtcdTestStorage(t *testing.T, prefix string) (*etcd3testing.EtcdTestServer, storage.Interface) {
	server, _ := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	storage := etcd3.New(server.V3Client, apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion), newPod, prefix, schema.GroupResource{Resource: "pods"}, value.IdentityTransformer, true, etcd3.NewDefaultLeaseManagerConfig())
	return server, storage
}

func newTestCacher(s storage.Interface) (*cacherstorage.Cacher, storage.Versioner, error) {
	return newTestCacherWithClock(s, clock.RealClock{})
}

func newTestCacherWithClock(s storage.Interface, clock clock.Clock) (*cacherstorage.Cacher, storage.Versioner, error) {
	prefix := "pods"
	v := etcd3.APIObjectVersioner{}
	config := cacherstorage.Config{
		Storage:        s,
		Versioner:      v,
		ResourcePrefix: prefix,
		KeyFunc:        func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc(prefix, obj) },
		GetAttrsFunc:   GetAttrs,
		NewFunc:        newPod,
		NewListFunc:    newPodList,
		Codec:          codecs.LegacyCodec(examplev1.SchemeGroupVersion),
		Clock:          clock,
	}
	cacher, err := cacherstorage.NewCacherFromConfig(config)
	return cacher, v, err
}

func makeTestPod(name string) *example.Pod {
	return &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: name},
		Spec:       storagetesting.DeepEqualSafePodSpec(),
	}
}

func createPod(s storage.Interface, obj *example.Pod) error {
	key := "pods/" + obj.Namespace + "/" + obj.Name
	out := &example.Pod{}
	return s.Create(context.TODO(), key, obj, out, 0)
}

func updatePod(t *testing.T, s storage.Interface, obj, old *example.Pod) *example.Pod {
	updateFn := func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		return obj.DeepCopyObject(), nil, nil
	}
	key := "pods/" + obj.Namespace + "/" + obj.Name
	if err := s.GuaranteedUpdate(context.TODO(), key, &example.Pod{}, old == nil, nil, updateFn, nil); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	obj.ResourceVersion = ""
	result := &example.Pod{}
	if err := s.Get(context.TODO(), key, storage.GetOptions{}, result); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	return result
}

func TestGet(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	defer server.Terminate(t)
	cacher, _, err := newTestCacher(etcdStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	podFoo := makeTestPod("foo")
	fooCreated := updatePod(t, etcdStorage, podFoo, nil)

	// We pass the ResourceVersion from the above Create() operation.
	result := &example.Pod{}
	if err := cacher.Get(context.TODO(), "pods/ns/foo", storage.GetOptions{IgnoreNotFound: true, ResourceVersion: fooCreated.ResourceVersion}, result); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := *fooCreated, *result; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected: %#v, got: %#v", e, a)
	}

	if err := cacher.Get(context.TODO(), "pods/ns/bar", storage.GetOptions{ResourceVersion: fooCreated.ResourceVersion, IgnoreNotFound: true}, result); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	emptyPod := example.Pod{}
	if e, a := emptyPod, *result; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected: %#v, got: %#v", e, a)
	}

	if err := cacher.Get(context.TODO(), "pods/ns/bar", storage.GetOptions{ResourceVersion: fooCreated.ResourceVersion}, result); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestGetToList(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	defer server.Terminate(t)
	cacher, _, err := newTestCacher(etcdStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	storedObj := updatePod(t, etcdStorage, makeTestPod("foo"), nil)
	key := "pods/" + storedObj.Namespace + "/" + storedObj.Name

	tests := []struct {
		key         string
		pred        storage.SelectionPredicate
		expectedOut []*example.Pod
	}{{ // test GetToList on existing key
		key:         key,
		pred:        storage.Everything,
		expectedOut: []*example.Pod{storedObj},
	}, { // test GetToList on non-existing key
		key:         "/non-existing",
		pred:        storage.Everything,
		expectedOut: nil,
	}, { // test GetToList with matching pod name
		key: "/non-existing",
		pred: storage.SelectionPredicate{
			Label: labels.Everything(),
			Field: fields.ParseSelectorOrDie("metadata.name!=" + storedObj.Name),
			GetAttrs: func(obj runtime.Object) (labels.Set, fields.Set, error) {
				pod := obj.(*example.Pod)
				return nil, fields.Set{"metadata.name": pod.Name}, nil
			},
		},
		expectedOut: nil,
	}}

	for i, tt := range tests {
		out := &example.PodList{}
		err := cacher.GetToList(context.TODO(), tt.key, storage.ListOptions{Predicate: tt.pred}, out)
		if err != nil {
			t.Fatalf("GetToList failed: %v", err)
		}
		if len(out.ResourceVersion) == 0 {
			t.Errorf("#%d: unset resourceVersion", i)
		}
		if len(out.Items) != len(tt.expectedOut) {
			t.Errorf("#%d: length of list want=%d, get=%d", i, len(tt.expectedOut), len(out.Items))
			continue
		}
		for j, wantPod := range tt.expectedOut {
			getPod := &out.Items[j]
			if !reflect.DeepEqual(wantPod, getPod) {
				t.Errorf("#%d: pod want=%#v, get=%#v", i, wantPod, getPod)
			}
		}
	}
}

func TestList(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	defer server.Terminate(t)
	cacher, _, err := newTestCacher(etcdStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	podFoo := makeTestPod("foo")
	podBar := makeTestPod("bar")
	podBaz := makeTestPod("baz")

	podFooPrime := makeTestPod("foo")
	podFooPrime.Spec.NodeName = "fakeNode"

	fooCreated := updatePod(t, etcdStorage, podFoo, nil)
	_ = updatePod(t, etcdStorage, podBar, nil)
	_ = updatePod(t, etcdStorage, podBaz, nil)

	_ = updatePod(t, etcdStorage, podFooPrime, fooCreated)

	// Create a pod in a namespace that contains "ns" as a prefix
	// Make sure it is not returned in a watch of "ns"
	podFooNS2 := makeTestPod("foo")
	podFooNS2.Namespace += "2"
	updatePod(t, etcdStorage, podFooNS2, nil)

	deleted := example.Pod{}
	if err := etcdStorage.Delete(context.TODO(), "pods/ns/bar", &deleted, nil, storage.ValidateAllObjectFunc, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// We first List directly from etcd by passing empty resourceVersion,
	// to get the current etcd resourceVersion.
	rvResult := &example.PodList{}
	if err := cacher.List(context.TODO(), "pods/ns", storage.ListOptions{Predicate: storage.Everything}, rvResult); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	deletedPodRV := rvResult.ListMeta.ResourceVersion

	result := &example.PodList{}
	// We pass the current etcd ResourceVersion received from the above List() operation,
	// since there is not easy way to get ResourceVersion of barPod deletion operation.
	if err := cacher.List(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: deletedPodRV, Predicate: storage.Everything}, result); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if result.ListMeta.ResourceVersion != deletedPodRV {
		t.Errorf("Incorrect resource version: %v", result.ListMeta.ResourceVersion)
	}
	if len(result.Items) != 2 {
		t.Errorf("Unexpected list result: %d", len(result.Items))
	}
	keys := sets.String{}
	for _, item := range result.Items {
		keys.Insert(item.Name)
	}
	if !keys.HasAll("foo", "baz") {
		t.Errorf("Unexpected list result: %#v", result)
	}
	for _, item := range result.Items {
		// unset fields that are set by the infrastructure
		item.ResourceVersion = ""
		item.CreationTimestamp = metav1.Time{}

		if item.Namespace != "ns" {
			t.Errorf("Unexpected namespace: %s", item.Namespace)
		}

		var expected *example.Pod
		switch item.Name {
		case "foo":
			expected = podFooPrime
		case "baz":
			expected = podBaz
		default:
			t.Errorf("Unexpected item: %v", item)
		}
		if e, a := *expected, item; !reflect.DeepEqual(e, a) {
			t.Errorf("Expected: %#v, got: %#v", e, a)
		}
	}
}

// TestTooLargeResourceVersionList ensures that a list request for a resource version higher than available
// in the watch cache completes (does not wait indefinitely) and results in a ResourceVersionTooLarge error.
func TestTooLargeResourceVersionList(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	defer server.Terminate(t)
	cacher, v, err := newTestCacher(etcdStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	podFoo := makeTestPod("foo")
	fooCreated := updatePod(t, etcdStorage, podFoo, nil)

	// Set up List at fooCreated.ResourceVersion + 10
	rv, err := v.ParseResourceVersion(fooCreated.ResourceVersion)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	listRV := strconv.Itoa(int(rv + 10))

	result := &example.PodList{}
	err = cacher.List(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: listRV, Predicate: storage.Everything}, result)
	if !errors.IsTimeout(err) {
		t.Errorf("Unexpected error: %v", err)
	}
	if !storage.IsTooLargeResourceVersion(err) {
		t.Errorf("expected 'Too large resource version' cause in error but got: %v", err)
	}
}

func verifyWatchEvent(t *testing.T, w watch.Interface, eventType watch.EventType, eventObject runtime.Object) {
	_, _, line, _ := goruntime.Caller(1)
	select {
	case event := <-w.ResultChan():
		if e, a := eventType, event.Type; e != a {
			t.Logf("(called from line %d)", line)
			t.Errorf("Expected: %s, got: %s", eventType, event.Type)
		}
		object := event.Object
		if co, ok := object.(runtime.CacheableObject); ok {
			object = co.GetObject()
		}
		if e, a := eventObject, object; !apiequality.Semantic.DeepDerivative(e, a) {
			t.Logf("(called from line %d)", line)
			t.Errorf("Expected (%s): %#v, got: %#v", eventType, e, a)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Logf("(called from line %d)", line)
		t.Errorf("Timed out waiting for an event")
	}
}

type injectListError struct {
	errors int
	storage.Interface
}

func (self *injectListError) List(ctx context.Context, key string, opts storage.ListOptions, listObj runtime.Object) error {
	if self.errors > 0 {
		self.errors--
		return fmt.Errorf("injected error")
	}
	return self.Interface.List(ctx, key, opts, listObj)
}

func TestWatch(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	// Inject one list error to make sure we test the relist case.
	etcdStorage = &injectListError{errors: 1, Interface: etcdStorage}
	defer server.Terminate(t)
	fakeClock := testingclock.NewFakeClock(time.Now())
	cacher, _, err := newTestCacherWithClock(etcdStorage, fakeClock)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	podFoo := makeTestPod("foo")
	podBar := makeTestPod("bar")

	podFooPrime := makeTestPod("foo")
	podFooPrime.Spec.NodeName = "fakeNode"

	podFooBis := makeTestPod("foo")
	podFooBis.Spec.NodeName = "anotherFakeNode"

	podFooNS2 := makeTestPod("foo")
	podFooNS2.Namespace += "2"

	// initialVersion is used to initate the watcher at the beginning of the world,
	// which is not defined precisely in etcd.
	initialVersion, err := cacher.LastSyncResourceVersion()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	startVersion := strconv.Itoa(int(initialVersion))

	// Set up Watch for object "podFoo".
	watcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", storage.ListOptions{ResourceVersion: startVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watcher.Stop()

	// Create in another namespace first to make sure events from other namespaces don't get delivered
	updatePod(t, etcdStorage, podFooNS2, nil)

	fooCreated := updatePod(t, etcdStorage, podFoo, nil)
	_ = updatePod(t, etcdStorage, podBar, nil)
	fooUpdated := updatePod(t, etcdStorage, podFooPrime, fooCreated)

	verifyWatchEvent(t, watcher, watch.Added, podFoo)
	verifyWatchEvent(t, watcher, watch.Modified, podFooPrime)

	initialWatcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", storage.ListOptions{ResourceVersion: fooCreated.ResourceVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer initialWatcher.Stop()

	verifyWatchEvent(t, initialWatcher, watch.Modified, podFooPrime)

	// Now test watch from "now".
	nowWatcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", storage.ListOptions{ResourceVersion: "0", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer nowWatcher.Stop()

	verifyWatchEvent(t, nowWatcher, watch.Added, podFooPrime)

	_ = updatePod(t, etcdStorage, podFooBis, fooUpdated)

	verifyWatchEvent(t, nowWatcher, watch.Modified, podFooBis)

	// Add watchCacheDefaultCapacity events to make current watch cache full.
	// Make start and last event duration exceed eventFreshDuration(current 75s) to ensure watch cache won't expand.
	for i := 0; i < watchCacheDefaultCapacity; i++ {
		fakeClock.SetTime(time.Now().Add(time.Duration(i) * time.Minute))
		podFoo := makeTestPod(fmt.Sprintf("foo-%d", i))
		updatePod(t, etcdStorage, podFoo, nil)
	}

	// Check whether we get too-old error via the watch channel
	tooOldWatcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", storage.ListOptions{ResourceVersion: "1", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Expected no direct error, got %v", err)
	}
	defer tooOldWatcher.Stop()

	// Ensure we get a "Gone" error.
	expectedResourceExpiredError := errors.NewResourceExpired("").ErrStatus
	verifyWatchEvent(t, tooOldWatcher, watch.Error, &expectedResourceExpiredError)
}

func TestWatcherTimeout(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	defer server.Terminate(t)
	cacher, _, err := newTestCacher(etcdStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// initialVersion is used to initate the watcher at the beginning of the world,
	// which is not defined precisely in etcd.
	initialVersion, err := cacher.LastSyncResourceVersion()
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	startVersion := strconv.Itoa(int(initialVersion))

	// Create a number of watchers that will not be reading any result.
	nonReadingWatchers := 50
	for i := 0; i < nonReadingWatchers; i++ {
		watcher, err := cacher.WatchList(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: startVersion, Predicate: storage.Everything})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		defer watcher.Stop()
	}

	// Create a second watcher that will be reading result.
	readingWatcher, err := cacher.WatchList(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: startVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer readingWatcher.Stop()

	startTime := time.Now()
	for i := 1; i <= 22; i++ {
		pod := makeTestPod(strconv.Itoa(i))
		_ = updatePod(t, etcdStorage, pod, nil)
		verifyWatchEvent(t, readingWatcher, watch.Added, pod)
	}
	if time.Since(startTime) > time.Duration(250*nonReadingWatchers)*time.Millisecond {
		t.Errorf("waiting for events took too long: %v", time.Since(startTime))
	}
}

func TestFiltering(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	defer server.Terminate(t)
	cacher, _, err := newTestCacher(etcdStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Ensure that the cacher is initialized, before creating any pods,
	// so that we are sure that all events will be present in cacher.
	syncWatcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", storage.ListOptions{ResourceVersion: "0", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	syncWatcher.Stop()

	podFoo := makeTestPod("foo")
	podFoo.Labels = map[string]string{"filter": "foo"}
	podFooFiltered := makeTestPod("foo")
	podFooPrime := makeTestPod("foo")
	podFooPrime.Labels = map[string]string{"filter": "foo"}
	podFooPrime.Spec.NodeName = "fakeNode"

	podFooNS2 := makeTestPod("foo")
	podFooNS2.Namespace += "2"
	podFooNS2.Labels = map[string]string{"filter": "foo"}

	// Create in another namespace first to make sure events from other namespaces don't get delivered
	updatePod(t, etcdStorage, podFooNS2, nil)

	fooCreated := updatePod(t, etcdStorage, podFoo, nil)
	fooFiltered := updatePod(t, etcdStorage, podFooFiltered, fooCreated)
	fooUnfiltered := updatePod(t, etcdStorage, podFoo, fooFiltered)
	_ = updatePod(t, etcdStorage, podFooPrime, fooUnfiltered)

	deleted := example.Pod{}
	if err := etcdStorage.Delete(context.TODO(), "pods/ns/foo", &deleted, nil, storage.ValidateAllObjectFunc, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Set up Watch for object "podFoo" with label filter set.
	pred := storage.SelectionPredicate{
		Label: labels.SelectorFromSet(labels.Set{"filter": "foo"}),
		Field: fields.Everything(),
		GetAttrs: func(obj runtime.Object) (label labels.Set, field fields.Set, err error) {
			metadata, err := meta.Accessor(obj)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			return labels.Set(metadata.GetLabels()), nil, nil
		},
	}
	watcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", storage.ListOptions{ResourceVersion: fooCreated.ResourceVersion, Predicate: pred})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watcher.Stop()

	verifyWatchEvent(t, watcher, watch.Deleted, podFooFiltered)
	verifyWatchEvent(t, watcher, watch.Added, podFoo)
	verifyWatchEvent(t, watcher, watch.Modified, podFooPrime)
	verifyWatchEvent(t, watcher, watch.Deleted, podFooPrime)
}

func TestEmptyWatchEventCache(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	defer server.Terminate(t)

	// add a few objects
	updatePod(t, etcdStorage, makeTestPod("pod1"), nil)
	updatePod(t, etcdStorage, makeTestPod("pod2"), nil)
	updatePod(t, etcdStorage, makeTestPod("pod3"), nil)
	updatePod(t, etcdStorage, makeTestPod("pod4"), nil)
	updatePod(t, etcdStorage, makeTestPod("pod5"), nil)

	fooCreated := updatePod(t, etcdStorage, makeTestPod("foo"), nil)

	cacher, v, err := newTestCacher(etcdStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// get rv of last pod created
	rv, err := v.ParseResourceVersion(fooCreated.ResourceVersion)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// We now have a cacher with an empty cache of watch events and a resourceVersion of rv.
	// It should support establishing watches from rv and higher, but not older.

	{
		watcher, err := cacher.Watch(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: strconv.Itoa(int(rv - 1)), Predicate: storage.Everything})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		defer watcher.Stop()
		expectedResourceExpiredError := errors.NewResourceExpired("").ErrStatus
		verifyWatchEvent(t, watcher, watch.Error, &expectedResourceExpiredError)
	}

	{
		watcher, err := cacher.Watch(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: strconv.Itoa(int(rv + 1)), Predicate: storage.Everything})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		defer watcher.Stop()
		select {
		case e := <-watcher.ResultChan():
			t.Errorf("unexpected event %#v", e)
		case <-time.After(3 * time.Second):
			// watch from rv+1 remained established successfully
		}
	}

	{
		watcher, err := cacher.Watch(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: strconv.Itoa(int(rv)), Predicate: storage.Everything})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		defer watcher.Stop()
		select {
		case e := <-watcher.ResultChan():
			t.Errorf("unexpected event %#v", e)
		case <-time.After(3 * time.Second):
			// watch from rv remained established successfully
		}
	}
}

func TestRandomWatchDeliver(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	defer server.Terminate(t)
	cacher, v, err := newTestCacher(etcdStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	fooCreated := updatePod(t, etcdStorage, makeTestPod("foo"), nil)
	rv, err := v.ParseResourceVersion(fooCreated.ResourceVersion)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	startVersion := strconv.Itoa(int(rv))

	watcher, err := cacher.WatchList(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: startVersion, Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// Now we can create exactly 21 events that should be delivered
	// to the watcher, before it will completely block cacher and as
	// a result will be dropped.
	for i := 0; i < 21; i++ {
		updatePod(t, etcdStorage, makeTestPod(fmt.Sprintf("foo-%d", i)), nil)
	}

	// Now stop the watcher and check if the consecutive events are being delivered.
	watcher.Stop()

	watched := 0
	for {
		event, ok := <-watcher.ResultChan()
		if !ok {
			break
		}
		object := event.Object
		if co, ok := object.(runtime.CacheableObject); ok {
			object = co.GetObject()
		}
		if a, e := object.(*example.Pod).Name, fmt.Sprintf("foo-%d", watched); e != a {
			t.Errorf("Unexpected object watched: %s, expected %s", a, e)
		}
		watched++
	}
}

func TestCacherListerWatcher(t *testing.T) {
	prefix := "pods"
	fn := func() runtime.Object { return &example.PodList{} }
	server, store := newEtcdTestStorage(t, prefix)
	defer server.Terminate(t)

	podFoo := makeTestPod("foo")
	podBar := makeTestPod("bar")
	podBaz := makeTestPod("baz")

	_ = updatePod(t, store, podFoo, nil)
	_ = updatePod(t, store, podBar, nil)
	_ = updatePod(t, store, podBaz, nil)

	lw := cacherstorage.NewCacherListerWatcher(store, prefix, fn)

	obj, err := lw.List(metav1.ListOptions{})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	pl, ok := obj.(*example.PodList)
	if !ok {
		t.Fatalf("Expected PodList but got %v", pl)
	}
	if len(pl.Items) != 3 {
		t.Errorf("Expected PodList of length 3 but got %d", len(pl.Items))
	}
}

func TestCacherListerWatcherPagination(t *testing.T) {
	prefix := "pods"
	fn := func() runtime.Object { return &example.PodList{} }
	server, store := newEtcdTestStorage(t, prefix)
	defer server.Terminate(t)

	podFoo := makeTestPod("foo")
	podBar := makeTestPod("bar")
	podBaz := makeTestPod("baz")

	_ = updatePod(t, store, podFoo, nil)
	_ = updatePod(t, store, podBar, nil)
	_ = updatePod(t, store, podBaz, nil)

	lw := cacherstorage.NewCacherListerWatcher(store, prefix, fn)

	obj1, err := lw.List(metav1.ListOptions{Limit: 2})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	limit1, ok := obj1.(*example.PodList)
	if !ok {
		t.Fatalf("Expected PodList but got %v", limit1)
	}
	if len(limit1.Items) != 2 {
		t.Errorf("Expected PodList of length 2 but got %d", len(limit1.Items))
	}
	if limit1.Continue == "" {
		t.Errorf("Expected list to have Continue but got none")
	}
	obj2, err := lw.List(metav1.ListOptions{Limit: 2, Continue: limit1.Continue})
	if err != nil {
		t.Fatalf("List failed: %v", err)
	}
	limit2, ok := obj2.(*example.PodList)
	if !ok {
		t.Fatalf("Expected PodList but got %v", limit2)
	}
	if limit2.Continue != "" {
		t.Errorf("Expected list not to have Continue, but got %s", limit1.Continue)
	}

	if limit1.Items[0].Name != podBar.Name {
		t.Errorf("Expected list1.Items[0] to be %s but got %s", podBar.Name, limit1.Items[0].Name)
	}
	if limit1.Items[1].Name != podBaz.Name {
		t.Errorf("Expected list1.Items[1] to be %s but got %s", podBaz.Name, limit1.Items[1].Name)
	}
	if limit2.Items[0].Name != podFoo.Name {
		t.Errorf("Expected list2.Items[0] to be %s but got %s", podFoo.Name, limit2.Items[0].Name)
	}

}

func TestWatchDispatchBookmarkEvents(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchBookmark, true)()

	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	defer server.Terminate(t)
	cacher, v, err := newTestCacher(etcdStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	fooCreated := updatePod(t, etcdStorage, makeTestPod("foo"), nil)
	rv, err := v.ParseResourceVersion(fooCreated.ResourceVersion)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	startVersion := strconv.Itoa(int(rv))

	tests := []struct {
		timeout            time.Duration
		expected           bool
		allowWatchBookmark bool
	}{
		{ // test old client won't get Bookmark event
			timeout:            3 * time.Second,
			expected:           false,
			allowWatchBookmark: false,
		},
		{
			timeout:            3 * time.Second,
			expected:           true,
			allowWatchBookmark: true,
		},
	}

	for i, c := range tests {
		pred := storage.Everything
		pred.AllowWatchBookmarks = c.allowWatchBookmark
		ctx, _ := context.WithTimeout(context.Background(), c.timeout)
		watcher, err := cacher.Watch(ctx, "pods/ns/foo", storage.ListOptions{ResourceVersion: startVersion, Predicate: pred})
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		// Create events of other pods
		updatePod(t, etcdStorage, makeTestPod(fmt.Sprintf("foo-whatever-%d", i)), nil)

		// Now wait for Bookmark event
		select {
		case event, ok := <-watcher.ResultChan():
			if !ok && c.expected {
				t.Errorf("Unexpected object watched (no objects)")
			}
			if c.expected && event.Type != watch.Bookmark {
				t.Errorf("Unexpected object watched %#v", event)
			}
		case <-time.After(time.Second * 3):
			if c.expected {
				t.Errorf("Unexpected object watched (timeout)")
			}
		}
		watcher.Stop()
	}
}

func TestWatchBookmarksWithCorrectResourceVersion(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchBookmark, true)()

	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	defer server.Terminate(t)
	cacher, v, err := newTestCacher(etcdStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	pred := storage.Everything
	pred.AllowWatchBookmarks = true
	ctx, _ := context.WithTimeout(context.Background(), 3*time.Second)
	watcher, err := cacher.WatchList(ctx, "pods/ns", storage.ListOptions{ResourceVersion: "0", Predicate: pred})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watcher.Stop()

	done := make(chan struct{})
	errc := make(chan error, 1)
	var wg sync.WaitGroup
	wg.Add(1)
	defer wg.Wait()   // We must wait for the waitgroup to exit before we terminate the cache or the server in prior defers
	defer close(done) // call close first, so the goroutine knows to exit
	go func() {
		defer wg.Done()
		for i := 0; i < 100; i++ {
			select {
			case <-done:
				return
			default:
				pod := fmt.Sprintf("foo-%d", i)
				err := createPod(etcdStorage, makeTestPod(pod))
				if err != nil {
					errc <- fmt.Errorf("failed to create pod %v: %v", pod, err)
					return
				}
				time.Sleep(time.Second / 100)
			}
		}
	}()

	bookmarkReceived := false
	lastObservedResourceVersion := uint64(0)

	for {
		select {
		case err := <-errc:
			t.Fatal(err)
		case event, ok := <-watcher.ResultChan():
			if !ok {
				// Make sure we have received a bookmark event
				if !bookmarkReceived {
					t.Fatalf("Unpexected error, we did not received a bookmark event")
				}
				return
			}
			rv, err := v.ObjectResourceVersion(event.Object)
			if err != nil {
				t.Fatalf("failed to parse resourceVersion from %#v", event)
			}
			if event.Type == watch.Bookmark {
				bookmarkReceived = true
				// bookmark event has a RV greater than or equal to the before one
				if rv < lastObservedResourceVersion {
					t.Fatalf("Unexpected bookmark resourceVersion %v less than observed %v)", rv, lastObservedResourceVersion)
				}
			} else {
				// non-bookmark event has a RV greater than anything before
				if rv <= lastObservedResourceVersion {
					t.Fatalf("Unexpected event resourceVersion %v less than or equal to bookmark %v)", rv, lastObservedResourceVersion)
				}
			}
			lastObservedResourceVersion = rv
		}
	}
}
