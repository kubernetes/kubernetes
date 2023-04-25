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
	goruntime "runtime"
	"strconv"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/api/apitesting"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage"
	cacherstorage "k8s.io/apiserver/pkg/storage/cacher"
	"k8s.io/apiserver/pkg/storage/etcd3"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	storagetesting "k8s.io/apiserver/pkg/storage/testing"
	"k8s.io/apiserver/pkg/storage/value/encrypt/identity"
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

// GetPodAttrs returns labels and fields of a given object for filtering purposes.
func GetPodAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
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
	storage := etcd3.New(server.V3Client, apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion), newPod, prefix, schema.GroupResource{Resource: "pods"}, identity.NewEncryptCheckTransformer(), true, etcd3.NewDefaultLeaseManagerConfig())
	return server, storage
}

func newTestCacher(s storage.Interface) (*cacherstorage.Cacher, storage.Versioner, error) {
	return newTestCacherWithClock(s, clock.RealClock{})
}

func newTestCacherWithClock(s storage.Interface, clock clock.Clock) (*cacherstorage.Cacher, storage.Versioner, error) {
	prefix := "pods"
	v := storage.APIObjectVersioner{}
	config := cacherstorage.Config{
		Storage:        s,
		Versioner:      v,
		GroupResource:  schema.GroupResource{Resource: "pods"},
		ResourcePrefix: prefix,
		KeyFunc:        func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc(prefix, obj) },
		GetAttrsFunc:   GetPodAttrs,
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
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGet(ctx, t, cacher)
}

func TestGetListNonRecursive(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGetListNonRecursive(ctx, t, cacher)
}

func TestList(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestList(ctx, t, cacher, true)
}

func TestClusterScopedWatch(t *testing.T) {
	ctx, cacher, terminate := testSetup(t, withClusterScopedKeyFunc, withSpecNodeNameIndexerFuncs)
	t.Cleanup(terminate)
	storagetesting.RunTestClusterScopedWatch(ctx, t, cacher)
}

func TestNamespaceScopedWatch(t *testing.T) {
	ctx, cacher, terminate := testSetup(t, withSpecNodeNameIndexerFuncs)
	t.Cleanup(terminate)
	storagetesting.RunTestNamespaceScopedWatch(ctx, t, cacher)
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

func TestWatch(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatch(ctx, t, cacher)
}

// TODO(wojtek-t): We should extend the generic RunTestWatch test to cover the
// scenarios that are not yet covered by it and get rid of this test.
func TestWatchDeprecated(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
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
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatcherTimeout(ctx, t, cacher)
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

func TestDelayedWatchDelivery(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDelayedWatchDelivery(ctx, t, cacher)
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
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatchDispatchBookmarkEvents(ctx, t, cacher)
}

func TestWatchBookmarksWithCorrectResourceVersion(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestOptionalWatchBookmarksWithCorrectResourceVersion(ctx, t, cacher)
}

// ===================================================
// Test-setup related function are following.
// ===================================================

type tearDownFunc func()

type setupOptions struct {
	resourcePrefix string
	keyFunc        func(runtime.Object) (string, error)
	indexerFuncs   map[string]storage.IndexerFunc
	clock          clock.Clock
}

type setupOption func(*setupOptions)

func withDefaults(options *setupOptions) {
	prefix := "/pods"

	options.resourcePrefix = prefix
	options.keyFunc = func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc(prefix, obj) }
	options.clock = clock.RealClock{}
}

func withClusterScopedKeyFunc(options *setupOptions) {
	options.keyFunc = func(obj runtime.Object) (string, error) {
		return storage.NoNamespaceKeyFunc(options.resourcePrefix, obj)
	}
}

func withSpecNodeNameIndexerFuncs(options *setupOptions) {
	options.indexerFuncs = map[string]storage.IndexerFunc{
		"spec.nodeName": func(obj runtime.Object) string {
			pod, ok := obj.(*example.Pod)
			if !ok {
				return ""
			}
			return pod.Spec.NodeName
		},
	}
}

func testSetup(t *testing.T, opts ...setupOption) (context.Context, *cacherstorage.Cacher, tearDownFunc) {
	setupOpts := setupOptions{}
	opts = append([]setupOption{withDefaults}, opts...)
	for _, opt := range opts {
		opt(&setupOpts)
	}

	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix())
	// Inject one list error to make sure we test the relist case.
	wrappedStorage := &storagetesting.StorageInjectingListErrors{
		Interface: etcdStorage,
		Errors:    1,
	}

	config := cacherstorage.Config{
		Storage:        wrappedStorage,
		Versioner:      storage.APIObjectVersioner{},
		GroupResource:  schema.GroupResource{Resource: "pods"},
		ResourcePrefix: setupOpts.resourcePrefix,
		KeyFunc:        setupOpts.keyFunc,
		GetAttrsFunc:   GetPodAttrs,
		NewFunc:        newPod,
		NewListFunc:    newPodList,
		IndexerFuncs:   setupOpts.indexerFuncs,
		Codec:          codecs.LegacyCodec(examplev1.SchemeGroupVersion),
		Clock:          setupOpts.clock,
	}
	cacher, err := cacherstorage.NewCacherFromConfig(config)
	if err != nil {
		t.Fatalf("Failed to initialize cacher: %v", err)
	}
	ctx := context.Background()
	terminate := func() {
		cacher.Stop()
		server.Terminate(t)
	}

	// Since some tests depend on the fact that GetList shouldn't fail,
	// we wait until the error from the underlying storage is consumed.
	if err := wait.PollInfinite(100*time.Millisecond, wrappedStorage.ErrorsConsumed); err != nil {
		t.Fatalf("Failed to inject list errors: %v", err)
	}

	return ctx, cacher, terminate
}
