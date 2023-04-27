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

func newEtcdTestStorage(t *testing.T, prefix string, pagingEnabled bool) (*etcd3testing.EtcdTestServer, storage.Interface) {
	server, _ := etcd3testing.NewUnsecuredEtcd3TestClientServer(t)
	storage := etcd3.New(
		server.V3Client,
		apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion),
		newPod,
		prefix,
		schema.GroupResource{Resource: "pods"},
		identity.NewEncryptCheckTransformer(),
		pagingEnabled,
		etcd3.NewDefaultLeaseManagerConfig())
	return server, storage
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

func checkStorageInvariants(ctx context.Context, t *testing.T, key string) {
	// No-op function since cacher simply passes object creation to the underlying storage.
}

func TestCreate(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCreate(ctx, t, cacher, checkStorageInvariants)
}

func TestCreateWithTTL(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCreateWithTTL(ctx, t, cacher)
}

func TestCreateWithKeyExist(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCreateWithKeyExist(ctx, t, cacher)
}

func TestGet(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGet(ctx, t, cacher)
}

func TestUnconditionalDelete(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestUnconditionalDelete(ctx, t, cacher)
}

func TestConditionalDelete(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestConditionalDelete(ctx, t, cacher)
}

func TestDeleteWithSuggestion(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDeleteWithSuggestion(ctx, t, cacher)
}

func TestDeleteWithSuggestionAndConflict(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDeleteWithSuggestionAndConflict(ctx, t, cacher)
}

func TestDeleteWithSuggestionOfDeletedObject(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDeleteWithSuggestionOfDeletedObject(ctx, t, cacher)
}

func TestValidateDeletionWithSuggestion(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestValidateDeletionWithSuggestion(ctx, t, cacher)
}

func TestPreconditionalDeleteWithSuggestion(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestPreconditionalDeleteWithSuggestion(ctx, t, cacher)
}

func TestList(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestList(ctx, t, cacher, true)
}

func TestListWithoutPaging(t *testing.T) {
	ctx, cacher, terminate := testSetup(t, withoutPaging)
	t.Cleanup(terminate)
	storagetesting.RunTestListWithoutPaging(ctx, t, cacher)
}

func TestGetListNonRecursive(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGetListNonRecursive(ctx, t, cacher)
}

func checkStorageCalls(t *testing.T, pageSize, estimatedProcessedObjects uint64) {
	// No-op function for now, since cacher passes pagination calls to underlying storage.
}

func TestListContinuation(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListContinuation(ctx, t, cacher, checkStorageCalls)
}

func TestListPaginationRareObject(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListPaginationRareObject(ctx, t, cacher, checkStorageCalls)
}

func TestListContinuationWithFilter(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestListContinuationWithFilter(ctx, t, cacher, checkStorageCalls)
}

func TestListInconsistentContinuation(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestConsistentList(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestGuaranteedUpdate(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestGuaranteedUpdateWithTTL(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGuaranteedUpdateWithTTL(ctx, t, cacher)
}

func TestGuaranteedUpdateChecksStoredData(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestGuaranteedUpdateWithConflict(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGuaranteedUpdateWithConflict(ctx, t, cacher)
}

func TestGuaranteedUpdateWithSuggestionAndConflict(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestGuaranteedUpdateWithSuggestionAndConflict(ctx, t, cacher)
}

func TestTransformationFailure(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestCount(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestCount(ctx, t, cacher)
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

func TestWatchFromZero(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestDeleteTriggerWatch(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDeleteTriggerWatch(ctx, t, cacher)
}

func TestWatchFromNonZero(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatchFromNonZero(ctx, t, cacher)
}

func TestDelayedWatchDelivery(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestDelayedWatchDelivery(ctx, t, cacher)
}

func TestWatchError(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestWatchContextCancel(t *testing.T) {
	// TODO(#109831): Enable use of this test and run it.
}

func TestWatcherTimeout(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatcherTimeout(ctx, t, cacher)
}

func TestWatchDeleteEventObjectHaveLatestRV(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatchDeleteEventObjectHaveLatestRV(ctx, t, cacher)
}

func TestWatchInitializationSignal(t *testing.T) {
	ctx, cacher, terminate := testSetup(t)
	t.Cleanup(terminate)
	storagetesting.RunTestWatchInitializationSignal(ctx, t, cacher)
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

// TODO(wojtek-t): We should extend the generic RunTestWatch test to cover the
// scenarios that are not yet covered by it and get rid of this test.
func TestWatchDeprecated(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix(), true)
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
	pagingEnabled  bool
	clock          clock.Clock
}

type setupOption func(*setupOptions)

func withDefaults(options *setupOptions) {
	prefix := "/pods"

	options.resourcePrefix = prefix
	options.keyFunc = func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc(prefix, obj) }
	options.pagingEnabled = true
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

func withoutPaging(options *setupOptions) {
	options.pagingEnabled = false
}

func testSetup(t *testing.T, opts ...setupOption) (context.Context, *cacherstorage.Cacher, tearDownFunc) {
	setupOpts := setupOptions{}
	opts = append([]setupOption{withDefaults}, opts...)
	for _, opt := range opts {
		opt(&setupOpts)
	}

	server, etcdStorage := newEtcdTestStorage(t, etcd3testing.PathPrefix(), setupOpts.pagingEnabled)
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
