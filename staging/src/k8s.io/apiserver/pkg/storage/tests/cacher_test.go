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
	"fmt"
	"reflect"
	goruntime "runtime"
	"strconv"
	"testing"
	"time"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	apitesting "k8s.io/apimachinery/pkg/api/testing"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage"
	etcdstorage "k8s.io/apiserver/pkg/storage/etcd"
	"k8s.io/apiserver/pkg/storage/etcd/etcdtest"
	etcdtesting "k8s.io/apiserver/pkg/storage/etcd/testing"
	"k8s.io/apiserver/pkg/storage/etcd3"
	"k8s.io/apiserver/pkg/storage/value"

	"golang.org/x/net/context"

	"k8s.io/apimachinery/pkg/runtime/serializer"
)

var (
	scheme = runtime.NewScheme()
	codecs = serializer.NewCodecFactory(scheme)
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	example.AddToScheme(scheme)
	examplev1.AddToScheme(scheme)
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, bool, error) {
	pod, ok := obj.(*example.Pod)
	if !ok {
		return nil, nil, false, fmt.Errorf("not a pod")
	}
	return labels.Set(pod.ObjectMeta.Labels), PodToSelectableFields(pod), pod.Initializers != nil, nil
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

func newEtcdTestStorage(t *testing.T, prefix string) (*etcdtesting.EtcdTestServer, storage.Interface) {
	server, _ := etcdtesting.NewUnsecuredEtcd3TestClientServer(t, scheme)
	storage := etcd3.New(server.V3Client, apitesting.TestCodec(codecs, examplev1.SchemeGroupVersion), prefix, value.IdentityTransformer)
	return server, storage
}

func newTestCacher(s storage.Interface, cap int) *storage.Cacher {
	prefix := "pods"
	config := storage.CacherConfig{
		CacheCapacity:  cap,
		Storage:        s,
		Versioner:      etcdstorage.APIObjectVersioner{},
		Copier:         scheme,
		Type:           &example.Pod{},
		ResourcePrefix: prefix,
		KeyFunc:        func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc(prefix, obj) },
		GetAttrsFunc:   GetAttrs,
		NewListFunc:    func() runtime.Object { return &example.PodList{} },
		Codec:          codecs.LegacyCodec(examplev1.SchemeGroupVersion),
	}
	return storage.NewCacherFromConfig(config)
}

func makeTestPod(name string) *example.Pod {
	return &example.Pod{
		ObjectMeta: metav1.ObjectMeta{Namespace: "ns", Name: name},
		Spec:       DeepEqualSafePodSpec(),
	}
}

func updatePod(t *testing.T, s storage.Interface, obj, old *example.Pod) *example.Pod {
	updateFn := func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		return obj.DeepCopyObject(), nil, nil
	}
	key := "pods/" + obj.Namespace + "/" + obj.Name
	if err := s.GuaranteedUpdate(context.TODO(), key, &example.Pod{}, old == nil, nil, updateFn); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	obj.ResourceVersion = ""
	result := &example.Pod{}
	if err := s.Get(context.TODO(), key, "", result, false); err != nil {
		t.Errorf("unexpected error: %v", err)
	}
	return result
}

func TestGet(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcdtest.PathPrefix())
	defer server.Terminate(t)
	cacher := newTestCacher(etcdStorage, 10)
	defer cacher.Stop()

	podFoo := makeTestPod("foo")
	fooCreated := updatePod(t, etcdStorage, podFoo, nil)

	// We pass the ResourceVersion from the above Create() operation.
	result := &example.Pod{}
	if err := cacher.Get(context.TODO(), "pods/ns/foo", fooCreated.ResourceVersion, result, true); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	if e, a := *fooCreated, *result; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected: %#v, got: %#v", e, a)
	}

	if err := cacher.Get(context.TODO(), "pods/ns/bar", fooCreated.ResourceVersion, result, true); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	emptyPod := example.Pod{}
	if e, a := emptyPod, *result; !reflect.DeepEqual(e, a) {
		t.Errorf("Expected: %#v, got: %#v", e, a)
	}

	if err := cacher.Get(context.TODO(), "pods/ns/bar", fooCreated.ResourceVersion, result, false); !storage.IsNotFound(err) {
		t.Errorf("Unexpected error: %v", err)
	}
}

func TestList(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcdtest.PathPrefix())
	defer server.Terminate(t)
	cacher := newTestCacher(etcdStorage, 10)
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
	if err := etcdStorage.Delete(context.TODO(), "pods/ns/bar", &deleted, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// We first List directly from etcd by passing empty resourceVersion,
	// to get the current etcd resourceVersion.
	rvResult := &example.PodList{}
	if err := cacher.List(context.TODO(), "pods/ns", "", storage.Everything, rvResult); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}
	deletedPodRV := rvResult.ListMeta.ResourceVersion

	result := &example.PodList{}
	// We pass the current etcd ResourceVersion received from the above List() operation,
	// since there is not easy way to get ResourceVersion of barPod deletion operation.
	if err := cacher.List(context.TODO(), "pods/ns", deletedPodRV, storage.Everything, result); err != nil {
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

func TestInfiniteList(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcdtest.PathPrefix())
	defer server.Terminate(t)
	cacher := newTestCacher(etcdStorage, 10)
	defer cacher.Stop()

	podFoo := makeTestPod("foo")
	fooCreated := updatePod(t, etcdStorage, podFoo, nil)

	// Set up List at fooCreated.ResourceVersion + 10
	rv, err := storage.ParseWatchResourceVersion(fooCreated.ResourceVersion)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	listRV := strconv.Itoa(int(rv + 10))

	result := &example.PodList{}
	err = cacher.List(context.TODO(), "pods/ns", listRV, storage.Everything, result)
	if !errors.IsTimeout(err) {
		t.Errorf("Unexpected error: %v", err)
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
		if e, a := eventObject, event.Object; !apiequality.Semantic.DeepDerivative(e, a) {
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

func (self *injectListError) List(ctx context.Context, key string, resourceVersion string, p storage.SelectionPredicate, listObj runtime.Object) error {
	if self.errors > 0 {
		self.errors--
		return fmt.Errorf("injected error")
	}
	return self.Interface.List(ctx, key, resourceVersion, p, listObj)
}

func TestWatch(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcdtest.PathPrefix())
	// Inject one list error to make sure we test the relist case.
	etcdStorage = &injectListError{errors: 1, Interface: etcdStorage}
	defer server.Terminate(t)
	cacher := newTestCacher(etcdStorage, 3) // small capacity to trigger "too old version" error
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
	watcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", startVersion, storage.Everything)
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

	// Check whether we get too-old error via the watch channel
	tooOldWatcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", "1", storage.Everything)
	if err != nil {
		t.Fatalf("Expected no direct error, got %v", err)
	}
	defer tooOldWatcher.Stop()
	// Ensure we get a "Gone" error
	expectedGoneError := errors.NewGone("").ErrStatus
	verifyWatchEvent(t, tooOldWatcher, watch.Error, &expectedGoneError)

	initialWatcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", fooCreated.ResourceVersion, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer initialWatcher.Stop()

	verifyWatchEvent(t, initialWatcher, watch.Modified, podFooPrime)

	// Now test watch from "now".
	nowWatcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", "0", storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer nowWatcher.Stop()

	verifyWatchEvent(t, nowWatcher, watch.Added, podFooPrime)

	_ = updatePod(t, etcdStorage, podFooBis, fooUpdated)

	verifyWatchEvent(t, nowWatcher, watch.Modified, podFooBis)
}

func TestWatcherTimeout(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcdtest.PathPrefix())
	defer server.Terminate(t)
	cacher := newTestCacher(etcdStorage, 10)
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
		watcher, err := cacher.WatchList(context.TODO(), "pods/ns", startVersion, storage.Everything)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		defer watcher.Stop()
	}

	// Create a second watcher that will be reading result.
	readingWatcher, err := cacher.WatchList(context.TODO(), "pods/ns", startVersion, storage.Everything)
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
	server, etcdStorage := newEtcdTestStorage(t, etcdtest.PathPrefix())
	defer server.Terminate(t)
	cacher := newTestCacher(etcdStorage, 10)
	defer cacher.Stop()

	// Ensure that the cacher is initialized, before creating any pods,
	// so that we are sure that all events will be present in cacher.
	syncWatcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", "0", storage.Everything)
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
	if err := etcdStorage.Delete(context.TODO(), "pods/ns/foo", &deleted, nil); err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	// Set up Watch for object "podFoo" with label filter set.
	pred := storage.SelectionPredicate{
		Label: labels.SelectorFromSet(labels.Set{"filter": "foo"}),
		Field: fields.Everything(),
		GetAttrs: func(obj runtime.Object) (label labels.Set, field fields.Set, uninitialized bool, err error) {
			metadata, err := meta.Accessor(obj)
			if err != nil {
				t.Fatalf("Unexpected error: %v", err)
			}
			return labels.Set(metadata.GetLabels()), nil, metadata.GetInitializers() != nil, nil
		},
	}
	watcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", fooCreated.ResourceVersion, pred)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watcher.Stop()

	verifyWatchEvent(t, watcher, watch.Deleted, podFooFiltered)
	verifyWatchEvent(t, watcher, watch.Added, podFoo)
	verifyWatchEvent(t, watcher, watch.Modified, podFooPrime)
	verifyWatchEvent(t, watcher, watch.Deleted, podFooPrime)
}

func TestStartingResourceVersion(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcdtest.PathPrefix())
	defer server.Terminate(t)
	cacher := newTestCacher(etcdStorage, 10)
	defer cacher.Stop()

	// add 1 object
	podFoo := makeTestPod("foo")
	fooCreated := updatePod(t, etcdStorage, podFoo, nil)

	// Set up Watch starting at fooCreated.ResourceVersion + 10
	rv, err := storage.ParseWatchResourceVersion(fooCreated.ResourceVersion)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	rv += 10
	startVersion := strconv.Itoa(int(rv))

	watcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", startVersion, storage.Everything)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watcher.Stop()

	lastFoo := fooCreated
	for i := 0; i < 11; i++ {
		podFooForUpdate := makeTestPod("foo")
		podFooForUpdate.Labels = map[string]string{"foo": strconv.Itoa(i)}
		lastFoo = updatePod(t, etcdStorage, podFooForUpdate, lastFoo)
	}

	select {
	case e := <-watcher.ResultChan():
		pod := e.Object.(*example.Pod)
		podRV, err := storage.ParseWatchResourceVersion(pod.ResourceVersion)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// event should have at least rv + 1, since we're starting the watch at rv
		if podRV <= rv {
			t.Errorf("expected event with resourceVersion of at least %d, got %d", rv+1, podRV)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("timed out waiting for event")
	}
}

func TestEmptyWatchEventCache(t *testing.T) {
	server, etcdStorage := newEtcdTestStorage(t, etcdtest.PathPrefix())
	defer server.Terminate(t)

	// add a few objects
	updatePod(t, etcdStorage, makeTestPod("pod1"), nil)
	updatePod(t, etcdStorage, makeTestPod("pod2"), nil)
	updatePod(t, etcdStorage, makeTestPod("pod3"), nil)
	updatePod(t, etcdStorage, makeTestPod("pod4"), nil)
	updatePod(t, etcdStorage, makeTestPod("pod5"), nil)

	fooCreated := updatePod(t, etcdStorage, makeTestPod("foo"), nil)

	// get rv of last pod created
	rv, err := storage.ParseWatchResourceVersion(fooCreated.ResourceVersion)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	cacher := newTestCacher(etcdStorage, 10)
	defer cacher.Stop()

	// We now have a cacher with an empty cache of watch events and a resourceVersion of rv.
	// It should support establishing watches from rv and higher, but not older.

	{
		watcher, err := cacher.Watch(context.TODO(), "pods/ns", strconv.Itoa(int(rv-1)), storage.Everything)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}
		defer watcher.Stop()
		expectedGoneError := errors.NewGone("").ErrStatus
		verifyWatchEvent(t, watcher, watch.Error, &expectedGoneError)
	}

	{
		watcher, err := cacher.Watch(context.TODO(), "pods/ns", strconv.Itoa(int(rv+1)), storage.Everything)
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
		watcher, err := cacher.Watch(context.TODO(), "pods/ns", strconv.Itoa(int(rv)), storage.Everything)
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
	server, etcdStorage := newEtcdTestStorage(t, etcdtest.PathPrefix())
	defer server.Terminate(t)
	cacher := newTestCacher(etcdStorage, 10)
	defer cacher.Stop()

	fooCreated := updatePod(t, etcdStorage, makeTestPod("foo"), nil)
	rv, err := storage.ParseWatchResourceVersion(fooCreated.ResourceVersion)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	startVersion := strconv.Itoa(int(rv))

	watcher, err := cacher.WatchList(context.TODO(), "pods/ns", startVersion, storage.Everything)
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
		if a, e := event.Object.(*example.Pod).Name, fmt.Sprintf("foo-%d", watched); e != a {
			t.Errorf("Unexpected object watched: %s, expected %s", a, e)
		}
		watched++
	}
}
