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

package cacher

import (
	"context"
	"errors"
	"fmt"
	"reflect"
	goruntime "runtime"
	"strconv"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/diff"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/storage"
	utilflowcontrol "k8s.io/apiserver/pkg/util/flowcontrol"
	"k8s.io/utils/clock"
	testingclock "k8s.io/utils/clock/testing"
)

var (
	objectType = reflect.TypeOf(&v1.Pod{})
)

// verifies the cacheWatcher.process goroutine is properly cleaned up even if
// the writes to cacheWatcher.result channel is blocked.
func TestCacheWatcherCleanupNotBlockedByResult(t *testing.T) {
	var lock sync.RWMutex
	var w *cacheWatcher
	count := 0
	filter := func(string, labels.Set, fields.Set) bool { return true }
	forget := func() {
		lock.Lock()
		defer lock.Unlock()
		count++
		// forget() has to stop the watcher, as only stopping the watcher
		// triggers stopping the process() goroutine which we are in the
		// end waiting for in this test.
		w.stopThreadUnsafe()
	}
	initEvents := []*watchCacheEvent{
		{Object: &v1.Pod{}},
		{Object: &v1.Pod{}},
	}
	// set the size of the buffer of w.result to 0, so that the writes to
	// w.result is blocked.
	w = newCacheWatcher(0, filter, forget, testVersioner{}, time.Now(), false, objectType, "")
	go w.processEvents(context.Background(), initEvents, 0)
	w.Stop()
	if err := wait.PollImmediate(1*time.Second, 5*time.Second, func() (bool, error) {
		lock.RLock()
		defer lock.RUnlock()
		return count == 2, nil
	}); err != nil {
		t.Fatalf("expected forget() to be called twice, because sendWatchCacheEvent should not be blocked by the result channel: %v", err)
	}
}

func TestCacheWatcherHandlesFiltering(t *testing.T) {
	filter := func(_ string, _ labels.Set, field fields.Set) bool {
		return field["spec.nodeName"] == "host"
	}
	forget := func() {}

	testCases := []struct {
		events   []*watchCacheEvent
		expected []watch.Event
	}{
		// properly handle starting with the filter, then being deleted, then re-added
		{
			events: []*watchCacheEvent{
				{
					Type:            watch.Added,
					Object:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
					ObjFields:       fields.Set{"spec.nodeName": "host"},
					ResourceVersion: 1,
				},
				{
					Type:            watch.Modified,
					PrevObject:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
					PrevObjFields:   fields.Set{"spec.nodeName": "host"},
					Object:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},
					ObjFields:       fields.Set{"spec.nodeName": ""},
					ResourceVersion: 2,
				},
				{
					Type:            watch.Modified,
					PrevObject:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},
					PrevObjFields:   fields.Set{"spec.nodeName": ""},
					Object:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "3"}},
					ObjFields:       fields.Set{"spec.nodeName": "host"},
					ResourceVersion: 3,
				},
			},
			expected: []watch.Event{
				{Type: watch.Added, Object: &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}}},
				{Type: watch.Deleted, Object: &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}}},
				{Type: watch.Added, Object: &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "3"}}},
			},
		},
		// properly handle ignoring changes prior to the filter, then getting added, then deleted
		{
			events: []*watchCacheEvent{
				{
					Type:            watch.Added,
					Object:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
					ObjFields:       fields.Set{"spec.nodeName": ""},
					ResourceVersion: 1,
				},
				{
					Type:            watch.Modified,
					PrevObject:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1"}},
					PrevObjFields:   fields.Set{"spec.nodeName": ""},
					Object:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},
					ObjFields:       fields.Set{"spec.nodeName": ""},
					ResourceVersion: 2,
				},
				{
					Type:            watch.Modified,
					PrevObject:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "2"}},
					PrevObjFields:   fields.Set{"spec.nodeName": ""},
					Object:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "3"}},
					ObjFields:       fields.Set{"spec.nodeName": "host"},
					ResourceVersion: 3,
				},
				{
					Type:            watch.Modified,
					PrevObject:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "3"}},
					PrevObjFields:   fields.Set{"spec.nodeName": "host"},
					Object:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "4"}},
					ObjFields:       fields.Set{"spec.nodeName": "host"},
					ResourceVersion: 4,
				},
				{
					Type:            watch.Modified,
					PrevObject:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "4"}},
					PrevObjFields:   fields.Set{"spec.nodeName": "host"},
					Object:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5"}},
					ObjFields:       fields.Set{"spec.nodeName": ""},
					ResourceVersion: 5,
				},
				{
					Type:            watch.Modified,
					PrevObject:      &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5"}},
					PrevObjFields:   fields.Set{"spec.nodeName": ""},
					Object:          &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "6"}},
					ObjFields:       fields.Set{"spec.nodeName": ""},
					ResourceVersion: 6,
				},
			},
			expected: []watch.Event{
				{Type: watch.Added, Object: &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "3"}}},
				{Type: watch.Modified, Object: &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "4"}}},
				{Type: watch.Deleted, Object: &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "5"}}},
			},
		},
	}

TestCase:
	for i, testCase := range testCases {
		// set the size of the buffer of w.result to 0, so that the writes to
		// w.result is blocked.
		for j := range testCase.events {
			testCase.events[j].ResourceVersion = uint64(j) + 1
		}

		w := newCacheWatcher(0, filter, forget, testVersioner{}, time.Now(), false, objectType, "")
		go w.processEvents(context.Background(), testCase.events, 0)

		ch := w.ResultChan()
		for j, event := range testCase.expected {
			e := <-ch
			if !reflect.DeepEqual(event, e) {
				t.Errorf("%d: unexpected event %d: %s", i, j, diff.ObjectReflectDiff(event, e))
				break TestCase
			}
		}
		select {
		case obj, ok := <-ch:
			t.Errorf("%d: unexpected excess event: %#v %t", i, obj, ok)
			break TestCase
		default:
		}
		w.stopThreadUnsafe()
	}
}

type testVersioner struct{}

func (testVersioner) UpdateObject(obj runtime.Object, resourceVersion uint64) error {
	return meta.NewAccessor().SetResourceVersion(obj, strconv.FormatUint(resourceVersion, 10))
}
func (testVersioner) UpdateList(obj runtime.Object, resourceVersion uint64, continueValue string, count *int64) error {
	listAccessor, err := meta.ListAccessor(obj)
	if err != nil || listAccessor == nil {
		return err
	}
	listAccessor.SetResourceVersion(strconv.FormatUint(resourceVersion, 10))
	listAccessor.SetContinue(continueValue)
	listAccessor.SetRemainingItemCount(count)
	return nil
}
func (testVersioner) PrepareObjectForStorage(obj runtime.Object) error {
	return fmt.Errorf("unimplemented")
}
func (testVersioner) ObjectResourceVersion(obj runtime.Object) (uint64, error) {
	accessor, err := meta.Accessor(obj)
	if err != nil {
		return 0, err
	}
	version := accessor.GetResourceVersion()
	if len(version) == 0 {
		return 0, nil
	}
	return strconv.ParseUint(version, 10, 64)
}
func (testVersioner) ParseResourceVersion(resourceVersion string) (uint64, error) {
	return strconv.ParseUint(resourceVersion, 10, 64)
}

var (
	scheme   = runtime.NewScheme()
	codecs   = serializer.NewCodecFactory(scheme)
	errDummy = fmt.Errorf("dummy error")
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
}

func newTestCacher(s storage.Interface) (*Cacher, storage.Versioner, error) {
	prefix := "pods"
	config := Config{
		Storage:        s,
		Versioner:      testVersioner{},
		ResourcePrefix: prefix,
		KeyFunc:        func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc(prefix, obj) },
		GetAttrsFunc:   storage.DefaultNamespaceScopedAttr,
		NewFunc:        func() runtime.Object { return &example.Pod{} },
		NewListFunc:    func() runtime.Object { return &example.PodList{} },
		Codec:          codecs.LegacyCodec(examplev1.SchemeGroupVersion),
		Clock:          clock.RealClock{},
	}
	cacher, err := NewCacherFromConfig(config)
	return cacher, testVersioner{}, err
}

type dummyStorage struct {
	err error
}

type dummyWatch struct {
	ch chan watch.Event
}

func (w *dummyWatch) ResultChan() <-chan watch.Event {
	return w.ch
}

func (w *dummyWatch) Stop() {
	close(w.ch)
}

func newDummyWatch() watch.Interface {
	return &dummyWatch{
		ch: make(chan watch.Event),
	}
}

func (d *dummyStorage) Versioner() storage.Versioner { return nil }
func (d *dummyStorage) Create(_ context.Context, _ string, _, _ runtime.Object, _ uint64) error {
	return fmt.Errorf("unimplemented")
}
func (d *dummyStorage) Delete(_ context.Context, _ string, _ runtime.Object, _ *storage.Preconditions, _ storage.ValidateObjectFunc, _ runtime.Object) error {
	return fmt.Errorf("unimplemented")
}
func (d *dummyStorage) Watch(_ context.Context, _ string, _ storage.ListOptions) (watch.Interface, error) {
	return newDummyWatch(), nil
}
func (d *dummyStorage) WatchList(_ context.Context, _ string, _ storage.ListOptions) (watch.Interface, error) {
	return newDummyWatch(), nil
}
func (d *dummyStorage) Get(_ context.Context, _ string, _ storage.GetOptions, _ runtime.Object) error {
	return d.err
}
func (d *dummyStorage) GetToList(_ context.Context, _ string, _ storage.ListOptions, _ runtime.Object) error {
	return d.err
}
func (d *dummyStorage) List(_ context.Context, _ string, _ storage.ListOptions, listObj runtime.Object) error {
	podList := listObj.(*example.PodList)
	podList.ListMeta = metav1.ListMeta{ResourceVersion: "100"}
	return d.err
}
func (d *dummyStorage) GuaranteedUpdate(_ context.Context, _ string, _ runtime.Object, _ bool, _ *storage.Preconditions, _ storage.UpdateFunc, _ runtime.Object) error {
	return fmt.Errorf("unimplemented")
}
func (d *dummyStorage) Count(_ string) (int64, error) {
	return 0, fmt.Errorf("unimplemented")
}

func TestListCacheBypass(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	pred := storage.SelectionPredicate{
		Limit: 500,
	}
	result := &example.PodList{}

	// Wait until cacher is initialized.
	cacher.ready.wait()

	// Inject error to underlying layer and check if cacher is not bypassed.
	backingStorage.err = errDummy
	err = cacher.List(context.TODO(), "pods/ns", storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred,
	}, result)
	if err != nil {
		t.Errorf("List with Limit and RV=0 should be served from cache: %v", err)
	}

	err = cacher.List(context.TODO(), "pods/ns", storage.ListOptions{
		ResourceVersion: "",
		Predicate:       pred,
	}, result)
	if err != errDummy {
		t.Errorf("List with Limit without RV=0 should bypass cacher: %v", err)
	}
}

func TestGetToListCacheBypass(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	pred := storage.SelectionPredicate{
		Limit: 500,
	}
	result := &example.PodList{}

	// Wait until cacher is initialized.
	cacher.ready.wait()

	// Inject error to underlying layer and check if cacher is not bypassed.
	backingStorage.err = errDummy
	err = cacher.GetToList(context.TODO(), "pods/ns", storage.ListOptions{
		ResourceVersion: "0",
		Predicate:       pred,
	}, result)
	if err != nil {
		t.Errorf("GetToList with Limit and RV=0 should be served from cache: %v", err)
	}

	err = cacher.GetToList(context.TODO(), "pods/ns", storage.ListOptions{
		ResourceVersion: "",
		Predicate:       pred,
	}, result)
	if err != errDummy {
		t.Errorf("List with Limit without RV=0 should bypass cacher: %v", err)
	}
}

func TestGetCacheBypass(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	result := &example.Pod{}

	// Wait until cacher is initialized.
	cacher.ready.wait()

	// Inject error to underlying layer and check if cacher is not bypassed.
	backingStorage.err = errDummy
	err = cacher.Get(context.TODO(), "pods/ns/pod-0", storage.GetOptions{
		IgnoreNotFound:  true,
		ResourceVersion: "0",
	}, result)
	if err != nil {
		t.Errorf("Get with RV=0 should be served from cache: %v", err)
	}

	err = cacher.Get(context.TODO(), "pods/ns/pod-0", storage.GetOptions{
		IgnoreNotFound:  true,
		ResourceVersion: "",
	}, result)
	if err != errDummy {
		t.Errorf("Get without RV=0 should bypass cacher: %v", err)
	}
}

func TestWatcherNotGoingBackInTime(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Wait until cacher is initialized.
	cacher.ready.wait()

	// Ensure there is some budget for slowing down processing.
	cacher.dispatchTimeoutBudget.returnUnused(100 * time.Millisecond)

	makePod := func(i int) *examplev1.Pod {
		return &examplev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%d", 1000+i),
				Namespace:       "ns",
				ResourceVersion: fmt.Sprintf("%d", 1000+i),
			},
		}
	}
	if err := cacher.watchCache.Add(makePod(0)); err != nil {
		t.Errorf("error: %v", err)
	}

	totalPods := 100

	// Create watcher that will be slowing down reading.
	w1, err := cacher.Watch(context.TODO(), "pods/ns", storage.ListOptions{
		ResourceVersion: "999",
		Predicate:       storage.Everything,
	})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	defer w1.Stop()
	go func() {
		a := 0
		for range w1.ResultChan() {
			time.Sleep(time.Millisecond)
			a++
			if a == 100 {
				break
			}
		}
	}()

	// Now push a ton of object to cache.
	for i := 1; i < totalPods; i++ {
		cacher.watchCache.Add(makePod(i))
	}

	// Create fast watcher and ensure it will get each object exactly once.
	w2, err := cacher.Watch(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: "999", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	defer w2.Stop()

	shouldContinue := true
	currentRV := uint64(0)
	for shouldContinue {
		select {
		case event, ok := <-w2.ResultChan():
			if !ok {
				shouldContinue = false
				break
			}
			rv, err := testVersioner{}.ParseResourceVersion(event.Object.(metaRuntimeInterface).GetResourceVersion())
			if err != nil {
				t.Errorf("unexpected parsing error: %v", err)
			} else {
				if rv < currentRV {
					t.Errorf("watcher going back in time")
				}
				currentRV = rv
			}
		case <-time.After(time.Second):
			w2.Stop()
		}
	}
}

func TestCacheWatcherStoppedInAnotherGoroutine(t *testing.T) {
	var w *cacheWatcher
	done := make(chan struct{})
	filter := func(string, labels.Set, fields.Set) bool { return true }
	forget := func() {
		w.stopThreadUnsafe()
		done <- struct{}{}
	}

	maxRetriesToProduceTheRaceCondition := 1000
	// Simulating the timer is fired and stopped concurrently by set time
	// timeout to zero and run the Stop goroutine concurrently.
	// May sure that the watch will not be blocked on Stop.
	for i := 0; i < maxRetriesToProduceTheRaceCondition; i++ {
		w = newCacheWatcher(0, filter, forget, testVersioner{}, time.Now(), false, objectType, "")
		go w.Stop()
		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("stop is blocked when the timer is fired concurrently")
		}
	}

	deadline := time.Now().Add(time.Hour)
	// After that, verifies the cacheWatcher.process goroutine works correctly.
	for i := 0; i < maxRetriesToProduceTheRaceCondition; i++ {
		w = newCacheWatcher(2, filter, emptyFunc, testVersioner{}, deadline, false, objectType, "")
		w.input <- &watchCacheEvent{Object: &v1.Pod{}, ResourceVersion: uint64(i + 1)}
		ctx, _ := context.WithDeadline(context.Background(), deadline)
		go w.processEvents(ctx, nil, 0)
		select {
		case <-w.ResultChan():
		case <-time.After(time.Second):
			t.Fatal("expected received a event on ResultChan")
		}
		w.stopThreadUnsafe()
	}
}

func TestCacheWatcherStoppedOnDestroy(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Wait until cacher is initialized.
	cacher.ready.wait()

	w, err := cacher.Watch(context.Background(), "pods/ns", storage.ListOptions{ResourceVersion: "0", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}

	watchClosed := make(chan struct{})
	go func() {
		defer close(watchClosed)
		for event := range w.ResultChan() {
			switch event.Type {
			case watch.Added, watch.Modified, watch.Deleted:
				// ok
			default:
				t.Errorf("unexpected event %#v", event)
			}
		}
	}()

	cacher.Stop()

	select {
	case <-watchClosed:
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("timed out waiting for watch to close")
	}

}

func TestTimeBucketWatchersBasic(t *testing.T) {
	filter := func(_ string, _ labels.Set, _ fields.Set) bool {
		return true
	}
	forget := func() {}

	newWatcher := func(deadline time.Time) *cacheWatcher {
		return newCacheWatcher(0, filter, forget, testVersioner{}, deadline, true, objectType, "")
	}

	clock := testingclock.NewFakeClock(time.Now())
	watchers := newTimeBucketWatchers(clock, defaultBookmarkFrequency)
	now := clock.Now()
	watchers.addWatcher(newWatcher(now.Add(10 * time.Second)))
	watchers.addWatcher(newWatcher(now.Add(20 * time.Second)))
	watchers.addWatcher(newWatcher(now.Add(20 * time.Second)))

	if len(watchers.watchersBuckets) != 2 {
		t.Errorf("unexpected bucket size: %#v", watchers.watchersBuckets)
	}
	watchers0 := watchers.popExpiredWatchers()
	if len(watchers0) != 0 {
		t.Errorf("unexpected bucket size: %#v", watchers0)
	}

	clock.Step(10 * time.Second)
	watchers1 := watchers.popExpiredWatchers()
	if len(watchers1) != 1 || len(watchers1[0]) != 1 {
		t.Errorf("unexpected bucket size: %v", watchers1)
	}
	watchers1 = watchers.popExpiredWatchers()
	if len(watchers1) != 0 {
		t.Errorf("unexpected bucket size: %#v", watchers1)
	}

	clock.Step(12 * time.Second)
	watchers2 := watchers.popExpiredWatchers()
	if len(watchers2) != 1 || len(watchers2[0]) != 2 {
		t.Errorf("unexpected bucket size: %#v", watchers2)
	}
}

func TestCacherNoLeakWithMultipleWatchers(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Wait until cacher is initialized.
	cacher.ready.wait()
	pred := storage.Everything
	pred.AllowWatchBookmarks = true

	// run the collision test for 3 seconds to let ~2 buckets expire
	stopCh := make(chan struct{})
	var watchErr error
	time.AfterFunc(3*time.Second, func() { close(stopCh) })

	wg := &sync.WaitGroup{}

	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stopCh:
				return
			default:
				ctx, _ := context.WithTimeout(context.Background(), 3*time.Second)
				w, err := cacher.Watch(ctx, "pods/ns", storage.ListOptions{ResourceVersion: "0", Predicate: pred})
				if err != nil {
					watchErr = fmt.Errorf("Failed to create watch: %v", err)
					return
				}
				w.Stop()
			}
		}
	}()

	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			select {
			case <-stopCh:
				return
			default:
				cacher.bookmarkWatchers.popExpiredWatchers()
			}
		}
	}()

	// wait for adding/removing watchers to end
	wg.Wait()

	if watchErr != nil {
		t.Fatal(watchErr)
	}

	// wait out the expiration period and pop expired watchers
	time.Sleep(2 * time.Second)
	cacher.bookmarkWatchers.popExpiredWatchers()
	cacher.bookmarkWatchers.lock.Lock()
	defer cacher.bookmarkWatchers.lock.Unlock()
	if len(cacher.bookmarkWatchers.watchersBuckets) != 0 {
		numWatchers := 0
		for bucketID, v := range cacher.bookmarkWatchers.watchersBuckets {
			numWatchers += len(v)
			t.Errorf("there are %v watchers at bucket Id %v with start Id %v", len(v), bucketID, cacher.bookmarkWatchers.startBucketID)
		}
		t.Errorf("unexpected bookmark watchers %v", numWatchers)
	}
}

func TestWatchInitializationSignal(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	ctx, _ := context.WithTimeout(context.Background(), 5*time.Second)
	initSignal := utilflowcontrol.NewInitializationSignal()
	ctx = utilflowcontrol.WithInitializationSignal(ctx, initSignal)

	_, err = cacher.Watch(ctx, "pods/ns", storage.ListOptions{ResourceVersion: "0", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}

	initSignal.Wait()
}

func testCacherSendBookmarkEvents(t *testing.T, allowWatchBookmarks, expectedBookmarks bool) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Wait until cacher is initialized.
	cacher.ready.wait()
	pred := storage.Everything
	pred.AllowWatchBookmarks = allowWatchBookmarks

	ctx, _ := context.WithTimeout(context.Background(), 3*time.Second)
	w, err := cacher.Watch(ctx, "pods/ns", storage.ListOptions{ResourceVersion: "0", Predicate: pred})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}

	resourceVersion := uint64(1000)
	errc := make(chan error, 1)
	go func() {
		deadline := time.Now().Add(time.Second)
		for i := 0; time.Now().Before(deadline); i++ {
			err := cacher.watchCache.Add(&examplev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            fmt.Sprintf("pod-%d", i),
					Namespace:       "ns",
					ResourceVersion: fmt.Sprintf("%v", resourceVersion+uint64(i)),
				}})
			if err != nil {
				errc <- fmt.Errorf("failed to add a pod: %v", err)
				return
			}
			time.Sleep(100 * time.Millisecond)
		}
	}()

	timeoutCh := time.After(2 * time.Second)
	lastObservedRV := uint64(0)
	for {
		select {
		case err := <-errc:
			t.Fatal(err)
			return
		case event, ok := <-w.ResultChan():
			if !ok {
				t.Fatal("Unexpected closed")
			}
			rv, err := cacher.versioner.ObjectResourceVersion(event.Object)
			if err != nil {
				t.Errorf("failed to parse resource version from %#v: %v", event.Object, err)
			}
			if event.Type == watch.Bookmark {
				if !expectedBookmarks {
					t.Fatalf("Unexpected bookmark events received")
				}

				if rv < lastObservedRV {
					t.Errorf("Unexpected bookmark event resource version %v (last %v)", rv, lastObservedRV)
				}
				return
			}
			lastObservedRV = rv
		case <-timeoutCh:
			if expectedBookmarks {
				t.Fatal("Unexpected timeout to receive a bookmark event")
			}
			return
		}
	}
}

func TestCacherSendBookmarkEvents(t *testing.T) {
	testCases := []struct {
		allowWatchBookmarks bool
		expectedBookmarks   bool
	}{
		{
			allowWatchBookmarks: true,
			expectedBookmarks:   true,
		},
		{
			allowWatchBookmarks: false,
			expectedBookmarks:   false,
		},
	}

	for _, tc := range testCases {
		testCacherSendBookmarkEvents(t, tc.allowWatchBookmarks, tc.expectedBookmarks)
	}
}

func TestCacherSendsMultipleWatchBookmarks(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()
	// Update bookmarkFrequency to speed up test.
	// Note that the frequency lower than 1s doesn't change much due to
	// resolution how frequency we recompute.
	cacher.bookmarkWatchers.bookmarkFrequency = time.Second

	// Wait until cacher is initialized.
	cacher.ready.wait()
	pred := storage.Everything
	pred.AllowWatchBookmarks = true

	makePod := func(index int) *examplev1.Pod {
		return &examplev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%d", index),
				Namespace:       "ns",
				ResourceVersion: fmt.Sprintf("%v", 100+index),
			},
		}
	}

	// Create pod to initialize watch cache.
	if err := cacher.watchCache.Add(makePod(0)); err != nil {
		t.Fatalf("failed to add a pod: %v", err)
	}

	ctx, _ := context.WithTimeout(context.Background(), 10*time.Second)
	w, err := cacher.Watch(ctx, "pods/ns", storage.ListOptions{ResourceVersion: "100", Predicate: pred})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}

	// Create one more pod, to ensure that current RV is higher and thus
	// bookmarks will be delievere (events are delivered for RV higher
	// than the max from init events).
	if err := cacher.watchCache.Add(makePod(1)); err != nil {
		t.Fatalf("failed to add a pod: %v", err)
	}

	timeoutCh := time.After(5 * time.Second)
	lastObservedRV := uint64(0)
	// Ensure that a watcher gets two bookmarks.
	for observedBookmarks := 0; observedBookmarks < 2; {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				t.Fatal("Unexpected closed")
			}
			rv, err := cacher.versioner.ObjectResourceVersion(event.Object)
			if err != nil {
				t.Errorf("failed to parse resource version from %#v: %v", event.Object, err)
			}
			if event.Type == watch.Bookmark {
				observedBookmarks++
				if rv < lastObservedRV {
					t.Errorf("Unexpected bookmark event resource version %v (last %v)", rv, lastObservedRV)
				}
			}
			lastObservedRV = rv
		case <-timeoutCh:
			t.Fatal("Unexpected timeout to receive bookmark events")
		}
	}
}

func TestDispatchingBookmarkEventsWithConcurrentStop(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Wait until cacher is initialized.
	cacher.ready.wait()

	// Ensure there is some budget for slowing down processing.
	cacher.dispatchTimeoutBudget.returnUnused(100 * time.Millisecond)

	resourceVersion := uint64(1000)
	err = cacher.watchCache.Add(&examplev1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:            "pod-0",
			Namespace:       "ns",
			ResourceVersion: fmt.Sprintf("%v", resourceVersion),
		}})
	if err != nil {
		t.Fatalf("failed to add a pod: %v", err)
	}

	for i := 0; i < 1000; i++ {
		pred := storage.Everything
		pred.AllowWatchBookmarks = true
		ctx, _ := context.WithTimeout(context.Background(), time.Second)
		w, err := cacher.Watch(ctx, "pods/ns", storage.ListOptions{ResourceVersion: "999", Predicate: pred})
		if err != nil {
			t.Fatalf("Failed to create watch: %v", err)
		}
		bookmark := &watchCacheEvent{
			Type:            watch.Bookmark,
			ResourceVersion: uint64(i),
			Object:          cacher.newFunc(),
		}
		err = cacher.versioner.UpdateObject(bookmark.Object, bookmark.ResourceVersion)
		if err != nil {
			t.Fatalf("failure to update version of object (%d) %#v", bookmark.ResourceVersion, bookmark.Object)
		}

		wg := sync.WaitGroup{}
		wg.Add(2)
		go func() {
			cacher.processEvent(bookmark)
			wg.Done()
		}()

		go func() {
			w.Stop()
			wg.Done()
		}()

		done := make(chan struct{})
		go func() {
			for range w.ResultChan() {
			}
			close(done)
		}()

		select {
		case <-done:
		case <-time.After(time.Second):
			t.Fatal("receive result timeout")
		}
		w.Stop()
		wg.Wait()
	}
}

func TestBookmarksOnResourceVersionUpdates(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Ensure that bookmarks are sent more frequently than every 1m.
	cacher.bookmarkWatchers = newTimeBucketWatchers(clock.RealClock{}, 2*time.Second)

	// Wait until cacher is initialized.
	cacher.ready.wait()

	makePod := func(i int) *examplev1.Pod {
		return &examplev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%d", i),
				Namespace:       "ns",
				ResourceVersion: fmt.Sprintf("%d", i),
			},
		}
	}
	if err := cacher.watchCache.Add(makePod(1000)); err != nil {
		t.Errorf("error: %v", err)
	}

	pred := storage.Everything
	pred.AllowWatchBookmarks = true

	w, err := cacher.Watch(context.TODO(), "/pods/ns", storage.ListOptions{
		ResourceVersion: "1000",
		Predicate:       pred,
	})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}

	expectedRV := 2000

	var rcErr error

	wg := sync.WaitGroup{}
	wg.Add(1)
	go func() {
		defer wg.Done()
		for {
			event, ok := <-w.ResultChan()
			if !ok {
				rcErr = errors.New("Unexpected closed channel")
				return
			}
			rv, err := cacher.versioner.ObjectResourceVersion(event.Object)
			if err != nil {
				t.Errorf("failed to parse resource version from %#v: %v", event.Object, err)
			}
			if event.Type == watch.Bookmark && rv == uint64(expectedRV) {
				return
			}
		}
	}()

	// Simulate progress notify event.
	cacher.watchCache.UpdateResourceVersion(strconv.Itoa(expectedRV))

	wg.Wait()
	if rcErr != nil {
		t.Fatal(rcErr)
	}
}

type fakeTimeBudget struct{}

func (f *fakeTimeBudget) takeAvailable() time.Duration {
	return 2 * time.Second
}

func (f *fakeTimeBudget) returnUnused(_ time.Duration) {}

func TestStartingResourceVersion(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Wait until cacher is initialized.
	cacher.ready.wait()

	// Ensure there is some budget for slowing down processing.
	// We use the fakeTimeBudget to prevent this test from flaking under
	// the following conditions:
	// 1) in total we create 11 events that has to be processed by the watcher
	// 2) the size of the channels are set to 10 for the watcher
	// 3) if the test is cpu-starved and the internal goroutine is not picking
	//    up these events from the channel, after consuming the whole time
	//    budget (defaulted to 100ms) on waiting, we will simply close the watch,
	//    which will cause the test failure
	// Using fakeTimeBudget gives us always a budget to wait and have a test
	// pick up something from ResultCh in the meantime.
	//
	// The same can potentially happen in production, but in that case a watch
	// can be resumed by the client. This doesn't work in the case of this test,
	// because we explicitly want to test the behavior that object changes are
	// happening after the watch was initiated.
	cacher.dispatchTimeoutBudget = &fakeTimeBudget{}

	makePod := func(i int) *examplev1.Pod {
		return &examplev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "foo",
				Namespace:       "ns",
				Labels:          map[string]string{"foo": strconv.Itoa(i)},
				ResourceVersion: fmt.Sprintf("%d", i),
			},
		}
	}

	if err := cacher.watchCache.Add(makePod(1000)); err != nil {
		t.Errorf("error: %v", err)
	}
	// Advance RV by 10.
	startVersion := uint64(1010)

	watcher, err := cacher.Watch(context.TODO(), "pods/ns/foo", storage.ListOptions{ResourceVersion: strconv.FormatUint(startVersion, 10), Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer watcher.Stop()

	for i := 1; i <= 11; i++ {
		if err := cacher.watchCache.Update(makePod(1000 + i)); err != nil {
			t.Errorf("error: %v", err)
		}
	}

	select {
	case e, ok := <-watcher.ResultChan():
		if !ok {
			t.Errorf("unexpectedly closed watch")
			break
		}
		object := e.Object
		if co, ok := object.(runtime.CacheableObject); ok {
			object = co.GetObject()
		}
		pod := object.(*examplev1.Pod)
		podRV, err := cacher.versioner.ParseResourceVersion(pod.ResourceVersion)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}

		// event should have at least rv + 1, since we're starting the watch at rv
		if podRV <= startVersion {
			t.Errorf("expected event with resourceVersion of at least %d, got %d", startVersion+1, podRV)
		}
	}
}

func TestDispatchEventWillNotBeBlockedByTimedOutWatcher(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Wait until cacher is initialized.
	cacher.ready.wait()

	// Ensure there is some budget for slowing down processing.
	// We use the fakeTimeBudget to prevent this test from flaking under
	// the following conditions:
	// 1) the watch w1 is blocked, so we were consuming the whole budget once
	//    its buffer was filled in (10 items)
	// 2) the budget is refreshed once per second, so it basically wasn't
	//    happening in the test at all
	// 3) if the test was cpu-starved and we weren't able to consume events
	//    from w2 ResultCh it could have happened that its buffer was also
	//    filling in and given we no longer had timeBudget (consumed in (1))
	//    trying to put next item was simply breaking the watch
	// Using fakeTimeBudget gives us always a budget to wait and have a test
	// pick up something from ResultCh in the meantime.
	cacher.dispatchTimeoutBudget = &fakeTimeBudget{}

	makePod := func(i int) *examplev1.Pod {
		return &examplev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            fmt.Sprintf("pod-%d", 1000+i),
				Namespace:       "ns",
				ResourceVersion: fmt.Sprintf("%d", 1000+i),
			},
		}
	}
	if err := cacher.watchCache.Add(makePod(0)); err != nil {
		t.Errorf("error: %v", err)
	}

	totalPods := 50

	// Create watcher that will be blocked.
	w1, err := cacher.Watch(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: "999", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	defer w1.Stop()

	// Create fast watcher and ensure it will get all objects.
	w2, err := cacher.Watch(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: "999", Predicate: storage.Everything})
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	defer w2.Stop()

	// Now push a ton of object to cache.
	for i := 1; i < totalPods; i++ {
		cacher.watchCache.Add(makePod(i))
	}

	shouldContinue := true
	eventsCount := 0
	for shouldContinue {
		select {
		case event, ok := <-w2.ResultChan():
			if !ok {
				shouldContinue = false
				break
			}
			if event.Type == watch.Added {
				eventsCount++
				if eventsCount == totalPods {
					shouldContinue = false
				}
			}
		case <-time.After(wait.ForeverTestTimeout):
			shouldContinue = false
			w2.Stop()
		}
	}
	if eventsCount != totalPods {
		t.Errorf("watcher is blocked by slower one (count: %d)", eventsCount)
	}
}

func verifyEvents(t *testing.T, w watch.Interface, events []watch.Event) {
	_, _, line, _ := goruntime.Caller(1)
	for _, expectedEvent := range events {
		select {
		case event := <-w.ResultChan():
			if e, a := expectedEvent.Type, event.Type; e != a {
				t.Logf("(called from line %d)", line)
				t.Errorf("Expected: %s, got: %s", e, a)
			}
			object := event.Object
			if co, ok := object.(runtime.CacheableObject); ok {
				object = co.GetObject()
			}
			if e, a := expectedEvent.Object, object; !apiequality.Semantic.DeepEqual(e, a) {
				t.Logf("(called from line %d)", line)
				t.Errorf("Expected: %#v, got: %#v", e, a)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Logf("(called from line %d)", line)
			t.Errorf("Timed out waiting for an event")
		}
	}
}

func TestCachingDeleteEvents(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Wait until cacher is initialized.
	cacher.ready.wait()

	fooPredicate := storage.SelectionPredicate{
		Label: labels.SelectorFromSet(map[string]string{"foo": "true"}),
		Field: fields.Everything(),
	}
	barPredicate := storage.SelectionPredicate{
		Label: labels.SelectorFromSet(map[string]string{"bar": "true"}),
		Field: fields.Everything(),
	}

	createWatch := func(pred storage.SelectionPredicate) watch.Interface {
		w, err := cacher.Watch(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: "999", Predicate: pred})
		if err != nil {
			t.Fatalf("Failed to create watch: %v", err)
		}
		return w
	}

	allEventsWatcher := createWatch(storage.Everything)
	defer allEventsWatcher.Stop()
	fooEventsWatcher := createWatch(fooPredicate)
	defer fooEventsWatcher.Stop()
	barEventsWatcher := createWatch(barPredicate)
	defer barEventsWatcher.Stop()

	makePod := func(labels map[string]string, rv string) *examplev1.Pod {
		return &examplev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            "pod",
				Namespace:       "ns",
				Labels:          labels,
				ResourceVersion: rv,
			},
		}
	}
	pod1 := makePod(map[string]string{"foo": "true", "bar": "true"}, "1001")
	pod2 := makePod(map[string]string{"foo": "true"}, "1002")
	pod3 := makePod(map[string]string{}, "1003")
	pod4 := makePod(map[string]string{}, "1004")
	pod1DeletedAt2 := pod1.DeepCopyObject().(*examplev1.Pod)
	pod1DeletedAt2.ResourceVersion = "1002"
	pod2DeletedAt3 := pod2.DeepCopyObject().(*examplev1.Pod)
	pod2DeletedAt3.ResourceVersion = "1003"

	allEvents := []watch.Event{
		{Type: watch.Added, Object: pod1.DeepCopy()},
		{Type: watch.Modified, Object: pod2.DeepCopy()},
		{Type: watch.Modified, Object: pod3.DeepCopy()},
		{Type: watch.Deleted, Object: pod4.DeepCopy()},
	}
	fooEvents := []watch.Event{
		{Type: watch.Added, Object: pod1.DeepCopy()},
		{Type: watch.Modified, Object: pod2.DeepCopy()},
		{Type: watch.Deleted, Object: pod2DeletedAt3.DeepCopy()},
	}
	barEvents := []watch.Event{
		{Type: watch.Added, Object: pod1.DeepCopy()},
		{Type: watch.Deleted, Object: pod1DeletedAt2.DeepCopy()},
	}

	cacher.watchCache.Add(pod1)
	cacher.watchCache.Update(pod2)
	cacher.watchCache.Update(pod3)
	cacher.watchCache.Delete(pod4)

	verifyEvents(t, allEventsWatcher, allEvents)
	verifyEvents(t, fooEventsWatcher, fooEvents)
	verifyEvents(t, barEventsWatcher, barEvents)
}

func testCachingObjects(t *testing.T, watchersCount int) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Wait until cacher is initialized.
	cacher.ready.wait()

	dispatchedEvents := []*watchCacheEvent{}
	cacher.watchCache.eventHandler = func(event *watchCacheEvent) {
		dispatchedEvents = append(dispatchedEvents, event)
		cacher.processEvent(event)
	}

	watchers := make([]watch.Interface, 0, watchersCount)
	for i := 0; i < watchersCount; i++ {
		w, err := cacher.Watch(context.TODO(), "pods/ns", storage.ListOptions{ResourceVersion: "1000", Predicate: storage.Everything})
		if err != nil {
			t.Fatalf("Failed to create watch: %v", err)
		}
		defer w.Stop()
		watchers = append(watchers, w)
	}

	makePod := func(name, rv string) *examplev1.Pod {
		return &examplev1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:            name,
				Namespace:       "ns",
				ResourceVersion: rv,
			},
		}
	}
	pod1 := makePod("pod", "1001")
	pod2 := makePod("pod", "1002")
	pod3 := makePod("pod", "1003")

	cacher.watchCache.Add(pod1)
	cacher.watchCache.Update(pod2)
	cacher.watchCache.Delete(pod3)

	// At this point, we already have dispatchedEvents fully propagated.

	verifyEvents := func(w watch.Interface) {
		var event watch.Event
		for index := range dispatchedEvents {
			select {
			case event = <-w.ResultChan():
			case <-time.After(wait.ForeverTestTimeout):
				t.Fatalf("timeout watiching for the event")
			}

			var object runtime.Object
			if watchersCount >= 3 {
				if _, ok := event.Object.(runtime.CacheableObject); !ok {
					t.Fatalf("Object in %s event should support caching: %#v", event.Type, event.Object)
				}
				object = event.Object.(runtime.CacheableObject).GetObject()
			} else {
				if _, ok := event.Object.(runtime.CacheableObject); ok {
					t.Fatalf("Object in %s event should not support caching: %#v", event.Type, event.Object)
				}
				object = event.Object.DeepCopyObject()
			}

			if event.Type == watch.Deleted {
				resourceVersion, err := cacher.versioner.ObjectResourceVersion(cacher.watchCache.cache[index].PrevObject)
				if err != nil {
					t.Fatalf("Failed to parse resource version: %v", err)
				}
				updateResourceVersionIfNeeded(object, cacher.versioner, resourceVersion)
			}

			var e runtime.Object
			switch event.Type {
			case watch.Added, watch.Modified:
				e = cacher.watchCache.cache[index].Object
			case watch.Deleted:
				e = cacher.watchCache.cache[index].PrevObject
			default:
				t.Errorf("unexpected watch event: %#v", event)
			}
			if a := object; !reflect.DeepEqual(a, e) {
				t.Errorf("event object messed up for %s: %#v, expected: %#v", event.Type, a, e)
			}
		}
	}

	for i := range watchers {
		verifyEvents(watchers[i])
	}
}

func TestCachingObjects(t *testing.T) {
	t.Run("single watcher", func(t *testing.T) { testCachingObjects(t, 1) })
	t.Run("many watcher", func(t *testing.T) { testCachingObjects(t, 3) })
}
