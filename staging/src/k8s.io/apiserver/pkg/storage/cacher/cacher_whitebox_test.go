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
	"fmt"
	"reflect"
	"strconv"
	"sync"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/diff"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	"k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/storage"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
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
	w = newCacheWatcher(0, filter, forget, testVersioner{}, time.Now(), false, objectType)
	go w.process(context.Background(), initEvents, 0)
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

		w := newCacheWatcher(0, filter, forget, testVersioner{}, time.Now(), false, objectType)
		go w.process(context.Background(), testCase.events, 0)

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
		w.Stop()
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

func newTestCacher(s storage.Interface, cap int) (*Cacher, storage.Versioner, error) {
	prefix := "pods"
	config := Config{
		CacheCapacity:  cap,
		Storage:        s,
		Versioner:      testVersioner{},
		ResourcePrefix: prefix,
		KeyFunc:        func(obj runtime.Object) (string, error) { return storage.NamespaceKeyFunc(prefix, obj) },
		GetAttrsFunc:   func(obj runtime.Object) (labels.Set, fields.Set, error) { return nil, nil, nil },
		NewFunc:        func() runtime.Object { return &example.Pod{} },
		NewListFunc:    func() runtime.Object { return &example.PodList{} },
		Codec:          codecs.LegacyCodec(examplev1.SchemeGroupVersion),
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
func (d *dummyStorage) Delete(_ context.Context, _ string, _ runtime.Object, _ *storage.Preconditions, _ storage.ValidateObjectFunc) error {
	return fmt.Errorf("unimplemented")
}
func (d *dummyStorage) Watch(_ context.Context, _ string, _ string, _ storage.SelectionPredicate) (watch.Interface, error) {
	return newDummyWatch(), nil
}
func (d *dummyStorage) WatchList(_ context.Context, _ string, _ string, _ storage.SelectionPredicate) (watch.Interface, error) {
	return newDummyWatch(), nil
}
func (d *dummyStorage) Get(_ context.Context, _ string, _ string, _ runtime.Object, _ bool) error {
	return fmt.Errorf("unimplemented")
}
func (d *dummyStorage) GetToList(_ context.Context, _ string, _ string, _ storage.SelectionPredicate, _ runtime.Object) error {
	return d.err
}
func (d *dummyStorage) List(_ context.Context, _ string, _ string, _ storage.SelectionPredicate, listObj runtime.Object) error {
	podList := listObj.(*example.PodList)
	podList.ListMeta = metav1.ListMeta{ResourceVersion: "100"}
	return d.err
}
func (d *dummyStorage) GuaranteedUpdate(_ context.Context, _ string, _ runtime.Object, _ bool, _ *storage.Preconditions, _ storage.UpdateFunc, _ ...runtime.Object) error {
	return fmt.Errorf("unimplemented")
}
func (d *dummyStorage) Count(_ string) (int64, error) {
	return 0, fmt.Errorf("unimplemented")
}

func TestListWithLimitAndRV0(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage, 0)
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
	err = cacher.List(context.TODO(), "pods/ns", "0", pred, result)
	if err != nil {
		t.Errorf("List with Limit and RV=0 should be served from cache: %v", err)
	}

	err = cacher.List(context.TODO(), "pods/ns", "", pred, result)
	if err != errDummy {
		t.Errorf("List with Limit without RV=0 should bypass cacher: %v", err)
	}
}

func TestGetToListWithLimitAndRV0(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage, 0)
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
	err = cacher.GetToList(context.TODO(), "pods/ns", "0", pred, result)
	if err != nil {
		t.Errorf("GetToList with Limit and RV=0 should be served from cache: %v", err)
	}

	err = cacher.GetToList(context.TODO(), "pods/ns", "", pred, result)
	if err != errDummy {
		t.Errorf("List with Limit without RV=0 should bypass cacher: %v", err)
	}
}

func TestWatcherNotGoingBackInTime(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage, 1000)
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
	w1, err := cacher.Watch(context.TODO(), "pods/ns", "999", storage.Everything)
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
	w2, err := cacher.Watch(context.TODO(), "pods/ns", "999", storage.Everything)
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
			rv, err := testVersioner{}.ParseResourceVersion(event.Object.(*examplev1.Pod).ResourceVersion)
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
		w = newCacheWatcher(0, filter, forget, testVersioner{}, time.Now(), false, objectType)
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
		w = newCacheWatcher(2, filter, emptyFunc, testVersioner{}, deadline, false, objectType)
		w.input <- &watchCacheEvent{Object: &v1.Pod{}, ResourceVersion: uint64(i + 1)}
		ctx, _ := context.WithDeadline(context.Background(), deadline)
		go w.process(ctx, nil, 0)
		select {
		case <-w.ResultChan():
		case <-time.After(time.Second):
			t.Fatal("expected received a event on ResultChan")
		}
		w.Stop()
	}
}

func TestCacheWatcherStoppedOnDestroy(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage, 1000)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Wait until cacher is initialized.
	cacher.ready.wait()

	w, err := cacher.Watch(context.Background(), "pods/ns", "0", storage.Everything)
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
		return newCacheWatcher(0, filter, forget, testVersioner{}, deadline, true, objectType)
	}

	clock := clock.NewFakeClock(time.Now())
	watchers := newTimeBucketWatchers(clock)
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
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchBookmark, true)()
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage, 1000)
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
				w, err := cacher.Watch(ctx, "pods/ns", "0", pred)
				if err != nil {
					t.Fatalf("Failed to create watch: %v", err)
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

func testCacherSendBookmarkEvents(t *testing.T, watchCacheEnabled, allowWatchBookmarks, expectedBookmarks bool) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchBookmark, watchCacheEnabled)()
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage, 1000)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Wait until cacher is initialized.
	cacher.ready.wait()
	pred := storage.Everything
	pred.AllowWatchBookmarks = allowWatchBookmarks

	ctx, _ := context.WithTimeout(context.Background(), 3*time.Second)
	w, err := cacher.Watch(ctx, "pods/ns", "0", pred)
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}

	resourceVersion := uint64(1000)
	go func() {
		deadline := time.Now().Add(time.Second)
		for i := 0; time.Now().Before(deadline); i++ {
			err = cacher.watchCache.Add(&examplev1.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:            fmt.Sprintf("pod-%d", i),
					Namespace:       "ns",
					ResourceVersion: fmt.Sprintf("%v", resourceVersion+uint64(i)),
				}})
			if err != nil {
				t.Fatalf("failed to add a pod: %v", err)
			}
			time.Sleep(100 * time.Millisecond)
		}
	}()

	timeoutCh := time.After(2 * time.Second)
	lastObservedRV := uint64(0)
	for {
		select {
		case event, ok := <-w.ResultChan():
			if !ok {
				t.Fatal("Unexpected closed")
			}
			rv, err := cacher.versioner.ObjectResourceVersion(event.Object)
			if err != nil {
				t.Errorf("failed to parse resource version from %#v", event.Object)
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
		watchCacheEnabled   bool
		allowWatchBookmarks bool
		expectedBookmarks   bool
	}{
		{
			watchCacheEnabled:   true,
			allowWatchBookmarks: true,
			expectedBookmarks:   true,
		},
		{
			watchCacheEnabled:   true,
			allowWatchBookmarks: false,
			expectedBookmarks:   false,
		},
		{
			watchCacheEnabled:   false,
			allowWatchBookmarks: true,
			expectedBookmarks:   false,
		},
		{
			watchCacheEnabled:   false,
			allowWatchBookmarks: false,
			expectedBookmarks:   false,
		},
	}

	for _, tc := range testCases {
		testCacherSendBookmarkEvents(t, tc.watchCacheEnabled, tc.allowWatchBookmarks, tc.expectedBookmarks)
	}
}

func TestDispatchingBookmarkEventsWithConcurrentStop(t *testing.T) {
	defer featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.WatchBookmark, true)()
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage, 1000)
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
			Name:            fmt.Sprintf("pod-0"),
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
		w, err := cacher.Watch(ctx, "pods/ns", "999", pred)
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
			cacher.dispatchEvent(bookmark)
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
			break
		case <-time.After(time.Second):
			t.Fatal("receive result timeout")
		}
		w.Stop()
		wg.Wait()
	}
}

func TestDispatchEventWillNotBeBlockedByTimedOutWatcher(t *testing.T) {
	backingStorage := &dummyStorage{}
	cacher, _, err := newTestCacher(backingStorage, 1000)
	if err != nil {
		t.Fatalf("Couldn't create cacher: %v", err)
	}
	defer cacher.Stop()

	// Wait until cacher is initialized.
	cacher.ready.wait()

	// Ensure there is some budget for slowing down processing.
	cacher.dispatchTimeoutBudget.returnUnused(50 * time.Millisecond)

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
	w1, err := cacher.Watch(context.TODO(), "pods/ns", "999", storage.Everything)
	if err != nil {
		t.Fatalf("Failed to create watch: %v", err)
	}
	defer w1.Stop()

	// Create fast watcher and ensure it will get all objects.
	w2, err := cacher.Watch(context.TODO(), "pods/ns", "999", storage.Everything)
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
			// Ensure there is some budget for fast watcher after slower one is blocked.
			cacher.dispatchTimeoutBudget.returnUnused(50 * time.Millisecond)
			if event.Type == watch.Added {
				eventsCount++
				if eventsCount == totalPods {
					shouldContinue = false
				}
			}
		case <-time.After(2 * time.Second):
			shouldContinue = false
			w2.Stop()
		}
	}
	if eventsCount != totalPods {
		t.Errorf("watcher is blocked by slower one (count: %d)", eventsCount)
	}
}
