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

package cache

import (
	"context"
	"errors"
	"fmt"
	"math/rand"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	fcache "k8s.io/client-go/tools/cache/testing"
	"k8s.io/klog/v2/ktesting"

	"sigs.k8s.io/randfill"

	"k8s.io/apimachinery/pkg/api/meta"
)

func Example() {
	// source simulates an apiserver object endpoint.
	source := fcache.NewFakeControllerSource()
	defer source.Shutdown()

	// This will hold the downstream state, as we know it.
	downstream := NewStore(DeletionHandlingMetaNamespaceKeyFunc)

	// This will hold incoming changes. Note how we pass downstream in as a
	// KeyLister, that way resync operations will result in the correct set
	// of update/delete deltas.
	fifo := NewRealFIFO(MetaNamespaceKeyFunc, downstream, nil)

	// Let's do threadsafe output to get predictable test results.
	deletionCounter := make(chan string, 1000)

	cfg := &Config{
		Queue:            fifo,
		ListerWatcher:    source,
		ObjectType:       &v1.Pod{},
		FullResyncPeriod: time.Millisecond * 100,

		// Let's implement a simple controller that just deletes
		// everything that comes in.
		Process: func(obj interface{}, isInInitialList bool) error {
			// Obj is from the Pop method of the Queue we make above.
			newest := obj.(Deltas).Newest()

			if newest.Type != Deleted {
				// Update our downstream store.
				err := downstream.Add(newest.Object)
				if err != nil {
					return err
				}

				// Delete this object.
				source.Delete(newest.Object.(runtime.Object))
			} else {
				// Update our downstream store.
				err := downstream.Delete(newest.Object)
				if err != nil {
					return err
				}

				// fifo's KeyOf is easiest, because it handles
				// DeletedFinalStateUnknown markers.
				key, err := fifo.keyOf(newest.Object)
				if err != nil {
					return err
				}

				// Report this deletion.
				deletionCounter <- key
			}
			return nil
		},
	}

	// Create the controller and run it until we cancel.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go New(cfg).RunWithContext(ctx)

	// Let's add a few objects to the source.
	testIDs := []string{"a-hello", "b-controller", "c-framework"}
	for _, name := range testIDs {
		// Note that these pods are not valid-- the fake source doesn't
		// call validation or anything.
		source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name}})
	}

	// Let's wait for the controller to process the things we just added.
	outputSet := sets.Set[string]{}
	for i := 0; i < len(testIDs); i++ {
		outputSet.Insert(<-deletionCounter)
	}

	for _, key := range sets.List(outputSet) {
		fmt.Println(key)
	}
	// Output:
	// a-hello
	// b-controller
	// c-framework
}

func ExampleNewInformer() {
	// source simulates an apiserver object endpoint.
	source := fcache.NewFakeControllerSource()
	defer source.Shutdown()

	// Let's do threadsafe output to get predictable test results.
	deletionCounter := make(chan string, 1000)

	// Make a controller that immediately deletes anything added to it, and
	// logs anything deleted.
	_, controller := NewInformer(
		source,
		&v1.Pod{},
		time.Millisecond*100,
		ResourceEventHandlerDetailedFuncs{
			AddFunc: func(obj interface{}, isInInitialList bool) {
				source.Delete(obj.(runtime.Object))
			},
			DeleteFunc: func(obj interface{}) {
				key, err := DeletionHandlingMetaNamespaceKeyFunc(obj)
				if err != nil {
					key = "oops something went wrong with the key"
				}

				// Report this deletion.
				deletionCounter <- key
			},
		},
	)

	// Run the controller and run it until we cancel.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go controller.RunWithContext(ctx)

	// Let's add a few objects to the source.
	testIDs := []string{"a-hello", "b-controller", "c-framework"}
	for _, name := range testIDs {
		// Note that these pods are not valid-- the fake source doesn't
		// call validation or anything.
		source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name}})
	}

	// Let's wait for the controller to process the things we just added.
	outputSet := sets.Set[string]{}
	for range testIDs {
		outputSet.Insert(<-deletionCounter)
	}

	for _, key := range sets.List(outputSet) {
		fmt.Println(key)
	}
	// Output:
	// a-hello
	// b-controller
	// c-framework
}

func TestHammerController(t *testing.T) {
	// This test executes a bunch of requests through the fake source and
	// controller framework to make sure there's no locking/threading
	// errors. If an error happens, it should hang forever or trigger the
	// race detector.

	// source simulates an apiserver object endpoint.
	source := newFakeControllerSource(t)

	// Let's do threadsafe output to get predictable test results.
	outputSetLock := sync.Mutex{}
	// map of key to operations done on the key
	outputSet := map[string][]string{}

	recordFunc := func(eventType string, obj interface{}) {
		key, err := DeletionHandlingMetaNamespaceKeyFunc(obj)
		if err != nil {
			t.Errorf("something wrong with key: %v", err)
			key = "oops something went wrong with the key"
		}

		// Record some output when items are deleted.
		outputSetLock.Lock()
		defer outputSetLock.Unlock()
		outputSet[key] = append(outputSet[key], eventType)
	}

	// Make a controller which just logs all the changes it gets.
	_, controller := NewInformer(
		source,
		&v1.Pod{},
		time.Millisecond*100,
		ResourceEventHandlerDetailedFuncs{
			AddFunc:    func(obj interface{}, isInInitialList bool) { recordFunc("add", obj) },
			UpdateFunc: func(oldObj, newObj interface{}) { recordFunc("update", newObj) },
			DeleteFunc: func(obj interface{}) { recordFunc("delete", obj) },
		},
	)

	if controller.HasSynced() {
		t.Errorf("Expected HasSynced() to return false before we started the controller")
	}

	// Run the controller and run it until we cancel.
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	var controllerWG sync.WaitGroup
	controllerWG.Add(1)
	go func() {
		defer controllerWG.Done()
		controller.RunWithContext(ctx)
	}()

	// Let's wait for the controller to do its initial sync
	wait.Poll(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
		return controller.HasSynced(), nil
	})
	if !controller.HasSynced() {
		t.Errorf("Expected HasSynced() to return true after the initial sync")
	}

	wg := sync.WaitGroup{}
	const threads = 3
	wg.Add(threads)
	for i := 0; i < threads; i++ {
		go func() {
			defer wg.Done()
			// Let's add a few objects to the source.
			currentNames := sets.Set[string]{}
			rs := rand.NewSource(rand.Int63())
			f := randfill.New().NilChance(.5).NumElements(0, 2).RandSource(rs)
			for i := 0; i < 100; i++ {
				var name string
				var isNew bool
				if currentNames.Len() == 0 || rand.Intn(3) == 1 {
					f.Fill(&name)
					isNew = true
				} else {
					l := sets.List(currentNames)
					name = l[rand.Intn(len(l))]
				}

				pod := &v1.Pod{}
				f.Fill(pod)
				pod.ObjectMeta.Name = name
				pod.ObjectMeta.Namespace = "default"
				// Add, update, or delete randomly.
				// Note that these pods are not valid-- the fake source doesn't
				// call validation or perform any other checking.
				if isNew {
					currentNames.Insert(name)
					source.Add(pod)
					continue
				}
				switch rand.Intn(2) {
				case 0:
					currentNames.Insert(name)
					source.Modify(pod)
				case 1:
					currentNames.Delete(name)
					source.Delete(pod)
				}
			}
		}()
	}
	wg.Wait()

	// Let's wait for the controller to finish processing the things we just added.
	// TODO: look in the queue to see how many items need to be processed.
	time.Sleep(100 * time.Millisecond)
	cancel()

	// Before we permanently lock this mutex, we have to be sure
	// that the controller has stopped running. At this point,
	// all goroutines should have stopped. Leak checking is
	// done by TestMain.
	controllerWG.Wait()

	outputSetLock.Lock()
	t.Logf("got: %#v", outputSet)
}

func TestUpdate(t *testing.T) {
	// This test is going to exercise the various paths that result in a
	// call to update.

	// source simulates an apiserver object endpoint.
	source := newFakeControllerSource(t)

	const (
		FROM = "from"
		TO   = "to"
	)

	// These are the transitions we expect to see; because this is
	// asynchronous, there are a lot of valid possibilities.
	type pair struct{ from, to string }
	allowedTransitions := map[pair]bool{
		{FROM, TO}: true,

		// Because a resync can happen when we've already observed one
		// of the above but before the item is deleted.
		{TO, TO}: true,
		// Because a resync could happen before we observe an update.
		{FROM, FROM}: true,
	}

	pod := func(name, check string, final bool) *v1.Pod {
		p := &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:   name,
				Labels: map[string]string{"check": check},
			},
		}
		if final {
			p.Labels["final"] = "true"
		}
		return p
	}
	deletePod := func(p *v1.Pod) bool {
		return p.Labels["final"] == "true"
	}

	tests := []func(string){
		func(name string) {
			name = "a-" + name
			source.Add(pod(name, FROM, false))
			source.Modify(pod(name, TO, true))
		},
	}

	const threads = 3

	var testDoneWG sync.WaitGroup
	testDoneWG.Add(threads * len(tests))

	// Make a controller that deletes things once it observes an update.
	// It calls Done() on the wait group on deletions so we can tell when
	// everything we've added has been deleted.
	watchCh := make(chan struct{})
	_, controller := NewInformer(
		toListWatcherWithUnSupportedWatchListSemantics(&ListWatch{
			WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
				watch, err := source.Watch(options)
				close(watchCh)
				return watch, err
			},
			ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
				return source.List(options)
			},
		}),
		&v1.Pod{},
		0,
		ResourceEventHandlerFuncs{
			UpdateFunc: func(oldObj, newObj interface{}) {
				o, n := oldObj.(*v1.Pod), newObj.(*v1.Pod)
				from, to := o.Labels["check"], n.Labels["check"]
				if !allowedTransitions[pair{from, to}] {
					t.Errorf("observed transition %q -> %q for %v", from, to, n.Name)
				}
				if deletePod(n) {
					source.Delete(n)
				}
			},
			DeleteFunc: func(obj interface{}) {
				testDoneWG.Done()
			},
		},
	)

	// Run the controller and run it until we cancel.
	// Once Run() is called, calls to testDoneWG.Done() might start, so
	// all testDoneWG.Add() calls must happen before this point
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go controller.RunWithContext(ctx)
	<-watchCh

	// run every test a few times, in parallel
	var wg sync.WaitGroup
	wg.Add(threads * len(tests))
	for i := 0; i < threads; i++ {
		for j, f := range tests {
			go func(name string, f func(string)) {
				defer wg.Done()
				f(name)
			}(fmt.Sprintf("%v-%v", i, j), f)
		}
	}
	wg.Wait()

	// Let's wait for the controller to process the things we just added.
	testDoneWG.Wait()
}

func TestPanicPropagated(t *testing.T) {
	// source simulates an apiserver object endpoint.
	source := newFakeControllerSource(t)

	// Make a controller that just panic if the AddFunc is called.
	_, controller := NewInformer(
		source,
		&v1.Pod{},
		time.Millisecond*100,
		ResourceEventHandlerDetailedFuncs{
			AddFunc: func(obj interface{}, isInInitialList bool) {
				// Create a panic.
				panic("Just panic.")
			},
		},
	)

	// Run the controller and run it until we cancel.
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	propagated := make(chan interface{})
	go func() {
		defer func() {
			if r := recover(); r != nil {
				propagated <- r
			}
		}()
		controller.RunWithContext(ctx)
	}()
	// Let's add a object to the source. It will trigger a panic.
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "test"}})

	// Check if the panic propagated up.
	select {
	case p := <-propagated:
		if p == "Just panic." {
			t.Logf("Test Passed")
		} else {
			t.Errorf("unrecognized panic in controller run: %v", p)
		}
	case <-time.After(wait.ForeverTestTimeout):
		t.Errorf("timeout: the panic failed to propagate from the controller run method!")
	}
}

func TestTransformingInformer(t *testing.T) {
	// source simulates an apiserver object endpoint.
	source := newFakeControllerSource(t)

	makePod := func(name, generation string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: "namespace",
				Labels:    map[string]string{"generation": generation},
			},
			Spec: v1.PodSpec{
				Hostname:  "hostname",
				Subdomain: "subdomain",
			},
		}
	}
	expectedPod := func(name, generation string) *v1.Pod {
		pod := makePod(name, generation)
		pod.Spec.Hostname = "new-hostname"
		pod.Spec.Subdomain = ""
		pod.Spec.NodeName = "nodename"
		return pod
	}

	source.Add(makePod("pod1", "1"))
	source.Modify(makePod("pod1", "2"))

	type event struct {
		eventType watch.EventType
		previous  interface{}
		current   interface{}
	}
	events := make(chan event, 10)
	recordEvent := func(eventType watch.EventType, previous, current interface{}) {
		events <- event{eventType: eventType, previous: previous, current: current}
	}
	verifyEvent := func(eventType watch.EventType, previous, current interface{}) {
		select {
		case event := <-events:
			if event.eventType != eventType {
				t.Errorf("expected type %v, got %v", eventType, event.eventType)
			}
			if !apiequality.Semantic.DeepEqual(event.previous, previous) {
				t.Errorf("expected previous object %#v, got %#v", previous, event.previous)
			}
			if !apiequality.Semantic.DeepEqual(event.current, current) {
				t.Errorf("expected object %#v, got %#v", current, event.current)
			}
		case <-time.After(wait.ForeverTestTimeout):
			t.Errorf("failed to get event")
		}
	}

	podTransformer := func(obj interface{}) (interface{}, error) {
		pod, ok := obj.(*v1.Pod)
		if !ok {
			return nil, fmt.Errorf("unexpected object type: %T", obj)
		}
		pod.Spec.Hostname = "new-hostname"
		pod.Spec.Subdomain = ""
		pod.Spec.NodeName = "nodename"

		// Clear out ResourceVersion to simplify comparisons.
		pod.ResourceVersion = ""

		return pod, nil
	}

	store, controller := NewTransformingInformer(
		source,
		&v1.Pod{},
		0,
		ResourceEventHandlerDetailedFuncs{
			AddFunc:    func(obj interface{}, isInInitialList bool) { recordEvent(watch.Added, nil, obj) },
			UpdateFunc: func(oldObj, newObj interface{}) { recordEvent(watch.Modified, oldObj, newObj) },
			DeleteFunc: func(obj interface{}) { recordEvent(watch.Deleted, obj, nil) },
		},
		podTransformer,
	)

	verifyStore := func(expectedItems []interface{}) {
		items := store.List()
		if len(items) != len(expectedItems) {
			t.Errorf("unexpected items %v, expected %v", items, expectedItems)
		}
		for _, expectedItem := range expectedItems {
			found := false
			for _, item := range items {
				if apiequality.Semantic.DeepEqual(item, expectedItem) {
					found = true
				}
			}
			if !found {
				t.Errorf("expected item %v not found in %v", expectedItem, items)
			}
		}
	}

	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	go controller.RunWithContext(ctx)

	verifyEvent(watch.Added, nil, expectedPod("pod1", "2"))
	verifyStore([]interface{}{expectedPod("pod1", "2")})

	source.Add(makePod("pod2", "1"))
	verifyEvent(watch.Added, nil, expectedPod("pod2", "1"))
	verifyStore([]interface{}{expectedPod("pod1", "2"), expectedPod("pod2", "1")})

	source.Add(makePod("pod3", "1"))
	verifyEvent(watch.Added, nil, expectedPod("pod3", "1"))

	source.Modify(makePod("pod2", "2"))
	verifyEvent(watch.Modified, expectedPod("pod2", "1"), expectedPod("pod2", "2"))

	source.Delete(makePod("pod1", "2"))
	verifyEvent(watch.Deleted, expectedPod("pod1", "2"), nil)
	verifyStore([]interface{}{expectedPod("pod2", "2"), expectedPod("pod3", "1")})
}

func TestTransformingInformerRace(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	// Canceled *only* when the test is done.
	testCtx, cancel := context.WithCancel(ctx)
	defer cancel()
	// Canceled *also* during the test.
	ctx, cancel = context.WithCancel(ctx)
	defer cancel()

	// source simulates an apiserver object endpoint.
	source := newFakeControllerSource(t)

	label := "to-be-transformed"
	makePod := func(name string) *v1.Pod {
		return &v1.Pod{
			ObjectMeta: metav1.ObjectMeta{
				Name:      name,
				Namespace: "namespace",
				Labels:    map[string]string{label: "true"},
			},
			Spec: v1.PodSpec{
				Hostname: "hostname",
			},
		}
	}

	badTransform := atomic.Bool{}
	podTransformer := func(obj interface{}) (interface{}, error) {
		pod, ok := obj.(*v1.Pod)
		if !ok {
			return nil, fmt.Errorf("unexpected object type: %T", obj)
		}
		if pod.ObjectMeta.Labels[label] != "true" {
			badTransform.Store(true)
			return nil, fmt.Errorf("object already transformed: %#v", obj)
		}
		pod.ObjectMeta.Labels[label] = "false"
		return pod, nil
	}

	numObjs := 5
	for i := 0; i < numObjs; i++ {
		source.Add(makePod(fmt.Sprintf("pod-%d", i)))
	}

	type event struct{}
	events := make(chan event, numObjs)
	recordEvent := func(eventType watch.EventType, previous, current interface{}) {
		select {
		case events <- event{}:
		case <-testCtx.Done():
			// Don't block forever in the write above when test is already done.
		}
	}
	checkEvents := func(count int) {
		for i := 0; i < count; i++ {
			<-events
		}
	}
	store, controller := NewTransformingInformer(
		source,
		&v1.Pod{},
		5*time.Millisecond,
		ResourceEventHandlerDetailedFuncs{
			AddFunc:    func(obj interface{}, isInInitialList bool) { recordEvent(watch.Added, nil, obj) },
			UpdateFunc: func(oldObj, newObj interface{}) { recordEvent(watch.Modified, oldObj, newObj) },
			DeleteFunc: func(obj interface{}) { recordEvent(watch.Deleted, obj, nil) },
		},
		podTransformer,
	)

	go controller.RunWithContext(ctx)

	checkEvents(numObjs)

	// Periodically fetch objects to ensure no access races.
	wg := sync.WaitGroup{}
	errors := make(chan error, numObjs)
	for i := 0; i < numObjs; i++ {
		wg.Add(1)
		go func(index int) {
			defer wg.Done()
			key := fmt.Sprintf("namespace/pod-%d", index)
			for {
				select {
				case <-ctx.Done():
					return
				default:
				}

				obj, ok, err := store.GetByKey(key)
				if !ok || err != nil {
					errors <- fmt.Errorf("couldn't get the object for %v", key)
					return
				}
				pod := obj.(*v1.Pod)
				if pod.ObjectMeta.Labels[label] != "false" {
					errors <- fmt.Errorf("unexpected object: %#v", pod)
					return
				}
			}
		}(i)
	}

	// Let resyncs to happen for some time.
	time.Sleep(time.Second)

	cancel()
	wg.Wait()
	close(errors)
	for err := range errors {
		t.Error(err)
	}

	if badTransform.Load() {
		t.Errorf("unexpected transformation happened")
	}
}

func TestDeletionHandlingObjectToName(t *testing.T) {
	cm := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "testname",
			Namespace: "testnamespace",
		},
	}
	stringKey, err := MetaNamespaceKeyFunc(cm)
	if err != nil {
		t.Error(err)
	}
	deleted := DeletedFinalStateUnknown{
		Key: stringKey,
		Obj: cm,
	}
	expected, err := ObjectToName(cm)
	if err != nil {
		t.Error(err)
	}
	actual, err := DeletionHandlingObjectToName(deleted)
	if err != nil {
		t.Error(err)
	}
	if expected != actual {
		t.Errorf("Expected %#v, got %#v", expected, actual)
	}
}

type listWatchWithUnSupportedWatchListSemanticsWrapper struct {
	*ListWatch
}

func (lw listWatchWithUnSupportedWatchListSemanticsWrapper) IsWatchListSemanticsUnSupported() bool {
	return true
}

func toListWatcherWithUnSupportedWatchListSemantics(lw *ListWatch) ListerWatcher {
	return listWatchWithUnSupportedWatchListSemanticsWrapper{
		lw,
	}
}

func TestProcessDeltasInBatch(t *testing.T) {
	cm1 := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "testname1",
			Namespace: "testnamespace",
		},
	}
	cm2 := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "testname2",
			Namespace: "testnamespace",
		},
	}
	cm3 := &v1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "testname3",
			Namespace: "testnamespace",
		},
	}
	testDelta1 := Delta{
		Type:   Added,
		Object: cm1,
	}
	testDelta2 := Delta{
		Type:   Added,
		Object: cm2,
	}
	testDelta3 := Delta{
		Type:   Added,
		Object: cm3,
	}
	testCases := []struct {
		name                            string
		deltaList                       []Delta
		succeedingObjects               []runtime.Object
		failingObjects                  []runtime.Object
		expectedListenerReceivedObjects []interface{}
		expectedSuccessCount            int
		assertErr                       func(error) bool
	}{
		{
			name:                            "all transaction succeeding should works",
			deltaList:                       []Delta{testDelta1},
			expectedSuccessCount:            1,
			expectedListenerReceivedObjects: []interface{}{cm1},
		},
		{
			name:                 "all transaction failing should not trigger listener",
			deltaList:            []Delta{testDelta1},
			failingObjects:       []runtime.Object{cm1},
			expectedSuccessCount: 0,
			assertErr: func(err error) bool {
				return assert.Contains(t, err.Error(), "failed to execute (1/1) transactions")
			},
			expectedListenerReceivedObjects: make([]interface{}, 0),
		},
		{
			name:                 "partial transaction failing should only trigger successful events to listener #1",
			deltaList:            []Delta{testDelta1, testDelta2},
			failingObjects:       []runtime.Object{cm2},
			expectedSuccessCount: 1,
			assertErr: func(err error) bool {
				return assert.Contains(t, err.Error(), "failed to execute (1/2) transactions")
			},
			expectedListenerReceivedObjects: []interface{}{cm1},
		},
		{
			name:                 "partial transaction failing should only trigger successful events to listener #2",
			deltaList:            []Delta{testDelta1, testDelta2, testDelta3},
			failingObjects:       []runtime.Object{cm2},
			expectedSuccessCount: 2,
			assertErr: func(err error) bool {
				return assert.Contains(t, err.Error(), "failed to execute (1/3) transactions")
			},
			expectedListenerReceivedObjects: []interface{}{cm1, cm3},
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mockStore := &mockTxnStore{
				Store:       NewStore(MetaNamespaceKeyFunc),
				failingObjs: tc.failingObjects,
			}
			actualListenerReceivedObjects := make([]interface{}, 0)
			dummyListener := ResourceEventHandlerFuncs{
				AddFunc: func(obj interface{}) {
					actualListenerReceivedObjects = append(actualListenerReceivedObjects, obj)
				},
				UpdateFunc: func(oldObj, newObj interface{}) {
					actualListenerReceivedObjects = append(actualListenerReceivedObjects, newObj)
				},
				DeleteFunc: func(obj interface{}) {
					actualListenerReceivedObjects = append(actualListenerReceivedObjects, obj)
				},
			}
			err := processDeltasInBatch(
				dummyListener,
				mockStore,
				tc.deltaList,
				true)
			if tc.assertErr != nil {
				assert.True(t, tc.assertErr(err))
			}
			assert.Equal(t, tc.expectedSuccessCount, mockStore.succeedCount)
			assert.Equal(t, tc.expectedListenerReceivedObjects, actualListenerReceivedObjects)
		})
	}
}

var _ TransactionStore = &mockTxnStore{}

type mockTxnStore struct {
	failingObjs  []runtime.Object
	succeedCount int
	Store
}

func (m *mockTxnStore) Transaction(txns ...Transaction) *TransactionError {
	successfuls := make([]int, 0)
	fails := make([]int, 0)
	errs := make([]error, 0)
	for i := range txns {
		txn := txns[i]
		for _, fail := range m.failingObjs {
			if apiequality.Semantic.DeepEqual(fail, txn.Object) {
				fails = append(fails, i)
				errs = append(errs, errors.New("test error"))
			} else {
				successfuls = append(successfuls, i)
			}
		}
	}
	if len(fails) > 0 {
		m.succeedCount = len(successfuls)
		return &TransactionError{
			TotalTransactions: len(txns),
			SuccessfulIndices: successfuls,
			Errors:            errs,
		}
	}
	m.succeedCount = len(txns)
	return nil
}

func TestReplaceEvents(t *testing.T) {
	_, ctx := ktesting.NewTestContext(t)
	ctx, cancel := context.WithCancel(ctx)
	defer cancel()

	source := fcache.NewFakeControllerSource()
	store := NewStore(DeletionHandlingMetaNamespaceKeyFunc)
	t.Cleanup(func() {
		source.Shutdown()
	})

	recorder := newEventRecorder(store)

	fifo := NewRealFIFOWithOptions(RealFIFOOptions{
		KnownObjects: store,
	})

	cfg := &Config{
		Queue:            fifo,
		ListerWatcher:    source,
		ObjectType:       &v1.Pod{},
		FullResyncPeriod: 0,

		Process: func(obj interface{}, isInInitialList bool) error {
			if deltas, ok := obj.(Deltas); ok {
				return processDeltas(recorder, store, deltas, isInInitialList)
			}
			return errors.New("object given as Process argument is not Deltas")
		},
		ProcessBatch: func(deltaList []Delta, isInInitialList bool) error {
			return processDeltasInBatch(recorder, store, deltaList, isInInitialList)
		},
	}

	c := New(cfg)
	go c.RunWithContext(ctx)
	if !WaitForCacheSync(ctx.Done(), c.HasSynced) {
		t.Fatal("Timed out waiting for cache sync")
	}
	testReplaceEvents(t, ctx, fifo, recorder, store)
}

func testReplaceEvents(t *testing.T, ctx context.Context, fifo Queue, m *eventRecorder, store Store) {
	tcs := []struct {
		name            string
		initialObjs     []metav1.Object
		replacedItems   []interface{}
		replaceRV       string
		expectedHistory []eventRecord
	}{
		{
			name: "Create with no initial objects",
			replacedItems: []interface{}{
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "default", ResourceVersion: "1"}},
			},
			expectedHistory: []eventRecord{
				{Action: "add", Key: "default/pod-1", EventRV: "1", StoreRV: "1"},
			},
			replaceRV: "1",
		},
		{
			name: "Delete all objects",
			initialObjs: []metav1.Object{
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "default", ResourceVersion: "1"}},
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-3", Namespace: "default", ResourceVersion: "2"}},
			},
			replacedItems: []interface{}{},
			expectedHistory: []eventRecord{
				{Action: "delete", Key: "default/pod-1", EventRV: "1", StoreRV: ""},
				{Action: "delete", Key: "default/pod-3", EventRV: "2", StoreRV: ""},
			},
			replaceRV: "2",
		},
		{
			name: "Create, update, delete",
			initialObjs: []metav1.Object{
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "default", ResourceVersion: "1"}},
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-2", Namespace: "default", Labels: map[string]string{"ver": "1"}, ResourceVersion: "2"}},
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-3", Namespace: "default", ResourceVersion: "3"}},
			},
			replacedItems: []interface{}{
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-1", Namespace: "default", ResourceVersion: "4"}},
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-2", Namespace: "default", Labels: map[string]string{"ver": "2"}, ResourceVersion: "5"}},
				&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod-4", Namespace: "default", ResourceVersion: "6"}},
			},
			expectedHistory: []eventRecord{
				{Action: "delete", Key: "default/pod-3", EventRV: "3", StoreRV: ""},
				{Action: "update", Key: "default/pod-1", EventRV: "4", StoreRV: "4"},
				{Action: "update", Key: "default/pod-2", EventRV: "5", StoreRV: "5"},
				{Action: "add", Key: "default/pod-4", EventRV: "6", StoreRV: "6"},
			},
			replaceRV: "6",
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			// Clear history so we don't get noise from previous tests.
			m.clearHistory()

			for _, obj := range tc.initialObjs {
				require.NoError(t, fifo.Add(obj), "failed to add")
			}

			// Wait for all create events to appear
			require.NoError(t, m.waitForEventCount(ctx, len(tc.initialObjs), 5*time.Second), "Controller failed to receive initial setup all events, got: %v", m.getHistory())

			// Clear history so we only compare what happens during the RESYNC/REPLACE
			m.clearHistory()

			// Run replace
			require.NoError(t, fifo.Replace(tc.replacedItems, tc.replaceRV), "failed to replace")

			// Wait for all expected events to be received
			require.NoError(t, m.waitForEventCount(ctx, len(tc.expectedHistory), 5*time.Second), "Controller failed to receive all events, got: %v", m.getHistory())

			items := sortItemsByKey(store.List(), DeletionHandlingMetaNamespaceKeyFunc)
			assert.Equal(t, tc.replacedItems, items)
			// synthetic delete events from a replace are not ordered, sort by key to make the test deterministic
			history := sortDeleteEventsByKey(m.getHistory())
			assert.Equal(t, tc.expectedHistory, history)
		})
	}
}

type eventRecord struct {
	Action  string
	Key     string
	EventRV string
	StoreRV string
}

type eventRecorder struct {
	historyLock sync.Mutex
	history     []eventRecord
	store       Store

	updateCh chan bool
}

func newEventRecorder(store Store) *eventRecorder {
	return &eventRecorder{
		store:    store,
		updateCh: make(chan bool, 1),
	}
}

func (m *eventRecorder) OnAdd(obj interface{}, _ bool) {
	m.record("add", obj)
}
func (m *eventRecorder) OnUpdate(_, obj interface{}) {
	m.record("update", obj)
}
func (m *eventRecorder) OnDelete(obj interface{}) {
	m.record("delete", obj)
}

func (m *eventRecorder) record(action string, obj interface{}) {
	m.historyLock.Lock()
	defer m.historyLock.Unlock()
	key, _ := DeletionHandlingMetaNamespaceKeyFunc(obj)

	if deleted, ok := obj.(DeletedFinalStateUnknown); ok {
		obj = deleted.Obj
	}

	eventRV := ""
	if accessor, err := meta.Accessor(obj); err == nil {
		eventRV = accessor.GetResourceVersion()
	}

	storeRV := ""
	if storeObj, exists, err := m.store.GetByKey(key); err == nil && exists {
		if accessor, err := meta.Accessor(storeObj); err == nil {
			storeRV = accessor.GetResourceVersion()
		}
	}

	m.history = append(m.history, eventRecord{
		Action:  action,
		Key:     key,
		EventRV: eventRV,
		StoreRV: storeRV,
	})
	select {
	case m.updateCh <- true:
	default:
	}
}

func (m *eventRecorder) clearHistory() {
	m.historyLock.Lock()
	defer m.historyLock.Unlock()
	m.history = []eventRecord{}
}

func (m *eventRecorder) getHistory() []eventRecord {
	m.historyLock.Lock()
	historyCopy := make([]eventRecord, len(m.history))
	copy(historyCopy, m.history)
	m.historyLock.Unlock()

	return historyCopy
}

func (m *eventRecorder) waitForEventCount(ctx context.Context, count int, timeout time.Duration) error {
	ctx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	for {
		m.historyLock.Lock()
		currentCount := len(m.history)
		m.historyLock.Unlock()

		if currentCount >= count {
			return nil
		}

		select {
		case <-m.updateCh:
			continue
		case <-ctx.Done():
			return fmt.Errorf("context canceled waiting for %d events, currently have %d", count, currentCount)
		}
	}
}

func sortItemsByKey(items []interface{}, keyFunc KeyFunc) []interface{} {
	sort.Slice(items, func(i, j int) bool {
		keyI, _ := keyFunc(items[i])
		keyJ, _ := keyFunc(items[j])
		return keyI < keyJ
	})
	return items
}

func sortDeleteEventsByKey(events []eventRecord) []eventRecord {
	sort.Slice(events, func(i, j int) bool {
		eventI := events[i]
		eventJ := events[j]
		// sort delete events for different objects by key
		if eventI.Action == "delete" && eventJ.Action == "delete" && eventI.Key != eventJ.Key {
			return eventI.Key < eventJ.Key
		}
		// keep existing order otherwise
		return i < j
	})
	return events
}
