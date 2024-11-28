/*
Copyright 2017 The Kubernetes Authors.

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
	"fmt"
	"math/rand"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	testingclock "k8s.io/utils/clock/testing"
)

type testListener struct {
	lock              sync.RWMutex
	resyncPeriod      time.Duration
	expectedItemNames sets.String
	receivedItemNames []string
	name              string
}

func newTestListener(name string, resyncPeriod time.Duration, expected ...string) *testListener {
	l := &testListener{
		resyncPeriod:      resyncPeriod,
		expectedItemNames: sets.NewString(expected...),
		name:              name,
	}
	return l
}

func (l *testListener) OnAdd(obj interface{}, isInInitialList bool) {
	l.handle(obj)
}

func (l *testListener) OnUpdate(old, new interface{}) {
	l.handle(new)
}

func (l *testListener) OnDelete(obj interface{}) {
}

func (l *testListener) handle(obj interface{}) {
	key, _ := MetaNamespaceKeyFunc(obj)
	fmt.Printf("%s: handle: %v\n", l.name, key)
	l.lock.Lock()
	defer l.lock.Unlock()
	objectMeta, _ := meta.Accessor(obj)
	l.receivedItemNames = append(l.receivedItemNames, objectMeta.GetName())
}

func (l *testListener) ok() bool {
	fmt.Println("polling")
	err := wait.PollImmediate(100*time.Millisecond, 2*time.Second, func() (bool, error) {
		if l.satisfiedExpectations() {
			return true, nil
		}
		return false, nil
	})
	if err != nil {
		return false
	}

	// wait just a bit to allow any unexpected stragglers to come in
	fmt.Println("sleeping")
	time.Sleep(1 * time.Second)
	fmt.Println("final check")
	return l.satisfiedExpectations()
}

func (l *testListener) satisfiedExpectations() bool {
	l.lock.RLock()
	defer l.lock.RUnlock()

	return sets.NewString(l.receivedItemNames...).Equal(l.expectedItemNames)
}

func eventHandlerCount(i SharedInformer) int {
	s := i.(*sharedIndexInformer)
	s.startedLock.Lock()
	defer s.startedLock.Unlock()
	return len(s.processor.listeners)
}

func isStarted(i SharedInformer) bool {
	s := i.(*sharedIndexInformer)
	s.startedLock.Lock()
	defer s.startedLock.Unlock()
	return s.started
}

func isRegistered(i SharedInformer, h ResourceEventHandlerRegistration) bool {
	s := i.(*sharedIndexInformer)
	return s.processor.getListener(h) != nil
}

func TestIndexer(t *testing.T) {
	assert := assert.New(t)
	// source simulates an apiserver object endpoint.
	source := newFakeControllerSource(t)
	pod1 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", Labels: map[string]string{"a": "a-val", "b": "b-val1"}}}
	pod2 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod2", Labels: map[string]string{"b": "b-val2"}}}
	pod3 := &v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod3", Labels: map[string]string{"a": "a-val2"}}}
	source.Add(pod1)
	source.Add(pod2)

	// create the shared informer and resync every 1s
	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)
	err := informer.AddIndexers(map[string]IndexFunc{
		"labels": func(obj interface{}) ([]string, error) {
			res := []string{}
			for k := range obj.(*v1.Pod).Labels {
				res = append(res, k)
			}
			return res, nil
		},
	})
	if err != nil {
		t.Fatal(err)
	}
	stop := make(chan struct{})
	defer close(stop)

	go informer.Run(stop)
	WaitForCacheSync(stop, informer.HasSynced)

	cmpOps := cmpopts.SortSlices(func(a, b any) bool {
		return a.(*v1.Pod).Name < b.(*v1.Pod).Name
	})

	// We should be able to lookup by index
	res, err := informer.GetIndexer().ByIndex("labels", "a")
	assert.NoError(err)
	if diff := cmp.Diff([]any{pod1}, res); diff != "" {
		t.Fatal(diff)
	}

	// Adding an item later is fine as well
	source.Add(pod3)
	// Event is async, need to poll
	assert.Eventually(func() bool {
		res, _ := informer.GetIndexer().ByIndex("labels", "a")
		return cmp.Diff([]any{pod1, pod3}, res, cmpOps) == ""
	}, time.Second*3, time.Millisecond)

	// Adding an index later is also fine
	err = informer.AddIndexers(map[string]IndexFunc{
		"labels-again": func(obj interface{}) ([]string, error) {
			res := []string{}
			for k := range obj.(*v1.Pod).Labels {
				res = append(res, k)
			}
			return res, nil
		},
	})
	assert.NoError(err)

	// Should be immediately available
	res, err = informer.GetIndexer().ByIndex("labels-again", "a")
	assert.NoError(err)
	if diff := cmp.Diff([]any{pod1, pod3}, res, cmpOps); diff != "" {
		t.Fatal(diff)
	}
	if got := informer.GetIndexer().ListIndexFuncValues("labels"); !sets.New(got...).Equal(sets.New("a", "b")) {
		t.Fatalf("got %v", got)
	}
	if got := informer.GetIndexer().ListIndexFuncValues("labels-again"); !sets.New(got...).Equal(sets.New("a", "b")) {
		t.Fatalf("got %v", got)
	}
}

func TestListenerResyncPeriods(t *testing.T) {
	// source simulates an apiserver object endpoint.
	source := newFakeControllerSource(t)
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod2"}})

	// create the shared informer and resync every 1s
	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)

	clock := testingclock.NewFakeClock(time.Now())
	informer.clock = clock
	informer.processor.clock = clock

	// listener 1, never resync
	listener1 := newTestListener("listener1", 0, "pod1", "pod2")
	informer.AddEventHandlerWithResyncPeriod(listener1, listener1.resyncPeriod)

	// listener 2, resync every 2s
	listener2 := newTestListener("listener2", 2*time.Second, "pod1", "pod2")
	informer.AddEventHandlerWithResyncPeriod(listener2, listener2.resyncPeriod)

	// listener 3, resync every 3s
	listener3 := newTestListener("listener3", 3*time.Second, "pod1", "pod2")
	informer.AddEventHandlerWithResyncPeriod(listener3, listener3.resyncPeriod)
	listeners := []*testListener{listener1, listener2, listener3}

	stop := make(chan struct{})
	defer close(stop)

	go informer.Run(stop)

	// ensure all listeners got the initial List
	for _, listener := range listeners {
		if !listener.ok() {
			t.Errorf("%s: expected %v, got %v", listener.name, listener.expectedItemNames, listener.receivedItemNames)
		}
	}

	// reset
	for _, listener := range listeners {
		listener.receivedItemNames = []string{}
	}

	// advance so listener2 gets a resync
	clock.Step(2 * time.Second)

	// make sure listener2 got the resync
	if !listener2.ok() {
		t.Errorf("%s: expected %v, got %v", listener2.name, listener2.expectedItemNames, listener2.receivedItemNames)
	}

	// wait a bit to give errant items a chance to go to 1 and 3
	time.Sleep(1 * time.Second)

	// make sure listeners 1 and 3 got nothing
	if len(listener1.receivedItemNames) != 0 {
		t.Errorf("listener1: should not have resynced (got %d)", len(listener1.receivedItemNames))
	}
	if len(listener3.receivedItemNames) != 0 {
		t.Errorf("listener3: should not have resynced (got %d)", len(listener3.receivedItemNames))
	}

	// reset
	for _, listener := range listeners {
		listener.receivedItemNames = []string{}
	}

	// advance so listener3 gets a resync
	clock.Step(1 * time.Second)

	// make sure listener3 got the resync
	if !listener3.ok() {
		t.Errorf("%s: expected %v, got %v", listener3.name, listener3.expectedItemNames, listener3.receivedItemNames)
	}

	// wait a bit to give errant items a chance to go to 1 and 2
	time.Sleep(1 * time.Second)

	// make sure listeners 1 and 2 got nothing
	if len(listener1.receivedItemNames) != 0 {
		t.Errorf("listener1: should not have resynced (got %d)", len(listener1.receivedItemNames))
	}
	if len(listener2.receivedItemNames) != 0 {
		t.Errorf("listener2: should not have resynced (got %d)", len(listener2.receivedItemNames))
	}
}

func TestResyncCheckPeriod(t *testing.T) {
	// source simulates an apiserver object endpoint.
	source := newFakeControllerSource(t)

	// create the shared informer and resync every 12 hours
	informer := NewSharedInformer(source, &v1.Pod{}, 12*time.Hour).(*sharedIndexInformer)
	gl := informer.processor.getListener

	clock := testingclock.NewFakeClock(time.Now())
	informer.clock = clock
	informer.processor.clock = clock

	// listener 1, never resync
	listener1 := newTestListener("listener1", 0)
	handler1, _ := informer.AddEventHandlerWithResyncPeriod(listener1, listener1.resyncPeriod)

	if e, a := 12*time.Hour, informer.resyncCheckPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := time.Duration(0), gl(handler1).resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}

	// listener 2, resync every minute
	listener2 := newTestListener("listener2", 1*time.Minute)
	handler2, _ := informer.AddEventHandlerWithResyncPeriod(listener2, listener2.resyncPeriod)
	if e, a := 1*time.Minute, informer.resyncCheckPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := time.Duration(0), gl(handler1).resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := 1*time.Minute, gl(handler2).resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}

	// listener 3, resync every 55 seconds
	listener3 := newTestListener("listener3", 55*time.Second)
	handler3, _ := informer.AddEventHandlerWithResyncPeriod(listener3, listener3.resyncPeriod)
	if e, a := 55*time.Second, informer.resyncCheckPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := time.Duration(0), gl(handler1).resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := 1*time.Minute, gl(handler2).resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := 55*time.Second, gl(handler3).resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}

	// listener 4, resync every 5 seconds
	listener4 := newTestListener("listener4", 5*time.Second)
	handler4, _ := informer.AddEventHandlerWithResyncPeriod(listener4, listener4.resyncPeriod)
	if e, a := 5*time.Second, informer.resyncCheckPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := time.Duration(0), gl(handler1).resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := 1*time.Minute, gl(handler2).resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := 55*time.Second, gl(handler3).resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := 5*time.Second, gl(handler4).resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
}

// verify that https://github.com/kubernetes/kubernetes/issues/59822 is fixed
func TestSharedInformerInitializationRace(t *testing.T) {
	source := newFakeControllerSource(t)
	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)
	listener := newTestListener("raceListener", 0)

	stop := make(chan struct{})
	go informer.AddEventHandlerWithResyncPeriod(listener, listener.resyncPeriod)
	go informer.Run(stop)
	close(stop)
}

// TestSharedInformerWatchDisruption simulates a watch that was closed
// with updates to the store during that time. We ensure that handlers with
// resync and no resync see the expected state.
func TestSharedInformerWatchDisruption(t *testing.T) {
	// source simulates an apiserver object endpoint.
	source := newFakeControllerSource(t)

	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: "pod1", ResourceVersion: "1"}})
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod2", UID: "pod2", ResourceVersion: "2"}})

	// create the shared informer and resync every 1s
	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)

	clock := testingclock.NewFakeClock(time.Now())
	informer.clock = clock
	informer.processor.clock = clock

	// listener, never resync
	listenerNoResync := newTestListener("listenerNoResync", 0, "pod1", "pod2")
	informer.AddEventHandlerWithResyncPeriod(listenerNoResync, listenerNoResync.resyncPeriod)

	listenerResync := newTestListener("listenerResync", 1*time.Second, "pod1", "pod2")
	informer.AddEventHandlerWithResyncPeriod(listenerResync, listenerResync.resyncPeriod)
	listeners := []*testListener{listenerNoResync, listenerResync}

	stop := make(chan struct{})
	defer close(stop)

	go informer.Run(stop)

	for _, listener := range listeners {
		if !listener.ok() {
			t.Errorf("%s: expected %v, got %v", listener.name, listener.expectedItemNames, listener.receivedItemNames)
		}
	}

	// Add pod3, bump pod2 but don't broadcast it, so that the change will be seen only on relist
	source.AddDropWatch(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod3", UID: "pod3", ResourceVersion: "3"}})
	source.ModifyDropWatch(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod2", UID: "pod2", ResourceVersion: "4"}})

	// Ensure that nobody saw any changes
	for _, listener := range listeners {
		if !listener.ok() {
			t.Errorf("%s: expected %v, got %v", listener.name, listener.expectedItemNames, listener.receivedItemNames)
		}
	}

	for _, listener := range listeners {
		listener.receivedItemNames = []string{}
	}

	listenerNoResync.expectedItemNames = sets.NewString("pod2", "pod3")
	listenerResync.expectedItemNames = sets.NewString("pod1", "pod2", "pod3")

	// This calls shouldSync, which deletes noResync from the list of syncingListeners
	clock.Step(1 * time.Second)

	// Simulate a connection loss (or even just a too-old-watch)
	source.ResetWatch()

	// Wait long enough for the reflector to exit and the backoff function to start waiting
	// on the fake clock, otherwise advancing the fake clock will have no effect.
	// TODO: Make this deterministic by counting the number of waiters on FakeClock
	time.Sleep(10 * time.Millisecond)

	// Advance the clock to cause the backoff wait to expire.
	clock.Step(1601 * time.Millisecond)

	// Wait long enough for backoff to invoke ListWatch a second time and distribute events
	// to listeners.
	time.Sleep(10 * time.Millisecond)

	for _, listener := range listeners {
		if !listener.ok() {
			t.Errorf("%s: expected %v, got %v", listener.name, listener.expectedItemNames, listener.receivedItemNames)
		}
	}
}

func TestSharedInformerErrorHandling(t *testing.T) {
	source := newFakeControllerSource(t)
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})
	source.ListError = fmt.Errorf("Access Denied")

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)

	errCh := make(chan error)
	_ = informer.SetWatchErrorHandler(func(_ *Reflector, err error) {
		errCh <- err
	})

	stop := make(chan struct{})
	go informer.Run(stop)

	select {
	case err := <-errCh:
		if !strings.Contains(err.Error(), "Access Denied") {
			t.Errorf("Expected 'Access Denied' error. Actual: %v", err)
		}
	case <-time.After(time.Second):
		t.Errorf("Timeout waiting for error handler call")
	}
	close(stop)
}

// TestSharedInformerStartRace is a regression test to ensure there is no race between
// Run and SetWatchErrorHandler, and Run and SetTransform.
func TestSharedInformerStartRace(t *testing.T) {
	source := newFakeControllerSource(t)
	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)
	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				return
			default:
			}
			// Set dummy functions, just to test for race
			informer.SetTransform(func(i interface{}) (interface{}, error) {
				return i, nil
			})
			informer.SetWatchErrorHandler(func(r *Reflector, err error) {
			})
		}
	}()

	go informer.Run(stop)

	close(stop)
}

func TestSharedInformerTransformer(t *testing.T) {
	// source simulates an apiserver object endpoint.
	source := newFakeControllerSource(t)

	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1", UID: "pod1", ResourceVersion: "1"}})
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod2", UID: "pod2", ResourceVersion: "2"}})

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)
	informer.SetTransform(func(obj interface{}) (interface{}, error) {
		if pod, ok := obj.(*v1.Pod); ok {
			name := pod.GetName()

			if upper := strings.ToUpper(name); upper != name {
				pod.SetName(upper)
				return pod, nil
			}
		}
		return obj, nil
	})

	listenerTransformer := newTestListener("listenerTransformer", 0, "POD1", "POD2")
	informer.AddEventHandler(listenerTransformer)

	stop := make(chan struct{})
	go informer.Run(stop)
	defer close(stop)

	if !listenerTransformer.ok() {
		t.Errorf("%s: expected %v, got %v", listenerTransformer.name, listenerTransformer.expectedItemNames, listenerTransformer.receivedItemNames)
	}
}

func TestSharedInformerRemoveHandler(t *testing.T) {
	source := newFakeControllerSource(t)
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second)

	handler1 := &ResourceEventHandlerFuncs{}
	handle1, err := informer.AddEventHandler(handler1)
	if err != nil {
		t.Errorf("informer did not add handler1: %s", err)
		return
	}
	handler2 := &ResourceEventHandlerFuncs{}
	handle2, err := informer.AddEventHandler(handler2)
	if err != nil {
		t.Errorf("informer did not add handler2: %s", err)
		return
	}

	if eventHandlerCount(informer) != 2 {
		t.Errorf("informer has %d registered handler, instead of 2", eventHandlerCount(informer))
	}

	if err := informer.RemoveEventHandler(handle2); err != nil {
		t.Errorf("removing of second pointer handler failed: %s", err)
	}
	if eventHandlerCount(informer) != 1 {
		t.Errorf("after removing handler informer has %d registered handler(s), instead of 1", eventHandlerCount(informer))
	}

	if err := informer.RemoveEventHandler(handle1); err != nil {
		t.Errorf("removing of first pointer handler failed: %s", err)
	}
	if eventHandlerCount(informer) != 0 {
		t.Errorf("informer still has registered handlers after removing both handlers")
	}
}

func TestSharedInformerRemoveForeignHandler(t *testing.T) {
	source := newFakeControllerSource(t)
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)

	source2 := newFakeControllerSource(t)
	source2.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	informer2 := NewSharedInformer(source2, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)

	handler1 := &ResourceEventHandlerFuncs{}
	handle1, err := informer.AddEventHandler(handler1)
	if err != nil {
		t.Errorf("informer did not add handler1: %s", err)
		return
	}
	handler2 := &ResourceEventHandlerFuncs{}
	handle2, err := informer.AddEventHandler(handler2)
	if err != nil {
		t.Errorf("informer did not add handler2: %s", err)
		return
	}

	if eventHandlerCount(informer) != 2 {
		t.Errorf("informer has %d registered handler, instead of 2", eventHandlerCount(informer))
	}
	if eventHandlerCount(informer2) != 0 {
		t.Errorf("informer2 has %d registered handler, instead of 0", eventHandlerCount(informer2))
	}

	// remove handle at foreign informer
	if isRegistered(informer2, handle1) {
		t.Errorf("handle1 registered for informer2")
	}
	if isRegistered(informer2, handle2) {
		t.Errorf("handle2 registered for informer2")
	}
	if err := informer2.RemoveEventHandler(handle1); err != nil {
		t.Errorf("removing of second pointer handler failed: %s", err)
	}
	if eventHandlerCount(informer) != 2 {
		t.Errorf("informer has %d registered handler, instead of 2", eventHandlerCount(informer))
	}
	if eventHandlerCount(informer2) != 0 {
		t.Errorf("informer2 has %d registered handler, instead of 0", eventHandlerCount(informer2))
	}
	if !isRegistered(informer, handle1) {
		t.Errorf("handle1 not registered anymore for informer")
	}
	if !isRegistered(informer, handle2) {
		t.Errorf("handle2 not registered anymore for informer")
	}

	if eventHandlerCount(informer) != 2 {
		t.Errorf("informer has %d registered handler, instead of 2", eventHandlerCount(informer))
	}
	if eventHandlerCount(informer2) != 0 {
		t.Errorf("informer2 has %d registered handler, instead of 0", eventHandlerCount(informer2))
	}
	if !isRegistered(informer, handle1) {
		t.Errorf("handle1 not registered anymore for informer")
	}
	if !isRegistered(informer, handle2) {
		t.Errorf("handle2 not registered anymore for informer")
	}

	if err := informer.RemoveEventHandler(handle2); err != nil {
		t.Errorf("removing of second pointer handler failed: %s", err)
	}
	if eventHandlerCount(informer) != 1 {
		t.Errorf("after removing handler informer has %d registered handler(s), instead of 1", eventHandlerCount(informer))
	}

	if err := informer.RemoveEventHandler(handle1); err != nil {
		t.Errorf("removing of first pointer handler failed: %s", err)
	}
	if eventHandlerCount(informer) != 0 {
		t.Errorf("informer still has registered handlers after removing both handlers")
	}
}

func TestSharedInformerMultipleRegistration(t *testing.T) {
	source := newFakeControllerSource(t)
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)

	handler1 := &ResourceEventHandlerFuncs{}
	reg1, err := informer.AddEventHandler(handler1)
	if err != nil {
		t.Errorf("informer did not add handler for the first time: %s", err)
		return
	}

	if !isRegistered(informer, reg1) {
		t.Errorf("handle1 is not active after successful registration")
		return
	}

	reg2, err := informer.AddEventHandler(handler1)
	if err != nil {
		t.Errorf("informer did not add handler for the second: %s", err)
		return
	}

	if !isRegistered(informer, reg2) {
		t.Errorf("handle2 is not active after successful registration")
		return
	}

	if eventHandlerCount(informer) != 2 {
		t.Errorf("informer has %d registered handler(s), instead of 2", eventHandlerCount(informer))
	}

	if err := informer.RemoveEventHandler(reg1); err != nil {
		t.Errorf("removing of duplicate handler registration failed: %s", err)
	}

	if isRegistered(informer, reg1) {
		t.Errorf("handle1 is still active after successful remove")
		return
	}
	if !isRegistered(informer, reg2) {
		t.Errorf("handle2 is not active after removing handle1")
		return
	}

	if eventHandlerCount(informer) != 1 {
		if eventHandlerCount(informer) == 0 {
			t.Errorf("informer has no registered handler anymore after removal of duplicate registrations")
		} else {
			t.Errorf("informer has unexpected number (%d) of handlers after removal of duplicate handler registration", eventHandlerCount(informer))
		}
	}

	if err := informer.RemoveEventHandler(reg2); err != nil {
		t.Errorf("removing of second handler registration failed: %s", err)
	}

	if isRegistered(informer, reg2) {
		t.Errorf("handle2 is still active after successful remove")
		return
	}

	if eventHandlerCount(informer) != 0 {
		t.Errorf("informer has unexpected number (%d) of handlers after removal of second handler registrations", eventHandlerCount(informer))
	}
}

func TestRemovingRemovedSharedInformer(t *testing.T) {
	source := newFakeControllerSource(t)
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)
	handler := &ResourceEventHandlerFuncs{}
	reg, err := informer.AddEventHandler(handler)

	if err != nil {
		t.Errorf("informer did not add handler for the first time: %s", err)
		return
	}
	if err := informer.RemoveEventHandler(reg); err != nil {
		t.Errorf("removing of handler registration failed: %s", err)
		return
	}
	if isRegistered(informer, reg) {
		t.Errorf("handle is still active after successful remove")
		return
	}
	if err := informer.RemoveEventHandler(reg); err != nil {
		t.Errorf("removing of already removed registration yields unexpected error: %s", err)
	}
	if isRegistered(informer, reg) {
		t.Errorf("handle is still active after second remove")
		return
	}
}

// Shows that many concurrent goroutines can be manipulating shared informer
// listeners without tripping it up. There are not really many assertions in this
// test. Meant to be run with -race to find race conditions
func TestSharedInformerHandlerAbuse(t *testing.T) {
	source := newFakeControllerSource(t)
	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)

	ctx, cancel := context.WithCancel(context.Background())
	informerCtx, informerCancel := context.WithCancel(context.Background())
	go func() {
		informer.Run(informerCtx.Done())
		cancel()
	}()

	worker := func() {
		// Keep adding and removing handler
		// Make sure no duplicate events?
		funcs := ResourceEventHandlerDetailedFuncs{
			AddFunc:    func(obj interface{}, isInInitialList bool) {},
			UpdateFunc: func(oldObj, newObj interface{}) {},
			DeleteFunc: func(obj interface{}) {},
		}
		handles := []ResourceEventHandlerRegistration{}

		for {
			select {
			case <-ctx.Done():
				return
			default:
				switch rand.Intn(2) {
				case 0:
					// Register handler again
					reg, err := informer.AddEventHandlerWithResyncPeriod(funcs, 1*time.Second)
					if err != nil {
						if strings.Contains(err.Error(), "stopped already") {
							// test is over
							return
						}
						t.Errorf("failed to add handler: %v", err)
						return
					}
					handles = append(handles, reg)
				case 1:
					//  Remove a random handler
					if len(handles) == 0 {
						continue
					}

					idx := rand.Intn(len(handles))
					err := informer.RemoveEventHandler(handles[idx])
					if err != nil {
						if strings.Contains(err.Error(), "stopped already") {
							// test is over
							return
						}
						t.Errorf("failed to remove handler: %v", err)
						return
					}
					handles = append(handles[:idx], handles[idx+1:]...)
				}
			}
		}
	}

	wg := sync.WaitGroup{}
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			worker()
			wg.Done()
		}()
	}

	objs := []*v1.Pod{}

	// While workers run, randomly create events for the informer
	for i := 0; i < 10000; i++ {
		if len(objs) == 0 {
			// Make sure there is always an object
			obj := &v1.Pod{ObjectMeta: metav1.ObjectMeta{
				Name: "pod" + strconv.Itoa(i),
			}}
			objs = append(objs, obj)

			// deep copy before adding since the Modify function mutates the obj
			source.Add(obj.DeepCopy())
		}

		switch rand.Intn(3) {
		case 0:
			// Add Object
			obj := &v1.Pod{ObjectMeta: metav1.ObjectMeta{
				Name: "pod" + strconv.Itoa(i),
			}}
			objs = append(objs, obj)
			source.Add(obj.DeepCopy())
		case 1:
			// Update Object
			idx := rand.Intn(len(objs))
			source.Modify(objs[idx].DeepCopy())

		case 2:
			// Remove Object
			idx := rand.Intn(len(objs))
			source.Delete(objs[idx].DeepCopy())
			objs = append(objs[:idx], objs[idx+1:]...)
		}
	}

	// sotp informer which stops workers. stopping informer first to exercise
	// contention for informer while it is closing
	informerCancel()

	// wait for workers to finish since they may throw errors
	wg.Wait()
}

func TestStateSharedInformer(t *testing.T) {
	source := newFakeControllerSource(t)
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)
	listener := newTestListener("listener", 0, "pod1")
	informer.AddEventHandlerWithResyncPeriod(listener, listener.resyncPeriod)

	if isStarted(informer) {
		t.Errorf("informer already started after creation")
		return
	}
	if informer.IsStopped() {
		t.Errorf("informer already stopped after creation")
		return
	}
	stop := make(chan struct{})
	go informer.Run(stop)
	if !listener.ok() {
		t.Errorf("informer did not report initial objects")
		close(stop)
		return
	}

	if !isStarted(informer) {
		t.Errorf("informer does not report to be started although handling events")
		close(stop)
		return
	}
	if informer.IsStopped() {
		t.Errorf("informer reports to be stopped although stop channel not closed")
		close(stop)
		return
	}

	close(stop)
	fmt.Println("sleeping")
	time.Sleep(1 * time.Second)

	if !informer.IsStopped() {
		t.Errorf("informer reports not to be stopped although stop channel closed")
		return
	}
	if !isStarted(informer) {
		t.Errorf("informer reports not to be started after it has been started and stopped")
		return
	}
}

func TestAddOnStoppedSharedInformer(t *testing.T) {
	source := newFakeControllerSource(t)
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)
	listener := newTestListener("listener", 0, "pod1")
	stop := make(chan struct{})
	go informer.Run(stop)
	close(stop)

	err := wait.PollImmediate(100*time.Millisecond, 2*time.Second, func() (bool, error) {
		if informer.IsStopped() {
			return true, nil
		}
		return false, nil
	})

	if err != nil {
		t.Errorf("informer reports not to be stopped although stop channel closed")
		return
	}

	_, err = informer.AddEventHandlerWithResyncPeriod(listener, listener.resyncPeriod)
	if err == nil {
		t.Errorf("stopped informer did not reject add handler")
		return
	}
	if !strings.HasSuffix(err.Error(), "was not added to shared informer because it has stopped already") {
		t.Errorf("adding handler to a stopped informer yields unexpected error: %s", err)
		return
	}
}

func TestRemoveOnStoppedSharedInformer(t *testing.T) {
	source := newFakeControllerSource(t)
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)
	listener := newTestListener("listener", 0, "pod1")
	handle, err := informer.AddEventHandlerWithResyncPeriod(listener, listener.resyncPeriod)
	if err != nil {
		t.Errorf("informer did not add handler: %s", err)
		return
	}
	stop := make(chan struct{})
	go informer.Run(stop)
	close(stop)
	fmt.Println("sleeping")
	time.Sleep(1 * time.Second)

	if !informer.IsStopped() {
		t.Errorf("informer reports not to be stopped although stop channel closed")
		return
	}
	err = informer.RemoveEventHandler(handle)
	if err != nil {
		t.Errorf("informer does not remove handler on stopped informer")
		return
	}
}

func TestRemoveWhileActive(t *testing.T) {
	// source simulates an apiserver object endpoint.
	source := newFakeControllerSource(t)

	// create the shared informer and resync every 12 hours
	informer := NewSharedInformer(source, &v1.Pod{}, 0).(*sharedIndexInformer)

	listener := newTestListener("listener", 0, "pod1")
	handle, _ := informer.AddEventHandler(listener)

	stop := make(chan struct{})
	defer close(stop)

	go informer.Run(stop)
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	if !listener.ok() {
		t.Errorf("event did not occur")
		return
	}

	informer.RemoveEventHandler(handle)

	if isRegistered(informer, handle) {
		t.Errorf("handle is still active after successful remove")
		return
	}

	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod2"}})

	if !listener.ok() {
		t.Errorf("unexpected event occurred")
		return
	}
}

func TestAddWhileActive(t *testing.T) {
	// source simulates an apiserver object endpoint.
	source := newFakeControllerSource(t)

	// create the shared informer and resync every 12 hours
	informer := NewSharedInformer(source, &v1.Pod{}, 0).(*sharedIndexInformer)
	listener1 := newTestListener("originalListener", 0, "pod1")
	listener2 := newTestListener("listener2", 0, "pod1", "pod2")
	handle1, _ := informer.AddEventHandler(listener1)

	if handle1.HasSynced() {
		t.Error("Synced before Run??")
	}

	stop := make(chan struct{})
	defer close(stop)

	go informer.Run(stop)
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	if !listener1.ok() {
		t.Errorf("events on listener1 did not occur")
		return
	}

	if !handle1.HasSynced() {
		t.Error("Not synced after Run??")
	}

	listener2.lock.Lock() // ensure we observe it before it has synced
	handle2, _ := informer.AddEventHandler(listener2)
	if handle2.HasSynced() {
		t.Error("Synced before processing anything?")
	}
	listener2.lock.Unlock() // permit it to proceed and sync

	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod2"}})

	if !listener2.ok() {
		t.Errorf("event on listener2 did not occur")
		return
	}

	if !handle2.HasSynced() {
		t.Error("Not synced even after processing?")
	}

	if !isRegistered(informer, handle1) {
		t.Errorf("handle1 is not active")
		return
	}
	if !isRegistered(informer, handle2) {
		t.Errorf("handle2 is not active")
		return
	}

	listener1.expectedItemNames = listener2.expectedItemNames
	if !listener1.ok() {
		t.Errorf("events on listener1 did not occur")
		return
	}
}

// TestShutdown depends on goleak.VerifyTestMain in main_test.go to verify that
// all goroutines really have stopped in the different scenarios.
func TestShutdown(t *testing.T) {
	t.Run("no-context", func(t *testing.T) {
		source := newFakeControllerSource(t)
		stop := make(chan struct{})
		defer close(stop)

		informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second)
		handler, err := informer.AddEventHandler(ResourceEventHandlerFuncs{
			AddFunc: func(_ any) {},
		})
		require.NoError(t, err)
		defer func() {
			assert.NoError(t, informer.RemoveEventHandler(handler))
		}()
		go informer.Run(stop)
		require.Eventually(t, informer.HasSynced, time.Minute, time.Millisecond, "informer has synced")
	})

	t.Run("no-context-later", func(t *testing.T) {
		source := newFakeControllerSource(t)
		stop := make(chan struct{})
		defer close(stop)

		informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second)
		go informer.Run(stop)
		require.Eventually(t, informer.HasSynced, time.Minute, time.Millisecond, "informer has synced")

		handler, err := informer.AddEventHandler(ResourceEventHandlerFuncs{
			AddFunc: func(_ any) {},
		})
		require.NoError(t, err)
		assert.NoError(t, informer.RemoveEventHandler(handler))
	})

	t.Run("no-run", func(t *testing.T) {
		source := newFakeControllerSource(t)
		informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second)
		_, err := informer.AddEventHandler(ResourceEventHandlerFuncs{
			AddFunc: func(_ any) {},
		})
		require.NoError(t, err)

		// At this point, neither informer nor handler have any goroutines running
		// and it doesn't matter that nothing gets stopped or removed.
	})
}
