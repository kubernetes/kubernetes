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
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	fcache "k8s.io/client-go/tools/cache/testing"
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

func (l *testListener) OnAdd(obj interface{}) {
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
	return s.isRegistered(h)
}

func TestListenerResyncPeriods(t *testing.T) {
	// source simulates an apiserver object endpoint.
	source := fcache.NewFakeControllerSource()
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
	source := fcache.NewFakeControllerSource()

	// create the shared informer and resync every 12 hours
	informer := NewSharedInformer(source, &v1.Pod{}, 12*time.Hour).(*sharedIndexInformer)

	clock := testingclock.NewFakeClock(time.Now())
	informer.clock = clock
	informer.processor.clock = clock

	// listener 1, never resync
	listener1 := newTestListener("listener1", 0)
	informer.AddEventHandlerWithResyncPeriod(listener1, listener1.resyncPeriod)
	if e, a := 12*time.Hour, informer.resyncCheckPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := time.Duration(0), informer.processor.listeners[0].resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}

	// listener 2, resync every minute
	listener2 := newTestListener("listener2", 1*time.Minute)
	informer.AddEventHandlerWithResyncPeriod(listener2, listener2.resyncPeriod)
	if e, a := 1*time.Minute, informer.resyncCheckPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := time.Duration(0), informer.processor.listeners[0].resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := 1*time.Minute, informer.processor.listeners[1].resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}

	// listener 3, resync every 55 seconds
	listener3 := newTestListener("listener3", 55*time.Second)
	informer.AddEventHandlerWithResyncPeriod(listener3, listener3.resyncPeriod)
	if e, a := 55*time.Second, informer.resyncCheckPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := time.Duration(0), informer.processor.listeners[0].resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := 1*time.Minute, informer.processor.listeners[1].resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := 55*time.Second, informer.processor.listeners[2].resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}

	// listener 4, resync every 5 seconds
	listener4 := newTestListener("listener4", 5*time.Second)
	informer.AddEventHandlerWithResyncPeriod(listener4, listener4.resyncPeriod)
	if e, a := 5*time.Second, informer.resyncCheckPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := time.Duration(0), informer.processor.listeners[0].resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := 1*time.Minute, informer.processor.listeners[1].resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := 55*time.Second, informer.processor.listeners[2].resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
	if e, a := 5*time.Second, informer.processor.listeners[3].resyncPeriod; e != a {
		t.Errorf("expected %d, got %d", e, a)
	}
}

// verify that https://github.com/kubernetes/kubernetes/issues/59822 is fixed
func TestSharedInformerInitializationRace(t *testing.T) {
	source := fcache.NewFakeControllerSource()
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
	source := fcache.NewFakeControllerSource()

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

	for _, listener := range listeners {
		if !listener.ok() {
			t.Errorf("%s: expected %v, got %v", listener.name, listener.expectedItemNames, listener.receivedItemNames)
		}
	}
}

func TestSharedInformerErrorHandling(t *testing.T) {
	source := fcache.NewFakeControllerSource()
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

func TestSharedInformerRemoveHandler(t *testing.T) {
	source := fcache.NewFakeControllerSource()
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

	if err := informer.RemoveEventHandlerByRegistration(handle2); err != nil {
		t.Errorf("removing of first pointer handler failed: %s", err)
	}
	if eventHandlerCount(informer) != 1 {
		t.Errorf("after removing handler informer has %d registered handler(s), instead of 1", eventHandlerCount(informer))
	}

	if err := informer.RemoveEventHandlerByRegistration(handle1); err != nil {
		t.Errorf("removing of second pointer handler failed: %s", err)
	}
	if eventHandlerCount(informer) != 0 {
		t.Errorf("informer still has registered handlers after removing both handlers")
	}
}

func TestSharedInformerRemoveNonComparableHandler(t *testing.T) {
	source := fcache.NewFakeControllerSource()
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second)

	handler1 := ResourceEventHandlerFuncs{}
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
		t.Errorf("informer has %d registered handler(s), instead of 2", eventHandlerCount(informer))
	}

	if err := informer.RemoveEventHandlerByRegistration(handle2); err != nil {
		t.Errorf("removing of pointer handler failed: %s", err)
	}
	if eventHandlerCount(informer) != 1 {
		t.Errorf("after removal informer has %d registered handler(s), instead of 1", eventHandlerCount(informer))
	}

	if err := informer.RemoveEventHandlerByRegistration(handle1); err != nil {
		t.Errorf("removing of non-pointer handler failed: %s", err)
	}
	if eventHandlerCount(informer) != 0 {
		t.Errorf("after removal informer has %d registered handler(s), instead of 0", eventHandlerCount(informer))
	}
}

func TestSharedInformerMultipleRegistration(t *testing.T) {
	source := fcache.NewFakeControllerSource()
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second)

	handler1 := &ResourceEventHandlerFuncs{}
	reg1, err := informer.AddEventHandler(handler1)
	if err != nil {
		t.Errorf("informer did not add handler for the first time: %s", err)
		return
	}

	if !isRegistered(informer,reg1) {
		t.Errorf("handle1 is not active after successful registration")
		return
	}

	reg2, err := informer.AddEventHandler(handler1)
	if err != nil {
		t.Errorf("informer did not add handler for the second: %s", err)
		return
	}

	if !isRegistered(informer,reg2) {
		t.Errorf("handle2 is not active after successful registration")
		return
	}

	if eventHandlerCount(informer) != 2 {
		t.Errorf("informer has %d registered handler(s), instead of 1", eventHandlerCount(informer))
	}

	if err := informer.RemoveEventHandlerByRegistration(reg1); err != nil {
		t.Errorf("removing of duplicate handler registration failed: %s", err)
	}

	if isRegistered(informer,reg1) {
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

	if err := informer.RemoveEventHandlerByRegistration(reg2); err != nil {
		t.Errorf("removing of second handler registration failed: %s", err)
	}

	if isRegistered(informer,reg2) {
		t.Errorf("handle2 is still active after successful remove")
		return
	}

	if eventHandlerCount(informer) != 0 {
		t.Errorf("informer has unexpected number (%d) of handlers after removal of second handler registrations", eventHandlerCount(informer))
	}
}

func TestRemovingRemovedSharedInformer(t *testing.T) {
	source := fcache.NewFakeControllerSource()
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)
	handler := &ResourceEventHandlerFuncs{}
	reg, err := informer.AddEventHandler(handler)

	if err != nil {
		t.Errorf("informer did not add handler for the first time: %s", err)
		return
	}
	if err := informer.RemoveEventHandlerByRegistration(reg); err != nil {
		t.Errorf("removing of handler registration failed: %s", err)
		return
	}
	if isRegistered(informer,reg) {
		t.Errorf("handle is still active after successful remove")
		return
	}
	if err := informer.RemoveEventHandlerByRegistration(reg); err != nil {
		t.Errorf("removing of already removed registration yields unexpected error: %s", err)
	}
	if isRegistered(informer,reg) {
		t.Errorf("handle is still active after second remove")
		return
	}
}

func TestStateSharedInformer(t *testing.T) {
	source := fcache.NewFakeControllerSource()
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
	source := fcache.NewFakeControllerSource()
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod1"}})

	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)
	listener := newTestListener("listener", 0, "pod1")
	stop := make(chan struct{})
	go informer.Run(stop)
	close(stop)
	fmt.Println("sleeping")
	time.Sleep(1 * time.Second)
	if !informer.IsStopped() {
		t.Errorf("informer reports not to be stopped although stop channel closed")
		return
	}

	_, err := informer.AddEventHandlerWithResyncPeriod(listener, listener.resyncPeriod)
	if err == nil {
		t.Errorf("stopped informer did not reject add handler")
		return
	}
	if !strings.HasSuffix(err.Error(), "is not added to shared informer because it has stopped already") {
		t.Errorf("adding handler to a stopped informer yields unexpected error: %s", err)
		return
	}
}

func TestRemoveOnStoppedSharedInformer(t *testing.T) {
	source := fcache.NewFakeControllerSource()
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
	err = informer.RemoveEventHandlerByRegistration(handle)
	if err == nil {
		t.Errorf("informer removes handler on stopped informer")
		return
	}
	if !strings.HasSuffix(err.Error(), "is not removed from shared informer because it has stopped already") {
		t.Errorf("removing handler from stopped informer yield unexpected error: %s", err)
		return
	}
}
