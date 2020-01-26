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
	"math"
	"strings"
	"sync"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/clock"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	fcache "k8s.io/client-go/tools/cache/testing"
)

const timeFmt = "15:04:05.999999999"

type testListener struct {
	t                *testing.T
	name             string
	clock            clock.PassiveClock
	resyncPeriod     time.Duration
	checkTiming      bool // whether handle() checks timing
	expectedItemKeys sets.String
	lock             sync.RWMutex
	whenReceived     map[string]time.Time // maps key to when last received
}

func (l *testListener) receivedItemKeys() sets.String {
	l.lock.Lock()
	defer l.lock.Unlock()
	return sets.StringKeySet(l.whenReceived)
}

func (l *testListener) reset() {
	l.lock.Lock()
	defer l.lock.Unlock()
	l.whenReceived = make(map[string]time.Time)
}

func newTestListener(t *testing.T, name string, clock clock.PassiveClock, resyncPeriod time.Duration, expected ...string) *testListener {
	return newTimableListener(t, name, clock, resyncPeriod, false, expected...)
}

func newTimableListener(t *testing.T, name string, clock clock.PassiveClock, resyncPeriod time.Duration, checkTiming bool, expected ...string) *testListener {
	l := &testListener{
		t:                t,
		name:             name,
		clock:            clock,
		resyncPeriod:     resyncPeriod,
		checkTiming:      checkTiming,
		expectedItemKeys: sets.NewString(expected...),
	}
	l.reset()
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
	now := l.clock.Now()
	l.t.Logf("%s %s: handle: %v\n", now.Format(timeFmt), l.name, key)
	l.lock.Lock()
	defer l.lock.Unlock()
	prev, wasThere := l.whenReceived[key]
	l.whenReceived[key] = now
	if l.checkTiming && wasThere && (l.resyncPeriod <= 0 || now.Sub(prev) < l.resyncPeriod) {
		l.t.Errorf("%s %s: prev receipt of %v was at %s but requested period is %s", now.Format(timeFmt), l.name, key, prev.Format(timeFmt), l.resyncPeriod)
	}
}

func (l *testListener) checkRecency(checkPeriod time.Duration) {
	for key := range l.expectedItemKeys {
		var prev, now time.Time
		var good bool
		// time since last receipt should not go above this quantity
		expectedResyncPeriod := checkPeriod * time.Duration(math.Ceil(float64(l.resyncPeriod)/float64(checkPeriod)))
		err := wait.PollImmediate(100*time.Millisecond, 2*time.Second, func() (bool, error) {
			l.lock.Lock()
			defer l.lock.Unlock()
			var got bool
			prev, got = l.whenReceived[key]
			if !got {
				return false, nil
			}
			if l.resyncPeriod <= 0 {
				good = true
				return true, nil
			}
			now = l.clock.Now()
			lat := now.Sub(prev)
			good = lat <= expectedResyncPeriod
			return good, nil
		})
		if err != nil {
			l.t.Errorf("%s %s: for %s, failed: %s", now.Format(timeFmt), l.name, key, err.Error())
		} else if !good {
			l.t.Errorf("%s %s: prev receipt of %v was at %s but requested period is %s", now.Format(timeFmt), l.name, key, prev.Format(timeFmt), l.resyncPeriod)
		}
	}
}

func (l *testListener) ok() bool {
	l.t.Logf("%s %s: polling\n", l.clock.Now().Format(timeFmt), l.name)
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
	l.t.Logf("%s %s: sleeping\n", l.clock.Now().Format(timeFmt), l.name)
	time.Sleep(1 * time.Second)
	l.t.Logf("%s %s: final check\n", l.clock.Now().Format(timeFmt), l.name)
	return l.satisfiedExpectations()
}

func (l *testListener) satisfiedExpectations() bool {
	l.lock.RLock()
	defer l.lock.RUnlock()

	return sets.StringKeySet(l.whenReceived).Equal(l.expectedItemKeys)
}

func TestListenerResyncPeriods(t *testing.T) {
	for _, defaultCheckPeriodMsec := range []int{1000, 1300, 1500, 5000} {
		t.Run(fmt.Sprintf("defaultCheckPeriod=%dms", defaultCheckPeriodMsec), func(t *testing.T) { checkListenerResyncPeriods(t, defaultCheckPeriodMsec) })
	}
}

func checkListenerResyncPeriods(t *testing.T, defaultCheckPeriodMsec int) {
	sfx := fmt.Sprintf("-%dms", defaultCheckPeriodMsec)
	defaultCheckPeriod := time.Duration(defaultCheckPeriodMsec) * time.Millisecond
	name1 := "pod1" + sfx
	name2 := "pod2" + sfx
	// source simulates an apiserver object endpoint.
	source := fcache.NewFakeControllerSource()
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name1}})
	source.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: name2}})

	// create the shared informer and resync every 1s
	informer := NewSharedInformer(source, &v1.Pod{}, defaultCheckPeriod).(*sharedIndexInformer)

	startTime := time.Now()
	clock := clock.NewFakeClock(startTime)
	informer.clock = clock
	informer.processor.clock = clock

	// listener 1, never resync
	listener1 := newTimableListener(t, "listener1"+sfx, clock, 0, true, name1, name2)
	informer.AddEventHandlerWithResyncPeriod(listener1, listener1.resyncPeriod)

	// listener 2, resync every 2s
	listener2 := newTimableListener(t, "listener2"+sfx, clock, 2*time.Second, true, name1, name2)
	informer.AddEventHandlerWithResyncPeriod(listener2, listener2.resyncPeriod)

	// listener 3, resync every 3s
	listener3 := newTimableListener(t, "listener3"+sfx, clock, 3*time.Second, true, name1, name2)
	informer.AddEventHandlerWithResyncPeriod(listener3, listener3.resyncPeriod)

	expectedCheckPeriod := defaultCheckPeriod
	if expectedCheckPeriod > 2*time.Second {
		expectedCheckPeriod = 2 * time.Second
	}

	stop := make(chan struct{})
	defer close(stop)

	go informer.Run(stop)
	// Ensure informer is started before adding the last listener
	WaitForCacheSync(stop, informer.HasSynced)

	// listener 4, resync every 600ms
	listener4 := newTimableListener(t, "listener4"+sfx, clock, 600*time.Millisecond, true, name1, name2)
	informer.AddEventHandlerWithResyncPeriod(listener4, listener4.resyncPeriod)
	listeners := []*testListener{listener1, listener2, listener3, listener4}

	// ensure all listeners got the initial List
	for _, listener := range listeners {
		if !listener.ok() {
			t.Errorf("%s: expected %v, got %v", listener.name, listener.expectedItemKeys, listener.receivedItemKeys())
		}
	}

	// Run the test through two of the longest cycles
	testDuration := expectedCheckPeriod * time.Duration(math.Ceil(float64(3*time.Second)/float64(expectedCheckPeriod))) * 2
	for dt := expectedCheckPeriod; dt <= testDuration; dt += expectedCheckPeriod {
		clock.SetTime(startTime.Add(dt))
		time.Sleep(1 * time.Second) // give unexpected stuff a chance to happen
		for _, listener := range listeners {
			listener.checkRecency(expectedCheckPeriod)
		}
	}
}

// verify that https://github.com/kubernetes/kubernetes/issues/59822 is fixed
func TestSharedInformerInitializationRace(t *testing.T) {
	source := fcache.NewFakeControllerSource()
	informer := NewSharedInformer(source, &v1.Pod{}, 1*time.Second).(*sharedIndexInformer)
	listener := newTestListener(t, "raceListener", clock.RealClock{}, 0)

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

	clock := clock.NewFakeClock(time.Now())
	informer.clock = clock
	informer.processor.clock = clock

	// listener, never resync
	listenerNoResync := newTestListener(t, "listenerNoResync", clock, 0, "pod1", "pod2")
	informer.AddEventHandlerWithResyncPeriod(listenerNoResync, listenerNoResync.resyncPeriod)

	listenerResync := newTestListener(t, "listenerResync", clock, 1*time.Second, "pod1", "pod2")
	informer.AddEventHandlerWithResyncPeriod(listenerResync, listenerResync.resyncPeriod)
	listeners := []*testListener{listenerNoResync, listenerResync}

	stop := make(chan struct{})
	defer close(stop)

	go informer.Run(stop)

	for _, listener := range listeners {
		if !listener.ok() {
			t.Errorf("%s: expected %v, got %v", listener.name, listener.expectedItemKeys, listener.receivedItemKeys())
		}
	}

	// Add pod3, bump pod2 but don't broadcast it, so that the change will be seen only on relist
	source.AddDropWatch(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod3", UID: "pod3", ResourceVersion: "3"}})
	source.ModifyDropWatch(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod2", UID: "pod2", ResourceVersion: "4"}})

	// Ensure that nobody saw any changes
	for _, listener := range listeners {
		if !listener.ok() {
			t.Errorf("%s: expected %v, got %v", listener.name, listener.expectedItemKeys, listener.receivedItemKeys())
		}
	}

	for _, listener := range listeners {
		listener.reset()
	}

	listenerNoResync.expectedItemKeys = sets.NewString("pod2", "pod3")
	listenerResync.expectedItemKeys = sets.NewString("pod1", "pod2", "pod3")

	// This calls shouldSync, which deletes noResync from the list of syncingListeners
	clock.Step(1 * time.Second)

	// Simulate a connection loss (or even just a too-old-watch)
	source.ResetWatch()

	for _, listener := range listeners {
		if !listener.ok() {
			t.Errorf("%s: expected %v, got %v", listener.name, listener.expectedItemKeys, listener.receivedItemKeys())
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
