/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package proc

import (
	"fmt"
	"sync"
	"testing"
	"time"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
)

// logs a testing.Fatalf if the elapsed time d passes before signal chan done is closed
func fatalAfter(t *testing.T, done <-chan struct{}, d time.Duration, msg string, args ...interface{}) {
	select {
	case <-done:
	case <-time.After(d):
		t.Fatalf(msg, args...)
	}
}

func errorAfter(errOnce ErrorOnce, done <-chan struct{}, d time.Duration, msg string, args ...interface{}) {
	select {
	case <-done:
	case <-time.After(d):
		//errOnce.Reportf(msg, args...)
		panic(fmt.Sprintf(msg, args...))
	}
}

// logs a testing.Fatalf if the signal chan closes before the elapsed time d passes
func fatalOn(t *testing.T, done <-chan struct{}, d time.Duration, msg string, args ...interface{}) {
	select {
	case <-done:
		t.Fatalf(msg, args...)
	case <-time.After(d):
	}
}

func TestProc_manyEndings(t *testing.T) {
	p := New()
	const COUNT = 20
	var wg sync.WaitGroup
	wg.Add(COUNT)
	for i := 0; i < COUNT; i++ {
		runtime.On(p.End(), wg.Done)
	}
	fatalAfter(t, runtime.After(wg.Wait), 5*time.Second, "timed out waiting for loose End()s")
	fatalAfter(t, p.Done(), 5*time.Second, "timed out waiting for process death")
}

func TestProc_singleAction(t *testing.T) {
	p := New()
	scheduled := make(chan struct{})
	called := make(chan struct{})

	go func() {
		log.Infof("do'ing deferred action")
		defer close(scheduled)
		err := p.Do(func() {
			defer close(called)
			log.Infof("deferred action invoked")
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}()

	fatalAfter(t, scheduled, 5*time.Second, "timed out waiting for deferred action to be scheduled")
	fatalAfter(t, called, 5*time.Second, "timed out waiting for deferred action to be invoked")

	p.End()

	fatalAfter(t, p.Done(), 5*time.Second, "timed out waiting for process death")
}

func TestProc_singleActionEnd(t *testing.T) {
	p := New()
	scheduled := make(chan struct{})
	called := make(chan struct{})

	go func() {
		log.Infof("do'ing deferred action")
		defer close(scheduled)
		err := p.Do(func() {
			defer close(called)
			log.Infof("deferred action invoked")
			p.End()
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}()

	fatalAfter(t, scheduled, 5*time.Second, "timed out waiting for deferred action to be scheduled")
	fatalAfter(t, called, 5*time.Second, "timed out waiting for deferred action to be invoked")
	fatalAfter(t, p.Done(), 5*time.Second, "timed out waiting for process death")
}

func TestProc_multiAction(t *testing.T) {
	p := New()
	const COUNT = 10
	var called sync.WaitGroup
	called.Add(COUNT)

	// test FIFO property
	next := 0
	for i := 0; i < COUNT; i++ {
		log.Infof("do'ing deferred action %d", i)
		idx := i
		err := p.Do(func() {
			defer called.Done()
			log.Infof("deferred action invoked")
			if next != idx {
				t.Fatalf("expected index %d instead of %d", idx, next)
			}
			next++
		})
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	}

	fatalAfter(t, runtime.After(called.Wait), 2*time.Second, "timed out waiting for deferred actions to be invoked")

	p.End()

	fatalAfter(t, p.Done(), 5*time.Second, "timed out waiting for process death")
}

func TestProc_goodLifecycle(t *testing.T) {
	p := New()
	p.End()
	fatalAfter(t, p.Done(), 5*time.Second, "timed out waiting for process death")
}

func TestProc_doWithDeadProc(t *testing.T) {
	p := New()
	p.End()
	time.Sleep(100 * time.Millisecond)

	errUnexpected := fmt.Errorf("unexpected execution of delegated action")
	decorated := DoWith(p, DoerFunc(func(_ Action) <-chan error {
		return ErrorChan(errUnexpected)
	}))

	decorated.Do(func() {})
	fatalAfter(t, decorated.Done(), 5*time.Second, "timed out waiting for process death")
}

func TestProc_doWith(t *testing.T) {
	p := New()

	delegated := false
	decorated := DoWith(p, DoerFunc(func(a Action) <-chan error {
		delegated = true
		a()
		return nil
	}))

	executed := make(chan struct{})
	err := decorated.Do(func() {
		defer close(executed)
		if !delegated {
			t.Fatalf("expected delegated execution")
		}
	})
	if err == nil {
		t.Fatalf("expected !nil error chan")
	}

	fatalAfter(t, executed, 5*time.Second, "timed out waiting deferred execution")
	fatalAfter(t, decorated.OnError(err, func(e error) {
		t.Fatalf("unexpected error: %v", err)
	}), 1*time.Second, "timed out waiting for doer result")

	decorated.End()
	fatalAfter(t, p.Done(), 5*time.Second, "timed out waiting for process death")
}

func TestProc_doWithNestedTwice(t *testing.T) {
	p := New()

	delegated := false
	decorated := DoWith(p, DoerFunc(func(a Action) <-chan error {
		a()
		return nil
	}))

	decorated2 := DoWith(decorated, DoerFunc(func(a Action) <-chan error {
		delegated = true
		a()
		return nil
	}))

	executed := make(chan struct{})
	err := decorated2.Do(func() {
		defer close(executed)
		if !delegated {
			t.Fatalf("expected delegated execution")
		}
	})
	if err == nil {
		t.Fatalf("expected !nil error chan")
	}

	fatalAfter(t, executed, 5*time.Second, "timed out waiting deferred execution")
	fatalAfter(t, decorated2.OnError(err, func(e error) {
		t.Fatalf("unexpected error: %v", err)
	}), 1*time.Second, "timed out waiting for doer result")

	decorated2.End()
	fatalAfter(t, p.Done(), 5*time.Second, "timed out waiting for process death")
}

func TestProc_doWithNestedErrorPropagation(t *testing.T) {
	p := New()

	delegated := false
	decorated := DoWith(p, DoerFunc(func(a Action) <-chan error {
		a()
		return nil
	}))

	expectedErr := fmt.Errorf("expecting this")
	errOnce := NewErrorOnce(p.Done())
	decorated2 := DoWith(decorated, DoerFunc(func(a Action) <-chan error {
		delegated = true
		a()
		errOnce.Reportf("unexpected error in decorator2")
		return ErrorChanf("another unexpected error in decorator2")
	}))

	executed := make(chan struct{})
	err := decorated2.Do(func() {
		defer close(executed)
		if !delegated {
			t.Fatalf("expected delegated execution")
		}
		errOnce.Report(expectedErr)
	})
	if err == nil {
		t.Fatalf("expected !nil error chan")
	}
	errOnce.Send(err)

	foundError := false
	fatalAfter(t, executed, 1*time.Second, "timed out waiting deferred execution")
	fatalAfter(t, decorated2.OnError(errOnce.Err(), func(e error) {
		if e != expectedErr {
			t.Fatalf("unexpected error: %v", err)
		} else {
			foundError = true
		}
	}), 1*time.Second, "timed out waiting for doer result")

	if !foundError {
		t.Fatalf("expected a propagated error")
	}

	decorated2.End()
	fatalAfter(t, p.Done(), 5*time.Second, "timed out waiting for process death")
}

func runDelegationTest(t *testing.T, p Process, name string, errOnce ErrorOnce, timeout time.Duration) {
	t.Logf("starting test case " + name + " at " + time.Now().String())
	defer func() {
		t.Logf("runDelegationTest finished at " + time.Now().String())
	}()
	var decorated Process
	decorated = p

	const DEPTH = 100
	var wg sync.WaitGroup
	wg.Add(DEPTH)
	y := 0

	for x := 1; x <= DEPTH; x++ {
		x := x
		nextp := DoWith(decorated, DoerFunc(func(a Action) <-chan error {
			if x == 1 {
				t.Logf("delegate chain invoked for " + name + " at " + time.Now().String())
			}
			y++
			if y != x {
				return ErrorChanf("out of order delegated execution for " + name)
			}
			defer wg.Done()
			a()
			return nil
		}))
		decorated = nextp
	}

	executed := make(chan struct{})
	errCh := decorated.Do(func() {
		defer close(executed)
		if y != DEPTH {
			errOnce.Reportf("expected delegated execution for " + name)
		}
		t.Logf("executing deferred action: " + name + " at " + time.Now().String())
		errOnce.Send(nil) // we completed without error, let the listener know
	})
	if errCh == nil {
		t.Fatalf("expected !nil error chan")
	}

	// forward any scheduling errors to the listener; NOTHING else should attempt to read
	// from errCh after this point
	errOnce.Send(errCh)

	errorAfter(errOnce, executed, timeout, "timed out waiting deferred execution of "+name)
	t.Logf("runDelegationTest received executed signal at " + time.Now().String())
}

func TestProc_doWithNestedX(t *testing.T) {
	t.Logf("starting test case at " + time.Now().String())
	p := New()
	errOnce := NewErrorOnce(p.Done())
	timeout := 5 * time.Second
	runDelegationTest(t, p, "nested", errOnce, timeout)
	<-p.End()
	select {
	case err := <-errOnce.Err():
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	case <-time.After(2 * timeout):
		t.Fatalf("timed out waiting for doer result")
	}
	fatalAfter(t, p.Done(), 5*time.Second, "timed out waiting for process death")
}

// TODO(jdef): find a way to test this without killing CI builds.
// intended to be run with -race
func TestProc_doWithNestedXConcurrent(t *testing.T) {
	t.Skip("disabled for causing CI timeouts.")
	config := defaultConfig
	config.actionQueueDepth = 0
	p := newConfigured(config)

	var wg sync.WaitGroup
	const CONC = 20
	wg.Add(CONC)

	// this test spins up TONS of goroutines that can take a little while to execute on a busy
	// CI server. drawing the line at 10s because I've never seen it take anywhere near that long.
	timeout := 10 * time.Second

	for i := 0; i < CONC; i++ {
		i := i
		errOnce := NewErrorOnce(p.Done())
		runtime.After(func() { runDelegationTest(t, p, fmt.Sprintf("nested%d", i), errOnce, timeout) }).Then(wg.Done)
		go func() {
			select {
			case err := <-errOnce.Err():
				if err != nil {
					t.Fatalf("delegate %d: unexpected error: %v", i, err)
				}
			case <-time.After(2 * timeout):
				t.Fatalf("delegate %d: timed out waiting for doer result", i)
			}
		}()
	}
	ch := runtime.After(wg.Wait)
	fatalAfter(t, ch, 2*timeout, "timed out waiting for concurrent delegates")

	<-p.End()
	fatalAfter(t, p.Done(), 5*time.Second, "timed out waiting for process death")
}

func TestProcWithExceededActionQueueDepth(t *testing.T) {
	config := defaultConfig
	config.actionQueueDepth = 0
	p := newConfigured(config)

	errOnce := NewErrorOnce(p.Done())
	timeout := 5 * time.Second
	runDelegationTest(t, p, "nested", errOnce, timeout)
	<-p.End()
	select {
	case err := <-errOnce.Err():
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
	case <-time.After(2 * timeout):
		t.Fatalf("timed out waiting for doer result")
	}
	fatalAfter(t, p.Done(), 5*time.Second, "timed out waiting for process death")
}
