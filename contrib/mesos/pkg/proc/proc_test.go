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

package proc

import (
	"fmt"
	"sync"
	"testing"
	"time"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
)

func failOnError(t *testing.T, errOnce ErrorOnce) {
	err, _ := <-errOnce.Err()
	if err != nil {
		t.Errorf("unexpected action scheduling error: %v", err)
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
	wg.Wait()
	<-p.Done()
}

func TestProc_singleAction(t *testing.T) {
	p := New()
	scheduled := make(chan struct{})
	called := make(chan struct{})

	errOnce := NewErrorOnce(p.Done())
	go func() {
		log.Infof("do'ing deferred action")
		defer close(scheduled)
		errCh := p.Do(func() {
			defer close(called)
			log.Infof("deferred action invoked")
		})
		errOnce.Send(errCh)
	}()

	failOnError(t, errOnce)

	<-scheduled
	<-called
	p.End()
	<-p.Done()
}

func TestProc_singleActionThatPanics(t *testing.T) {
	p := New()
	scheduled := make(chan struct{})
	called := make(chan struct{})

	errOnce := NewErrorOnce(p.Done())
	go func() {
		log.Infof("do'ing deferred action")
		defer close(scheduled)
		errCh := p.Do(func() {
			defer close(called)
			panic("panicing here")
		})
		errOnce.Send(errCh)
	}()

	failOnError(t, errOnce)

	<-scheduled
	<-called
	p.End()
	<-p.Done()
}

func TestProc_singleActionEndsProcess(t *testing.T) {
	p := New()
	called := make(chan struct{})

	errOnce := NewErrorOnce(p.Done())
	go func() {
		log.Infof("do'ing deferred action")
		errCh := p.Do(func() {
			defer close(called)
			log.Infof("deferred action invoked")
			p.End()
		})
		errOnce.Send(errCh)
	}()

	<-called

	failOnError(t, errOnce)

	<-p.Done()
}

func TestProc_multiAction(t *testing.T) {
	p := New()
	const COUNT = 10
	var called sync.WaitGroup
	called.Add(COUNT * 2)

	// test FIFO property
	next := 0
	for i := 0; i < COUNT; i++ {
		log.Infof("do'ing deferred action %d", i)
		idx := i
		errOnce := NewErrorOnce(p.Done())
		errCh := p.Do(func() {
			defer called.Done()
			log.Infof("deferred action invoked")
			if next != idx {
				t.Errorf("expected index %d instead of %d", idx, next)
			}
			next++
		})
		errOnce.Send(errCh)
		go func() {
			defer called.Done()
			failOnError(t, errOnce)
		}()
	}

	called.Wait()
	p.End()
	<-p.Done()
}

func TestProc_goodLifecycle(t *testing.T) {
	p := New()
	p.End()
	<-p.Done()
}

func TestProc_doWithDeadProc(t *testing.T) {
	p := New()
	p.End()
	<-p.Done()

	errUnexpected := fmt.Errorf("unexpected execution of delegated action")
	decorated := DoWith(p, DoerFunc(func(_ Action) <-chan error {
		return ErrorChan(errUnexpected)
	}))

	decorated.Do(func() {})
	<-decorated.Done()
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
			t.Errorf("expected delegated execution")
		}
	})
	if err == nil {
		t.Fatalf("expected !nil error chan")
	}

	<-executed
	<-decorated.OnError(err, func(e error) {
		t.Errorf("unexpected error: %v", err)
	})
	decorated.End()
	<-p.Done()
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
			t.Errorf("expected delegated execution")
		}
	})
	if err == nil {
		t.Fatalf("expected !nil error chan")
	}

	<-executed

	<-decorated2.OnError(err, func(e error) {
		t.Errorf("unexpected error: %v", err)
	})

	decorated2.End()
	<-p.Done()
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
	errCh := decorated2.Do(func() {
		defer close(executed)
		if !delegated {
			t.Errorf("expected delegated execution")
		}
		errOnce.Report(expectedErr)
	})
	if errCh == nil {
		t.Fatalf("expected !nil error chan")
	}
	errOnce.Send(errCh)

	foundError := false
	<-executed

	<-decorated2.OnError(errOnce.Err(), func(e error) {
		if e != expectedErr {
			t.Errorf("unexpected error: %v", e)
		} else {
			foundError = true
		}
	})

	// this has been flaky in the past. recent changes to error handling in
	// processAdapter.Do should have fixed it.
	if !foundError {
		t.Fatalf("expected a propagated error")
	}

	decorated2.End()
	<-p.Done()
}

func runDelegationTest(t *testing.T, p Process, name string, errOnce ErrorOnce) {
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

	<-executed
	t.Logf("runDelegationTest received executed signal at " + time.Now().String())

	wg.Wait()
	t.Logf("runDelegationTest nested decorators finished at " + time.Now().String())
}

func TestProc_doWithNestedX(t *testing.T) {
	t.Logf("starting test case at " + time.Now().String())
	p := New()
	errOnce := NewErrorOnce(p.Done())
	runDelegationTest(t, p, "nested", errOnce)
	<-p.End()

	err, _ := <-errOnce.Err()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	<-p.Done()
}

// intended to be run with -race
func TestProc_doWithNestedXConcurrent(t *testing.T) {
	config := defaultConfig
	config.actionQueueDepth = 4000
	p := newConfigured(config)

	var wg sync.WaitGroup
	const CONC = 20
	wg.Add(CONC)

	for i := 0; i < CONC; i++ {
		i := i
		errOnce := NewErrorOnce(p.Done())
		runtime.After(func() { runDelegationTest(t, p, fmt.Sprintf("nested%d", i), errOnce) }).Then(wg.Done)
		go func() {
			err, _ := <-errOnce.Err()
			if err != nil {
				t.Errorf("delegate %d: unexpected error: %v", i, err)
			}
		}()
	}
	wg.Wait()
	<-p.End()
	<-p.Done()
}

func TestProcWithExceededActionQueueDepth(t *testing.T) {
	config := defaultConfig
	config.actionQueueDepth = 0
	p := newConfigured(config)

	errOnce := NewErrorOnce(p.Done())
	runDelegationTest(t, p, "nested", errOnce)
	<-p.End()

	err, _ := <-errOnce.Err()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	<-p.Done()
}
