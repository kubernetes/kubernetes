// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package singleflight

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestDo(t *testing.T) {
	t.Skip("singleflight tests not stable")
	var g Group
	v, err, _ := g.Do("key", func() (interface{}, error) {
		return "bar", nil
	})
	if got, want := fmt.Sprintf("%v (%T)", v, v), "bar (string)"; got != want {
		t.Errorf("Do = %v; want %v", got, want)
	}
	if err != nil {
		t.Errorf("Do error = %v", err)
	}
}

func TestDoErr(t *testing.T) {
	t.Skip("singleflight tests not stable")
	var g Group
	someErr := errors.New("Some error")
	v, err, _ := g.Do("key", func() (interface{}, error) {
		return nil, someErr
	})
	if err != someErr {
		t.Errorf("Do error = %v; want someErr %v", err, someErr)
	}
	if v != nil {
		t.Errorf("unexpected non-nil value %#v", v)
	}
}

func TestDoDupSuppress(t *testing.T) {
	t.Skip("singleflight tests not stable")
	var g Group
	var wg1, wg2 sync.WaitGroup
	c := make(chan string, 1)
	var calls int32
	fn := func() (interface{}, error) {
		if atomic.AddInt32(&calls, 1) == 1 {
			// First invocation.
			wg1.Done()
		}
		v := <-c
		c <- v // pump; make available for any future calls

		time.Sleep(10 * time.Millisecond) // let more goroutines enter Do

		return v, nil
	}

	const n = 10
	wg1.Add(1)
	for i := 0; i < n; i++ {
		wg1.Add(1)
		wg2.Add(1)
		go func() {
			defer wg2.Done()
			wg1.Done()
			v, err, _ := g.Do("key", fn)
			if err != nil {
				t.Errorf("Do error: %v", err)
				return
			}
			if s, _ := v.(string); s != "bar" {
				t.Errorf("Do = %T %v; want %q", v, v, "bar")
			}
		}()
	}
	wg1.Wait()
	// At least one goroutine is in fn now and all of them have at
	// least reached the line before the Do.
	c <- "bar"
	wg2.Wait()
	if got := atomic.LoadInt32(&calls); got <= 0 || got >= n {
		t.Errorf("number of calls = %d; want over 0 and less than %d", got, n)
	}
}

// Test that singleflight behaves correctly after Forget called.
// See https://github.com/golang/go/issues/31420
func TestForget(t *testing.T) {
	t.Skip("singleflight tests not stable")
	var g Group

	var firstStarted, firstFinished sync.WaitGroup

	firstStarted.Add(1)
	firstFinished.Add(1)

	firstCh := make(chan struct{})
	go func() {
		g.Do("key", func() (i interface{}, e error) {
			firstStarted.Done()
			<-firstCh
			firstFinished.Done()
			return
		})
	}()

	firstStarted.Wait()
	g.Forget("key") // from this point no two function using same key should be executed concurrently

	var secondStarted int32
	var secondFinished int32
	var thirdStarted int32

	secondCh := make(chan struct{})
	secondRunning := make(chan struct{})
	go func() {
		g.Do("key", func() (i interface{}, e error) {
			defer func() {
			}()
			atomic.AddInt32(&secondStarted, 1)
			// Notify that we started
			secondCh <- struct{}{}
			// Wait other get above signal
			<-secondRunning
			<-secondCh
			atomic.AddInt32(&secondFinished, 1)
			return 2, nil
		})
	}()

	close(firstCh)
	firstFinished.Wait() // wait for first execution (which should not affect execution after Forget)

	<-secondCh
	// Notify second that we got the signal that it started
	secondRunning <- struct{}{}
	if atomic.LoadInt32(&secondStarted) != 1 {
		t.Fatal("Second execution should be executed due to usage of forget")
	}

	if atomic.LoadInt32(&secondFinished) == 1 {
		t.Fatal("Second execution should be still active")
	}

	close(secondCh)
	result, _, _ := g.Do("key", func() (i interface{}, e error) {
		atomic.AddInt32(&thirdStarted, 1)
		return 3, nil
	})

	if atomic.LoadInt32(&thirdStarted) != 0 {
		t.Error("Third call should not be started because was started during second execution")
	}
	if result != 2 {
		t.Errorf("We should receive result produced by second call, expected: 2, got %d", result)
	}
}
