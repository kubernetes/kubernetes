// Copyright 2014 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// +build !appengine

package internal

import (
	"sync"
	"testing"
	"time"

	netcontext "golang.org/x/net/context"

	basepb "google.golang.org/appengine/internal/base"
)

func TestDialLimit(t *testing.T) {
	// Fill up semaphore with false acquisitions to permit only two TCP connections at a time.
	// We don't replace limitSem because that results in a data race when net/http lazily closes connections.
	nFake := cap(limitSem) - 2
	for i := 0; i < nFake; i++ {
		limitSem <- 1
	}
	defer func() {
		for i := 0; i < nFake; i++ {
			<-limitSem
		}
	}()

	f, c, cleanup := setup() // setup is in api_test.go
	defer cleanup()
	f.hang = make(chan int)

	// If we make two RunSlowly RPCs (which will wait for f.hang to be strobed),
	// then the simple Non200 RPC should hang.
	var wg sync.WaitGroup
	wg.Add(2)
	for i := 0; i < 2; i++ {
		go func() {
			defer wg.Done()
			Call(toContext(c), "errors", "RunSlowly", &basepb.VoidProto{}, &basepb.VoidProto{})
		}()
	}
	time.Sleep(50 * time.Millisecond) // let those two RPCs start

	ctx, _ := netcontext.WithTimeout(toContext(c), 50*time.Millisecond)
	err := Call(ctx, "errors", "Non200", &basepb.VoidProto{}, &basepb.VoidProto{})
	if err != errTimeout {
		t.Errorf("Non200 RPC returned with err %v, want errTimeout", err)
	}

	// Drain the two RunSlowly calls.
	f.hang <- 1
	f.hang <- 1
	wg.Wait()
}
