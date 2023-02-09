/*
Copyright 2022 The Kubernetes Authors.

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
	"sync"
)

type status int

const (
	Pending status = iota
	Ready
	Stopped
)

// ready is a three state condition variable that blocks until is Ready if is not Stopped.
// Its initial state is Pending.
type ready struct {
	state status
	c     *sync.Cond
}

func newReady() *ready {
	return &ready{
		c:     sync.NewCond(&sync.RWMutex{}),
		state: Pending,
	}
}

// wait blocks until it is Ready or Stopped, it returns an error if is Stopped.
func (r *ready) wait(ctx context.Context) error {
	r.c.L.Lock()
	defer r.c.L.Unlock()

	if r.state == Pending {
		// We're still waiting for initialization.
		// Ensure that context is honored in case it is cancelled
		// before initialization happens.
		// However avoid spanning additional goroutine in a healthy case.
		// To prevent that this goroutine will be leaked if the context
		// is never cancelled, force cancel it on exit from the function.
		waitCtx, cancel := context.WithCancel(ctx)
		defer cancel()

		go func() {
			<-waitCtx.Done()

			r.c.L.Lock()
			defer r.c.L.Unlock()
			r.c.Broadcast()
		}()
	}
	for r.state == Pending {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}
		r.c.Wait()
	}
	switch r.state {
	case Ready:
		return nil
	case Stopped:
		return fmt.Errorf("apiserver cacher is stopped")
	default:
		return fmt.Errorf("unexpected apiserver cache state: %v", r.state)
	}
}

// check returns true only if it is Ready.
func (r *ready) check() bool {
	rwMutex := r.c.L.(*sync.RWMutex)
	rwMutex.RLock()
	defer rwMutex.RUnlock()
	return r.state == Ready
}

// set the state to Pending (false) or Ready (true), it does not have effect if the state is Stopped.
func (r *ready) set(ok bool) {
	r.c.L.Lock()
	defer r.c.L.Unlock()
	if r.state == Stopped {
		return
	}
	if ok {
		r.state = Ready
	} else {
		r.state = Pending
	}
	r.c.Broadcast()
}

// stop the condition variable and set it as Stopped. This state is irreversible.
func (r *ready) stop() {
	r.c.L.Lock()
	defer r.c.L.Unlock()
	if r.state != Stopped {
		r.state = Stopped
		r.c.Broadcast()
	}
}
