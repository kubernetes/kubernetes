/*
Copyright 2019 The Kubernetes Authors.

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

package controller

import (
	"fmt"
	"sync"
)

// Limiter is used by an task barrier to limit the rate that it spawns tasks.
// Calls to limiter are synchonized by the taskBarrier.
type Limiter interface {
	// reserve attempts reserve a new slot for the task. It returns true if the
	// slot was successfully reserved.
	reserve() bool

	// release is called to inform to the limiter that a single task has
	// completed. The limiter is expected to call Signal or Broadcast on the
	// condition to wake goroutines that may be waiting for a reservation slot.
	release(*sync.Cond)
}

// TaskBarrier is an incremental barrier that makes it easy to create multiple
// async operations and wait for them as a group. It allows creation to be rate
// limited by a Limiter. It reports errors to Policy, which aggregates the
// errors and decides whether the barrier should continue or bail out early. It
// takes care of error prone requirements, such as remembering to count the
// start tasks with a waitgroup and synchronizing completion reporting out of
// the task loop.
type TaskBarrier struct {
	// cond guards all state including the policy.
	cond *sync.Cond
	// Policy is used to propagate errors from tasks started by a barrier, and
	// also to signal to the barrier that it should terminate early (i.e. refuse
	// to create more tasks). It receives a (possibly nil) error on each
	// completion of a task. It returns a bool indicating whether the barrier
	// should continue creating tasks.
	//
	// The barrier will synchronize calls to Policy and make no more calls once
	// Wait returns. If state modified by Policy needs to be accessed outside of
	// Policy but before Wait returns, it's Policy's responsibility to
	// synchronize access to this state.
	Policy func(error) bool
	// Limiter is the limiter that the task barrier will use to limit the
	// creation of tasks.
	Limiter Limiter
	// the barrier becomes done when:
	// * a call to policy returns false, indicating that the barrier should not
	//   continue creating tasks.
	// * on the first return of Wait
	done bool
	// each active task places a hold on the task barrier. Wait will not return
	// until all holds are removed.
	holds int
}

// NewTaskBarrier creates a task barrier with a limiter that never blocks and
// ignores all errors. The Policy and Limiter can be replace before the task
// barrier is used.
func NewTaskBarrier() *TaskBarrier {
	return &TaskBarrier{
		cond: sync.NewCond(&sync.Mutex{}),
		// ignore all errors by default
		Policy:  func(error) bool { return true },
		Limiter: NewNoLimiter(),
	}
}

// Go requests that the barrier start a new task. The barrier will ignore
// requests to Go if the barrier is already done (e.g. if a call to Wait has
// returned).
func (tb *TaskBarrier) Go(task func() error) {
	tb.cond.L.Lock()
	defer tb.cond.L.Unlock()

	// Wait for the limiter to give the goahead or the task barrier to
	// transition to done.
	for !tb.Limiter.reserve() && !tb.done {
		tb.cond.Wait()
	}

	// bail if the barrier is done
	if tb.done {
		return
	}

	// add a hold and start the task.
	tb.holds++

	go func() {
		// Run the task, then under lock:
		// * release the task's hold on the barrier
		// * release the task's reservation back to the limiter
		// * report the task's return to Policy. If the Policy indicates
		//   that we shouldn't continue, set the barrier to done.
		// * if holds drop to zero or the barrier is done, wake everyone up.
		err := task()

		tb.cond.L.Lock()
		defer tb.cond.L.Unlock()

		tb.holds--
		tb.Limiter.release(tb.cond)

		if ok := tb.Policy(err); !ok {
			tb.done = true
		}
		if tb.holds == 0 || tb.done {
			tb.cond.Broadcast()
		}
	}()
}

// Wait waits for all holds tasks to complete. Once Wait has returned, the
// barrier is done and all calls to Go() will be ignored.
func (tb *TaskBarrier) Wait() {
	tb.cond.L.Lock()
	defer tb.cond.L.Unlock()

	for tb.holds > 0 {
		tb.cond.Wait()
	}
	tb.done = true
}

// NewSlowStarter returns a Limiter that rate limits a barrier to allow loops
// with parallelism to start slowly and speed up. It uses a token bucket that
// is refilled with two tokens for every token returned allowing a loop to
// incrementally double.
//
// For example, this can be used to gracefully handle attempts to start large
// numbers of pods that would likely all fail with the same error. If a
// DaemonSet in namespace with no quota attempts to create a large number of
// pods, TaskBarrier will prevent the DaemonSet controller from spamming the
// API service with the pod create requests after one of its pods fails.
// Conveniently, this also prevents the event spam that those failures would
// generate.
func NewSlowStarter() Limiter {
	return &slowStartLimiter{
		tokens: 1,
	}
}

type slowStartLimiter struct {
	tokens int
}

// calls to reserve must be guarded by the task barrier's condition.
func (s *slowStartLimiter) reserve() bool {
	// Negative available tokens will happen if more tokens are returned to
	// TaskBarrier than it hands out. This can't happen through misuse, so we
	// made a mistake somewhere around here. We can either:
	//
	// * Panic proactively
	// * Continue to wait for the tokens reach positive again, potentially
	//   deadlocking the consumer
	// * Protect against this condition with a sync.Once that escapes the stack
	//   and is captured by a release callback. This doesn't prevent bugs.
	if s.tokens < 0 {
		panic(fmt.Errorf("slowStartLimiter reached negative available tokens: %v", s.tokens))
	}

	if s.tokens == 0 {
		return false
	}

	s.tokens--
	return true
}

// calls to release must be guarded by the task barrier's condition.
func (s *slowStartLimiter) release(cond *sync.Cond) {
	// slow starter doubles its rate over usage. for every slot released, make
	// two slots available, and signal two waiters.
	s.tokens += 2
	cond.Signal()
	cond.Signal()
}

// NewNoLimiter returns a barrier that never blocks a limiter from starting a
// new task.
func NewNoLimiter() Limiter {
	return noLimiter{}
}

type noLimiter struct{}

func (noLimiter) reserve() bool {
	return true
}

func (noLimiter) release(cond *sync.Cond) {
	// no need to signal because nothing should ever be blocked on us.
}
