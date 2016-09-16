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
	"sync"

	"k8s.io/kubernetes/pkg/util/runtime"
)

const (
	// how many actions we can store in the backlog
	defaultActionQueueDepth = 1024
)

type Config struct {
	// determines the size of the deferred action backlog
	actionQueueDepth uint32
}

var (
	// default process configuration, used in the creation of all new processes
	defaultConfig = Config{
		actionQueueDepth: defaultActionQueueDepth,
	}
)

type scheduledAction struct {
	action Action
	errCh  chan error
}

type processState struct {
	actions    chan *scheduledAction           // scheduled action backlog
	running    chan struct{}                   // closes upon start of action backlog processing
	terminated chan struct{}                   // closes upon termination of run()
	doer       Doer                            // delegate that schedules actions
	guardDoer  sync.RWMutex                    // protect doer
	end        chan struct{}                   // closes upon invocation of End()
	closeEnd   sync.Once                       // guard: only close end chan once
	nextAction func() (*scheduledAction, bool) // return false if actions queue is closed
}

func New() Process {
	return newConfigured(defaultConfig)
}

func newConfigured(c Config) Process {
	ps := &processState{
		actions:    make(chan *scheduledAction, c.actionQueueDepth),
		running:    make(chan struct{}),
		terminated: make(chan struct{}),
		end:        make(chan struct{}),
	}
	ps.doer = DoerFunc(ps.defaultDoer)
	go ps.run()
	return ps
}

type stateFn func(*processState, *scheduledAction) stateFn

func stateRun(ps *processState, a *scheduledAction) stateFn {
	// it's only possible to ever receive this once because we transition
	// to state "shutdown", permanently
	if a == nil {
		ps.shutdown()
		return stateShutdown
	}

	close(a.errCh) // signal that action was scheduled
	func() {
		// we don't trust clients of this package
		defer func() {
			if x := recover(); x != nil {
				// HandleCrash has already logged this, so
				// nothing to do.
			}
		}()
		defer runtime.HandleCrash()
		a.action()
	}()
	return stateRun
}

func (ps *processState) shutdown() {
	// all future attemps to schedule actions must fail immediately
	ps.guardDoer.Lock()
	ps.doer = DoerFunc(func(_ Action) <-chan error {
		return ErrorChan(errProcessTerminated)
	})
	ps.guardDoer.Unlock()

	// no more actions may be scheduled
	close(ps.actions)

	// no need to check ps.end anymore
	ps.nextAction = func() (a *scheduledAction, ok bool) {
		a, ok = <-ps.actions
		return
	}
}

// stateShutdown doesn't run any actions because the process is shutting down.
// instead it clears the action backlog. newly scheduled actions are rejected.
func stateShutdown(ps *processState, a *scheduledAction) stateFn {
	if a != nil {
		a.errCh <- errProcessTerminated
	}
	return stateShutdown
}

func (ps *processState) run() {
	defer close(ps.terminated)
	close(ps.running)

	// main state machine loop: process actions as they come,
	// updating the state func along the way.
	f := stateRun
	ps.nextAction = func() (a *scheduledAction, ok bool) {
		// if we successfully read from ps.end,  we don't know if the
		// actions queue is closed. assume it's not: the state machine
		// shouldn't terminate yet.
		// also, give preference to ps.end: we want to avoid processing
		// actions if both ps.actions and ps.end are ready
		select {
		case <-ps.end:
			ok = true
		default:
			select {
			case <-ps.end:
				ok = true
			case a, ok = <-ps.actions:
			}
		}
		return
	}
	for {
		a, ok := ps.nextAction()
		if !ok {
			return
		}
		g := f(ps, a)
		if g == nil {
			panic("state machine stateFn is not allowed to be nil")
		}
		f = g
	}
}

func (ps *processState) Running() <-chan struct{} {
	return ps.running
}

func (ps *processState) Done() <-chan struct{} {
	return ps.terminated
}

func (ps *processState) End() <-chan struct{} {
	ps.closeEnd.Do(func() {
		close(ps.end)
	})
	return ps.terminated
}

func (ps *processState) Do(a Action) <-chan error {
	ps.guardDoer.RLock()
	defer ps.guardDoer.RUnlock()
	return ps.doer.Do(a)
}

func (ps *processState) defaultDoer(a Action) <-chan error {
	ch := make(chan error, 1)
	ps.actions <- &scheduledAction{
		action: a,
		errCh:  ch,
	}
	return ch
}

func (ps *processState) OnError(ch <-chan error, f func(error)) <-chan struct{} {
	return OnError(ch, f, ps.terminated)
}
