// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package testutil

import (
	"errors"
	"fmt"
	"sync"
	"time"
)

type Action struct {
	Name   string
	Params []interface{}
}

type Recorder interface {
	// Record publishes an Action (e.g., function call) which will
	// be reflected by Wait() or Chan()
	Record(a Action)
	// Wait waits until at least n Actions are available or returns with error
	Wait(n int) ([]Action, error)
	// Action returns immediately available Actions
	Action() []Action
	// Chan returns the channel for actions published by Record
	Chan() <-chan Action
}

// RecorderBuffered appends all Actions to a slice
type RecorderBuffered struct {
	sync.Mutex
	actions []Action
}

func (r *RecorderBuffered) Record(a Action) {
	r.Lock()
	r.actions = append(r.actions, a)
	r.Unlock()
}

func (r *RecorderBuffered) Action() []Action {
	r.Lock()
	cpy := make([]Action, len(r.actions))
	copy(cpy, r.actions)
	r.Unlock()
	return cpy
}

func (r *RecorderBuffered) Wait(n int) (acts []Action, err error) {
	// legacy racey behavior
	WaitSchedule()
	acts = r.Action()
	if len(acts) < n {
		err = newLenErr(n, len(acts))
	}
	return acts, err
}

func (r *RecorderBuffered) Chan() <-chan Action {
	ch := make(chan Action)
	go func() {
		acts := r.Action()
		for i := range acts {
			ch <- acts[i]
		}
		close(ch)
	}()
	return ch
}

// RecorderStream writes all Actions to an unbuffered channel
type recorderStream struct {
	ch          chan Action
	waitTimeout time.Duration
}

func NewRecorderStream() Recorder {
	return NewRecorderStreamWithWaitTimout(time.Duration(5 * time.Second))
}

func NewRecorderStreamWithWaitTimout(waitTimeout time.Duration) Recorder {
	return &recorderStream{ch: make(chan Action), waitTimeout: waitTimeout}
}

func (r *recorderStream) Record(a Action) {
	r.ch <- a
}

func (r *recorderStream) Action() (acts []Action) {
	for {
		select {
		case act := <-r.ch:
			acts = append(acts, act)
		default:
			return acts
		}
	}
}

func (r *recorderStream) Chan() <-chan Action {
	return r.ch
}

func (r *recorderStream) Wait(n int) ([]Action, error) {
	acts := make([]Action, n)
	timeoutC := time.After(r.waitTimeout)
	for i := 0; i < n; i++ {
		select {
		case acts[i] = <-r.ch:
		case <-timeoutC:
			acts = acts[:i]
			return acts, newLenErr(n, i)
		}
	}
	// extra wait to catch any Action spew
	select {
	case act := <-r.ch:
		acts = append(acts, act)
	case <-time.After(10 * time.Millisecond):
	}
	return acts, nil
}

func newLenErr(expected int, actual int) error {
	s := fmt.Sprintf("len(actions) = %d, expected >= %d", actual, expected)
	return errors.New(s)
}
