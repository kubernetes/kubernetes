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

package tasks

import (
	"bytes"
	"errors"
	"fmt"
	"io"
	"sync"
	"syscall"
	"testing"
	"time"

	log "github.com/golang/glog"
	"github.com/stretchr/testify/assert"
)

type badWriteCloser struct {
	err error
}

func (b *badWriteCloser) Write(_ []byte) (int, error) { return 0, b.err }
func (b *badWriteCloser) Close() error                { return b.err }

type discardCloser int

func (d discardCloser) Write(b []byte) (int, error) { return len(b), nil }
func (d discardCloser) Close() error                { return nil }

var devNull = func() io.WriteCloser { return discardCloser(0) }

type fakeExitError uint32

func (f fakeExitError) Sys() interface{} { return syscall.WaitStatus(f << 8) }
func (f fakeExitError) Error() string    { return fmt.Sprintf("fake-exit-error: %d", f) }

type fakeProcess struct {
	done chan struct{}
	pid  int
	err  error
}

func (f *fakeProcess) Wait() error {
	<-f.done
	return f.err
}
func (f *fakeProcess) Kill(_ bool) (int, error) {
	close(f.done)
	return f.pid, f.err
}
func (f *fakeProcess) exit(code int) {
	f.err = fakeExitError(code)
	close(f.done)
}

func newFakeProcess() *fakeProcess {
	return &fakeProcess{
		done: make(chan struct{}),
	}
}

func TestBadLogger(t *testing.T) {
	err := errors.New("qux")
	fp := newFakeProcess()
	tt := New("foo", "bar", nil, func() io.WriteCloser {
		defer func() {
			fp.pid = 123   // sanity check
			fp.Kill(false) // this causes Wait() to return
		}()
		return &badWriteCloser{err}
	})

	tt.RestartDelay = 0 // don't slow the test down for no good reason

	finishCalled := make(chan struct{})
	tt.Finished = func(ok bool) bool {
		log.Infof("tt.Finished: ok %t", ok)
		if ok {
			close(finishCalled)
		}
		return false // never respawn, this causes t.done to close
	}

	// abuse eventsImpl: we're not going to listen on the task completion or event chans,
	// and we don't want to block the state machine, so discard all task events as they happen
	ei := newEventsImpl(tt.completedCh, tt.done)
	ei.Close()

	go tt.run(func(_ *Task) taskStateFn {
		log.Infof("tt initialized")
		tt.initLogging(bytes.NewBuffer(([]byte)("unlogged bytes")))
		tt.cmd = fp
		return taskRunning
	})

	// if the logger fails the task will be killed
	// badWriteLogger generates an error immediately and results in a task kill
	<-finishCalled
	<-tt.done

	// this should never data race since the state machine is dead at this point
	if fp.pid != 123 {
		t.Fatalf("incorrect pid, expected 123 not %d", fp.pid)
	}

	// TODO(jdef) would be nice to check for a specific error that indicates the logger died
}

func TestMergeOutput(t *testing.T) {
	var tasksStarted, tasksDone sync.WaitGroup
	tasksDone.Add(2)
	tasksStarted.Add(2)

	t1 := New("foo", "", nil, devNull)
	t1exited := make(chan struct{})
	t1.RestartDelay = 0 // don't slow the test down for no good reason
	t1.Finished = func(ok bool) bool {
		// we expect each of these cases to happen exactly once
		if !ok {
			tasksDone.Done()
		} else {
			close(t1exited)
		}
		return ok
	}
	go t1.run(func(t *Task) taskStateFn {
		defer tasksStarted.Done()
		t.initLogging(bytes.NewBuffer([]byte{}))
		t.cmd = newFakeProcess()
		return taskRunning
	})

	t2 := New("bar", "", nil, devNull)
	t2exited := make(chan struct{})
	t2.RestartDelay = 0 // don't slow the test down for no good reason
	t2.Finished = func(ok bool) bool {
		// we expect each of these cases to happen exactly once
		if !ok {
			tasksDone.Done()
		} else {
			close(t2exited)
		}
		return ok
	}
	go t2.run(func(t *Task) taskStateFn {
		defer tasksStarted.Done()
		t.initLogging(bytes.NewBuffer([]byte{}))
		t.cmd = newFakeProcess()
		return taskRunning
	})

	shouldQuit := make(chan struct{})
	te := MergeOutput([]*Task{t1, t2}, shouldQuit)

	tasksStarted.Wait()
	tasksStarted.Add(2) // recycle the barrier

	// kill each task once, let it restart; make sure that we get the completion status?
	t1.cmd.(*fakeProcess).exit(1)
	t2.cmd.(*fakeProcess).exit(2)

	codes := map[int]struct{}{}
	for i := 0; i < 2; i++ {
		switch tc := <-te.Completion(); tc.code {
		case 1, 2:
			codes[tc.code] = struct{}{}
		default:
			if tc.err != nil {
				t.Errorf("unexpected task completion error: %v", tc.err)
			} else {
				t.Errorf("unexpected task completion code: %d", tc.code)
			}
		}
	}

	te.Close() // we're not going to read any other completion or error events

	if len(codes) != 2 {
		t.Fatalf("expected each task process to exit once")
	}

	// each task invokes Finished() once
	<-t1exited
	<-t2exited

	log.Infoln("each task process has completed one round")
	tasksStarted.Wait() // tasks will auto-restart their exited procs

	// assert that the tasks are not dead; TODO(jdef) not sure that these checks are useful
	select {
	case <-t1.done:
		t.Fatalf("t1 is unexpectedly dead")
	default:
	}
	select {
	case <-t2.done:
		t.Fatalf("t2 is unexpectedly dead")
	default:
	}

	log.Infoln("firing quit signal")
	close(shouldQuit) // fire shouldQuit, and everything should terminate gracefully

	log.Infoln("waiting for tasks to die")
	tasksDone.Wait() // our tasks should die

	log.Infoln("waiting for merge to complete")
	<-te.Done() // wait for the merge to complete
}

type fakeTimer struct {
	ch chan time.Time
}

func (t *fakeTimer) set(d time.Duration)     {}
func (t *fakeTimer) discard()                {}
func (t *fakeTimer) await() <-chan time.Time { return t.ch }
func (t *fakeTimer) expire()                 { t.ch = make(chan time.Time); close(t.ch) }
func (t *fakeTimer) reset()                  { t.ch = nil }

func TestAfterDeath(t *testing.T) {
	// test kill escalation since that's not covered by other unit tests
	t1 := New("foo", "", nil, devNull)
	kills := 0
	waitCh := make(chan *Completion, 1)
	timer := &fakeTimer{}
	timer.expire()
	t1.killFunc = func(force bool) (int, error) {
		// > 0 is intentional, multiple calls to close() should panic
		if kills > 0 {
			assert.True(t, force)
			timer.reset() // don't want to race w/ waitCh
			waitCh <- &Completion{name: t1.name, code: 123}
			close(waitCh)
		} else {
			assert.False(t, force)
		}
		kills++
		return 0, nil
	}
	wr := t1.awaitDeath(timer, 0, waitCh)
	assert.Equal(t, "foo", wr.name)
	assert.Equal(t, 123, wr.code)
	assert.NoError(t, wr.err)

	// test tie between shouldQuit and waitCh
	waitCh = make(chan *Completion, 1)
	waitCh <- &Completion{name: t1.name, code: 456}
	close(waitCh)
	t1.killFunc = func(force bool) (int, error) {
		t.Fatalf("should not attempt to kill a task that has already reported completion")
		return 0, nil
	}

	timer.reset() // don't race w/ waitCh
	wr = t1.awaitDeath(timer, 0, waitCh)
	assert.Equal(t, 456, wr.code)
	assert.NoError(t, wr.err)

	// test delayed killFunc failure
	kills = 0
	killFailed := errors.New("for some reason kill failed")
	t1.killFunc = func(force bool) (int, error) {
		// > 0 is intentional, multiple calls to close() should panic
		if kills > 0 {
			assert.True(t, force)
			return -1, killFailed
		} else {
			assert.False(t, force)
		}
		kills++
		return 0, nil
	}
	timer.expire()
	wr = t1.awaitDeath(timer, 0, nil)
	assert.Equal(t, "foo", wr.name)
	assert.Error(t, wr.err)

	// test initial killFunc failure
	kills = 0
	t1.killFunc = func(force bool) (int, error) {
		// > 0 is intentional, multiple calls to close() should panic
		if kills > 0 {
			assert.True(t, force)
			t.Fatalf("killFunc should only be invoked once, not again after is has already failed")
		} else {
			assert.False(t, force)
		}
		kills++
		return 0, killFailed
	}
	timer.expire()
	wr = t1.awaitDeath(timer, 0, nil)
	assert.Equal(t, "foo", wr.name)
	assert.Error(t, wr.err)
}
