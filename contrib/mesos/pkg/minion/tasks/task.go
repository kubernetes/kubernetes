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

package tasks

import (
	"fmt"
	"io"
	"os/exec"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
)

const defaultTaskRestartDelay = 5 * time.Second

// Completion represents the termination of a Task process. Each process execution should
// yield (barring drops because of an abort signal) exactly one Completion.
type Completion struct {
	name string // name of the task
	code int    // exit code that the task process completed with
	err  error  // process management errors are reported here
}

// systemProcess is a useful abstraction for testing
type systemProcess interface {
	// Wait works like exec.Cmd.Wait()
	Wait() error

	// Kill returns the pid of the process that was killed
	Kill() (int, error)
}

type cmdProcess struct {
	delegate *exec.Cmd
}

func (cp *cmdProcess) Wait() error {
	return cp.delegate.Wait()
}

func (cp *cmdProcess) Kill() (int, error) {
	// kill the entire process group, not just the one process
	pid := cp.delegate.Process.Pid
	processGroup := 0 - pid

	// we send a SIGTERM here for a graceful stop. users of this package should
	// wait for tasks to complete normally. as a fallback/safeguard, child procs
	// are spawned in notStartedTask to receive a SIGKILL when this process dies.
	rc := syscall.Kill(processGroup, syscall.SIGTERM)
	return pid, rc
}

// task is a specification for running a system process; it provides hooks for customizing
// logging and restart handling as well as provides event channels for communicating process
// termination and errors related to process management.
type Task struct {
	Finished     func(restarting bool) bool // callback invoked when a task process has completed; if `restarting` then it will be restarted if it returns true
	RestartDelay time.Duration              // interval between repeated task restarts

	name         string                // required: unique name for this task
	bin          string                // required: path to executable
	args         []string              // optional: process arguments
	env          []string              // optional: process environment override
	createLogger func() io.WriteCloser // factory func that builds a log writer
	cmd          systemProcess         // process that we started
	completedCh  chan *Completion      // reports exit codes encountered when task processes exit, or errors during process management
	shouldQuit   chan struct{}         // shouldQuit is closed to indicate that the task should stop its running process, if any
	done         chan struct{}         // done closes when all processes related to the task have terminated
	initialState taskStateFn           // prepare and start a new live process, defaults to notStartedTask; should be set by run()
	runLatch     int32                 // guard against multiple Task.run calls
}

// New builds a newly initialized task object but does not start any processes for it. callers
// are expected to invoke task.run(...) on their own.
func New(name, bin string, args, env []string, cl func() io.WriteCloser) *Task {
	return &Task{
		name:         name,
		bin:          bin,
		args:         args,
		env:          env,
		createLogger: cl,
		completedCh:  make(chan *Completion),
		shouldQuit:   make(chan struct{}),
		done:         make(chan struct{}),
		RestartDelay: defaultTaskRestartDelay,
		Finished:     func(restarting bool) bool { return restarting },
	}
}

// Start spawns a goroutine to execute the Task. Panics if invoked more than once.
func (t *Task) Start() {
	go t.run(notStartedTask)
}

// run executes the state machine responsible for starting, monitoring, and possibly restarting
// a system process for the task. The initialState func is the entry point of the state machine.
// Upon returning the done and completedCh chans are all closed.
func (t *Task) run(initialState taskStateFn) {
	if !atomic.CompareAndSwapInt32(&t.runLatch, 0, 1) {
		panic("Task.run() may only be invoked once")
	}
	t.initialState = initialState

	defer close(t.done)
	defer close(t.completedCh)

	state := initialState
	for state != nil {
		next := state(t)
		state = next
	}
}

func (t *Task) tryComplete(tc *Completion) {
	select {
	case <-t.shouldQuit:
		// best effort
		select {
		case t.completedCh <- tc:
		default:
		}
	case t.completedCh <- tc:
	}
}

// tryError is a convenience func that invokes tryComplete with a completion error
func (t *Task) tryError(err error) {
	t.tryComplete(&Completion{err: err})
}

type taskStateFn func(*Task) taskStateFn

func taskShouldRestart(t *Task) taskStateFn {
	// make our best effort to stop here if signalled (shouldQuit). not doing so here
	// could add cost later (a process might be launched).

	// sleep for a bit; then return t.initialState
	tm := time.NewTimer(t.RestartDelay)
	defer tm.Stop()
	select {
	case <-tm.C:
		select {
		case <-t.shouldQuit:
		default:
			if t.Finished(true) {
				select {
				case <-t.shouldQuit:
					// the world has changed, die
					return nil
				default:
				}
				return t.initialState
			}
			// finish call decided not to respawn, so die
			return nil
		}
	case <-t.shouldQuit:
	}

	// we're quitting, tell the Finished callback and then die
	t.Finished(false)
	return nil
}

func (t *Task) initLogging(r io.Reader) {
	writer := t.createLogger()
	go func() {
		defer writer.Close()
		_, err := io.Copy(writer, r)
		if err != nil && err != io.EOF {
			// using tryComplete is racy because the state machine closes completedCh and
			// so we don't want to attempt to write to a closed/closing chan. so
			// just log this for now.
			log.Errorf("logger for task %q crashed: %v", t.bin, err)
		}
	}()
}

// notStartedTask spawns the given task and transitions to a startedTask state
func notStartedTask(t *Task) taskStateFn {
	log.Infof("starting task process %q with args '%+v'", t.bin, t.args)

	// create command
	cmd := exec.Command(t.bin, t.args...)
	if _, err := cmd.StdoutPipe(); err != nil {
		t.tryError(fmt.Errorf("error getting stdout of %v: %v", t.name, err))
		return taskShouldRestart
	}
	stderrLogs, err := cmd.StderrPipe()
	if err != nil {
		t.tryError(fmt.Errorf("error getting stderr of %v: %v", t.name, err))
		return taskShouldRestart
	}

	t.initLogging(stderrLogs)
	if len(t.env) > 0 {
		cmd.Env = t.env
	}
	cmd.SysProcAttr = sysProcAttr()

	// last min check for shouldQuit here
	select {
	case <-t.shouldQuit:
		t.tryError(fmt.Errorf("task execution canceled, aborting process launch"))
		return taskShouldRestart
	default:
	}

	if err := cmd.Start(); err != nil {
		t.tryError(fmt.Errorf("failed to start task process %q: %v", t.bin, err))
		return taskShouldRestart
	}
	log.Infoln("task started", t.name)
	t.cmd = &cmdProcess{delegate: cmd}
	return taskRunning
}

type exitError interface {
	error

	// see os.ProcessState.Sys: returned value can be converted to something like syscall.WaitStatus
	Sys() interface{}
}

func taskRunning(t *Task) taskStateFn {
	waiter := t.cmd.Wait
	var sendOnce sync.Once
	trySend := func(wr *Completion) {
		// guarded with once because we're only allowed to send a single "result" for each
		// process termination. for example, if Kill() results in an error because Wait()
		// already completed we only want to return a single result for the process.
		sendOnce.Do(func() {
			t.tryComplete(wr)
		})
	}
	// listen for normal process completion in a goroutine; don't block because we need to listen for shouldQuit
	waitCh := make(chan *Completion, 1)
	go func() {
		wr := &Completion{}
		defer func() {
			waitCh <- wr
			close(waitCh)
		}()

		if err := waiter(); err != nil {
			if exitError, ok := err.(exitError); ok {
				if waitStatus, ok := exitError.Sys().(syscall.WaitStatus); ok {
					wr.name = t.name
					wr.code = waitStatus.ExitStatus()
					return
				}
			}
			wr.err = fmt.Errorf("task wait ended strangely for %q: %v", t.bin, err)
		} else {
			wr.name = t.name
		}
	}()

	// wait for the process to complete, or else for a "quit" signal on the task (at which point we'll attempt to kill manually)
	select {
	case <-t.shouldQuit:
		// check for tie
		select {
		case wr := <-waitCh:
			// we got a signal to quit, but we're already finished; attempt best effort delvery
			trySend(wr)
		default:
			// Wait() has not exited yet, kill the process
			log.Infof("killing %s : %s", t.name, t.bin)
			pid, err := t.cmd.Kill()
			if err != nil {
				trySend(&Completion{err: fmt.Errorf("failed to kill process: %q pid %d: %v", t.bin, pid, err)})
			}
			// else, Wait() should complete and send a completion event
		}
	case wr := <-waitCh:
		// task has completed before we were told to quit, pass along completion and error information
		trySend(wr)
	}
	return taskShouldRestart
}

// forwardUntil forwards task process completion status and errors to the given output
// chans until either the task terminates or abort is closed.
func (t *Task) forwardUntil(tch chan<- *Completion, abort <-chan struct{}) {
	// merge task completion and error until we're told to die, then
	// tell the task to stop
	defer close(t.shouldQuit)
	forwardCompletionUntil(t.completedCh, tch, nil, abort, nil)
}

// MergeOutput waits for the given tasks to complete. meanwhile it logs each time a task
// process completes or generates an error. when shouldQuit closes, tasks are canceled and this
// func eventually returns once all ongoing event handlers have completed running.
func MergeOutput(tasks []*Task, shouldQuit <-chan struct{}) Events {
	tc := make(chan *Completion)

	var waitForTasks sync.WaitGroup
	waitForTasks.Add(len(tasks))

	for _, t := range tasks {
		t := t
		// translate task dead signal into Done
		go func() {
			<-t.done
			waitForTasks.Done()
		}()
		// fan-in task completion and error events to tc, ec
		go t.forwardUntil(tc, shouldQuit)
	}

	tclistener := make(chan *Completion)
	done := runtime.After(func() {
		completionFinished := runtime.After(func() {
			defer close(tclistener)
			forwardCompletionUntil(tc, tclistener, nil, shouldQuit, func(tt *Completion, shutdown bool) {
				prefix := ""
				if shutdown {
					prefix = "(shutdown) "
				}
				log.Infof(prefix+"task %q exited with status %d", tt.name, tt.code)
			})
		})
		waitForTasks.Wait()
		close(tc)
		<-completionFinished
	})
	ei := newEventsImpl(tclistener, done)
	return ei
}
