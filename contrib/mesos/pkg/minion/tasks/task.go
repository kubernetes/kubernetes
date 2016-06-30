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
	"fmt"
	"io"
	"io/ioutil"
	"os/exec"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	log "github.com/golang/glog"
	"k8s.io/kubernetes/contrib/mesos/pkg/runtime"
)

const (
	defaultTaskRestartDelay = 5 * time.Second

	// TODO(jdef) there's no easy way for us to discover the grace period that we actually
	// have, from mesos: it's simply a missing core feature. there's a MESOS-xyz ticket for
	// this somewhere. if it was discoverable then we could come up with a better strategy.
	// there are some comments in the executor regarding this as well (because there we're
	// concerned about cleaning up pods within the grace period). we could pick a some
	// higher (arbitrary) value but without knowing when the slave will forcibly kill us
	// it seems a somewhat futile exercise.
	defaultKillGracePeriod = 5 * time.Second
)

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
	Kill(force bool) (int, error)
}

type cmdProcess struct {
	delegate *exec.Cmd
}

func (cp *cmdProcess) Wait() error {
	return cp.delegate.Wait()
}

func (cp *cmdProcess) Kill(force bool) (int, error) {
	// kill the entire process group, not just the one process
	pid := cp.delegate.Process.Pid
	processGroup := 0 - pid

	// we send a SIGTERM here for a graceful stop. users of this package should
	// wait for tasks to complete normally. as a fallback/safeguard, child procs
	// are spawned in notStartedTask to receive a SIGKILL when this process dies.
	sig := syscall.SIGTERM
	if force {
		sig = syscall.SIGKILL
	}
	rc := syscall.Kill(processGroup, sig)
	return pid, rc
}

// task is a specification for running a system process; it provides hooks for customizing
// logging and restart handling as well as provides event channels for communicating process
// termination and errors related to process management.
type Task struct {
	Env          []string                   // optional: process environment override
	Finished     func(restarting bool) bool // callback invoked when a task process has completed; if `restarting` then it will be restarted if it returns true
	RestartDelay time.Duration              // interval between repeated task restarts

	name         string                // required: unique name for this task
	bin          string                // required: path to executable
	args         []string              // optional: process arguments
	createLogger func() io.WriteCloser // factory func that builds a log writer
	cmd          systemProcess         // process that we started
	completedCh  chan *Completion      // reports exit codes encountered when task processes exit, or errors during process management
	shouldQuit   chan struct{}         // shouldQuit is closed to indicate that the task should stop its running process, if any
	done         chan struct{}         // done closes when all processes related to the task have terminated
	initialState taskStateFn           // prepare and start a new live process, defaults to notStartedTask; should be set by run()
	runLatch     int32                 // guard against multiple Task.run calls
	killFunc     func(bool) (int, error)
}

// New builds a newly initialized task object but does not start any processes for it. callers
// are expected to invoke task.run(...) on their own.
func New(name, bin string, args []string, cl func() io.WriteCloser, options ...Option) *Task {
	t := &Task{
		name:         name,
		bin:          bin,
		args:         args,
		createLogger: cl,
		completedCh:  make(chan *Completion),
		shouldQuit:   make(chan struct{}),
		done:         make(chan struct{}),
		RestartDelay: defaultTaskRestartDelay,
		Finished:     func(restarting bool) bool { return restarting },
	}
	t.killFunc = func(force bool) (int, error) { return t.cmd.Kill(force) }
	for _, opt := range options {
		opt(t)
	}
	return t
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
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.tryError(fmt.Errorf("error getting stdout of %v: %v", t.name, err))
		return taskShouldRestart
	}
	go func() {
		defer stdout.Close()
		io.Copy(ioutil.Discard, stdout) // TODO(jdef) we might want to save this at some point
	}()
	stderrLogs, err := cmd.StderrPipe()
	if err != nil {
		t.tryError(fmt.Errorf("error getting stderr of %v: %v", t.name, err))
		return taskShouldRestart
	}

	t.initLogging(stderrLogs)
	if len(t.Env) > 0 {
		cmd.Env = t.Env
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
	// listen for normal process completion in a goroutine; don't block because we need to listen for shouldQuit
	waitCh := make(chan *Completion, 1)
	go func() {
		wr := &Completion{name: t.name}
		defer func() {
			waitCh <- wr
			close(waitCh)
		}()

		if err := t.cmd.Wait(); err != nil {
			if exitError, ok := err.(exitError); ok {
				if waitStatus, ok := exitError.Sys().(syscall.WaitStatus); ok {
					wr.code = waitStatus.ExitStatus()
					return
				}
			}
			wr.err = fmt.Errorf("task wait ended strangely for %q: %v", t.bin, err)
		}
	}()

	select {
	case <-t.shouldQuit:
		t.tryComplete(t.awaitDeath(&realTimer{}, defaultKillGracePeriod, waitCh))
	case wr := <-waitCh:
		t.tryComplete(wr)
	}
	return taskShouldRestart
}

// awaitDeath waits for the process to complete, or else for a "quit" signal on the task-
// at which point we'll attempt to kill manually.
func (t *Task) awaitDeath(timer timer, gracePeriod time.Duration, waitCh <-chan *Completion) *Completion {
	defer timer.discard()

	select {
	case wr := <-waitCh:
		// got a signal to quit, but we're already finished
		return wr
	default:
	}

	forceKill := false
	wr := &Completion{name: t.name, err: fmt.Errorf("failed to kill process: %q", t.bin)}

	// the loop is here in case we receive a shouldQuit signal; we need to kill the task.
	// in this case, first send a SIGTERM (force=false) to the task and then wait for it
	// to die (within the gracePeriod). if it doesn't die, then we loop around only this
	// time we'll send a SIGKILL (force=true) and wait for a reduced gracePeriod. There
	// does exist a slim chance that the underlying wait4() syscall won't complete before
	// this process dies, in which case a zombie will rise. Starting the mesos slave with
	// pid namespace isolation should mitigate this.
waitLoop:
	for i := 0; i < 2; i++ {
		log.Infof("killing %s (force=%t) : %s", t.name, forceKill, t.bin)
		pid, err := t.killFunc(forceKill)
		if err != nil {
			log.Warningf("failed to kill process: %q pid %d: %v", t.bin, pid, err)
			break waitLoop
		}

		// Wait for the kill to be processed, and child proc resources cleaned up; try to avoid zombies!
		timer.set(gracePeriod)
		select {
		case wr = <-waitCh:
			break waitLoop
		case <-timer.await():
			// want a timeout, but a shorter one than we used initially.
			// using /= 2 is deterministic and yields the desirable effect.
			gracePeriod /= 2
			forceKill = true
			continue waitLoop
		}
	}
	return wr
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

// Option is a functional option type for a Task that returns an "undo" Option after upon modifying the Task
type Option func(*Task) Option

// NoRespawn configures the Task lifecycle such that it will not respawn upon termination
func NoRespawn(listener chan<- struct{}) Option {
	return func(t *Task) Option {
		finished, restartDelay := t.Finished, t.RestartDelay

		t.Finished = func(_ bool) bool {
			// this func implements the task.finished spec, so when the task exits
			// we return false to indicate that it should not be restarted. we also
			// close execDied to signal interested listeners.
			if listener != nil {
				close(listener)
				listener = nil
			}
			return false
		}

		// since we only expect to die once, and there is no restart; don't delay any longer than needed
		t.RestartDelay = 0

		return func(t2 *Task) Option {
			t2.Finished, t2.RestartDelay = finished, restartDelay
			return NoRespawn(listener)
		}
	}
}

// Environment customizes the process runtime environment for a Task
func Environment(env []string) Option {
	return func(t *Task) Option {
		oldenv := t.Env
		t.Env = env[:]
		return Environment(oldenv)
	}
}
