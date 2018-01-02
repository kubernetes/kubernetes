// +build !windows

package shim

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"path/filepath"
	"sync"
	"syscall"
	"time"

	"golang.org/x/sys/unix"

	"github.com/containerd/console"
	"github.com/containerd/containerd/identifiers"
	shimapi "github.com/containerd/containerd/linux/shim/v1"
	"github.com/containerd/fifo"
	runc "github.com/containerd/go-runc"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
)

type execProcess struct {
	wg sync.WaitGroup

	processState

	mu      sync.Mutex
	id      string
	console console.Console
	io      runc.IO
	status  int
	exited  time.Time
	pid     int
	closers []io.Closer
	stdin   io.Closer
	stdio   stdio
	path    string
	spec    specs.Process

	parent    *initProcess
	waitBlock chan struct{}
}

func newExecProcess(context context.Context, path string, r *shimapi.ExecProcessRequest, parent *initProcess, id string) (process, error) {
	if err := identifiers.Validate(id); err != nil {
		return nil, errors.Wrapf(err, "invalid exec id")
	}
	// process exec request
	var spec specs.Process
	if err := json.Unmarshal(r.Spec.Value, &spec); err != nil {
		return nil, err
	}
	spec.Terminal = r.Terminal

	e := &execProcess{
		id:     id,
		path:   path,
		parent: parent,
		spec:   spec,
		stdio: stdio{
			stdin:    r.Stdin,
			stdout:   r.Stdout,
			stderr:   r.Stderr,
			terminal: r.Terminal,
		},
		waitBlock: make(chan struct{}),
	}
	e.processState = &execCreatedState{p: e}
	return e, nil
}

func (e *execProcess) Wait() {
	<-e.waitBlock
}

func (e *execProcess) ID() string {
	return e.id
}

func (e *execProcess) Pid() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.pid
}

func (e *execProcess) ExitStatus() int {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.status
}

func (e *execProcess) ExitedAt() time.Time {
	e.mu.Lock()
	defer e.mu.Unlock()
	return e.exited
}

func (e *execProcess) setExited(status int) {
	e.status = status
	e.exited = time.Now()
	e.parent.platform.shutdownConsole(context.Background(), e.console)
	close(e.waitBlock)
}

func (e *execProcess) delete(ctx context.Context) error {
	e.wg.Wait()
	if e.io != nil {
		for _, c := range e.closers {
			c.Close()
		}
		e.io.Close()
	}
	return nil
}

func (e *execProcess) resize(ws console.WinSize) error {
	if e.console == nil {
		return nil
	}
	return e.console.Resize(ws)
}

func (e *execProcess) kill(ctx context.Context, sig uint32, _ bool) error {
	pid := e.pid
	if pid != 0 {
		if err := unix.Kill(pid, syscall.Signal(sig)); err != nil {
			return errors.Wrapf(checkKillError(err), "exec kill error")
		}
	}
	return nil
}

func (e *execProcess) Stdin() io.Closer {
	return e.stdin
}

func (e *execProcess) Stdio() stdio {
	return e.stdio
}

func (e *execProcess) start(ctx context.Context) (err error) {
	var (
		socket  *runc.Socket
		pidfile = filepath.Join(e.path, fmt.Sprintf("%s.pid", e.id))
	)
	if e.stdio.terminal {
		if socket, err = runc.NewTempConsoleSocket(); err != nil {
			return errors.Wrap(err, "failed to create runc console socket")
		}
		defer socket.Close()
	} else if e.stdio.isNull() {
		if e.io, err = runc.NewNullIO(); err != nil {
			return errors.Wrap(err, "creating new NULL IO")
		}
	} else {
		if e.io, err = runc.NewPipeIO(e.parent.IoUID, e.parent.IoGID); err != nil {
			return errors.Wrap(err, "failed to create runc io pipes")
		}
	}
	opts := &runc.ExecOpts{
		PidFile: pidfile,
		IO:      e.io,
		Detach:  true,
	}
	if socket != nil {
		opts.ConsoleSocket = socket
	}
	if err := e.parent.runtime.Exec(ctx, e.parent.id, e.spec, opts); err != nil {
		return e.parent.runtimeError(err, "OCI runtime exec failed")
	}
	if e.stdio.stdin != "" {
		sc, err := fifo.OpenFifo(ctx, e.stdio.stdin, syscall.O_WRONLY|syscall.O_NONBLOCK, 0)
		if err != nil {
			return errors.Wrapf(err, "failed to open stdin fifo %s", e.stdio.stdin)
		}
		e.closers = append(e.closers, sc)
		e.stdin = sc
	}
	var copyWaitGroup sync.WaitGroup
	if socket != nil {
		console, err := socket.ReceiveMaster()
		if err != nil {
			return errors.Wrap(err, "failed to retrieve console master")
		}
		if e.console, err = e.parent.platform.copyConsole(ctx, console, e.stdio.stdin, e.stdio.stdout, e.stdio.stderr, &e.wg, &copyWaitGroup); err != nil {
			return errors.Wrap(err, "failed to start console copy")
		}
	} else if !e.stdio.isNull() {
		if err := copyPipes(ctx, e.io, e.stdio.stdin, e.stdio.stdout, e.stdio.stderr, &e.wg, &copyWaitGroup); err != nil {
			return errors.Wrap(err, "failed to start io pipe copy")
		}
	}
	copyWaitGroup.Wait()
	pid, err := runc.ReadPidFile(opts.PidFile)
	if err != nil {
		return errors.Wrap(err, "failed to retrieve OCI runtime exec pid")
	}
	e.pid = pid
	return nil
}

func (e *execProcess) Status(ctx context.Context) (string, error) {
	s, err := e.parent.Status(ctx)
	if err != nil {
		return "", err
	}
	// if the container as a whole is in the pausing/paused state, so are all
	// other processes inside the container, use container state here
	switch s {
	case "paused", "pausing":
		return s, nil
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	// if we don't have a pid then the exec process has just been created
	if e.pid == 0 {
		return "created", nil
	}
	// if we have a pid and it can be signaled, the process is running
	if err := unix.Kill(e.pid, 0); err == nil {
		return "running", nil
	}
	// else if we have a pid but it can nolonger be signaled, it has stopped
	return "stopped", nil
}
