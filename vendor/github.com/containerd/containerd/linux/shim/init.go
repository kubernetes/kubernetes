// +build !windows

package shim

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"syscall"
	"time"

	"github.com/containerd/console"
	"github.com/containerd/containerd/identifiers"
	"github.com/containerd/containerd/linux/runcopts"
	shimapi "github.com/containerd/containerd/linux/shim/v1"
	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/mount"
	"github.com/containerd/fifo"
	runc "github.com/containerd/go-runc"
	"github.com/containerd/typeurl"
	specs "github.com/opencontainers/runtime-spec/specs-go"
	"github.com/pkg/errors"
)

// InitPidFile name of the file that contains the init pid
const InitPidFile = "init.pid"

type initProcess struct {
	wg sync.WaitGroup
	initState

	// mu is used to ensure that `Start()` and `Exited()` calls return in
	// the right order when invoked in separate go routines.
	// This is the case within the shim implementation as it makes use of
	// the reaper interface.
	mu sync.Mutex

	waitBlock chan struct{}

	workDir string

	id       string
	bundle   string
	console  console.Console
	platform platform
	io       runc.IO
	runtime  *runc.Runc
	status   int
	exited   time.Time
	pid      int
	closers  []io.Closer
	stdin    io.Closer
	stdio    stdio
	rootfs   string
	IoUID    int
	IoGID    int
}

func (s *Service) newInitProcess(context context.Context, r *shimapi.CreateTaskRequest) (*initProcess, error) {
	var success bool

	if err := identifiers.Validate(r.ID); err != nil {
		return nil, errors.Wrapf(err, "invalid task id")
	}
	var options runcopts.CreateOptions
	if r.Options != nil {
		v, err := typeurl.UnmarshalAny(r.Options)
		if err != nil {
			return nil, err
		}
		options = *v.(*runcopts.CreateOptions)
	}

	rootfs := filepath.Join(s.config.Path, "rootfs")
	// count the number of successful mounts so we can undo
	// what was actually done rather than what should have been
	// done.
	defer func() {
		if success {
			return
		}
		if err2 := mount.UnmountAll(rootfs, 0); err2 != nil {
			log.G(context).WithError(err2).Warn("Failed to cleanup rootfs mount")
		}
	}()
	for _, rm := range r.Rootfs {
		m := &mount.Mount{
			Type:    rm.Type,
			Source:  rm.Source,
			Options: rm.Options,
		}
		if err := m.Mount(rootfs); err != nil {
			return nil, errors.Wrapf(err, "failed to mount rootfs component %v", m)
		}
	}
	root := s.config.RuntimeRoot
	if root == "" {
		root = RuncRoot
	}
	runtime := &runc.Runc{
		Command:       r.Runtime,
		Log:           filepath.Join(s.config.Path, "log.json"),
		LogFormat:     runc.JSON,
		PdeathSignal:  syscall.SIGKILL,
		Root:          filepath.Join(root, s.config.Namespace),
		Criu:          s.config.Criu,
		SystemdCgroup: s.config.SystemdCgroup,
	}
	p := &initProcess{
		id:       r.ID,
		bundle:   r.Bundle,
		runtime:  runtime,
		platform: s.platform,
		stdio: stdio{
			stdin:    r.Stdin,
			stdout:   r.Stdout,
			stderr:   r.Stderr,
			terminal: r.Terminal,
		},
		rootfs:    rootfs,
		workDir:   s.config.WorkDir,
		status:    0,
		waitBlock: make(chan struct{}),
		IoUID:     int(options.IoUid),
		IoGID:     int(options.IoGid),
	}
	p.initState = &createdState{p: p}
	var (
		err    error
		socket *runc.Socket
	)
	if r.Terminal {
		if socket, err = runc.NewTempConsoleSocket(); err != nil {
			return nil, errors.Wrap(err, "failed to create OCI runtime console socket")
		}
		defer socket.Close()
	} else if hasNoIO(r) {
		if p.io, err = runc.NewNullIO(); err != nil {
			return nil, errors.Wrap(err, "creating new NULL IO")
		}
	} else {
		if p.io, err = runc.NewPipeIO(int(options.IoUid), int(options.IoGid)); err != nil {
			return nil, errors.Wrap(err, "failed to create OCI runtime io pipes")
		}
	}
	pidFile := filepath.Join(s.config.Path, InitPidFile)
	if r.Checkpoint != "" {
		opts := &runc.RestoreOpts{
			CheckpointOpts: runc.CheckpointOpts{
				ImagePath:  r.Checkpoint,
				WorkDir:    p.workDir,
				ParentPath: r.ParentCheckpoint,
			},
			PidFile:     pidFile,
			IO:          p.io,
			NoPivot:     options.NoPivotRoot,
			Detach:      true,
			NoSubreaper: true,
		}
		p.initState = &createdCheckpointState{
			p:    p,
			opts: opts,
		}
		success = true
		return p, nil
	}
	opts := &runc.CreateOpts{
		PidFile:      pidFile,
		IO:           p.io,
		NoPivot:      options.NoPivotRoot,
		NoNewKeyring: options.NoNewKeyring,
	}
	if socket != nil {
		opts.ConsoleSocket = socket
	}
	if err := p.runtime.Create(context, r.ID, r.Bundle, opts); err != nil {
		return nil, p.runtimeError(err, "OCI runtime create failed")
	}
	if r.Stdin != "" {
		sc, err := fifo.OpenFifo(context, r.Stdin, syscall.O_WRONLY|syscall.O_NONBLOCK, 0)
		if err != nil {
			return nil, errors.Wrapf(err, "failed to open stdin fifo %s", r.Stdin)
		}
		p.stdin = sc
		p.closers = append(p.closers, sc)
	}
	var copyWaitGroup sync.WaitGroup
	if socket != nil {
		console, err := socket.ReceiveMaster()
		if err != nil {
			return nil, errors.Wrap(err, "failed to retrieve console master")
		}
		console, err = s.platform.copyConsole(context, console, r.Stdin, r.Stdout, r.Stderr, &p.wg, &copyWaitGroup)
		if err != nil {
			return nil, errors.Wrap(err, "failed to start console copy")
		}
		p.console = console
	} else if !hasNoIO(r) {
		if err := copyPipes(context, p.io, r.Stdin, r.Stdout, r.Stderr, &p.wg, &copyWaitGroup); err != nil {
			return nil, errors.Wrap(err, "failed to start io pipe copy")
		}
	}

	copyWaitGroup.Wait()
	pid, err := runc.ReadPidFile(pidFile)
	if err != nil {
		return nil, errors.Wrap(err, "failed to retrieve OCI runtime container pid")
	}
	p.pid = pid
	success = true
	return p, nil
}

func (p *initProcess) Wait() {
	<-p.waitBlock
}

func (p *initProcess) ID() string {
	return p.id
}

func (p *initProcess) Pid() int {
	return p.pid
}

func (p *initProcess) ExitStatus() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.status
}

func (p *initProcess) ExitedAt() time.Time {
	p.mu.Lock()
	defer p.mu.Unlock()
	return p.exited
}

func (p *initProcess) Status(ctx context.Context) (string, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	c, err := p.runtime.State(ctx, p.id)
	if err != nil {
		if os.IsNotExist(err) {
			return "stopped", nil
		}
		return "", p.runtimeError(err, "OCI runtime state failed")
	}
	return c.Status, nil
}

func (p *initProcess) start(context context.Context) error {
	err := p.runtime.Start(context, p.id)
	return p.runtimeError(err, "OCI runtime start failed")
}

func (p *initProcess) setExited(status int) {
	p.exited = time.Now()
	p.status = status
	p.platform.shutdownConsole(context.Background(), p.console)
	close(p.waitBlock)
}

func (p *initProcess) delete(context context.Context) error {
	p.killAll(context)
	p.wg.Wait()
	err := p.runtime.Delete(context, p.id, nil)
	// ignore errors if a runtime has already deleted the process
	// but we still hold metadata and pipes
	//
	// this is common during a checkpoint, runc will delete the container state
	// after a checkpoint and the container will no longer exist within runc
	if err != nil {
		if strings.Contains(err.Error(), "does not exist") {
			err = nil
		} else {
			err = p.runtimeError(err, "failed to delete task")
		}
	}
	if p.io != nil {
		for _, c := range p.closers {
			c.Close()
		}
		p.io.Close()
	}
	if err2 := mount.UnmountAll(p.rootfs, 0); err2 != nil {
		log.G(context).WithError(err2).Warn("failed to cleanup rootfs mount")
		if err == nil {
			err = errors.Wrap(err2, "failed rootfs umount")
		}
	}
	return err
}

func (p *initProcess) resize(ws console.WinSize) error {
	if p.console == nil {
		return nil
	}
	return p.console.Resize(ws)
}

func (p *initProcess) pause(context context.Context) error {
	err := p.runtime.Pause(context, p.id)
	return p.runtimeError(err, "OCI runtime pause failed")
}

func (p *initProcess) resume(context context.Context) error {
	err := p.runtime.Resume(context, p.id)
	return p.runtimeError(err, "OCI runtime resume failed")
}

func (p *initProcess) kill(context context.Context, signal uint32, all bool) error {
	err := p.runtime.Kill(context, p.id, int(signal), &runc.KillOpts{
		All: all,
	})
	return checkKillError(err)
}

func (p *initProcess) killAll(context context.Context) error {
	err := p.runtime.Kill(context, p.id, int(syscall.SIGKILL), &runc.KillOpts{
		All: true,
	})
	return p.runtimeError(err, "OCI runtime killall failed")
}

func (p *initProcess) Stdin() io.Closer {
	return p.stdin
}

func (p *initProcess) checkpoint(context context.Context, r *shimapi.CheckpointTaskRequest) error {
	var options runcopts.CheckpointOptions
	if r.Options != nil {
		v, err := typeurl.UnmarshalAny(r.Options)
		if err != nil {
			return err
		}
		options = *v.(*runcopts.CheckpointOptions)
	}
	var actions []runc.CheckpointAction
	if !options.Exit {
		actions = append(actions, runc.LeaveRunning)
	}
	work := filepath.Join(p.workDir, "criu-work")
	defer os.RemoveAll(work)
	if err := p.runtime.Checkpoint(context, p.id, &runc.CheckpointOpts{
		WorkDir:                  work,
		ImagePath:                r.Path,
		AllowOpenTCP:             options.OpenTcp,
		AllowExternalUnixSockets: options.ExternalUnixSockets,
		AllowTerminal:            options.Terminal,
		FileLocks:                options.FileLocks,
		EmptyNamespaces:          options.EmptyNamespaces,
	}, actions...); err != nil {
		dumpLog := filepath.Join(p.bundle, "criu-dump.log")
		if cerr := copyFile(dumpLog, filepath.Join(work, "dump.log")); cerr != nil {
			log.G(context).Error(err)
		}
		return fmt.Errorf("%s path= %s", criuError(err), dumpLog)
	}
	return nil
}

func (p *initProcess) update(context context.Context, r *shimapi.UpdateTaskRequest) error {
	var resources specs.LinuxResources
	if err := json.Unmarshal(r.Resources.Value, &resources); err != nil {
		return err
	}
	return p.runtime.Update(context, p.id, &resources)
}

func (p *initProcess) Stdio() stdio {
	return p.stdio
}

func (p *initProcess) runtimeError(rErr error, msg string) error {
	if rErr == nil {
		return nil
	}

	rMsg, err := getLastRuntimeError(p.runtime)
	switch {
	case err != nil:
		return errors.Wrapf(rErr, "%s: %s (%s)", msg, "unable to retrieve OCI runtime error", err.Error())
	case rMsg == "":
		return errors.Wrap(rErr, msg)
	default:
		return errors.Errorf("%s: %s", msg, rMsg)
	}
}
