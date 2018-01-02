// +build windows

package windows

import (
	"context"
	"io"
	"sync"
	"syscall"
	"time"

	"github.com/Microsoft/hcsshim"
	eventsapi "github.com/containerd/containerd/api/services/events/v1"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/runtime"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// process implements containerd.Process and containerd.State
type process struct {
	sync.Mutex

	hcs hcsshim.Process

	id   string
	pid  uint32
	io   *pipeSet
	task *task

	exitCh   chan struct{}
	exitCode uint32
	exitTime time.Time
	conf     *hcsshim.ProcessConfig
}

func (p *process) ID() string {
	return p.id
}

func (p *process) State(ctx context.Context) (runtime.State, error) {
	return runtime.State{
		Status:     p.Status(),
		Pid:        p.pid,
		Stdin:      p.io.src.Stdin,
		Stdout:     p.io.src.Stdout,
		Stderr:     p.io.src.Stderr,
		Terminal:   p.io.src.Terminal,
		ExitStatus: p.exitCode,
	}, nil
}

func (p *process) Status() runtime.Status {
	p.Lock()
	defer p.Unlock()

	if p.task.getStatus() == runtime.PausedStatus {
		return runtime.PausedStatus
	}

	var status runtime.Status
	select {
	case <-p.exitCh:
		status = runtime.StoppedStatus
	default:
		if p.hcs == nil {
			return runtime.CreatedStatus
		}
		status = runtime.RunningStatus
	}
	return status
}

func (p *process) Kill(ctx context.Context, sig uint32, all bool) error {
	// On windows all signals kill the process
	if p.Status() == runtime.CreatedStatus {
		return errors.Wrap(errdefs.ErrFailedPrecondition, "process was not started")
	}
	return errors.Wrap(p.hcs.Kill(), "failed to kill process")
}

func (p *process) ResizePty(ctx context.Context, size runtime.ConsoleSize) error {
	if p.Status() == runtime.CreatedStatus {
		return errors.Wrap(errdefs.ErrFailedPrecondition, "process was not started")
	}
	err := p.hcs.ResizeConsole(uint16(size.Width), uint16(size.Height))
	return errors.Wrap(err, "failed to resize process console")
}

func (p *process) CloseIO(ctx context.Context) error {
	if p.Status() == runtime.CreatedStatus {
		return errors.Wrap(errdefs.ErrFailedPrecondition, "process was not started")
	}
	return errors.Wrap(p.hcs.CloseStdin(), "failed to close stdin")
}

func (p *process) Pid() uint32 {
	return p.pid
}

func (p *process) ExitCode() (uint32, time.Time, error) {
	if s := p.Status(); s != runtime.StoppedStatus && s != runtime.CreatedStatus {
		return 255, time.Time{}, errors.Wrapf(errdefs.ErrFailedPrecondition, "process is not stopped: %s", s)
	}
	return p.exitCode, p.exitTime, nil
}

func (p *process) Start(ctx context.Context) (err error) {
	p.Lock()
	defer p.Unlock()

	if p.hcs != nil {
		return errors.Wrap(errdefs.ErrFailedPrecondition, "process already started")
	}

	// If we fail, close the io right now
	defer func() {
		if err != nil {
			p.io.Close()
		}
	}()

	var hp hcsshim.Process
	if hp, err = p.task.hcsContainer.CreateProcess(p.conf); err != nil {
		return errors.Wrapf(err, "failed to create process")
	}

	stdin, stdout, stderr, err := hp.Stdio()
	if err != nil {
		hp.Kill()
		return errors.Wrapf(err, "failed to retrieve init process stdio")
	}

	ioCopy := func(name string, dst io.WriteCloser, src io.ReadCloser) {
		log.G(ctx).WithFields(logrus.Fields{"id": p.id, "pid": p.pid}).
			Debugf("%s: copy started", name)
		io.Copy(dst, src)
		log.G(ctx).WithFields(logrus.Fields{"id": p.id, "pid": p.pid}).
			Debugf("%s: copy done", name)
		dst.Close()
		src.Close()
	}

	if p.io.stdin != nil {
		go ioCopy("stdin", stdin, p.io.stdin)
	}

	if p.io.stdout != nil {
		go ioCopy("stdout", p.io.stdout, stdout)
	}

	if p.io.stderr != nil {
		go ioCopy("stderr", p.io.stderr, stderr)
	}
	p.hcs = hp

	// Wait for the process to exit to get the exit status
	go func() {
		if err := hp.Wait(); err != nil {
			herr, ok := err.(*hcsshim.ProcessError)
			if ok && herr.Err != syscall.ERROR_BROKEN_PIPE {
				log.G(ctx).
					WithError(err).
					WithFields(logrus.Fields{"id": p.id, "pid": p.pid}).
					Warnf("hcsshim wait failed (process may have been killed)")
			}
			// Try to get the exit code nonetheless
		}
		p.exitTime = time.Now()

		ec, err := hp.ExitCode()
		if err != nil {
			log.G(ctx).
				WithError(err).
				WithFields(logrus.Fields{"id": p.id, "pid": p.pid}).
				Warnf("hcsshim could not retrieve exit code")
			// Use the unknown exit code
			ec = 255
		}
		p.exitCode = uint32(ec)

		p.task.publisher.Publish(ctx,
			runtime.TaskExitEventTopic,
			&eventsapi.TaskExit{
				ContainerID: p.task.id,
				ID:          p.id,
				Pid:         p.pid,
				ExitStatus:  p.exitCode,
				ExitedAt:    p.exitTime,
			})

		close(p.exitCh)
		// Ensure io's are closed
		p.io.Close()
		// Cleanup HCS resources
		hp.Close()
	}()
	return nil
}

func (p *process) Wait(ctx context.Context) (*runtime.Exit, error) {
	<-p.exitCh

	ec, ea, err := p.ExitCode()
	if err != nil {
		return nil, err
	}
	return &runtime.Exit{
		Status:    ec,
		Timestamp: ea,
	}, nil
}
