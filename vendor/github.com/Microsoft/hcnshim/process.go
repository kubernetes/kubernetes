//go:build windows

package hcnshim

import (
	"context"
	"io"
	"sync"
	"time"

	"github.com/Microsoft/hcnshim/internal/hcs"
)

// ContainerError is an error encountered in HCS
type process struct {
	p        *hcs.Process
	waitOnce sync.Once
	waitCh   chan struct{}
	waitErr  error
}

// Pid returns the process ID of the process within the container.
func (process *process) Pid() int {
	return process.p.Pid()
}

// Kill signals the process to terminate but does not wait for it to finish terminating.
func (process *process) Kill() error {
	found, err := process.p.Kill(context.Background())
	if err != nil {
		return convertProcessError(err, process)
	}
	if !found {
		return &ProcessError{Process: process, Err: ErrElementNotFound, Operation: "hcnshim::Process::Kill"}
	}
	return nil
}

// Wait waits for the process to exit.
func (process *process) Wait() error {
	return convertProcessError(process.p.Wait(), process)
}

// WaitTimeout waits for the process to exit or the duration to elapse. It returns
// false if timeout occurs.
func (process *process) WaitTimeout(timeout time.Duration) error {
	process.waitOnce.Do(func() {
		process.waitCh = make(chan struct{})
		go func() {
			process.waitErr = process.Wait()
			close(process.waitCh)
		}()
	})
	t := time.NewTimer(timeout)
	defer t.Stop()
	select {
	case <-t.C:
		return &ProcessError{Process: process, Err: ErrTimeout, Operation: "hcnshim::Process::Wait"}
	case <-process.waitCh:
		return process.waitErr
	}
}

// ExitCode returns the exit code of the process. The process must have
// already terminated.
func (process *process) ExitCode() (int, error) {
	code, err := process.p.ExitCode()
	if err != nil {
		err = convertProcessError(err, process)
	}
	return code, err
}

// ResizeConsole resizes the console of the process.
func (process *process) ResizeConsole(width, height uint16) error {
	return convertProcessError(process.p.ResizeConsole(context.Background(), width, height), process)
}

// Stdio returns the stdin, stdout, and stderr pipes, respectively. Closing
// these pipes does not close the underlying pipes; it should be possible to
// call this multiple times to get multiple interfaces.
func (process *process) Stdio() (io.WriteCloser, io.ReadCloser, io.ReadCloser, error) {
	stdin, stdout, stderr, err := process.p.StdioLegacy()
	if err != nil {
		err = convertProcessError(err, process)
	}
	return stdin, stdout, stderr, err
}

// CloseStdin closes the write side of the stdin pipe so that the process is
// notified on the read side that there is no more data in stdin.
func (process *process) CloseStdin() error {
	return convertProcessError(process.p.CloseStdin(context.Background()), process)
}

// Close cleans up any state associated with the process but does not kill
// or wait on it.
func (process *process) Close() error {
	return convertProcessError(process.p.Close(), process)
}
