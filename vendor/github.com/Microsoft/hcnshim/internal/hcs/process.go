//go:build windows

package hcs

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"os"
	"sync"
	"syscall"
	"time"

	"go.opencensus.io/trace"

	"github.com/Microsoft/hcnshim/internal/cow"
	hcsschema "github.com/Microsoft/hcnshim/internal/hcs/schema2"
	"github.com/Microsoft/hcnshim/internal/log"
	"github.com/Microsoft/hcnshim/internal/oc"
	"github.com/Microsoft/hcnshim/internal/protocol/guestrequest"
	"github.com/Microsoft/hcnshim/internal/vmcompute"
)

type Process struct {
	handleLock          sync.RWMutex
	handle              vmcompute.HcsProcess
	processID           int
	system              *System
	hasCachedStdio      bool
	stdioLock           sync.Mutex
	stdin               io.WriteCloser
	stdout              io.ReadCloser
	stderr              io.ReadCloser
	callbackNumber      uintptr
	killSignalDelivered bool

	closedWaitOnce sync.Once
	waitBlock      chan struct{}
	exitCode       int
	waitError      error
}

var _ cow.Process = &Process{}

func newProcess(process vmcompute.HcsProcess, processID int, computeSystem *System) *Process {
	return &Process{
		handle:    process,
		processID: processID,
		system:    computeSystem,
		waitBlock: make(chan struct{}),
	}
}

// Pid returns the process ID of the process within the container.
func (process *Process) Pid() int {
	return process.processID
}

// SystemID returns the ID of the process's compute system.
func (process *Process) SystemID() string {
	return process.system.ID()
}

func (process *Process) processSignalResult(ctx context.Context, err error) (bool, error) {
	switch err { //nolint:errorlint
	case nil:
		return true, nil
	case ErrVmcomputeOperationInvalidState, ErrComputeSystemDoesNotExist, ErrElementNotFound:
		if !process.stopped() {
			// The process should be gone, but we have not received the notification.
			// After a second, force unblock the process wait to work around a possible
			// deadlock in the HCS.
			go func() {
				time.Sleep(time.Second)
				process.closedWaitOnce.Do(func() {
					log.G(ctx).WithError(err).Warn("force unblocking process waits")
					process.exitCode = -1
					process.waitError = err
					close(process.waitBlock)
				})
			}()
		}
		return false, nil
	default:
		return false, err
	}
}

// Signal signals the process with `options`.
//
// For LCOW `guestresource.SignalProcessOptionsLCOW`.
//
// For WCOW `guestresource.SignalProcessOptionsWCOW`.
func (process *Process) Signal(ctx context.Context, options interface{}) (bool, error) {
	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	operation := "hcs::Process::Signal"

	if process.handle == 0 {
		return false, makeProcessError(process, operation, ErrAlreadyClosed, nil)
	}

	optionsb, err := json.Marshal(options)
	if err != nil {
		return false, err
	}

	resultJSON, err := vmcompute.HcsSignalProcess(ctx, process.handle, string(optionsb))
	events := processHcsResult(ctx, resultJSON)
	delivered, err := process.processSignalResult(ctx, err)
	if err != nil {
		err = makeProcessError(process, operation, err, events)
	}
	return delivered, err
}

// Kill signals the process to terminate but does not wait for it to finish terminating.
func (process *Process) Kill(ctx context.Context) (bool, error) {
	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	operation := "hcs::Process::Kill"

	if process.handle == 0 {
		return false, makeProcessError(process, operation, ErrAlreadyClosed, nil)
	}

	if process.stopped() {
		return false, makeProcessError(process, operation, ErrProcessAlreadyStopped, nil)
	}

	if process.killSignalDelivered {
		// A kill signal has already been sent to this process. Sending a second
		// one offers no real benefit, as processes cannot stop themselves from
		// being terminated, once a TerminateProcess has been issued. Sending a
		// second kill may result in a number of errors (two of which detailed bellow)
		// and which we can avoid handling.
		return true, nil
	}

	// HCS serializes the signals sent to a target pid per compute system handle.
	// To avoid SIGKILL being serialized behind other signals, we open a new compute
	// system handle to deliver the kill signal.
	// If the calls to opening a new compute system handle fail, we forcefully
	// terminate the container itself so that no container is left behind
	hcsSystem, err := OpenComputeSystem(ctx, process.system.id)
	if err != nil {
		// log error and force termination of container
		log.G(ctx).WithField("err", err).Error("OpenComputeSystem() call failed")
		err = process.system.Terminate(ctx)
		// if the Terminate() call itself ever failed, log and return error
		if err != nil {
			log.G(ctx).WithField("err", err).Error("Terminate() call failed")
			return false, err
		}
		process.system.Close()
		return true, nil
	}
	defer hcsSystem.Close()

	newProcessHandle, err := hcsSystem.OpenProcess(ctx, process.Pid())
	if err != nil {
		// Return true only if the target process has either already
		// exited, or does not exist.
		if IsAlreadyStopped(err) {
			return true, nil
		} else {
			return false, err
		}
	}
	defer newProcessHandle.Close()

	resultJSON, err := vmcompute.HcsTerminateProcess(ctx, newProcessHandle.handle)
	if err != nil {
		// We still need to check these two cases, as processes may still be killed by an
		// external actor (human operator, OOM, random script etc).
		if errors.Is(err, os.ErrPermission) || IsAlreadyStopped(err) {
			// There are two cases where it should be safe to ignore an error returned
			// by HcsTerminateProcess. The first one is cause by the fact that
			// HcsTerminateProcess ends up calling TerminateProcess in the context
			// of a container. According to the TerminateProcess documentation:
			// https://docs.microsoft.com/en-us/windows/win32/api/processthreadsapi/nf-processthreadsapi-terminateprocess#remarks
			// After a process has terminated, call to TerminateProcess with open
			// handles to the process fails with ERROR_ACCESS_DENIED (5) error code.
			// It's safe to ignore this error here. HCS should always have permissions
			// to kill processes inside any container. So an ERROR_ACCESS_DENIED
			// is unlikely to be anything else than what the ending remarks in the
			// documentation states.
			//
			// The second case is generated by hcs itself, if for any reason HcsTerminateProcess
			// is called twice in a very short amount of time. In such cases, hcs may return
			// HCS_E_PROCESS_ALREADY_STOPPED.
			return true, nil
		}
	}
	events := processHcsResult(ctx, resultJSON)
	delivered, err := newProcessHandle.processSignalResult(ctx, err)
	if err != nil {
		err = makeProcessError(newProcessHandle, operation, err, events)
	}

	process.killSignalDelivered = delivered
	return delivered, err
}

// waitBackground waits for the process exit notification. Once received sets
// `process.waitError` (if any) and unblocks all `Wait` calls.
//
// This MUST be called exactly once per `process.handle` but `Wait` is safe to
// call multiple times.
func (process *Process) waitBackground() {
	operation := "hcs::Process::waitBackground"
	ctx, span := oc.StartSpan(context.Background(), operation)
	defer span.End()
	span.AddAttributes(
		trace.StringAttribute("cid", process.SystemID()),
		trace.Int64Attribute("pid", int64(process.processID)))

	var (
		err            error
		exitCode       = -1
		propertiesJSON string
		resultJSON     string
	)

	err = waitForNotification(ctx, process.callbackNumber, hcsNotificationProcessExited, nil)
	if err != nil {
		err = makeProcessError(process, operation, err, nil)
		log.G(ctx).WithError(err).Error("failed wait")
	} else {
		process.handleLock.RLock()
		defer process.handleLock.RUnlock()

		// Make sure we didn't race with Close() here
		if process.handle != 0 {
			propertiesJSON, resultJSON, err = vmcompute.HcsGetProcessProperties(ctx, process.handle)
			events := processHcsResult(ctx, resultJSON)
			if err != nil {
				err = makeProcessError(process, operation, err, events)
			} else {
				properties := &hcsschema.ProcessStatus{}
				err = json.Unmarshal([]byte(propertiesJSON), properties)
				if err != nil {
					err = makeProcessError(process, operation, err, nil)
				} else {
					if properties.LastWaitResult != 0 {
						log.G(ctx).WithField("wait-result", properties.LastWaitResult).Warning("non-zero last wait result")
					} else {
						exitCode = int(properties.ExitCode)
					}
				}
			}
		}
	}
	log.G(ctx).WithField("exitCode", exitCode).Debug("process exited")

	process.closedWaitOnce.Do(func() {
		process.exitCode = exitCode
		process.waitError = err
		close(process.waitBlock)
	})
	oc.SetSpanStatus(span, err)
}

// Wait waits for the process to exit. If the process has already exited returns
// the previous error (if any).
func (process *Process) Wait() error {
	<-process.waitBlock
	return process.waitError
}

// Exited returns if the process has stopped
func (process *Process) stopped() bool {
	select {
	case <-process.waitBlock:
		return true
	default:
		return false
	}
}

// ResizeConsole resizes the console of the process.
func (process *Process) ResizeConsole(ctx context.Context, width, height uint16) error {
	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	operation := "hcs::Process::ResizeConsole"

	if process.handle == 0 {
		return makeProcessError(process, operation, ErrAlreadyClosed, nil)
	}
	modifyRequest := hcsschema.ProcessModifyRequest{
		Operation: guestrequest.ModifyProcessConsoleSize,
		ConsoleSize: &hcsschema.ConsoleSize{
			Height: height,
			Width:  width,
		},
	}

	modifyRequestb, err := json.Marshal(modifyRequest)
	if err != nil {
		return err
	}

	resultJSON, err := vmcompute.HcsModifyProcess(ctx, process.handle, string(modifyRequestb))
	events := processHcsResult(ctx, resultJSON)
	if err != nil {
		return makeProcessError(process, operation, err, events)
	}

	return nil
}

// ExitCode returns the exit code of the process. The process must have
// already terminated.
func (process *Process) ExitCode() (int, error) {
	if !process.stopped() {
		return -1, makeProcessError(process, "hcs::Process::ExitCode", ErrInvalidProcessState, nil)
	}
	if process.waitError != nil {
		return -1, process.waitError
	}
	return process.exitCode, nil
}

// StdioLegacy returns the stdin, stdout, and stderr pipes, respectively. Closing
// these pipes does not close the underlying pipes. Once returned, these pipes
// are the responsibility of the caller to close.
func (process *Process) StdioLegacy() (_ io.WriteCloser, _ io.ReadCloser, _ io.ReadCloser, err error) {
	operation := "hcs::Process::StdioLegacy"
	ctx, span := oc.StartSpan(context.Background(), operation)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(
		trace.StringAttribute("cid", process.SystemID()),
		trace.Int64Attribute("pid", int64(process.processID)))

	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	if process.handle == 0 {
		return nil, nil, nil, makeProcessError(process, operation, ErrAlreadyClosed, nil)
	}

	process.stdioLock.Lock()
	defer process.stdioLock.Unlock()
	if process.hasCachedStdio {
		stdin, stdout, stderr := process.stdin, process.stdout, process.stderr
		process.stdin, process.stdout, process.stderr = nil, nil, nil
		process.hasCachedStdio = false
		return stdin, stdout, stderr, nil
	}

	processInfo, resultJSON, err := vmcompute.HcsGetProcessInfo(ctx, process.handle)
	events := processHcsResult(ctx, resultJSON)
	if err != nil {
		return nil, nil, nil, makeProcessError(process, operation, err, events)
	}

	pipes, err := makeOpenFiles([]syscall.Handle{processInfo.StdInput, processInfo.StdOutput, processInfo.StdError})
	if err != nil {
		return nil, nil, nil, makeProcessError(process, operation, err, nil)
	}

	return pipes[0], pipes[1], pipes[2], nil
}

// Stdio returns the stdin, stdout, and stderr pipes, respectively.
// To close them, close the process handle, or use the `CloseStd*` functions.
func (process *Process) Stdio() (stdin io.Writer, stdout, stderr io.Reader) {
	process.stdioLock.Lock()
	defer process.stdioLock.Unlock()
	return process.stdin, process.stdout, process.stderr
}

// CloseStdin closes the write side of the stdin pipe so that the process is
// notified on the read side that there is no more data in stdin.
func (process *Process) CloseStdin(ctx context.Context) (err error) {
	operation := "hcs::Process::CloseStdin"
	ctx, span := trace.StartSpan(ctx, operation)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(
		trace.StringAttribute("cid", process.SystemID()),
		trace.Int64Attribute("pid", int64(process.processID)))

	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	if process.handle == 0 {
		return makeProcessError(process, operation, ErrAlreadyClosed, nil)
	}

	//HcsModifyProcess request to close stdin will fail if the process has already exited
	if !process.stopped() {
		modifyRequest := hcsschema.ProcessModifyRequest{
			Operation: guestrequest.CloseProcessHandle,
			CloseHandle: &hcsschema.CloseHandle{
				Handle: guestrequest.STDInHandle,
			},
		}

		modifyRequestb, err := json.Marshal(modifyRequest)
		if err != nil {
			return err
		}

		resultJSON, err := vmcompute.HcsModifyProcess(ctx, process.handle, string(modifyRequestb))
		events := processHcsResult(ctx, resultJSON)
		if err != nil {
			return makeProcessError(process, operation, err, events)
		}
	}

	process.stdioLock.Lock()
	defer process.stdioLock.Unlock()
	if process.stdin != nil {
		process.stdin.Close()
		process.stdin = nil
	}

	return nil
}

func (process *Process) CloseStdout(ctx context.Context) (err error) {
	ctx, span := oc.StartSpan(ctx, "hcs::Process::CloseStdout") //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(
		trace.StringAttribute("cid", process.SystemID()),
		trace.Int64Attribute("pid", int64(process.processID)))

	process.handleLock.Lock()
	defer process.handleLock.Unlock()

	if process.handle == 0 {
		return nil
	}

	process.stdioLock.Lock()
	defer process.stdioLock.Unlock()
	if process.stdout != nil {
		process.stdout.Close()
		process.stdout = nil
	}
	return nil
}

func (process *Process) CloseStderr(ctx context.Context) (err error) {
	ctx, span := oc.StartSpan(ctx, "hcs::Process::CloseStderr") //nolint:ineffassign,staticcheck
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(
		trace.StringAttribute("cid", process.SystemID()),
		trace.Int64Attribute("pid", int64(process.processID)))

	process.handleLock.Lock()
	defer process.handleLock.Unlock()

	if process.handle == 0 {
		return nil
	}

	process.stdioLock.Lock()
	defer process.stdioLock.Unlock()
	if process.stderr != nil {
		process.stderr.Close()
		process.stderr = nil
	}
	return nil
}

// Close cleans up any state associated with the process but does not kill
// or wait on it.
func (process *Process) Close() (err error) {
	operation := "hcs::Process::Close"
	ctx, span := oc.StartSpan(context.Background(), operation)
	defer span.End()
	defer func() { oc.SetSpanStatus(span, err) }()
	span.AddAttributes(
		trace.StringAttribute("cid", process.SystemID()),
		trace.Int64Attribute("pid", int64(process.processID)))

	process.handleLock.Lock()
	defer process.handleLock.Unlock()

	// Don't double free this
	if process.handle == 0 {
		return nil
	}

	process.stdioLock.Lock()
	if process.stdin != nil {
		process.stdin.Close()
		process.stdin = nil
	}
	if process.stdout != nil {
		process.stdout.Close()
		process.stdout = nil
	}
	if process.stderr != nil {
		process.stderr.Close()
		process.stderr = nil
	}
	process.stdioLock.Unlock()

	if err = process.unregisterCallback(ctx); err != nil {
		return makeProcessError(process, operation, err, nil)
	}

	if err = vmcompute.HcsCloseProcess(ctx, process.handle); err != nil {
		return makeProcessError(process, operation, err, nil)
	}

	process.handle = 0
	process.closedWaitOnce.Do(func() {
		process.exitCode = -1
		process.waitError = ErrAlreadyClosed
		close(process.waitBlock)
	})

	return nil
}

func (process *Process) registerCallback(ctx context.Context) error {
	callbackContext := &notificationWatcherContext{
		channels:  newProcessChannels(),
		systemID:  process.SystemID(),
		processID: process.processID,
	}

	callbackMapLock.Lock()
	callbackNumber := nextCallback
	nextCallback++
	callbackMap[callbackNumber] = callbackContext
	callbackMapLock.Unlock()

	callbackHandle, err := vmcompute.HcsRegisterProcessCallback(ctx, process.handle, notificationWatcherCallback, callbackNumber)
	if err != nil {
		return err
	}
	callbackContext.handle = callbackHandle
	process.callbackNumber = callbackNumber

	return nil
}

func (process *Process) unregisterCallback(ctx context.Context) error {
	callbackNumber := process.callbackNumber

	callbackMapLock.RLock()
	callbackContext := callbackMap[callbackNumber]
	callbackMapLock.RUnlock()

	if callbackContext == nil {
		return nil
	}

	handle := callbackContext.handle

	if handle == 0 {
		return nil
	}

	// vmcompute.HcsUnregisterProcessCallback has its own synchronization to
	// wait for all callbacks to complete. We must NOT hold the callbackMapLock.
	err := vmcompute.HcsUnregisterProcessCallback(ctx, handle)
	if err != nil {
		return err
	}

	closeChannels(callbackContext.channels)

	callbackMapLock.Lock()
	delete(callbackMap, callbackNumber)
	callbackMapLock.Unlock()

	handle = 0 //nolint:ineffassign

	return nil
}
