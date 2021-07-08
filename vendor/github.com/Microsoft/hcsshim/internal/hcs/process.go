package hcs

import (
	"context"
	"encoding/json"
	"io"
	"sync"
	"syscall"
	"time"

	"github.com/Microsoft/hcsshim/internal/log"
	"github.com/Microsoft/hcsshim/internal/oc"
	"github.com/Microsoft/hcsshim/internal/vmcompute"
	"go.opencensus.io/trace"
)

// ContainerError is an error encountered in HCS
type Process struct {
	handleLock     sync.RWMutex
	handle         vmcompute.HcsProcess
	processID      int
	system         *System
	hasCachedStdio bool
	stdioLock      sync.Mutex
	stdin          io.WriteCloser
	stdout         io.ReadCloser
	stderr         io.ReadCloser
	callbackNumber uintptr

	closedWaitOnce sync.Once
	waitBlock      chan struct{}
	exitCode       int
	waitError      error
}

func newProcess(process vmcompute.HcsProcess, processID int, computeSystem *System) *Process {
	return &Process{
		handle:    process,
		processID: processID,
		system:    computeSystem,
		waitBlock: make(chan struct{}),
	}
}

type processModifyRequest struct {
	Operation   string
	ConsoleSize *consoleSize `json:",omitempty"`
	CloseHandle *closeHandle `json:",omitempty"`
}

type consoleSize struct {
	Height uint16
	Width  uint16
}

type closeHandle struct {
	Handle string
}

type processStatus struct {
	ProcessID      uint32
	Exited         bool
	ExitCode       uint32
	LastWaitResult int32
}

const stdIn string = "StdIn"

const (
	modifyConsoleSize string = "ConsoleSize"
	modifyCloseHandle string = "CloseHandle"
)

// Pid returns the process ID of the process within the container.
func (process *Process) Pid() int {
	return process.processID
}

// SystemID returns the ID of the process's compute system.
func (process *Process) SystemID() string {
	return process.system.ID()
}

func (process *Process) processSignalResult(ctx context.Context, err error) (bool, error) {
	switch err {
	case nil:
		return true, nil
	case ErrVmcomputeOperationInvalidState, ErrComputeSystemDoesNotExist, ErrElementNotFound:
		select {
		case <-process.waitBlock:
			// The process exit notification has already arrived.
		default:
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
// For LCOW `guestrequest.SignalProcessOptionsLCOW`.
//
// For WCOW `guestrequest.SignalProcessOptionsWCOW`.
func (process *Process) Signal(ctx context.Context, options interface{}) (bool, error) {
	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	operation := "hcsshim::Process::Signal"

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

	operation := "hcsshim::Process::Kill"

	if process.handle == 0 {
		return false, makeProcessError(process, operation, ErrAlreadyClosed, nil)
	}

	resultJSON, err := vmcompute.HcsTerminateProcess(ctx, process.handle)
	events := processHcsResult(ctx, resultJSON)
	delivered, err := process.processSignalResult(ctx, err)
	if err != nil {
		err = makeProcessError(process, operation, err, events)
	}
	return delivered, err
}

// waitBackground waits for the process exit notification. Once received sets
// `process.waitError` (if any) and unblocks all `Wait` calls.
//
// This MUST be called exactly once per `process.handle` but `Wait` is safe to
// call multiple times.
func (process *Process) waitBackground() {
	operation := "hcsshim::Process::waitBackground"
	ctx, span := trace.StartSpan(context.Background(), operation)
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

		// Make sure we didnt race with Close() here
		if process.handle != 0 {
			propertiesJSON, resultJSON, err = vmcompute.HcsGetProcessProperties(ctx, process.handle)
			events := processHcsResult(ctx, resultJSON)
			if err != nil {
				err = makeProcessError(process, operation, err, events) //nolint:ineffassign
			} else {
				properties := &processStatus{}
				err = json.Unmarshal([]byte(propertiesJSON), properties)
				if err != nil {
					err = makeProcessError(process, operation, err, nil) //nolint:ineffassign
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
// the pervious error (if any).
func (process *Process) Wait() error {
	<-process.waitBlock
	return process.waitError
}

// ResizeConsole resizes the console of the process.
func (process *Process) ResizeConsole(ctx context.Context, width, height uint16) error {
	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	operation := "hcsshim::Process::ResizeConsole"

	if process.handle == 0 {
		return makeProcessError(process, operation, ErrAlreadyClosed, nil)
	}

	modifyRequest := processModifyRequest{
		Operation: modifyConsoleSize,
		ConsoleSize: &consoleSize{
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
	select {
	case <-process.waitBlock:
		if process.waitError != nil {
			return -1, process.waitError
		}
		return process.exitCode, nil
	default:
		return -1, makeProcessError(process, "hcsshim::Process::ExitCode", ErrInvalidProcessState, nil)
	}
}

// StdioLegacy returns the stdin, stdout, and stderr pipes, respectively. Closing
// these pipes does not close the underlying pipes. Once returned, these pipes
// are the responsibility of the caller to close.
func (process *Process) StdioLegacy() (_ io.WriteCloser, _ io.ReadCloser, _ io.ReadCloser, err error) {
	operation := "hcsshim::Process::StdioLegacy"
	ctx, span := trace.StartSpan(context.Background(), operation)
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
// To close them, close the process handle.
func (process *Process) Stdio() (stdin io.Writer, stdout, stderr io.Reader) {
	process.stdioLock.Lock()
	defer process.stdioLock.Unlock()
	return process.stdin, process.stdout, process.stderr
}

// CloseStdin closes the write side of the stdin pipe so that the process is
// notified on the read side that there is no more data in stdin.
func (process *Process) CloseStdin(ctx context.Context) error {
	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	operation := "hcsshim::Process::CloseStdin"

	if process.handle == 0 {
		return makeProcessError(process, operation, ErrAlreadyClosed, nil)
	}

	modifyRequest := processModifyRequest{
		Operation: modifyCloseHandle,
		CloseHandle: &closeHandle{
			Handle: stdIn,
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

	process.stdioLock.Lock()
	if process.stdin != nil {
		process.stdin.Close()
		process.stdin = nil
	}
	process.stdioLock.Unlock()

	return nil
}

// Close cleans up any state associated with the process but does not kill
// or wait on it.
func (process *Process) Close() (err error) {
	operation := "hcsshim::Process::Close"
	ctx, span := trace.StartSpan(context.Background(), operation)
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
	callbackContext := &notifcationWatcherContext{
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
