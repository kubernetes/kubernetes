package hcs

import (
	"encoding/json"
	"io"
	"sync"
	"syscall"
	"time"

	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/Microsoft/hcsshim/internal/logfields"
	"github.com/sirupsen/logrus"
)

// ContainerError is an error encountered in HCS
type Process struct {
	handleLock     sync.RWMutex
	handle         hcsProcess
	processID      int
	system         *System
	cachedPipes    *cachedPipes
	callbackNumber uintptr

	logctx logrus.Fields

	closedWaitOnce sync.Once
	waitBlock      chan struct{}
	waitError      error
}

func newProcess(process hcsProcess, processID int, computeSystem *System) *Process {
	return &Process{
		handle:    process,
		processID: processID,
		system:    computeSystem,
		logctx: logrus.Fields{
			logfields.ContainerID: computeSystem.ID(),
			logfields.ProcessID:   processID,
		},
		waitBlock: make(chan struct{}),
	}
}

type cachedPipes struct {
	stdIn  syscall.Handle
	stdOut syscall.Handle
	stdErr syscall.Handle
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

type ProcessStatus struct {
	ProcessID      uint32
	Exited         bool
	ExitCode       uint32
	LastWaitResult int32
}

const (
	stdIn  string = "StdIn"
	stdOut string = "StdOut"
	stdErr string = "StdErr"
)

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

func (process *Process) logOperationBegin(operation string) {
	logOperationBegin(
		process.logctx,
		operation+" - Begin Operation")
}

func (process *Process) logOperationEnd(operation string, err error) {
	var result string
	if err == nil {
		result = "Success"
	} else {
		result = "Error"
	}

	logOperationEnd(
		process.logctx,
		operation+" - End Operation - "+result,
		err)
}

// Signal signals the process with `options`.
//
// For LCOW `guestrequest.SignalProcessOptionsLCOW`.
//
// For WCOW `guestrequest.SignalProcessOptionsWCOW`.
func (process *Process) Signal(options interface{}) (err error) {
	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	operation := "hcsshim::Process::Signal"
	process.logOperationBegin(operation)
	defer func() { process.logOperationEnd(operation, err) }()

	if process.handle == 0 {
		return makeProcessError(process, operation, ErrAlreadyClosed, nil)
	}

	optionsb, err := json.Marshal(options)
	if err != nil {
		return err
	}

	optionsStr := string(optionsb)

	var resultp *uint16
	syscallWatcher(process.logctx, func() {
		err = hcsSignalProcess(process.handle, optionsStr, &resultp)
	})
	events := processHcsResult(resultp)
	if err != nil {
		return makeProcessError(process, operation, err, events)
	}

	return nil
}

// Kill signals the process to terminate but does not wait for it to finish terminating.
func (process *Process) Kill() (err error) {
	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	operation := "hcsshim::Process::Kill"
	process.logOperationBegin(operation)
	defer func() { process.logOperationEnd(operation, err) }()

	if process.handle == 0 {
		return makeProcessError(process, operation, ErrAlreadyClosed, nil)
	}

	var resultp *uint16
	syscallWatcher(process.logctx, func() {
		err = hcsTerminateProcess(process.handle, &resultp)
	})
	events := processHcsResult(resultp)
	if err != nil {
		return makeProcessError(process, operation, err, events)
	}

	return nil
}

// waitBackground waits for the process exit notification. Once received sets
// `process.waitError` (if any) and unblocks all `Wait` and `WaitTimeout` calls.
//
// This MUST be called exactly once per `process.handle` but `Wait` and
// `WaitTimeout` are safe to call multiple times.
func (process *Process) waitBackground() {
	process.waitError = waitForNotification(process.callbackNumber, hcsNotificationProcessExited, nil)
	process.closedWaitOnce.Do(func() {
		close(process.waitBlock)
	})
}

// Wait waits for the process to exit. If the process has already exited returns
// the pervious error (if any).
func (process *Process) Wait() (err error) {
	operation := "hcsshim::Process::Wait"
	process.logOperationBegin(operation)
	defer func() { process.logOperationEnd(operation, err) }()

	<-process.waitBlock
	if process.waitError != nil {
		return makeProcessError(process, operation, process.waitError, nil)
	}
	return nil
}

// WaitTimeout waits for the process to exit or the duration to elapse. If the
// process has already exited returns the pervious error (if any). If a timeout
// occurs returns `ErrTimeout`.
func (process *Process) WaitTimeout(timeout time.Duration) (err error) {
	operation := "hcssshim::Process::WaitTimeout"
	process.logOperationBegin(operation)
	defer func() { process.logOperationEnd(operation, err) }()

	select {
	case <-process.waitBlock:
		if process.waitError != nil {
			return makeProcessError(process, operation, process.waitError, nil)
		}
		return nil
	case <-time.After(timeout):
		return makeProcessError(process, operation, ErrTimeout, nil)
	}
}

// ResizeConsole resizes the console of the process.
func (process *Process) ResizeConsole(width, height uint16) (err error) {
	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	operation := "hcsshim::Process::ResizeConsole"
	process.logOperationBegin(operation)
	defer func() { process.logOperationEnd(operation, err) }()

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

	modifyRequestStr := string(modifyRequestb)

	var resultp *uint16
	err = hcsModifyProcess(process.handle, modifyRequestStr, &resultp)
	events := processHcsResult(resultp)
	if err != nil {
		return makeProcessError(process, operation, err, events)
	}

	return nil
}

func (process *Process) Properties() (_ *ProcessStatus, err error) {
	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	operation := "hcsshim::Process::Properties"
	process.logOperationBegin(operation)
	defer func() { process.logOperationEnd(operation, err) }()

	if process.handle == 0 {
		return nil, makeProcessError(process, operation, ErrAlreadyClosed, nil)
	}

	var (
		resultp     *uint16
		propertiesp *uint16
	)
	syscallWatcher(process.logctx, func() {
		err = hcsGetProcessProperties(process.handle, &propertiesp, &resultp)
	})
	events := processHcsResult(resultp)
	if err != nil {
		return nil, makeProcessError(process, operation, err, events)
	}

	if propertiesp == nil {
		return nil, ErrUnexpectedValue
	}
	propertiesRaw := interop.ConvertAndFreeCoTaskMemBytes(propertiesp)

	properties := &ProcessStatus{}
	if err := json.Unmarshal(propertiesRaw, properties); err != nil {
		return nil, makeProcessError(process, operation, err, nil)
	}

	return properties, nil
}

// ExitCode returns the exit code of the process. The process must have
// already terminated.
func (process *Process) ExitCode() (_ int, err error) {
	operation := "hcsshim::Process::ExitCode"
	process.logOperationBegin(operation)
	defer func() { process.logOperationEnd(operation, err) }()

	properties, err := process.Properties()
	if err != nil {
		return -1, makeProcessError(process, operation, err, nil)
	}

	if properties.Exited == false {
		return -1, makeProcessError(process, operation, ErrInvalidProcessState, nil)
	}

	if properties.LastWaitResult != 0 {
		logrus.WithFields(logrus.Fields{
			logfields.ContainerID: process.SystemID(),
			logfields.ProcessID:   process.processID,
			"wait-result":         properties.LastWaitResult,
		}).Warn("hcsshim::Process::ExitCode - Non-zero last wait result")
		return -1, nil
	}

	return int(properties.ExitCode), nil
}

// Stdio returns the stdin, stdout, and stderr pipes, respectively. Closing
// these pipes does not close the underlying pipes; it should be possible to
// call this multiple times to get multiple interfaces.
func (process *Process) Stdio() (_ io.WriteCloser, _ io.ReadCloser, _ io.ReadCloser, err error) {
	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	operation := "hcsshim::Process::Stdio"
	process.logOperationBegin(operation)
	defer func() { process.logOperationEnd(operation, err) }()

	if process.handle == 0 {
		return nil, nil, nil, makeProcessError(process, operation, ErrAlreadyClosed, nil)
	}

	var stdIn, stdOut, stdErr syscall.Handle

	if process.cachedPipes == nil {
		var (
			processInfo hcsProcessInformation
			resultp     *uint16
		)
		err = hcsGetProcessInfo(process.handle, &processInfo, &resultp)
		events := processHcsResult(resultp)
		if err != nil {
			return nil, nil, nil, makeProcessError(process, operation, err, events)
		}

		stdIn, stdOut, stdErr = processInfo.StdInput, processInfo.StdOutput, processInfo.StdError
	} else {
		// Use cached pipes
		stdIn, stdOut, stdErr = process.cachedPipes.stdIn, process.cachedPipes.stdOut, process.cachedPipes.stdErr

		// Invalidate the cache
		process.cachedPipes = nil
	}

	pipes, err := makeOpenFiles([]syscall.Handle{stdIn, stdOut, stdErr})
	if err != nil {
		return nil, nil, nil, makeProcessError(process, operation, err, nil)
	}

	return pipes[0], pipes[1], pipes[2], nil
}

// CloseStdin closes the write side of the stdin pipe so that the process is
// notified on the read side that there is no more data in stdin.
func (process *Process) CloseStdin() (err error) {
	process.handleLock.RLock()
	defer process.handleLock.RUnlock()

	operation := "hcsshim::Process::CloseStdin"
	process.logOperationBegin(operation)
	defer func() { process.logOperationEnd(operation, err) }()

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

	modifyRequestStr := string(modifyRequestb)

	var resultp *uint16
	err = hcsModifyProcess(process.handle, modifyRequestStr, &resultp)
	events := processHcsResult(resultp)
	if err != nil {
		return makeProcessError(process, operation, err, events)
	}

	return nil
}

// Close cleans up any state associated with the process but does not kill
// or wait on it.
func (process *Process) Close() (err error) {
	process.handleLock.Lock()
	defer process.handleLock.Unlock()

	operation := "hcsshim::Process::Close"
	process.logOperationBegin(operation)
	defer func() { process.logOperationEnd(operation, err) }()

	// Don't double free this
	if process.handle == 0 {
		return nil
	}

	if err = process.unregisterCallback(); err != nil {
		return makeProcessError(process, operation, err, nil)
	}

	if err = hcsCloseProcess(process.handle); err != nil {
		return makeProcessError(process, operation, err, nil)
	}

	process.handle = 0
	process.closedWaitOnce.Do(func() {
		close(process.waitBlock)
	})

	return nil
}

func (process *Process) registerCallback() error {
	context := &notifcationWatcherContext{
		channels:  newProcessChannels(),
		systemID:  process.SystemID(),
		processID: process.processID,
	}

	callbackMapLock.Lock()
	callbackNumber := nextCallback
	nextCallback++
	callbackMap[callbackNumber] = context
	callbackMapLock.Unlock()

	var callbackHandle hcsCallback
	err := hcsRegisterProcessCallback(process.handle, notificationWatcherCallback, callbackNumber, &callbackHandle)
	if err != nil {
		return err
	}
	context.handle = callbackHandle
	process.callbackNumber = callbackNumber

	return nil
}

func (process *Process) unregisterCallback() error {
	callbackNumber := process.callbackNumber

	callbackMapLock.RLock()
	context := callbackMap[callbackNumber]
	callbackMapLock.RUnlock()

	if context == nil {
		return nil
	}

	handle := context.handle

	if handle == 0 {
		return nil
	}

	// hcsUnregisterProcessCallback has its own syncronization
	// to wait for all callbacks to complete. We must NOT hold the callbackMapLock.
	err := hcsUnregisterProcessCallback(handle)
	if err != nil {
		return err
	}

	closeChannels(context.channels)

	callbackMapLock.Lock()
	delete(callbackMap, callbackNumber)
	callbackMapLock.Unlock()

	handle = 0

	return nil
}
