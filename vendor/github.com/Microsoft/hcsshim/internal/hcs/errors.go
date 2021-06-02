package hcs

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net"
	"syscall"

	"github.com/Microsoft/hcsshim/internal/log"
)

var (
	// ErrComputeSystemDoesNotExist is an error encountered when the container being operated on no longer exists
	ErrComputeSystemDoesNotExist = syscall.Errno(0xc037010e)

	// ErrElementNotFound is an error encountered when the object being referenced does not exist
	ErrElementNotFound = syscall.Errno(0x490)

	// ErrElementNotFound is an error encountered when the object being referenced does not exist
	ErrNotSupported = syscall.Errno(0x32)

	// ErrInvalidData is an error encountered when the request being sent to hcs is invalid/unsupported
	// decimal -2147024883 / hex 0x8007000d
	ErrInvalidData = syscall.Errno(0xd)

	// ErrHandleClose is an error encountered when the handle generating the notification being waited on has been closed
	ErrHandleClose = errors.New("hcsshim: the handle generating this notification has been closed")

	// ErrAlreadyClosed is an error encountered when using a handle that has been closed by the Close method
	ErrAlreadyClosed = errors.New("hcsshim: the handle has already been closed")

	// ErrInvalidNotificationType is an error encountered when an invalid notification type is used
	ErrInvalidNotificationType = errors.New("hcsshim: invalid notification type")

	// ErrInvalidProcessState is an error encountered when the process is not in a valid state for the requested operation
	ErrInvalidProcessState = errors.New("the process is in an invalid state for the attempted operation")

	// ErrTimeout is an error encountered when waiting on a notification times out
	ErrTimeout = errors.New("hcsshim: timeout waiting for notification")

	// ErrUnexpectedContainerExit is the error encountered when a container exits while waiting for
	// a different expected notification
	ErrUnexpectedContainerExit = errors.New("unexpected container exit")

	// ErrUnexpectedProcessAbort is the error encountered when communication with the compute service
	// is lost while waiting for a notification
	ErrUnexpectedProcessAbort = errors.New("lost communication with compute service")

	// ErrUnexpectedValue is an error encountered when hcs returns an invalid value
	ErrUnexpectedValue = errors.New("unexpected value returned from hcs")

	// ErrVmcomputeAlreadyStopped is an error encountered when a shutdown or terminate request is made on a stopped container
	ErrVmcomputeAlreadyStopped = syscall.Errno(0xc0370110)

	// ErrVmcomputeOperationPending is an error encountered when the operation is being completed asynchronously
	ErrVmcomputeOperationPending = syscall.Errno(0xC0370103)

	// ErrVmcomputeOperationInvalidState is an error encountered when the compute system is not in a valid state for the requested operation
	ErrVmcomputeOperationInvalidState = syscall.Errno(0xc0370105)

	// ErrProcNotFound is an error encountered when the the process cannot be found
	ErrProcNotFound = syscall.Errno(0x7f)

	// ErrVmcomputeOperationAccessIsDenied is an error which can be encountered when enumerating compute systems in RS1/RS2
	// builds when the underlying silo might be in the process of terminating. HCS was fixed in RS3.
	ErrVmcomputeOperationAccessIsDenied = syscall.Errno(0x5)

	// ErrVmcomputeInvalidJSON is an error encountered when the compute system does not support/understand the messages sent by management
	ErrVmcomputeInvalidJSON = syscall.Errno(0xc037010d)

	// ErrVmcomputeUnknownMessage is an error encountered guest compute system doesn't support the message
	ErrVmcomputeUnknownMessage = syscall.Errno(0xc037010b)

	// ErrVmcomputeUnexpectedExit is an error encountered when the compute system terminates unexpectedly
	ErrVmcomputeUnexpectedExit = syscall.Errno(0xC0370106)

	// ErrNotSupported is an error encountered when hcs doesn't support the request
	ErrPlatformNotSupported = errors.New("unsupported platform request")
)

type ErrorEvent struct {
	Message    string `json:"Message,omitempty"`    // Fully formated error message
	StackTrace string `json:"StackTrace,omitempty"` // Stack trace in string form
	Provider   string `json:"Provider,omitempty"`
	EventID    uint16 `json:"EventId,omitempty"`
	Flags      uint32 `json:"Flags,omitempty"`
	Source     string `json:"Source,omitempty"`
	//Data       []EventData `json:"Data,omitempty"`  // Omit this as HCS doesn't encode this well. It's more confusing to include. It is however logged in debug mode (see processHcsResult function)
}

type hcsResult struct {
	Error        int32
	ErrorMessage string
	ErrorEvents  []ErrorEvent `json:"ErrorEvents,omitempty"`
}

func (ev *ErrorEvent) String() string {
	evs := "[Event Detail: " + ev.Message
	if ev.StackTrace != "" {
		evs += " Stack Trace: " + ev.StackTrace
	}
	if ev.Provider != "" {
		evs += " Provider: " + ev.Provider
	}
	if ev.EventID != 0 {
		evs = fmt.Sprintf("%s EventID: %d", evs, ev.EventID)
	}
	if ev.Flags != 0 {
		evs = fmt.Sprintf("%s flags: %d", evs, ev.Flags)
	}
	if ev.Source != "" {
		evs += " Source: " + ev.Source
	}
	evs += "]"
	return evs
}

func processHcsResult(ctx context.Context, resultJSON string) []ErrorEvent {
	if resultJSON != "" {
		result := &hcsResult{}
		if err := json.Unmarshal([]byte(resultJSON), result); err != nil {
			log.G(ctx).WithError(err).Warning("Could not unmarshal HCS result")
			return nil
		}
		return result.ErrorEvents
	}
	return nil
}

type HcsError struct {
	Op     string
	Err    error
	Events []ErrorEvent
}

var _ net.Error = &HcsError{}

func (e *HcsError) Error() string {
	s := e.Op + ": " + e.Err.Error()
	for _, ev := range e.Events {
		s += "\n" + ev.String()
	}
	return s
}

func (e *HcsError) Temporary() bool {
	err, ok := e.Err.(net.Error)
	return ok && err.Temporary()
}

func (e *HcsError) Timeout() bool {
	err, ok := e.Err.(net.Error)
	return ok && err.Timeout()
}

// ProcessError is an error encountered in HCS during an operation on a Process object
type ProcessError struct {
	SystemID string
	Pid      int
	Op       string
	Err      error
	Events   []ErrorEvent
}

var _ net.Error = &ProcessError{}

// SystemError is an error encountered in HCS during an operation on a Container object
type SystemError struct {
	ID     string
	Op     string
	Err    error
	Extra  string
	Events []ErrorEvent
}

var _ net.Error = &SystemError{}

func (e *SystemError) Error() string {
	s := e.Op + " " + e.ID + ": " + e.Err.Error()
	for _, ev := range e.Events {
		s += "\n" + ev.String()
	}
	if e.Extra != "" {
		s += "\n(extra info: " + e.Extra + ")"
	}
	return s
}

func (e *SystemError) Temporary() bool {
	err, ok := e.Err.(net.Error)
	return ok && err.Temporary()
}

func (e *SystemError) Timeout() bool {
	err, ok := e.Err.(net.Error)
	return ok && err.Timeout()
}

func makeSystemError(system *System, op string, extra string, err error, events []ErrorEvent) error {
	// Don't double wrap errors
	if _, ok := err.(*SystemError); ok {
		return err
	}
	return &SystemError{
		ID:     system.ID(),
		Op:     op,
		Extra:  extra,
		Err:    err,
		Events: events,
	}
}

func (e *ProcessError) Error() string {
	s := fmt.Sprintf("%s %s:%d: %s", e.Op, e.SystemID, e.Pid, e.Err.Error())
	for _, ev := range e.Events {
		s += "\n" + ev.String()
	}
	return s
}

func (e *ProcessError) Temporary() bool {
	err, ok := e.Err.(net.Error)
	return ok && err.Temporary()
}

func (e *ProcessError) Timeout() bool {
	err, ok := e.Err.(net.Error)
	return ok && err.Timeout()
}

func makeProcessError(process *Process, op string, err error, events []ErrorEvent) error {
	// Don't double wrap errors
	if _, ok := err.(*ProcessError); ok {
		return err
	}
	return &ProcessError{
		Pid:      process.Pid(),
		SystemID: process.SystemID(),
		Op:       op,
		Err:      err,
		Events:   events,
	}
}

// IsNotExist checks if an error is caused by the Container or Process not existing.
// Note: Currently, ErrElementNotFound can mean that a Process has either
// already exited, or does not exist. Both IsAlreadyStopped and IsNotExist
// will currently return true when the error is ErrElementNotFound or ErrProcNotFound.
func IsNotExist(err error) bool {
	err = getInnerError(err)
	return err == ErrComputeSystemDoesNotExist ||
		err == ErrElementNotFound ||
		err == ErrProcNotFound
}

// IsAlreadyClosed checks if an error is caused by the Container or Process having been
// already closed by a call to the Close() method.
func IsAlreadyClosed(err error) bool {
	err = getInnerError(err)
	return err == ErrAlreadyClosed
}

// IsPending returns a boolean indicating whether the error is that
// the requested operation is being completed in the background.
func IsPending(err error) bool {
	err = getInnerError(err)
	return err == ErrVmcomputeOperationPending
}

// IsTimeout returns a boolean indicating whether the error is caused by
// a timeout waiting for the operation to complete.
func IsTimeout(err error) bool {
	if err, ok := err.(net.Error); ok && err.Timeout() {
		return true
	}
	err = getInnerError(err)
	return err == ErrTimeout
}

// IsAlreadyStopped returns a boolean indicating whether the error is caused by
// a Container or Process being already stopped.
// Note: Currently, ErrElementNotFound can mean that a Process has either
// already exited, or does not exist. Both IsAlreadyStopped and IsNotExist
// will currently return true when the error is ErrElementNotFound or ErrProcNotFound.
func IsAlreadyStopped(err error) bool {
	err = getInnerError(err)
	return err == ErrVmcomputeAlreadyStopped ||
		err == ErrElementNotFound ||
		err == ErrProcNotFound
}

// IsNotSupported returns a boolean indicating whether the error is caused by
// unsupported platform requests
// Note: Currently Unsupported platform requests can be mean either
// ErrVmcomputeInvalidJSON, ErrInvalidData, ErrNotSupported or ErrVmcomputeUnknownMessage
// is thrown from the Platform
func IsNotSupported(err error) bool {
	err = getInnerError(err)
	// If Platform doesn't recognize or support the request sent, below errors are seen
	return err == ErrVmcomputeInvalidJSON ||
		err == ErrInvalidData ||
		err == ErrNotSupported ||
		err == ErrVmcomputeUnknownMessage
}

// IsOperationInvalidState returns true when err is caused by
// `ErrVmcomputeOperationInvalidState`.
func IsOperationInvalidState(err error) bool {
	err = getInnerError(err)
	return err == ErrVmcomputeOperationInvalidState
}

func getInnerError(err error) error {
	switch pe := err.(type) {
	case nil:
		return nil
	case *HcsError:
		err = pe.Err
	case *SystemError:
		err = pe.Err
	case *ProcessError:
		err = pe.Err
	}
	return err
}

func getOperationLogResult(err error) (string, error) {
	switch err {
	case nil:
		return "Success", nil
	default:
		return "Error", err
	}
}
