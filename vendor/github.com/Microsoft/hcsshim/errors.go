package hcsshim

import (
	"fmt"
	"syscall"

	"github.com/Microsoft/hcsshim/internal/hns"

	"github.com/Microsoft/hcsshim/internal/hcs"
	"github.com/Microsoft/hcsshim/internal/hcserror"
)

var (
	// ErrComputeSystemDoesNotExist is an error encountered when the container being operated on no longer exists = hcs.exist
	ErrComputeSystemDoesNotExist = hcs.ErrComputeSystemDoesNotExist

	// ErrElementNotFound is an error encountered when the object being referenced does not exist
	ErrElementNotFound = hcs.ErrElementNotFound

	// ErrElementNotFound is an error encountered when the object being referenced does not exist
	ErrNotSupported = hcs.ErrNotSupported

	// ErrInvalidData is an error encountered when the request being sent to hcs is invalid/unsupported
	// decimal -2147024883 / hex 0x8007000d
	ErrInvalidData = hcs.ErrInvalidData

	// ErrHandleClose is an error encountered when the handle generating the notification being waited on has been closed
	ErrHandleClose = hcs.ErrHandleClose

	// ErrAlreadyClosed is an error encountered when using a handle that has been closed by the Close method
	ErrAlreadyClosed = hcs.ErrAlreadyClosed

	// ErrInvalidNotificationType is an error encountered when an invalid notification type is used
	ErrInvalidNotificationType = hcs.ErrInvalidNotificationType

	// ErrInvalidProcessState is an error encountered when the process is not in a valid state for the requested operation
	ErrInvalidProcessState = hcs.ErrInvalidProcessState

	// ErrTimeout is an error encountered when waiting on a notification times out
	ErrTimeout = hcs.ErrTimeout

	// ErrUnexpectedContainerExit is the error encountered when a container exits while waiting for
	// a different expected notification
	ErrUnexpectedContainerExit = hcs.ErrUnexpectedContainerExit

	// ErrUnexpectedProcessAbort is the error encountered when communication with the compute service
	// is lost while waiting for a notification
	ErrUnexpectedProcessAbort = hcs.ErrUnexpectedProcessAbort

	// ErrUnexpectedValue is an error encountered when hcs returns an invalid value
	ErrUnexpectedValue = hcs.ErrUnexpectedValue

	// ErrVmcomputeAlreadyStopped is an error encountered when a shutdown or terminate request is made on a stopped container
	ErrVmcomputeAlreadyStopped = hcs.ErrVmcomputeAlreadyStopped

	// ErrVmcomputeOperationPending is an error encountered when the operation is being completed asynchronously
	ErrVmcomputeOperationPending = hcs.ErrVmcomputeOperationPending

	// ErrVmcomputeOperationInvalidState is an error encountered when the compute system is not in a valid state for the requested operation
	ErrVmcomputeOperationInvalidState = hcs.ErrVmcomputeOperationInvalidState

	// ErrProcNotFound is an error encountered when the the process cannot be found
	ErrProcNotFound = hcs.ErrProcNotFound

	// ErrVmcomputeOperationAccessIsDenied is an error which can be encountered when enumerating compute systems in RS1/RS2
	// builds when the underlying silo might be in the process of terminating. HCS was fixed in RS3.
	ErrVmcomputeOperationAccessIsDenied = hcs.ErrVmcomputeOperationAccessIsDenied

	// ErrVmcomputeInvalidJSON is an error encountered when the compute system does not support/understand the messages sent by management
	ErrVmcomputeInvalidJSON = hcs.ErrVmcomputeInvalidJSON

	// ErrVmcomputeUnknownMessage is an error encountered guest compute system doesn't support the message
	ErrVmcomputeUnknownMessage = hcs.ErrVmcomputeUnknownMessage

	// ErrNotSupported is an error encountered when hcs doesn't support the request
	ErrPlatformNotSupported = hcs.ErrPlatformNotSupported
)

type EndpointNotFoundError = hns.EndpointNotFoundError
type NetworkNotFoundError = hns.NetworkNotFoundError

// ProcessError is an error encountered in HCS during an operation on a Process object
type ProcessError struct {
	Process   *process
	Operation string
	ExtraInfo string
	Err       error
	Events    []hcs.ErrorEvent
}

// ContainerError is an error encountered in HCS during an operation on a Container object
type ContainerError struct {
	Container *container
	Operation string
	ExtraInfo string
	Err       error
	Events    []hcs.ErrorEvent
}

func (e *ContainerError) Error() string {
	if e == nil {
		return "<nil>"
	}

	if e.Container == nil {
		return "unexpected nil container for error: " + e.Err.Error()
	}

	s := "container " + e.Container.system.ID()

	if e.Operation != "" {
		s += " encountered an error during " + e.Operation
	}

	switch e.Err.(type) {
	case nil:
		break
	case syscall.Errno:
		s += fmt.Sprintf(": failure in a Windows system call: %s (0x%x)", e.Err, hcserror.Win32FromError(e.Err))
	default:
		s += fmt.Sprintf(": %s", e.Err.Error())
	}

	for _, ev := range e.Events {
		s += "\n" + ev.String()
	}

	if e.ExtraInfo != "" {
		s += " extra info: " + e.ExtraInfo
	}

	return s
}

func makeContainerError(container *container, operation string, extraInfo string, err error) error {
	// Don't double wrap errors
	if _, ok := err.(*ContainerError); ok {
		return err
	}
	containerError := &ContainerError{Container: container, Operation: operation, ExtraInfo: extraInfo, Err: err}
	return containerError
}

func (e *ProcessError) Error() string {
	if e == nil {
		return "<nil>"
	}

	if e.Process == nil {
		return "Unexpected nil process for error: " + e.Err.Error()
	}

	s := fmt.Sprintf("process %d in container %s", e.Process.p.Pid(), e.Process.p.SystemID())
	if e.Operation != "" {
		s += " encountered an error during " + e.Operation
	}

	switch e.Err.(type) {
	case nil:
		break
	case syscall.Errno:
		s += fmt.Sprintf(": failure in a Windows system call: %s (0x%x)", e.Err, hcserror.Win32FromError(e.Err))
	default:
		s += fmt.Sprintf(": %s", e.Err.Error())
	}

	for _, ev := range e.Events {
		s += "\n" + ev.String()
	}

	return s
}

func makeProcessError(process *process, operation string, extraInfo string, err error) error {
	// Don't double wrap errors
	if _, ok := err.(*ProcessError); ok {
		return err
	}
	processError := &ProcessError{Process: process, Operation: operation, ExtraInfo: extraInfo, Err: err}
	return processError
}

// IsNotExist checks if an error is caused by the Container or Process not existing.
// Note: Currently, ErrElementNotFound can mean that a Process has either
// already exited, or does not exist. Both IsAlreadyStopped and IsNotExist
// will currently return true when the error is ErrElementNotFound or ErrProcNotFound.
func IsNotExist(err error) bool {
	if _, ok := err.(EndpointNotFoundError); ok {
		return true
	}
	if _, ok := err.(NetworkNotFoundError); ok {
		return true
	}
	return hcs.IsNotExist(getInnerError(err))
}

// IsAlreadyClosed checks if an error is caused by the Container or Process having been
// already closed by a call to the Close() method.
func IsAlreadyClosed(err error) bool {
	return hcs.IsAlreadyClosed(getInnerError(err))
}

// IsPending returns a boolean indicating whether the error is that
// the requested operation is being completed in the background.
func IsPending(err error) bool {
	return hcs.IsPending(getInnerError(err))
}

// IsTimeout returns a boolean indicating whether the error is caused by
// a timeout waiting for the operation to complete.
func IsTimeout(err error) bool {
	return hcs.IsTimeout(getInnerError(err))
}

// IsAlreadyStopped returns a boolean indicating whether the error is caused by
// a Container or Process being already stopped.
// Note: Currently, ErrElementNotFound can mean that a Process has either
// already exited, or does not exist. Both IsAlreadyStopped and IsNotExist
// will currently return true when the error is ErrElementNotFound or ErrProcNotFound.
func IsAlreadyStopped(err error) bool {
	return hcs.IsAlreadyStopped(getInnerError(err))
}

// IsNotSupported returns a boolean indicating whether the error is caused by
// unsupported platform requests
// Note: Currently Unsupported platform requests can be mean either
// ErrVmcomputeInvalidJSON, ErrInvalidData, ErrNotSupported or ErrVmcomputeUnknownMessage
// is thrown from the Platform
func IsNotSupported(err error) bool {
	return hcs.IsNotSupported(getInnerError(err))
}

func getInnerError(err error) error {
	switch pe := err.(type) {
	case nil:
		return nil
	case *ContainerError:
		err = pe.Err
	case *ProcessError:
		err = pe.Err
	}
	return err
}

func convertSystemError(err error, c *container) error {
	if serr, ok := err.(*hcs.SystemError); ok {
		return &ContainerError{Container: c, Operation: serr.Op, ExtraInfo: serr.Extra, Err: serr.Err, Events: serr.Events}
	}
	return err
}

func convertProcessError(err error, p *process) error {
	if perr, ok := err.(*hcs.ProcessError); ok {
		return &ProcessError{Process: p, Operation: perr.Op, Err: perr.Err, Events: perr.Events}
	}
	return err
}
