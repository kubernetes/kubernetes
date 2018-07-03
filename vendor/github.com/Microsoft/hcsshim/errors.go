package hcsshim

import (
	"errors"
	"fmt"
	"syscall"
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

	// ErrNotSupported is an error encountered when hcs doesn't support the request
	ErrPlatformNotSupported = errors.New("unsupported platform request")
)

type EndpointNotFoundError struct {
	EndpointName string
}

func (e EndpointNotFoundError) Error() string {
	return fmt.Sprintf("Endpoint %s not found", e.EndpointName)
}

type NetworkNotFoundError struct {
	NetworkName string
}

func (e NetworkNotFoundError) Error() string {
	return fmt.Sprintf("Network %s not found", e.NetworkName)
}

// ProcessError is an error encountered in HCS during an operation on a Process object
type ProcessError struct {
	Process   *process
	Operation string
	ExtraInfo string
	Err       error
}

// ContainerError is an error encountered in HCS during an operation on a Container object
type ContainerError struct {
	Container *container
	Operation string
	ExtraInfo string
	Err       error
}

func (e *ContainerError) Error() string {
	if e == nil {
		return "<nil>"
	}

	if e.Container == nil {
		return "unexpected nil container for error: " + e.Err.Error()
	}

	s := "container " + e.Container.id

	if e.Operation != "" {
		s += " encountered an error during " + e.Operation
	}

	switch e.Err.(type) {
	case nil:
		break
	case syscall.Errno:
		s += fmt.Sprintf(": failure in a Windows system call: %s (0x%x)", e.Err, win32FromError(e.Err))
	default:
		s += fmt.Sprintf(": %s", e.Err.Error())
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

	s := fmt.Sprintf("process %d", e.Process.processID)

	if e.Process.container != nil {
		s += " in container " + e.Process.container.id
	}

	if e.Operation != "" {
		s += " encountered an error during " + e.Operation
	}

	switch e.Err.(type) {
	case nil:
		break
	case syscall.Errno:
		s += fmt.Sprintf(": failure in a Windows system call: %s (0x%x)", e.Err, win32FromError(e.Err))
	default:
		s += fmt.Sprintf(": %s", e.Err.Error())
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
	err = getInnerError(err)
	if _, ok := err.(EndpointNotFoundError); ok {
		return true
	}
	if _, ok := err.(NetworkNotFoundError); ok {
		return true
	}
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
