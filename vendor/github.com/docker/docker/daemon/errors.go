package daemon

import (
	"fmt"
	"strings"
	"syscall"

	"github.com/pkg/errors"
	"google.golang.org/grpc"
)

func errNotRunning(id string) error {
	return stateConflictError{errors.Errorf("Container %s is not running", id)}
}

func containerNotFound(id string) error {
	return objNotFoundError{"container", id}
}

func volumeNotFound(id string) error {
	return objNotFoundError{"volume", id}
}

type objNotFoundError struct {
	object string
	id     string
}

func (e objNotFoundError) Error() string {
	return "No such " + e.object + ": " + e.id
}

func (e objNotFoundError) NotFound() {}

type stateConflictError struct {
	cause error
}

func (e stateConflictError) Error() string {
	return e.cause.Error()
}

func (e stateConflictError) Cause() error {
	return e.cause
}

func (e stateConflictError) Conflict() {}

func errContainerIsRestarting(containerID string) error {
	cause := errors.Errorf("Container %s is restarting, wait until the container is running", containerID)
	return stateConflictError{cause}
}

func errExecNotFound(id string) error {
	return objNotFoundError{"exec instance", id}
}

func errExecPaused(id string) error {
	cause := errors.Errorf("Container %s is paused, unpause the container before exec", id)
	return stateConflictError{cause}
}

func errNotPaused(id string) error {
	cause := errors.Errorf("Container %s is already paused", id)
	return stateConflictError{cause}
}

type nameConflictError struct {
	id   string
	name string
}

func (e nameConflictError) Error() string {
	return fmt.Sprintf("Conflict. The container name %q is already in use by container %q. You have to remove (or rename) that container to be able to reuse that name.", e.name, e.id)
}

func (nameConflictError) Conflict() {}

type validationError struct {
	cause error
}

func (e validationError) Error() string {
	return e.cause.Error()
}

func (e validationError) InvalidParameter() {}

func (e validationError) Cause() error {
	return e.cause
}

type notAllowedError struct {
	cause error
}

func (e notAllowedError) Error() string {
	return e.cause.Error()
}

func (e notAllowedError) Forbidden() {}

func (e notAllowedError) Cause() error {
	return e.cause
}

type containerNotModifiedError struct {
	running bool
}

func (e containerNotModifiedError) Error() string {
	if e.running {
		return "Container is already started"
	}
	return "Container is already stopped"
}

func (e containerNotModifiedError) NotModified() {}

type systemError struct {
	cause error
}

func (e systemError) Error() string {
	return e.cause.Error()
}

func (e systemError) SystemError() {}

func (e systemError) Cause() error {
	return e.cause
}

type invalidIdentifier string

func (e invalidIdentifier) Error() string {
	return fmt.Sprintf("invalid name or ID supplied: %q", string(e))
}

func (invalidIdentifier) InvalidParameter() {}

type duplicateMountPointError string

func (e duplicateMountPointError) Error() string {
	return "Duplicate mount point: " + string(e)
}
func (duplicateMountPointError) InvalidParameter() {}

type containerFileNotFound struct {
	file      string
	container string
}

func (e containerFileNotFound) Error() string {
	return "Could not find the file " + e.file + " in container " + e.container
}

func (containerFileNotFound) NotFound() {}

type invalidFilter struct {
	filter string
	value  interface{}
}

func (e invalidFilter) Error() string {
	msg := "Invalid filter '" + e.filter
	if e.value != nil {
		msg += fmt.Sprintf("=%s", e.value)
	}
	return msg + "'"
}

func (e invalidFilter) InvalidParameter() {}

type unknownError struct {
	cause error
}

func (e unknownError) Error() string {
	return e.cause.Error()
}

func (unknownError) Unknown() {}

func (e unknownError) Cause() error {
	return e.cause
}

type startInvalidConfigError string

func (e startInvalidConfigError) Error() string {
	return string(e)
}

func (e startInvalidConfigError) InvalidParameter() {} // Is this right???

func translateContainerdStartErr(cmd string, setExitCode func(int), err error) error {
	errDesc := grpc.ErrorDesc(err)
	contains := func(s1, s2 string) bool {
		return strings.Contains(strings.ToLower(s1), s2)
	}
	var retErr error = unknownError{errors.New(errDesc)}
	// if we receive an internal error from the initial start of a container then lets
	// return it instead of entering the restart loop
	// set to 127 for container cmd not found/does not exist)
	if contains(errDesc, cmd) &&
		(contains(errDesc, "executable file not found") ||
			contains(errDesc, "no such file or directory") ||
			contains(errDesc, "system cannot find the file specified")) {
		setExitCode(127)
		retErr = startInvalidConfigError(errDesc)
	}
	// set to 126 for container cmd can't be invoked errors
	if contains(errDesc, syscall.EACCES.Error()) {
		setExitCode(126)
		retErr = startInvalidConfigError(errDesc)
	}

	// attempted to mount a file onto a directory, or a directory onto a file, maybe from user specified bind mounts
	if contains(errDesc, syscall.ENOTDIR.Error()) {
		errDesc += ": Are you trying to mount a directory onto a file (or vice-versa)? Check if the specified host path exists and is the expected type"
		setExitCode(127)
		retErr = startInvalidConfigError(errDesc)
	}

	// TODO: it would be nice to get some better errors from containerd so we can return better errors here
	return retErr
}

// TODO: cpuguy83 take care of it once the new library is ready
type errNotFound struct{ error }

func (errNotFound) NotFound() {}

func (e errNotFound) Cause() error {
	return e.error
}

// notFound is a helper to create an error of the class with the same name from any error type
func notFound(err error) error {
	if err == nil {
		return nil
	}
	return errNotFound{err}
}
