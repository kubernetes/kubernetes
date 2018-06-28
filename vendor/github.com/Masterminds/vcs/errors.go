package vcs

import (
	"errors"
	"fmt"
)

// The vcs package provides ways to work with errors that hide the underlying
// implementation details but make them accessible if needed. For basic errors
// that do not have underlying implementation specific details or the underlying
// details are not necessary there are errors for comparison.
//
// For example:
//
//     ci, err := repo.CommitInfo("123")
//     if err == vcs.ErrRevisionUnavailable {
//         // The commit id was not available in the VCS.
//     }
//
// There are other times where getting the details are more useful. For example,
// if you're performing a repo.Get() and an error occurs. In general you'll want
// to consistently know it failed. But, you may want to know the underlying
// details (opt-in) to them. For those cases there is a different form of error
// handling.
//
// For example:
//
//     err := repo.Get()
//     if err != nil {
//         // A RemoteError was returned. This has access to the output of the
//         // vcs command, original error, and has a consistent cross vcs message.
//     }
//
// The errors returned here can be used in type switches to detect the underlying
// error. For example:
//
//     switch err.(type) {
//     case *vcs.RemoteError:
//         // This an error connecting to a remote system.
//     }
//
// For more information on using type switches to detect error types you can
// read the Go wiki at https://github.com/golang/go/wiki/Errors

var (
	// ErrWrongVCS is returned when an action is tried on the wrong VCS.
	ErrWrongVCS = errors.New("Wrong VCS detected")

	// ErrCannotDetectVCS is returned when VCS cannot be detected from URI string.
	ErrCannotDetectVCS = errors.New("Cannot detect VCS")

	// ErrWrongRemote occurs when the passed in remote does not match the VCS
	// configured endpoint.
	ErrWrongRemote = errors.New("The Remote does not match the VCS endpoint")

	// ErrRevisionUnavailable happens when commit revision information is
	// unavailable.
	ErrRevisionUnavailable = errors.New("Revision unavailable")
)

// RemoteError is returned when an operation fails against a remote repo
type RemoteError struct {
	vcsError
}

// NewRemoteError constructs a RemoteError
func NewRemoteError(msg string, err error, out string) error {
	e := &RemoteError{}
	e.s = msg
	e.e = err
	e.o = out

	return e
}

// LocalError is returned when a local operation has an error
type LocalError struct {
	vcsError
}

// NewLocalError constructs a LocalError
func NewLocalError(msg string, err error, out string) error {
	e := &LocalError{}
	e.s = msg
	e.e = err
	e.o = out

	return e
}

type vcsError struct {
	s string
	e error  // The original error
	o string // The output from executing the command
}

// Error implements the Error interface
func (e *vcsError) Error() string {
	if e.e == nil {
		return e.s
	}

	return fmt.Sprintf("%s: %v", e.s, e.e)
}

// Original retrieves the underlying implementation specific error.
func (e *vcsError) Original() error {
	return e.e
}

// Out retrieves the output of the original command that was run.
func (e *vcsError) Out() string {
	return e.o
}
