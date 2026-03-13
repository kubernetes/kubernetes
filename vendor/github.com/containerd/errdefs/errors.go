/*
   Copyright The containerd Authors.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

// Package errdefs defines the common errors used throughout containerd
// packages.
//
// Use with fmt.Errorf to add context to an error.
//
// To detect an error class, use the IsXXX functions to tell whether an error
// is of a certain type.
package errdefs

import (
	"context"
	"errors"
)

// Definitions of common error types used throughout containerd. All containerd
// errors returned by most packages will map into one of these errors classes.
// Packages should return errors of these types when they want to instruct a
// client to take a particular action.
//
// These errors map closely to grpc errors.
var (
	ErrUnknown            = errUnknown{}
	ErrInvalidArgument    = errInvalidArgument{}
	ErrNotFound           = errNotFound{}
	ErrAlreadyExists      = errAlreadyExists{}
	ErrPermissionDenied   = errPermissionDenied{}
	ErrResourceExhausted  = errResourceExhausted{}
	ErrFailedPrecondition = errFailedPrecondition{}
	ErrConflict           = errConflict{}
	ErrNotModified        = errNotModified{}
	ErrAborted            = errAborted{}
	ErrOutOfRange         = errOutOfRange{}
	ErrNotImplemented     = errNotImplemented{}
	ErrInternal           = errInternal{}
	ErrUnavailable        = errUnavailable{}
	ErrDataLoss           = errDataLoss{}
	ErrUnauthenticated    = errUnauthorized{}
)

// cancelled maps to Moby's "ErrCancelled"
type cancelled interface {
	Cancelled()
}

// IsCanceled returns true if the error is due to `context.Canceled`.
func IsCanceled(err error) bool {
	return errors.Is(err, context.Canceled) || isInterface[cancelled](err)
}

type errUnknown struct{}

func (errUnknown) Error() string { return "unknown" }

func (errUnknown) Unknown() {}

func (e errUnknown) WithMessage(msg string) error {
	return customMessage{e, msg}
}

// unknown maps to Moby's "ErrUnknown"
type unknown interface {
	Unknown()
}

// IsUnknown returns true if the error is due to an unknown error,
// unhandled condition or unexpected response.
func IsUnknown(err error) bool {
	return errors.Is(err, errUnknown{}) || isInterface[unknown](err)
}

type errInvalidArgument struct{}

func (errInvalidArgument) Error() string { return "invalid argument" }

func (errInvalidArgument) InvalidParameter() {}

func (e errInvalidArgument) WithMessage(msg string) error {
	return customMessage{e, msg}
}

// invalidParameter maps to Moby's "ErrInvalidParameter"
type invalidParameter interface {
	InvalidParameter()
}

// IsInvalidArgument returns true if the error is due to an invalid argument
func IsInvalidArgument(err error) bool {
	return errors.Is(err, ErrInvalidArgument) || isInterface[invalidParameter](err)
}

// deadlineExceed maps to Moby's "ErrDeadline"
type deadlineExceeded interface {
	DeadlineExceeded()
}

// IsDeadlineExceeded returns true if the error is due to
// `context.DeadlineExceeded`.
func IsDeadlineExceeded(err error) bool {
	return errors.Is(err, context.DeadlineExceeded) || isInterface[deadlineExceeded](err)
}

type errNotFound struct{}

func (errNotFound) Error() string { return "not found" }

func (errNotFound) NotFound() {}

func (e errNotFound) WithMessage(msg string) error {
	return customMessage{e, msg}
}

// notFound maps to Moby's "ErrNotFound"
type notFound interface {
	NotFound()
}

// IsNotFound returns true if the error is due to a missing object
func IsNotFound(err error) bool {
	return errors.Is(err, ErrNotFound) || isInterface[notFound](err)
}

type errAlreadyExists struct{}

func (errAlreadyExists) Error() string { return "already exists" }

func (errAlreadyExists) AlreadyExists() {}

func (e errAlreadyExists) WithMessage(msg string) error {
	return customMessage{e, msg}
}

type alreadyExists interface {
	AlreadyExists()
}

// IsAlreadyExists returns true if the error is due to an already existing
// metadata item
func IsAlreadyExists(err error) bool {
	return errors.Is(err, ErrAlreadyExists) || isInterface[alreadyExists](err)
}

type errPermissionDenied struct{}

func (errPermissionDenied) Error() string { return "permission denied" }

func (errPermissionDenied) Forbidden() {}

func (e errPermissionDenied) WithMessage(msg string) error {
	return customMessage{e, msg}
}

// forbidden maps to Moby's "ErrForbidden"
type forbidden interface {
	Forbidden()
}

// IsPermissionDenied returns true if the error is due to permission denied
// or forbidden (403) response
func IsPermissionDenied(err error) bool {
	return errors.Is(err, ErrPermissionDenied) || isInterface[forbidden](err)
}

type errResourceExhausted struct{}

func (errResourceExhausted) Error() string { return "resource exhausted" }

func (errResourceExhausted) ResourceExhausted() {}

func (e errResourceExhausted) WithMessage(msg string) error {
	return customMessage{e, msg}
}

type resourceExhausted interface {
	ResourceExhausted()
}

// IsResourceExhausted returns true if the error is due to
// a lack of resources or too many attempts.
func IsResourceExhausted(err error) bool {
	return errors.Is(err, errResourceExhausted{}) || isInterface[resourceExhausted](err)
}

type errFailedPrecondition struct{}

func (e errFailedPrecondition) Error() string { return "failed precondition" }

func (errFailedPrecondition) FailedPrecondition() {}

func (e errFailedPrecondition) WithMessage(msg string) error {
	return customMessage{e, msg}
}

type failedPrecondition interface {
	FailedPrecondition()
}

// IsFailedPrecondition returns true if an operation could not proceed due to
// the lack of a particular condition
func IsFailedPrecondition(err error) bool {
	return errors.Is(err, errFailedPrecondition{}) || isInterface[failedPrecondition](err)
}

type errConflict struct{}

func (errConflict) Error() string { return "conflict" }

func (errConflict) Conflict() {}

func (e errConflict) WithMessage(msg string) error {
	return customMessage{e, msg}
}

// conflict maps to Moby's "ErrConflict"
type conflict interface {
	Conflict()
}

// IsConflict returns true if an operation could not proceed due to
// a conflict.
func IsConflict(err error) bool {
	return errors.Is(err, errConflict{}) || isInterface[conflict](err)
}

type errNotModified struct{}

func (errNotModified) Error() string { return "not modified" }

func (errNotModified) NotModified() {}

func (e errNotModified) WithMessage(msg string) error {
	return customMessage{e, msg}
}

// notModified maps to Moby's "ErrNotModified"
type notModified interface {
	NotModified()
}

// IsNotModified returns true if an operation could not proceed due
// to an object not modified from a previous state.
func IsNotModified(err error) bool {
	return errors.Is(err, errNotModified{}) || isInterface[notModified](err)
}

type errAborted struct{}

func (errAborted) Error() string { return "aborted" }

func (errAborted) Aborted() {}

func (e errAborted) WithMessage(msg string) error {
	return customMessage{e, msg}
}

type aborted interface {
	Aborted()
}

// IsAborted returns true if an operation was aborted.
func IsAborted(err error) bool {
	return errors.Is(err, errAborted{}) || isInterface[aborted](err)
}

type errOutOfRange struct{}

func (errOutOfRange) Error() string { return "out of range" }

func (errOutOfRange) OutOfRange() {}

func (e errOutOfRange) WithMessage(msg string) error {
	return customMessage{e, msg}
}

type outOfRange interface {
	OutOfRange()
}

// IsOutOfRange returns true if an operation could not proceed due
// to data being out of the expected range.
func IsOutOfRange(err error) bool {
	return errors.Is(err, errOutOfRange{}) || isInterface[outOfRange](err)
}

type errNotImplemented struct{}

func (errNotImplemented) Error() string { return "not implemented" }

func (errNotImplemented) NotImplemented() {}

func (e errNotImplemented) WithMessage(msg string) error {
	return customMessage{e, msg}
}

// notImplemented maps to Moby's "ErrNotImplemented"
type notImplemented interface {
	NotImplemented()
}

// IsNotImplemented returns true if the error is due to not being implemented
func IsNotImplemented(err error) bool {
	return errors.Is(err, errNotImplemented{}) || isInterface[notImplemented](err)
}

type errInternal struct{}

func (errInternal) Error() string { return "internal" }

func (errInternal) System() {}

func (e errInternal) WithMessage(msg string) error {
	return customMessage{e, msg}
}

// system maps to Moby's "ErrSystem"
type system interface {
	System()
}

// IsInternal returns true if the error returns to an internal or system error
func IsInternal(err error) bool {
	return errors.Is(err, errInternal{}) || isInterface[system](err)
}

type errUnavailable struct{}

func (errUnavailable) Error() string { return "unavailable" }

func (errUnavailable) Unavailable() {}

func (e errUnavailable) WithMessage(msg string) error {
	return customMessage{e, msg}
}

// unavailable maps to Moby's "ErrUnavailable"
type unavailable interface {
	Unavailable()
}

// IsUnavailable returns true if the error is due to a resource being unavailable
func IsUnavailable(err error) bool {
	return errors.Is(err, errUnavailable{}) || isInterface[unavailable](err)
}

type errDataLoss struct{}

func (errDataLoss) Error() string { return "data loss" }

func (errDataLoss) DataLoss() {}

func (e errDataLoss) WithMessage(msg string) error {
	return customMessage{e, msg}
}

// dataLoss maps to Moby's "ErrDataLoss"
type dataLoss interface {
	DataLoss()
}

// IsDataLoss returns true if data during an operation was lost or corrupted
func IsDataLoss(err error) bool {
	return errors.Is(err, errDataLoss{}) || isInterface[dataLoss](err)
}

type errUnauthorized struct{}

func (errUnauthorized) Error() string { return "unauthorized" }

func (errUnauthorized) Unauthorized() {}

func (e errUnauthorized) WithMessage(msg string) error {
	return customMessage{e, msg}
}

// unauthorized maps to Moby's "ErrUnauthorized"
type unauthorized interface {
	Unauthorized()
}

// IsUnauthorized returns true if the error indicates that the user was
// unauthenticated or unauthorized.
func IsUnauthorized(err error) bool {
	return errors.Is(err, errUnauthorized{}) || isInterface[unauthorized](err)
}

func isInterface[T any](err error) bool {
	for {
		switch x := err.(type) {
		case T:
			return true
		case customMessage:
			err = x.err
		case interface{ Unwrap() error }:
			err = x.Unwrap()
			if err == nil {
				return false
			}
		case interface{ Unwrap() []error }:
			for _, err := range x.Unwrap() {
				if isInterface[T](err) {
					return true
				}
			}
			return false
		default:
			return false
		}
	}
}

// customMessage is used to provide a defined error with a custom message.
// The message is not wrapped but can be compared by the `Is(error) bool` interface.
type customMessage struct {
	err error
	msg string
}

func (c customMessage) Is(err error) bool {
	return c.err == err
}

func (c customMessage) As(target any) bool {
	return errors.As(c.err, target)
}

func (c customMessage) Error() string {
	return c.msg
}
