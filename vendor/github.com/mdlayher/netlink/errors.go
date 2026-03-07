package netlink

import (
	"errors"
	"fmt"
	"net"
	"os"
	"strings"
)

// Error messages which can be returned by Validate.
var (
	errMismatchedSequence = errors.New("mismatched sequence in netlink reply")
	errMismatchedPID      = errors.New("mismatched PID in netlink reply")
	errShortErrorMessage  = errors.New("not enough data for netlink error code")
)

// Errors which can be returned by a Socket that does not implement
// all exposed methods of Conn.

var errNotSupported = errors.New("operation not supported")

// notSupported provides a concise constructor for "not supported" errors.
func notSupported(op string) error {
	return newOpError(op, errNotSupported)
}

// IsNotExist determines if an error is produced as the result of querying some
// file, object, resource, etc. which does not exist.
//
// Deprecated: use errors.Unwrap and/or `errors.Is(err, os.Permission)` in Go
// 1.13+.
func IsNotExist(err error) bool {
	switch err := err.(type) {
	case *OpError:
		// Unwrap the inner error and use the stdlib's logic.
		return os.IsNotExist(err.Err)
	default:
		return os.IsNotExist(err)
	}
}

var (
	_ error     = &OpError{}
	_ net.Error = &OpError{}
	// Ensure compatibility with Go 1.13+ errors package.
	_ interface{ Unwrap() error } = &OpError{}
)

// An OpError is an error produced as the result of a failed netlink operation.
type OpError struct {
	// Op is the operation which caused this OpError, such as "send"
	// or "receive".
	Op string

	// Err is the underlying error which caused this OpError.
	//
	// If Err was produced by a system call error, Err will be of type
	// *os.SyscallError. If Err was produced by an error code in a netlink
	// message, Err will contain a raw error value type such as a unix.Errno.
	//
	// Most callers should inspect Err using errors.Is from the standard
	// library.
	Err error

	// Message and Offset contain additional error information provided by the
	// kernel when the ExtendedAcknowledge option is set on a Conn and the
	// kernel indicates the AcknowledgeTLVs flag in a response. If this option
	// is not set, both of these fields will be empty.
	Message string
	Offset  int
}

// newOpError is a small wrapper for creating an OpError. As a convenience, it
// returns nil if the input err is nil: akin to os.NewSyscallError.
func newOpError(op string, err error) error {
	if err == nil {
		return nil
	}

	return &OpError{
		Op:  op,
		Err: err,
	}
}

func (e *OpError) Error() string {
	if e == nil {
		return "<nil>"
	}

	var sb strings.Builder
	_, _ = sb.WriteString(fmt.Sprintf("netlink %s: %v", e.Op, e.Err))

	if e.Message != "" || e.Offset != 0 {
		_, _ = sb.WriteString(fmt.Sprintf(", offset: %d, message: %q",
			e.Offset, e.Message))
	}

	return sb.String()
}

// Unwrap unwraps the internal Err field for use with errors.Unwrap.
func (e *OpError) Unwrap() error { return e.Err }

// Portions of this code taken from the Go standard library:
//
// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

type timeout interface {
	Timeout() bool
}

// Timeout reports whether the error was caused by an I/O timeout.
func (e *OpError) Timeout() bool {
	if ne, ok := e.Err.(*os.SyscallError); ok {
		t, ok := ne.Err.(timeout)
		return ok && t.Timeout()
	}
	t, ok := e.Err.(timeout)
	return ok && t.Timeout()
}

type temporary interface {
	Temporary() bool
}

// Temporary reports whether an operation may succeed if retried.
func (e *OpError) Temporary() bool {
	if ne, ok := e.Err.(*os.SyscallError); ok {
		t, ok := ne.Err.(temporary)
		return ok && t.Temporary()
	}
	t, ok := e.Err.(temporary)
	return ok && t.Temporary()
}
