//go:build go1.20
// +build go1.20

package jwt

import (
	"fmt"
)

// Unwrap implements the multiple error unwrapping for this error type, which is
// possible in Go 1.20.
func (je joinedError) Unwrap() []error {
	return je.errs
}

// newError creates a new error message with a detailed error message. The
// message will be prefixed with the contents of the supplied error type.
// Additionally, more errors, that provide more context can be supplied which
// will be appended to the message. This makes use of Go 1.20's possibility to
// include more than one %w formatting directive in [fmt.Errorf].
//
// For example,
//
//	newError("no keyfunc was provided", ErrTokenUnverifiable)
//
// will produce the error string
//
//	"token is unverifiable: no keyfunc was provided"
func newError(message string, err error, more ...error) error {
	var format string
	var args []any
	if message != "" {
		format = "%w: %s"
		args = []any{err, message}
	} else {
		format = "%w"
		args = []any{err}
	}

	for _, e := range more {
		format += ": %w"
		args = append(args, e)
	}

	err = fmt.Errorf(format, args...)
	return err
}
