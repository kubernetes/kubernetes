//go:build !go1.20
// +build !go1.20

package jwt

import (
	"errors"
	"fmt"
)

// Is implements checking for multiple errors using [errors.Is], since multiple
// error unwrapping is not possible in versions less than Go 1.20.
func (je joinedError) Is(err error) bool {
	for _, e := range je.errs {
		if errors.Is(e, err) {
			return true
		}
	}

	return false
}

// wrappedErrors is a workaround for wrapping multiple errors in environments
// where Go 1.20 is not available. It basically uses the already implemented
// functionality of joinedError to handle multiple errors with supplies a
// custom error message that is identical to the one we produce in Go 1.20 using
// multiple %w directives.
type wrappedErrors struct {
	msg string
	joinedError
}

// Error returns the stored error string
func (we wrappedErrors) Error() string {
	return we.msg
}

// newError creates a new error message with a detailed error message. The
// message will be prefixed with the contents of the supplied error type.
// Additionally, more errors, that provide more context can be supplied which
// will be appended to the message. Since we cannot use of Go 1.20's possibility
// to include more than one %w formatting directive in [fmt.Errorf], we have to
// emulate that.
//
// For example,
//
//	newError("no keyfunc was provided", ErrTokenUnverifiable)
//
// will produce the error string
//
//	"token is unverifiable: no keyfunc was provided"
func newError(message string, err error, more ...error) error {
	// We cannot wrap multiple errors here with %w, so we have to be a little
	// bit creative. Basically, we are using %s instead of %w to produce the
	// same error message and then throw the result into a custom error struct.
	var format string
	var args []any
	if message != "" {
		format = "%s: %s"
		args = []any{err, message}
	} else {
		format = "%s"
		args = []any{err}
	}
	errs := []error{err}

	for _, e := range more {
		format += ": %s"
		args = append(args, e)
		errs = append(errs, e)
	}

	err = &wrappedErrors{
		msg:         fmt.Sprintf(format, args...),
		joinedError: joinedError{errs: errs},
	}
	return err
}
