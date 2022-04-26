package multierror

import (
	"errors"
	"fmt"
)

// Error is an error type to track multiple errors. This is used to
// accumulate errors in cases and return them as a single "error".
type Error struct {
	Errors      []error
	ErrorFormat ErrorFormatFunc
}

func (e *Error) Error() string {
	fn := e.ErrorFormat
	if fn == nil {
		fn = ListFormatFunc
	}

	return fn(e.Errors)
}

// ErrorOrNil returns an error interface if this Error represents
// a list of errors, or returns nil if the list of errors is empty. This
// function is useful at the end of accumulation to make sure that the value
// returned represents the existence of errors.
func (e *Error) ErrorOrNil() error {
	if e == nil {
		return nil
	}
	if len(e.Errors) == 0 {
		return nil
	}

	return e
}

func (e *Error) GoString() string {
	return fmt.Sprintf("*%#v", *e)
}

// WrappedErrors returns the list of errors that this Error is wrapping. It is
// an implementation of the errwrap.Wrapper interface so that multierror.Error
// can be used with that library.
//
// This method is not safe to be called concurrently. Unlike accessing the
// Errors field directly, this function also checks if the multierror is nil to
// prevent a null-pointer panic. It satisfies the errwrap.Wrapper interface.
func (e *Error) WrappedErrors() []error {
	if e == nil {
		return nil
	}
	return e.Errors
}

// Unwrap returns an error from Error (or nil if there are no errors).
// This error returned will further support Unwrap to get the next error,
// etc. The order will match the order of Errors in the multierror.Error
// at the time of calling.
//
// The resulting error supports errors.As/Is/Unwrap so you can continue
// to use the stdlib errors package to introspect further.
//
// This will perform a shallow copy of the errors slice. Any errors appended
// to this error after calling Unwrap will not be available until a new
// Unwrap is called on the multierror.Error.
func (e *Error) Unwrap() error {
	// If we have no errors then we do nothing
	if e == nil || len(e.Errors) == 0 {
		return nil
	}

	// If we have exactly one error, we can just return that directly.
	if len(e.Errors) == 1 {
		return e.Errors[0]
	}

	// Shallow copy the slice
	errs := make([]error, len(e.Errors))
	copy(errs, e.Errors)
	return chain(errs)
}

// chain implements the interfaces necessary for errors.Is/As/Unwrap to
// work in a deterministic way with multierror. A chain tracks a list of
// errors while accounting for the current represented error. This lets
// Is/As be meaningful.
//
// Unwrap returns the next error. In the cleanest form, Unwrap would return
// the wrapped error here but we can't do that if we want to properly
// get access to all the errors. Instead, users are recommended to use
// Is/As to get the correct error type out.
//
// Precondition: []error is non-empty (len > 0)
type chain []error

// Error implements the error interface
func (e chain) Error() string {
	return e[0].Error()
}

// Unwrap implements errors.Unwrap by returning the next error in the
// chain or nil if there are no more errors.
func (e chain) Unwrap() error {
	if len(e) == 1 {
		return nil
	}

	return e[1:]
}

// As implements errors.As by attempting to map to the current value.
func (e chain) As(target interface{}) bool {
	return errors.As(e[0], target)
}

// Is implements errors.Is by comparing the current value directly.
func (e chain) Is(target error) bool {
	return errors.Is(e[0], target)
}
