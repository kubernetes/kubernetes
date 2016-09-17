// Package errwrap implements methods to formalize error wrapping in Go.
//
// All of the top-level functions that take an `error` are built to be able
// to take any error, not just wrapped errors. This allows you to use errwrap
// without having to type-check and type-cast everywhere.
package errwrap

import (
	"errors"
	"reflect"
	"strings"
)

// WalkFunc is the callback called for Walk.
type WalkFunc func(error)

// Wrapper is an interface that can be implemented by custom types to
// have all the Contains, Get, etc. functions in errwrap work.
//
// When Walk reaches a Wrapper, it will call the callback for every
// wrapped error in addition to the wrapper itself. Since all the top-level
// functions in errwrap use Walk, this means that all those functions work
// with your custom type.
type Wrapper interface {
	WrappedErrors() []error
}

// Wrap defines that outer wraps inner, returning an error type that
// can be cleanly used with the other methods in this package, such as
// Contains, GetAll, etc.
//
// This function won't modify the error message at all (the outer message
// will be used).
func Wrap(outer, inner error) error {
	return &wrappedError{
		Outer: outer,
		Inner: inner,
	}
}

// Wrapf wraps an error with a formatting message. This is similar to using
// `fmt.Errorf` to wrap an error. If you're using `fmt.Errorf` to wrap
// errors, you should replace it with this.
//
// format is the format of the error message. The string '{{err}}' will
// be replaced with the original error message.
func Wrapf(format string, err error) error {
	outerMsg := "<nil>"
	if err != nil {
		outerMsg = err.Error()
	}

	outer := errors.New(strings.Replace(
		format, "{{err}}", outerMsg, -1))

	return Wrap(outer, err)
}

// Contains checks if the given error contains an error with the
// message msg. If err is not a wrapped error, this will always return
// false unless the error itself happens to match this msg.
func Contains(err error, msg string) bool {
	return len(GetAll(err, msg)) > 0
}

// ContainsType checks if the given error contains an error with
// the same concrete type as v. If err is not a wrapped error, this will
// check the err itself.
func ContainsType(err error, v interface{}) bool {
	return len(GetAllType(err, v)) > 0
}

// Get is the same as GetAll but returns the deepest matching error.
func Get(err error, msg string) error {
	es := GetAll(err, msg)
	if len(es) > 0 {
		return es[len(es)-1]
	}

	return nil
}

// GetType is the same as GetAllType but returns the deepest matching error.
func GetType(err error, v interface{}) error {
	es := GetAllType(err, v)
	if len(es) > 0 {
		return es[len(es)-1]
	}

	return nil
}

// GetAll gets all the errors that might be wrapped in err with the
// given message. The order of the errors is such that the outermost
// matching error (the most recent wrap) is index zero, and so on.
func GetAll(err error, msg string) []error {
	var result []error

	Walk(err, func(err error) {
		if err.Error() == msg {
			result = append(result, err)
		}
	})

	return result
}

// GetAllType gets all the errors that are the same type as v.
//
// The order of the return value is the same as described in GetAll.
func GetAllType(err error, v interface{}) []error {
	var result []error

	var search string
	if v != nil {
		search = reflect.TypeOf(v).String()
	}
	Walk(err, func(err error) {
		var needle string
		if err != nil {
			needle = reflect.TypeOf(err).String()
		}

		if needle == search {
			result = append(result, err)
		}
	})

	return result
}

// Walk walks all the wrapped errors in err and calls the callback. If
// err isn't a wrapped error, this will be called once for err. If err
// is a wrapped error, the callback will be called for both the wrapper
// that implements error as well as the wrapped error itself.
func Walk(err error, cb WalkFunc) {
	if err == nil {
		return
	}

	switch e := err.(type) {
	case *wrappedError:
		cb(e.Outer)
		Walk(e.Inner, cb)
	case Wrapper:
		cb(err)

		for _, err := range e.WrappedErrors() {
			Walk(err, cb)
		}
	default:
		cb(err)
	}
}

// wrappedError is an implementation of error that has both the
// outer and inner errors.
type wrappedError struct {
	Outer error
	Inner error
}

func (w *wrappedError) Error() string {
	return w.Outer.Error()
}

func (w *wrappedError) WrappedErrors() []error {
	return []error{w.Outer, w.Inner}
}
