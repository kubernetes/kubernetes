// +build !go1.13

package errors

import (
	"reflect"
)

type unwrapper interface {
	Unwrap() error
}

// As assigns error or any wrapped error to the value target points
// to. If there is no value of the target type of target As returns
// false.
func As(err error, target interface{}) bool {
	targetType := reflect.TypeOf(target)

	for {
		errType := reflect.TypeOf(err)

		if errType == nil {
			return false
		}

		if reflect.PtrTo(errType) == targetType {
			reflect.ValueOf(target).Elem().Set(reflect.ValueOf(err))
			return true
		}

		wrapped, ok := err.(unwrapper)
		if ok {
			err = wrapped.Unwrap()
		} else {
			return false
		}
	}
}

// Is detects whether the error is equal to a given error. Errors
// are considered equal by this function if they are the same object,
// or if they both contain the same error inside an errors.Error.
func Is(e error, original error) bool {
	if e == original {
		return true
	}

	if e, ok := e.(*Error); ok {
		return Is(e.Err, original)
	}

	if original, ok := original.(*Error); ok {
		return Is(e, original.Err)
	}

	return false
}
