package errors

import (
	"fmt"
	"strings"

	"github.com/onsi/gomega/types"
)

// A stateful matcher that nests other matchers within it and preserves the error types of the
// nested matcher failures.
type NestingMatcher interface {
	types.GomegaMatcher

	// Returns the failures of nested matchers.
	Failures() []error
}

// An error type for labeling errors on deeply nested matchers.
type NestedError struct {
	Path string
	Err  error
}

func (e *NestedError) Error() string {
	// Indent Errors.
	indented := strings.Replace(e.Err.Error(), "\n", "\n\t", -1)
	return fmt.Sprintf("%s:\n\t%v", e.Path, indented)
}

// Create a NestedError with the given path.
// If err is a NestedError, prepend the path to it.
// If err is an AggregateError, recursively Nest each error.
func Nest(path string, err error) error {
	if ag, ok := err.(AggregateError); ok {
		var errs AggregateError
		for _, e := range ag {
			errs = append(errs, Nest(path, e))
		}
		return errs
	}
	if ne, ok := err.(*NestedError); ok {
		return &NestedError{
			Path: path + ne.Path,
			Err:  ne.Err,
		}
	}
	return &NestedError{
		Path: path,
		Err:  err,
	}
}

// An error type for treating multiple errors as a single error.
type AggregateError []error

// Error is part of the error interface.
func (err AggregateError) Error() string {
	if len(err) == 0 {
		// This should never happen, really.
		return ""
	}
	if len(err) == 1 {
		return err[0].Error()
	}
	result := fmt.Sprintf("[%s", err[0].Error())
	for i := 1; i < len(err); i++ {
		result += fmt.Sprintf(", %s", err[i].Error())
	}
	result += "]"
	return result
}
