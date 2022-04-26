package packfile

import "fmt"

// Error specifies errors returned during packfile parsing.
type Error struct {
	reason, details string
}

// NewError returns a new error.
func NewError(reason string) *Error {
	return &Error{reason: reason}
}

// Error returns a text representation of the error.
func (e *Error) Error() string {
	if e.details == "" {
		return e.reason
	}

	return fmt.Sprintf("%s: %s", e.reason, e.details)
}

// AddDetails adds details to an error, with additional text.
func (e *Error) AddDetails(format string, args ...interface{}) *Error {
	return &Error{
		reason:  e.reason,
		details: fmt.Sprintf(format, args...),
	}
}
