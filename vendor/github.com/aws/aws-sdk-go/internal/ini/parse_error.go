package ini

import "fmt"

const (
	// ErrCodeParseError is returned when a parsing error
	// has occurred.
	ErrCodeParseError = "INIParseError"
)

// ParseError is an error which is returned during any part of
// the parsing process.
type ParseError struct {
	msg string
}

// NewParseError will return a new ParseError where message
// is the description of the error.
func NewParseError(message string) *ParseError {
	return &ParseError{
		msg: message,
	}
}

// Code will return the ErrCodeParseError
func (err *ParseError) Code() string {
	return ErrCodeParseError
}

// Message returns the error's message
func (err *ParseError) Message() string {
	return err.msg
}

// OrigError return nothing since there will never be any
// original error.
func (err *ParseError) OrigError() error {
	return nil
}

func (err *ParseError) Error() string {
	return fmt.Sprintf("%s: %s", err.Code(), err.Message())
}
