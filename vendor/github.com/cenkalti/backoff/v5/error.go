package backoff

import (
	"fmt"
	"time"
)

// PermanentError signals that the operation should not be retried.
type PermanentError struct {
	Err error
}

// Permanent wraps the given err in a *PermanentError.
func Permanent(err error) error {
	if err == nil {
		return nil
	}
	return &PermanentError{
		Err: err,
	}
}

// Error returns a string representation of the Permanent error.
func (e *PermanentError) Error() string {
	return e.Err.Error()
}

// Unwrap returns the wrapped error.
func (e *PermanentError) Unwrap() error {
	return e.Err
}

// RetryAfterError signals that the operation should be retried after the given duration.
type RetryAfterError struct {
	Duration time.Duration
}

// RetryAfter returns a RetryAfter error that specifies how long to wait before retrying.
func RetryAfter(seconds int) error {
	return &RetryAfterError{Duration: time.Duration(seconds) * time.Second}
}

// Error returns a string representation of the RetryAfter error.
func (e *RetryAfterError) Error() string {
	return fmt.Sprintf("retry after %s", e.Duration)
}
