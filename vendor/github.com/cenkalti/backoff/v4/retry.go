package backoff

import (
	"errors"
	"time"
)

// An Operation is executing by Retry() or RetryNotify().
// The operation will be retried using a backoff policy if it returns an error.
type Operation func() error

// Notify is a notify-on-error function. It receives an operation error and
// backoff delay if the operation failed (with an error).
//
// NOTE that if the backoff policy stated to stop retrying,
// the notify function isn't called.
type Notify func(error, time.Duration)

// Retry the operation o until it does not return error or BackOff stops.
// o is guaranteed to be run at least once.
//
// If o returns a *PermanentError, the operation is not retried, and the
// wrapped error is returned.
//
// Retry sleeps the goroutine for the duration returned by BackOff after a
// failed operation returns.
func Retry(o Operation, b BackOff) error {
	return RetryNotify(o, b, nil)
}

// RetryNotify calls notify function with the error and wait duration
// for each failed attempt before sleep.
func RetryNotify(operation Operation, b BackOff, notify Notify) error {
	return RetryNotifyWithTimer(operation, b, notify, nil)
}

// RetryNotifyWithTimer calls notify function with the error and wait duration using the given Timer
// for each failed attempt before sleep.
// A default timer that uses system timer is used when nil is passed.
func RetryNotifyWithTimer(operation Operation, b BackOff, notify Notify, t Timer) error {
	var err error
	var next time.Duration
	if t == nil {
		t = &defaultTimer{}
	}

	defer func() {
		t.Stop()
	}()

	ctx := getContext(b)

	b.Reset()
	for {
		if err = operation(); err == nil {
			return nil
		}

		var permanent *PermanentError
		if errors.As(err, &permanent) {
			return permanent.Err
		}

		if next = b.NextBackOff(); next == Stop {
			if cerr := ctx.Err(); cerr != nil {
				return cerr
			}

			return err
		}

		if notify != nil {
			notify(err, next)
		}

		t.Start(next)

		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-t.C():
		}
	}
}

// PermanentError signals that the operation should not be retried.
type PermanentError struct {
	Err error
}

func (e *PermanentError) Error() string {
	return e.Err.Error()
}

func (e *PermanentError) Unwrap() error {
	return e.Err
}

func (e *PermanentError) Is(target error) bool {
	_, ok := target.(*PermanentError)
	return ok
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
