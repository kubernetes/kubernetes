package backoff

import (
	"errors"
	"time"
)

// An OperationWithData is executing by RetryWithData() or RetryNotifyWithData().
// The operation will be retried using a backoff policy if it returns an error.
type OperationWithData[T any] func() (T, error)

// An Operation is executing by Retry() or RetryNotify().
// The operation will be retried using a backoff policy if it returns an error.
type Operation func() error

func (o Operation) withEmptyData() OperationWithData[struct{}] {
	return func() (struct{}, error) {
		return struct{}{}, o()
	}
}

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

// RetryWithData is like Retry but returns data in the response too.
func RetryWithData[T any](o OperationWithData[T], b BackOff) (T, error) {
	return RetryNotifyWithData(o, b, nil)
}

// RetryNotify calls notify function with the error and wait duration
// for each failed attempt before sleep.
func RetryNotify(operation Operation, b BackOff, notify Notify) error {
	return RetryNotifyWithTimer(operation, b, notify, nil)
}

// RetryNotifyWithData is like RetryNotify but returns data in the response too.
func RetryNotifyWithData[T any](operation OperationWithData[T], b BackOff, notify Notify) (T, error) {
	return doRetryNotify(operation, b, notify, nil)
}

// RetryNotifyWithTimer calls notify function with the error and wait duration using the given Timer
// for each failed attempt before sleep.
// A default timer that uses system timer is used when nil is passed.
func RetryNotifyWithTimer(operation Operation, b BackOff, notify Notify, t Timer) error {
	_, err := doRetryNotify(operation.withEmptyData(), b, notify, t)
	return err
}

// RetryNotifyWithTimerAndData is like RetryNotifyWithTimer but returns data in the response too.
func RetryNotifyWithTimerAndData[T any](operation OperationWithData[T], b BackOff, notify Notify, t Timer) (T, error) {
	return doRetryNotify(operation, b, notify, t)
}

func doRetryNotify[T any](operation OperationWithData[T], b BackOff, notify Notify, t Timer) (T, error) {
	var (
		err  error
		next time.Duration
		res  T
	)
	if t == nil {
		t = &defaultTimer{}
	}

	defer func() {
		t.Stop()
	}()

	ctx := getContext(b)

	b.Reset()
	for {
		res, err = operation()
		if err == nil {
			return res, nil
		}

		var permanent *PermanentError
		if errors.As(err, &permanent) {
			return res, permanent.Err
		}

		if next = b.NextBackOff(); next == Stop {
			if cerr := ctx.Err(); cerr != nil {
				return res, cerr
			}

			return res, err
		}

		if notify != nil {
			notify(err, next)
		}

		t.Start(next)

		select {
		case <-ctx.Done():
			return res, ctx.Err()
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
