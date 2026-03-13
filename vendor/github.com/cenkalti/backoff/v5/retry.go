package backoff

import (
	"context"
	"errors"
	"time"
)

// DefaultMaxElapsedTime sets a default limit for the total retry duration.
const DefaultMaxElapsedTime = 15 * time.Minute

// Operation is a function that attempts an operation and may be retried.
type Operation[T any] func() (T, error)

// Notify is a function called on operation error with the error and backoff duration.
type Notify func(error, time.Duration)

// retryOptions holds configuration settings for the retry mechanism.
type retryOptions struct {
	BackOff        BackOff       // Strategy for calculating backoff periods.
	Timer          timer         // Timer to manage retry delays.
	Notify         Notify        // Optional function to notify on each retry error.
	MaxTries       uint          // Maximum number of retry attempts.
	MaxElapsedTime time.Duration // Maximum total time for all retries.
}

type RetryOption func(*retryOptions)

// WithBackOff configures a custom backoff strategy.
func WithBackOff(b BackOff) RetryOption {
	return func(args *retryOptions) {
		args.BackOff = b
	}
}

// withTimer sets a custom timer for managing delays between retries.
func withTimer(t timer) RetryOption {
	return func(args *retryOptions) {
		args.Timer = t
	}
}

// WithNotify sets a notification function to handle retry errors.
func WithNotify(n Notify) RetryOption {
	return func(args *retryOptions) {
		args.Notify = n
	}
}

// WithMaxTries limits the number of all attempts.
func WithMaxTries(n uint) RetryOption {
	return func(args *retryOptions) {
		args.MaxTries = n
	}
}

// WithMaxElapsedTime limits the total duration for retry attempts.
func WithMaxElapsedTime(d time.Duration) RetryOption {
	return func(args *retryOptions) {
		args.MaxElapsedTime = d
	}
}

// Retry attempts the operation until success, a permanent error, or backoff completion.
// It ensures the operation is executed at least once.
//
// Returns the operation result or error if retries are exhausted or context is cancelled.
func Retry[T any](ctx context.Context, operation Operation[T], opts ...RetryOption) (T, error) {
	// Initialize default retry options.
	args := &retryOptions{
		BackOff:        NewExponentialBackOff(),
		Timer:          &defaultTimer{},
		MaxElapsedTime: DefaultMaxElapsedTime,
	}

	// Apply user-provided options to the default settings.
	for _, opt := range opts {
		opt(args)
	}

	defer args.Timer.Stop()

	startedAt := time.Now()
	args.BackOff.Reset()
	for numTries := uint(1); ; numTries++ {
		// Execute the operation.
		res, err := operation()
		if err == nil {
			return res, nil
		}

		// Stop retrying if maximum tries exceeded.
		if args.MaxTries > 0 && numTries >= args.MaxTries {
			return res, err
		}

		// Handle permanent errors without retrying.
		var permanent *PermanentError
		if errors.As(err, &permanent) {
			return res, permanent.Unwrap()
		}

		// Stop retrying if context is cancelled.
		if cerr := context.Cause(ctx); cerr != nil {
			return res, cerr
		}

		// Calculate next backoff duration.
		next := args.BackOff.NextBackOff()
		if next == Stop {
			return res, err
		}

		// Reset backoff if RetryAfterError is encountered.
		var retryAfter *RetryAfterError
		if errors.As(err, &retryAfter) {
			next = retryAfter.Duration
			args.BackOff.Reset()
		}

		// Stop retrying if maximum elapsed time exceeded.
		if args.MaxElapsedTime > 0 && time.Since(startedAt)+next > args.MaxElapsedTime {
			return res, err
		}

		// Notify on error if a notifier function is provided.
		if args.Notify != nil {
			args.Notify(err, next)
		}

		// Wait for the next backoff period or context cancellation.
		args.Timer.Start(next)
		select {
		case <-args.Timer.C():
		case <-ctx.Done():
			return res, context.Cause(ctx)
		}
	}
}
