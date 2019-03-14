// Package retry provides a simple, stateless, functional mechanism to perform
// actions repetitively until successful.
//
// Copyright © 2016 Trevor N. Suarez (Rican7)
package retry

import "github.com/Rican7/retry/strategy"

// Action defines a callable function that package retry can handle.
type Action func(attempt uint) error

// Retry takes an action and performs it, repetitively, until successful.
//
// Optionally, strategies may be passed that assess whether or not an attempt
// should be made.
func Retry(action Action, strategies ...strategy.Strategy) error {
	var err error

	for attempt := uint(0); (0 == attempt || nil != err) && shouldAttempt(attempt, strategies...); attempt++ {
		err = action(attempt)
	}

	return err
}

// shouldAttempt evaluates the provided strategies with the given attempt to
// determine if the Retry loop should make another attempt.
func shouldAttempt(attempt uint, strategies ...strategy.Strategy) bool {
	shouldAttempt := true

	for i := 0; shouldAttempt && i < len(strategies); i++ {
		shouldAttempt = shouldAttempt && strategies[i](attempt)
	}

	return shouldAttempt
}
