// Package backoff provides stateless methods of calculating durations based on
// a number of attempts made.
//
// Copyright © 2016 Trevor N. Suarez (Rican7)
package backoff

import (
	"math"
	"time"
)

// Algorithm defines a function that calculates a time.Duration based on
// the given retry attempt number.
type Algorithm func(attempt uint) time.Duration

// Incremental creates a Algorithm that increments the initial duration
// by the given increment for each attempt.
func Incremental(initial, increment time.Duration) Algorithm {
	return func(attempt uint) time.Duration {
		return initial + (increment * time.Duration(attempt))
	}
}

// Linear creates a Algorithm that linearly multiplies the factor
// duration by the attempt number for each attempt.
func Linear(factor time.Duration) Algorithm {
	return func(attempt uint) time.Duration {
		return (factor * time.Duration(attempt))
	}
}

// Exponential creates a Algorithm that multiplies the factor duration by
// an exponentially increasing factor for each attempt, where the factor is
// calculated as the given base raised to the attempt number.
func Exponential(factor time.Duration, base float64) Algorithm {
	return func(attempt uint) time.Duration {
		return (factor * time.Duration(math.Pow(base, float64(attempt))))
	}
}

// BinaryExponential creates a Algorithm that multiplies the factor
// duration by an exponentially increasing factor for each attempt, where the
// factor is calculated as `2` raised to the attempt number (2^attempt).
func BinaryExponential(factor time.Duration) Algorithm {
	return Exponential(factor, 2)
}

// Fibonacci creates a Algorithm that multiplies the factor duration by
// an increasing factor for each attempt, where the factor is the Nth number in
// the Fibonacci sequence.
func Fibonacci(factor time.Duration) Algorithm {
	return func(attempt uint) time.Duration {
		return (factor * time.Duration(fibonacciNumber(attempt)))
	}
}

// fibonacciNumber calculates the Fibonacci sequence number for the given
// sequence position.
func fibonacciNumber(n uint) uint {
	if 0 == n {
		return 0
	} else if 1 == n {
		return 1
	} else {
		return fibonacciNumber(n-1) + fibonacciNumber(n-2)
	}
}
