package godog

import (
	"io"
)

// Options are suite run options
// flags are mapped to these options.
//
// It can also be used together with godog.RunWithOptions
// to run test suite from go source directly
//
// See the flags for more details
type Options struct {
	// Print step definitions found and exit
	ShowStepDefinitions bool

	// Randomize, if not `0`, will be used to run scenarios in a random order.
	//
	// Randomizing scenario order is especially helpful for detecting
	// situations where you have state leaking between scenarios, which can
	// cause flickering or fragile tests.
	//
	// The default value of `0` means "do not randomize".
	//
	// The magic value of `-1` means "pick a random seed for me", and godog will
	// assign a seed on it's own during the `RunWithOptions` phase, similar to if
	// you specified `--random` on the command line.
	//
	// Any other value will be used as the random seed for shuffling. Re-using the
	// same seed will allow you to reproduce the shuffle order of a previous run
	// to isolate an error condition.
	Randomize int64

	// Stops on the first failure
	StopOnFailure bool

	// Fail suite when there are pending or undefined steps
	Strict bool

	// Forces ansi color stripping
	NoColors bool

	// Various filters for scenarios parsed
	// from feature files
	Tags string

	// The formatter name
	Format string

	// Concurrency rate, not all formatters accepts this
	Concurrency int

	// All feature file paths
	Paths []string

	// Where it should print formatter output
	Output io.Writer
}
