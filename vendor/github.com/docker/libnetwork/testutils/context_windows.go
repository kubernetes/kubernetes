package testutils

import (
	"os"
	"testing"
)

// SetupTestOSContext joins a new network namespace, and returns its associated
// teardown function.
//
// Example usage:
//
//     defer SetupTestOSContext(t)()
//
func SetupTestOSContext(t *testing.T) func() {
	return func() {
	}
}

// RunningOnCircleCI returns true if being executed on libnetwork Circle CI setup
func RunningOnCircleCI() bool {
	return os.Getenv("CIRCLECI") != ""
}
