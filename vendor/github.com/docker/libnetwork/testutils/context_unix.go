// +build linux freebsd

package testutils

import (
	"os"
	"runtime"
	"syscall"
	"testing"

	"github.com/docker/libnetwork/ns"
)

// SetupTestOSContext joins a new network namespace, and returns its associated
// teardown function.
//
// Example usage:
//
//     defer SetupTestOSContext(t)()
//
func SetupTestOSContext(t *testing.T) func() {
	runtime.LockOSThread()
	if err := syscall.Unshare(syscall.CLONE_NEWNET); err != nil {
		t.Fatalf("Failed to enter netns: %v", err)
	}

	fd, err := syscall.Open("/proc/self/ns/net", syscall.O_RDONLY, 0)
	if err != nil {
		t.Fatal("Failed to open netns file")
	}

	// Since we are switching to a new test namespace make
	// sure to re-initialize initNs context
	ns.Init()

	runtime.LockOSThread()

	return func() {
		if err := syscall.Close(fd); err != nil {
			t.Logf("Warning: netns closing failed (%v)", err)
		}
		runtime.UnlockOSThread()
	}
}

// RunningOnCircleCI returns true if being executed on libnetwork Circle CI setup
func RunningOnCircleCI() bool {
	return os.Getenv("CIRCLECI") != ""
}
