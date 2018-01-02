// +build !windows

package testutil

import (
	"fmt"
	"os"
	"testing"

	"github.com/containerd/containerd/mount"
	"github.com/stretchr/testify/assert"
)

// Unmount unmounts a given mountPoint and sets t.Error if it fails
func Unmount(t *testing.T, mountPoint string) {
	t.Log("unmount", mountPoint)
	if err := mount.UnmountAll(mountPoint, umountflags); err != nil {
		t.Error("Could not umount", mountPoint, err)
	}
}

// RequiresRoot skips tests that require root, unless the test.root flag has
// been set
func RequiresRoot(t testing.TB) {
	if !rootEnabled {
		t.Skip("skipping test that requires root")
		return
	}
	assert.Equal(t, 0, os.Getuid(), "This test must be run as root.")
}

// RequiresRootM is similar to RequiresRoot but intended to be called from *testing.M.
func RequiresRootM() {
	if !rootEnabled {
		fmt.Fprintln(os.Stderr, "skipping test that requires root")
		os.Exit(0)
	}
	if 0 != os.Getuid() {
		fmt.Fprintln(os.Stderr, "This test must be run as root.")
		os.Exit(1)
	}
}
