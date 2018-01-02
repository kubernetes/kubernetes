// +build !windows

package daemon

import (
	"os"
	"path/filepath"

	"github.com/go-check/check"
	"golang.org/x/sys/unix"
)

func cleanupExecRoot(c *check.C, execRoot string) {
	// Cleanup network namespaces in the exec root of this
	// daemon because this exec root is specific to this
	// daemon instance and has no chance of getting
	// cleaned up when a new daemon is instantiated with a
	// new exec root.
	netnsPath := filepath.Join(execRoot, "netns")
	filepath.Walk(netnsPath, func(path string, info os.FileInfo, err error) error {
		if err := unix.Unmount(path, unix.MNT_FORCE); err != nil {
			c.Logf("unmount of %s failed: %v", path, err)
		}
		os.Remove(path)
		return nil
	})
}

// SignalDaemonDump sends a signal to the daemon to write a dump file
func SignalDaemonDump(pid int) {
	unix.Kill(pid, unix.SIGQUIT)
}

func signalDaemonReload(pid int) error {
	return unix.Kill(pid, unix.SIGHUP)
}
