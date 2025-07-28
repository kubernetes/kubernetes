//go:build !windows

package configfile

import (
	"os"
	"syscall"
)

// copyFilePermissions copies file ownership and permissions from "src" to "dst",
// ignoring any error during the process.
func copyFilePermissions(src, dst string) {
	var (
		mode     os.FileMode = 0o600
		uid, gid int
	)

	fi, err := os.Stat(src)
	if err != nil {
		return
	}
	if fi.Mode().IsRegular() {
		mode = fi.Mode()
	}
	if err := os.Chmod(dst, mode); err != nil {
		return
	}

	uid = int(fi.Sys().(*syscall.Stat_t).Uid)
	gid = int(fi.Sys().(*syscall.Stat_t).Gid)

	if uid > 0 && gid > 0 {
		_ = os.Chown(dst, uid, gid)
	}
}
