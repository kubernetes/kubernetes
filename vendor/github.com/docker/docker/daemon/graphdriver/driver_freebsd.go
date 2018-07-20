package graphdriver

import (
	"syscall"

	"golang.org/x/sys/unix"
)

var (
	// List of drivers that should be used in an order
	priority = "zfs"
)

// Mounted checks if the given path is mounted as the fs type
func Mounted(fsType FsMagic, mountPath string) (bool, error) {
	var buf unix.Statfs_t
	if err := syscall.Statfs(mountPath, &buf); err != nil {
		return false, err
	}
	return FsMagic(buf.Type) == fsType, nil
}
