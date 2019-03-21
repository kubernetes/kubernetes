// +build darwin linux

package godirwalk

import "syscall"

func inoFromDirent(de *syscall.Dirent) uint64 {
	return de.Ino
}
