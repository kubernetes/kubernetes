// +build !windows

package system

import (
	"syscall"
)

// Lstat takes a path to a file and returns
// a system.Stat_t type pertaining to that file.
//
// Throws an error if the file does not exist
func Lstat(path string) (*Stat_t, error) {
	s := &syscall.Stat_t{}
	if err := syscall.Lstat(path, s); err != nil {
		return nil, err
	}
	return fromStatT(s)
}
