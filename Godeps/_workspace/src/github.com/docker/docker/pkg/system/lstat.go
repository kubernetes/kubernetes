// +build !windows

package system

import (
	"syscall"
)

func Lstat(path string) (*Stat_t, error) {
	s := &syscall.Stat_t{}
	err := syscall.Lstat(path, s)
	if err != nil {
		return nil, err
	}
	return fromStatT(s)
}
