// +build windows

package system

import (
	"errors"
	"syscall"
)

func fromStatT(s *syscall.Win32FileAttributeData) (*Stat_t, error) {
	return nil, errors.New("fromStatT should not be called on windows path")
}

func Stat(path string) (*Stat_t, error) {
	// should not be called on cli code path
	return nil, ErrNotSupportedPlatform
}
