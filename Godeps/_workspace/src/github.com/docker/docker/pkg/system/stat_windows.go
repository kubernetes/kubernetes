// +build windows

package system

import (
	"errors"
	"syscall"
)

func fromStatT(s *syscall.Win32FileAttributeData) (*Stat, error) {
	return nil, errors.New("fromStatT should not be called on windows path")
}
