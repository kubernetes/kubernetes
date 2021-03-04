// +build !windows

package configs

import (
	"errors"

	"golang.org/x/sys/unix"
)

func (d *DeviceRule) Mkdev() (uint64, error) {
	if d.Major == Wildcard || d.Minor == Wildcard {
		return 0, errors.New("cannot mkdev() device with wildcards")
	}
	return unix.Mkdev(uint32(d.Major), uint32(d.Minor)), nil
}
