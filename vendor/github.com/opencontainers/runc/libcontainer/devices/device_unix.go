// +build !windows

package devices

import (
	"errors"

	"golang.org/x/sys/unix"
)

func (d *Rule) Mkdev() (uint64, error) {
	if d.Major == Wildcard || d.Minor == Wildcard {
		return 0, errors.New("cannot mkdev() device with wildcards")
	}
	return unix.Mkdev(uint32(d.Major), uint32(d.Minor)), nil
}
