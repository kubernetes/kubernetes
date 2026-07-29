package config

import (
	"errors"

	"golang.org/x/sys/unix"
)

func mkDev(d *Rule) (uint64, error) {
	if d.Major == Wildcard || d.Minor == Wildcard {
		return 0, errors.New("cannot mkdev() device with wildcards")
	}
	return unix.Mkdev(uint32(d.Major), uint32(d.Minor)), nil
}
