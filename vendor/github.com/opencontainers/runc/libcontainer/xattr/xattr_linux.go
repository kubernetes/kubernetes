// +build linux

package xattr

import (
	"github.com/opencontainers/runc/libcontainer/system"

	"golang.org/x/sys/unix"
)

func XattrEnabled(path string) bool {
	if Setxattr(path, "user.test", "") == unix.ENOTSUP {
		return false
	}
	return true
}

func stringsfromByte(buf []byte) (result []string) {
	offset := 0
	for index, b := range buf {
		if b == 0 {
			result = append(result, string(buf[offset:index]))
			offset = index + 1
		}
	}
	return
}

func Listxattr(path string) ([]string, error) {
	size, err := unix.Llistxattr(path, nil)
	if err != nil {
		return nil, err
	}
	buf := make([]byte, size)
	read, err := unix.Llistxattr(path, buf)
	if err != nil {
		return nil, err
	}
	names := stringsfromByte(buf[:read])
	return names, nil
}

func Getxattr(path, attr string) (string, error) {
	value, err := system.Lgetxattr(path, attr)
	if err != nil {
		return "", err
	}
	return string(value), nil
}

func Setxattr(path, xattr, value string) error {
	return unix.Lsetxattr(path, xattr, []byte(value), 0)
}
