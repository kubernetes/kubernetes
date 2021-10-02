// +build !linux !cgo !seccomp

package seccomp

import (
	"errors"

	"github.com/opencontainers/runc/libcontainer/configs"
)

var ErrSeccompNotEnabled = errors.New("seccomp: config provided but seccomp not supported")

// InitSeccomp does nothing because seccomp is not supported.
func InitSeccomp(config *configs.Seccomp) error {
	if config != nil {
		return ErrSeccompNotEnabled
	}
	return nil
}

// Version returns major, minor, and micro.
func Version() (uint, uint, uint) {
	return 0, 0, 0
}
