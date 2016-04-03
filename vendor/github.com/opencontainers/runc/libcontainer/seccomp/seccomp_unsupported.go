// +build !linux !cgo !seccomp

package seccomp

import (
	"errors"

	"github.com/opencontainers/runc/libcontainer/configs"
)

var ErrSeccompNotEnabled = errors.New("seccomp: config provided but seccomp not supported")

// Seccomp not supported, do nothing
func InitSeccomp(config *configs.Seccomp) error {
	if config != nil {
		return ErrSeccompNotEnabled
	}
	return nil
}

// IsEnabled returns false, because it is not supported.
func IsEnabled() bool {
	return false
}
