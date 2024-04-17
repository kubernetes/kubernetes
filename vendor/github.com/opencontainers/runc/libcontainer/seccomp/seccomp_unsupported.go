//go:build !linux || !cgo || !seccomp
// +build !linux !cgo !seccomp

package seccomp

import (
	"errors"
	"os"

	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/opencontainers/runtime-spec/specs-go"
)

var ErrSeccompNotEnabled = errors.New("seccomp: config provided but seccomp not supported")

// InitSeccomp does nothing because seccomp is not supported.
func InitSeccomp(config *configs.Seccomp) (*os.File, error) {
	if config != nil {
		return nil, ErrSeccompNotEnabled
	}
	return nil, nil
}

// FlagSupported tells if a provided seccomp flag is supported.
func FlagSupported(_ specs.LinuxSeccompFlag) error {
	return ErrSeccompNotEnabled
}

// Version returns major, minor, and micro.
func Version() (uint, uint, uint) {
	return 0, 0, 0
}

// Enabled is true if seccomp support is compiled in.
const Enabled = false
