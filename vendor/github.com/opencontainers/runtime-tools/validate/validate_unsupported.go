// +build !linux

package validate

import (
	"github.com/syndtr/gocapability/capability"
)

// LastCap return last cap of system
func LastCap() capability.Cap {
	return capability.Cap(-1)
}

// CheckLinux is a noop on this platform
func (v *Validator) CheckLinux() (errs error) {
	return nil
}
