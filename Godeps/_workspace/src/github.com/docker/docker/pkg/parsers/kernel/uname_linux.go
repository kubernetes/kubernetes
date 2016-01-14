package kernel

import (
	"syscall"
)

// Utsname represents the system name structure.
// It is passthgrouh for syscall.Utsname in order to make it portable with
// other platforms where it is not available.
type Utsname syscall.Utsname

func uname() (*syscall.Utsname, error) {
	uts := &syscall.Utsname{}

	if err := syscall.Uname(uts); err != nil {
		return nil, err
	}
	return uts, nil
}
