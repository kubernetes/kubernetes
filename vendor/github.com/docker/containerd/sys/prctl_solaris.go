// +build solaris

package sys

import (
	"errors"
)

//Solaris TODO

// GetSubreaper returns the subreaper setting for the calling process
func GetSubreaper() (int, error) {
	return 0, errors.New("osutils GetSubreaper not implemented on Solaris")
}

// SetSubreaper sets the value i as the subreaper setting for the calling process
func SetSubreaper(i int) error {
	return errors.New("osutils SetSubreaper not implemented on Solaris")
}
