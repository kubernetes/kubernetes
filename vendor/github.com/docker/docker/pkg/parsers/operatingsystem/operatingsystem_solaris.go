// +build solaris,cgo

package operatingsystem

/*
#include <zone.h>
*/
import "C"

import (
	"bytes"
	"errors"
	"io/ioutil"
)

var etcOsRelease = "/etc/release"

// GetOperatingSystem gets the name of the current operating system.
func GetOperatingSystem() (string, error) {
	b, err := ioutil.ReadFile(etcOsRelease)
	if err != nil {
		return "", err
	}
	if i := bytes.Index(b, []byte("\n")); i >= 0 {
		b = bytes.Trim(b[:i], " ")
		return string(b), nil
	}
	return "", errors.New("release not found")
}

// IsContainerized returns true if we are running inside a container.
func IsContainerized() (bool, error) {
	if C.getzoneid() != 0 {
		return true, nil
	}
	return false, nil
}
