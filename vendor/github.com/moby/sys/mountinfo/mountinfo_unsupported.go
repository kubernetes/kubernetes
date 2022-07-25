//go:build (!windows && !linux && !freebsd && !openbsd && !darwin) || (freebsd && !cgo) || (openbsd && !cgo) || (darwin && !cgo)
// +build !windows,!linux,!freebsd,!openbsd,!darwin freebsd,!cgo openbsd,!cgo darwin,!cgo

package mountinfo

import (
	"fmt"
	"runtime"
)

var errNotImplemented = fmt.Errorf("not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)

func parseMountTable(_ FilterFunc) ([]*Info, error) {
	return nil, errNotImplemented
}

func mounted(path string) (bool, error) {
	return false, errNotImplemented
}
