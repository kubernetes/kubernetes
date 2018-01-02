// +build windows

package mount

import (
	"fmt"
	"runtime"
)

// Lookup returns the mount info corresponds to the path.
func Lookup(dir string) (Info, error) {
	return Info{}, fmt.Errorf("mount.Lookup is not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
}
