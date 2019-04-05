// +build !windows,!linux,!freebsd freebsd,!cgo

package mount // import "github.com/docker/docker/pkg/mount"

import (
	"fmt"
	"runtime"
)

func parseMountTable(f FilterFunc) ([]*Info, error) {
	return nil, fmt.Errorf("mount.parseMountTable is not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
}
