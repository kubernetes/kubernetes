// +build !linux,!freebsd freebsd,!cgo

package mount

import (
	"fmt"
	"runtime"
)

func parseMountTable() ([]*MountInfo, error) {
	return nil, fmt.Errorf("mount.parseMountTable is not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
}
