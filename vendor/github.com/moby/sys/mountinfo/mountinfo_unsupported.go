// +build !windows,!linux,!freebsd freebsd,!cgo

package mountinfo

import (
	"fmt"
	"io"
	"runtime"
)

func parseMountTable(_ FilterFunc) ([]*Info, error) {
	return nil, fmt.Errorf("mount.parseMountTable is not implemented on %s/%s", runtime.GOOS, runtime.GOARCH)
}

func parseInfoFile(_ io.Reader, f FilterFunc) ([]*Info, error) {
	return parseMountTable(f)
}
