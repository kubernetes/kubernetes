// +build !linux

package volume

import (
	"fmt"
	"runtime"

	mounttypes "github.com/docker/docker/api/types/mount"
)

// ConvertTmpfsOptions converts *mounttypes.TmpfsOptions to the raw option string
// for mount(2).
func ConvertTmpfsOptions(opt *mounttypes.TmpfsOptions, readOnly bool) (string, error) {
	return "", fmt.Errorf("%s does not support tmpfs", runtime.GOOS)
}
