// +build !windows

package system

import (
	"os"
)

func MkdirAll(path string, perm os.FileMode) error {
	return os.MkdirAll(path, perm)
}
