// +build !windows

package symlink

import (
	"path/filepath"
)

func evalSymlinks(path string) (string, error) {
	return filepath.EvalSymlinks(path)
}
