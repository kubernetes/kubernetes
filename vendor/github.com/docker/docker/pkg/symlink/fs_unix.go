// +build !windows

package symlink

import (
	"path/filepath"
)

func evalSymlinks(path string) (string, error) {
	return filepath.EvalSymlinks(path)
}

func isDriveOrRoot(p string) bool {
	return p == string(filepath.Separator)
}
