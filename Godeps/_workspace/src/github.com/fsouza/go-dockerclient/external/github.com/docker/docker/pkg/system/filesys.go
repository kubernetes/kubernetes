// +build !windows

package system

import (
	"os"
	"path/filepath"
)

// MkdirAll creates a directory named path along with any necessary parents,
// with permission specified by attribute perm for all dir created.
func MkdirAll(path string, perm os.FileMode) error {
	return os.MkdirAll(path, perm)
}

// IsAbs is a platform-specific wrapper for filepath.IsAbs.
func IsAbs(path string) bool {
	return filepath.IsAbs(path)
}
