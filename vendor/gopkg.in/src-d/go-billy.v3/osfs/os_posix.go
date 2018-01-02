// +build !windows

package osfs

import (
	"os"
)

// Stat returns the FileInfo structure describing file.
func (fs *OS) Stat(filename string) (os.FileInfo, error) {
	return os.Stat(filename)
}
