// +build !windows

package meta

import "os"

// renameFile will rename the source to target using os function.
func renameFile(oldpath, newpath string) error {
	return os.Rename(oldpath, newpath)
}
