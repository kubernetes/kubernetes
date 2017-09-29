// +build !windows

package system

import (
	"io/ioutil"
	"os"
	"path/filepath"
)

// MkdirAllWithACL is a wrapper for MkdirAll on unix systems.
func MkdirAllWithACL(path string, perm os.FileMode, sddl string) error {
	return MkdirAll(path, perm, sddl)
}

// MkdirAll creates a directory named path along with any necessary parents,
// with permission specified by attribute perm for all dir created.
func MkdirAll(path string, perm os.FileMode, sddl string) error {
	return os.MkdirAll(path, perm)
}

// IsAbs is a platform-specific wrapper for filepath.IsAbs.
func IsAbs(path string) bool {
	return filepath.IsAbs(path)
}

// The functions below here are wrappers for the equivalents in the os and ioutils packages.
// They are passthrough on Unix platforms, and only relevant on Windows.

// CreateSequential creates the named file with mode 0666 (before umask), truncating
// it if it already exists. If successful, methods on the returned
// File can be used for I/O; the associated file descriptor has mode
// O_RDWR.
// If there is an error, it will be of type *PathError.
func CreateSequential(name string) (*os.File, error) {
	return os.Create(name)
}

// OpenSequential opens the named file for reading. If successful, methods on
// the returned file can be used for reading; the associated file
// descriptor has mode O_RDONLY.
// If there is an error, it will be of type *PathError.
func OpenSequential(name string) (*os.File, error) {
	return os.Open(name)
}

// OpenFileSequential is the generalized open call; most users will use Open
// or Create instead. It opens the named file with specified flag
// (O_RDONLY etc.) and perm, (0666 etc.) if applicable. If successful,
// methods on the returned File can be used for I/O.
// If there is an error, it will be of type *PathError.
func OpenFileSequential(name string, flag int, perm os.FileMode) (*os.File, error) {
	return os.OpenFile(name, flag, perm)
}

// TempFileSequential creates a new temporary file in the directory dir
// with a name beginning with prefix, opens the file for reading
// and writing, and returns the resulting *os.File.
// If dir is the empty string, TempFile uses the default directory
// for temporary files (see os.TempDir).
// Multiple programs calling TempFile simultaneously
// will not choose the same file. The caller can use f.Name()
// to find the pathname of the file. It is the caller's responsibility
// to remove the file when no longer needed.
func TempFileSequential(dir, prefix string) (f *os.File, err error) {
	return ioutil.TempFile(dir, prefix)
}
