package safetemp

import (
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
)

// Dir creates a new temporary directory that isn't yet created. This
// can be used with calls that expect a non-existent directory.
//
// The directory is created as a child of a temporary directory created
// within the directory dir starting with prefix. The temporary directory
// returned is always named "temp". The parent directory has the specified
// prefix.
//
// The returned io.Closer should be used to clean up the returned directory.
// This will properly remove the returned directory and any other temporary
// files created.
//
// If an error is returned, the Closer does not need to be called (and will
// be nil).
func Dir(dir, prefix string) (string, io.Closer, error) {
	// Create the temporary directory
	td, err := ioutil.TempDir(dir, prefix)
	if err != nil {
		return "", nil, err
	}

	return filepath.Join(td, "temp"), pathCloser(td), nil
}

// pathCloser implements io.Closer to remove the given path on Close.
type pathCloser string

// Close deletes this path.
func (p pathCloser) Close() error {
	return os.RemoveAll(string(p))
}
