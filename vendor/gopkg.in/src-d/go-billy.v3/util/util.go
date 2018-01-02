package util

import (
	"errors"
	"fmt"
	"io"
	"os"
	"sync/atomic"
	"time"

	"gopkg.in/src-d/go-billy.v3"
)

// RemoveAll removes path and any children it contains. It removes everything it
// can but returns the first error it encounters. If the path does not exist,
// RemoveAll returns nil (no error).
func RemoveAll(fs billy.Basic, path string) error {
	fs, path = getUnderlyingAndPath(fs, path)

	if r, ok := fs.(removerAll); ok {
		return r.RemoveAll(path)
	}

	return removeAll(fs, path)
}

type removerAll interface {
	RemoveAll(string) error
}

func removeAll(fs billy.Basic, path string) error {
	// This implementation is adapted from os.RemoveAll.

	// Simple case: if Remove works, we're done.
	err := fs.Remove(path)
	if err == nil || os.IsNotExist(err) {
		return nil
	}

	// Otherwise, is this a directory we need to recurse into?
	dir, serr := fs.Stat(path)
	if serr != nil {
		if os.IsNotExist(serr) {
			return nil
		}

		return serr
	}

	if !dir.IsDir() {
		// Not a directory; return the error from Remove.
		return err
	}

	dirfs, ok := fs.(billy.Dir)
	if !ok {
		return billy.ErrNotSupported
	}

	// Directory.
	fis, err := dirfs.ReadDir(path)
	if err != nil {
		if os.IsNotExist(err) {
			// Race. It was deleted between the Lstat and Open.
			// Return nil per RemoveAll's docs.
			return nil
		}

		return err
	}

	// Remove contents & return first error.
	err = nil
	for _, fi := range fis {
		cpath := fs.Join(path, fi.Name())
		err1 := removeAll(fs, cpath)
		if err == nil {
			err = err1
		}
	}

	// Remove directory.
	err1 := fs.Remove(path)
	if err1 == nil || os.IsNotExist(err1) {
		return nil
	}

	if err == nil {
		err = err1
	}

	return err

}

// WriteFile writes data to a file named by filename in the given filesystem.
// If the file does not exist, WriteFile creates it with permissions perm;
// otherwise WriteFile truncates it before writing.
func WriteFile(fs billy.Basic, filename string, data []byte, perm os.FileMode) error {
	f, err := fs.OpenFile(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
	if err != nil {
		return err
	}

	n, err := f.Write(data)
	if err == nil && n < len(data) {
		err = io.ErrShortWrite
	}

	if err1 := f.Close(); err == nil {
		err = err1
	}

	return err
}

var (
	MaxTempFiles int32 = 1024 * 4
	tempCount    int32
)

// TempFile creates a new temporary file in the directory dir with a name
// beginning with prefix, opens the file for reading and writing, and returns
// the resulting *os.File. If dir is the empty string, TempFile uses the default
// directory for temporary files (see os.TempDir).
//
// Multiple programs calling TempFile simultaneously will not choose the same
// file. The caller can use f.Name() to find the pathname of the file.
//
// It is the caller's responsibility to remove the file when no longer needed.
func TempFile(fs billy.Basic, dir, prefix string) (billy.File, error) {
	var fullpath string
	for {
		if tempCount >= MaxTempFiles {
			return nil, errors.New("max. number of tempfiles reached")
		}

		fullpath = getTempFilename(fs, dir, prefix)
		break
	}

	return fs.Create(fullpath)
}

func getTempFilename(fs billy.Basic, dir, prefix string) string {
	atomic.AddInt32(&tempCount, 1)
	filename := fmt.Sprintf("%s_%d_%d", prefix, tempCount, time.Now().UnixNano())
	return fs.Join(dir, filename)
}

type underlying interface {
	Underlying() billy.Basic
}

func getUnderlyingAndPath(fs billy.Basic, path string) (billy.Basic, string) {
	u, ok := fs.(underlying)
	if !ok {
		return fs, path
	}
	if ch, ok := fs.(billy.Chroot); ok {
		path = fs.Join(ch.Root(), path)
	}

	return u.Underlying(), path
}
