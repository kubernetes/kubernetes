package util

import (
	"io"
	"os"
	"path/filepath"
	"strconv"
	"sync"
	"time"

	"github.com/go-git/go-billy/v5"
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

// Random number state.
// We generate random temporary file names so that there's a good
// chance the file doesn't exist yet - keeps the number of tries in
// TempFile to a minimum.
var rand uint32
var randmu sync.Mutex

func reseed() uint32 {
	return uint32(time.Now().UnixNano() + int64(os.Getpid()))
}

func nextSuffix() string {
	randmu.Lock()
	r := rand
	if r == 0 {
		r = reseed()
	}
	r = r*1664525 + 1013904223 // constants from Numerical Recipes
	rand = r
	randmu.Unlock()
	return strconv.Itoa(int(1e9 + r%1e9))[1:]
}

// TempFile creates a new temporary file in the directory dir with a name
// beginning with prefix, opens the file for reading and writing, and returns
// the resulting *os.File. If dir is the empty string, TempFile uses the default
// directory for temporary files (see os.TempDir). Multiple programs calling
// TempFile simultaneously will not choose the same file. The caller can use
// f.Name() to find the pathname of the file. It is the caller's responsibility
// to remove the file when no longer needed.
func TempFile(fs billy.Basic, dir, prefix string) (f billy.File, err error) {
	// This implementation is based on stdlib ioutil.TempFile.

	if dir == "" {
		dir = os.TempDir()
	}

	nconflict := 0
	for i := 0; i < 10000; i++ {
		name := filepath.Join(dir, prefix+nextSuffix())
		f, err = fs.OpenFile(name, os.O_RDWR|os.O_CREATE|os.O_EXCL, 0600)
		if os.IsExist(err) {
			if nconflict++; nconflict > 10 {
				randmu.Lock()
				rand = reseed()
				randmu.Unlock()
			}
			continue
		}
		break
	}
	return
}

// TempDir creates a new temporary directory in the directory dir
// with a name beginning with prefix and returns the path of the
// new directory. If dir is the empty string, TempDir uses the
// default directory for temporary files (see os.TempDir).
// Multiple programs calling TempDir simultaneously
// will not choose the same directory. It is the caller's responsibility
// to remove the directory when no longer needed.
func TempDir(fs billy.Dir, dir, prefix string) (name string, err error) {
	// This implementation is based on stdlib ioutil.TempDir

	if dir == "" {
		dir = os.TempDir()
	}

	nconflict := 0
	for i := 0; i < 10000; i++ {
		try := filepath.Join(dir, prefix+nextSuffix())
		err = fs.MkdirAll(try, 0700)
		if os.IsExist(err) {
			if nconflict++; nconflict > 10 {
				randmu.Lock()
				rand = reseed()
				randmu.Unlock()
			}
			continue
		}
		if os.IsNotExist(err) {
			if _, err := os.Stat(dir); os.IsNotExist(err) {
				return "", err
			}
		}
		if err == nil {
			name = try
		}
		break
	}
	return
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
