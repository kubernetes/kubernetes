package ioutils

import (
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
)

// NewAtomicFileWriter returns WriteCloser so that writing to it writes to a
// temporary file and closing it atomically changes the temporary file to
// destination path. Writing and closing concurrently is not allowed.
func NewAtomicFileWriter(filename string, perm os.FileMode) (io.WriteCloser, error) {
	f, err := ioutil.TempFile(filepath.Dir(filename), ".tmp-"+filepath.Base(filename))
	if err != nil {
		return nil, err
	}

	abspath, err := filepath.Abs(filename)
	if err != nil {
		return nil, err
	}
	return &atomicFileWriter{
		f:    f,
		fn:   abspath,
		perm: perm,
	}, nil
}

// AtomicWriteFile atomically writes data to a file named by filename.
func AtomicWriteFile(filename string, data []byte, perm os.FileMode) error {
	f, err := NewAtomicFileWriter(filename, perm)
	if err != nil {
		return err
	}
	n, err := f.Write(data)
	if err == nil && n < len(data) {
		err = io.ErrShortWrite
		f.(*atomicFileWriter).writeErr = err
	}
	if err1 := f.Close(); err == nil {
		err = err1
	}
	return err
}

type atomicFileWriter struct {
	f        *os.File
	fn       string
	writeErr error
	perm     os.FileMode
}

func (w *atomicFileWriter) Write(dt []byte) (int, error) {
	n, err := w.f.Write(dt)
	if err != nil {
		w.writeErr = err
	}
	return n, err
}

func (w *atomicFileWriter) Close() (retErr error) {
	defer func() {
		if retErr != nil || w.writeErr != nil {
			os.Remove(w.f.Name())
		}
	}()
	if err := w.f.Sync(); err != nil {
		w.f.Close()
		return err
	}
	if err := w.f.Close(); err != nil {
		return err
	}
	if err := os.Chmod(w.f.Name(), w.perm); err != nil {
		return err
	}
	if w.writeErr == nil {
		return os.Rename(w.f.Name(), w.fn)
	}
	return nil
}

// AtomicWriteSet is used to atomically write a set
// of files and ensure they are visible at the same time.
// Must be committed to a new directory.
type AtomicWriteSet struct {
	root string
}

// NewAtomicWriteSet creates a new atomic write set to
// atomically create a set of files. The given directory
// is used as the base directory for storing files before
// commit. If no temporary directory is given the system
// default is used.
func NewAtomicWriteSet(tmpDir string) (*AtomicWriteSet, error) {
	td, err := ioutil.TempDir(tmpDir, "write-set-")
	if err != nil {
		return nil, err
	}

	return &AtomicWriteSet{
		root: td,
	}, nil
}

// WriteFile writes a file to the set, guaranteeing the file
// has been synced.
func (ws *AtomicWriteSet) WriteFile(filename string, data []byte, perm os.FileMode) error {
	f, err := ws.FileWriter(filename, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, perm)
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

type syncFileCloser struct {
	*os.File
}

func (w syncFileCloser) Close() error {
	err := w.File.Sync()
	if err1 := w.File.Close(); err == nil {
		err = err1
	}
	return err
}

// FileWriter opens a file writer inside the set. The file
// should be synced and closed before calling commit.
func (ws *AtomicWriteSet) FileWriter(name string, flag int, perm os.FileMode) (io.WriteCloser, error) {
	f, err := os.OpenFile(filepath.Join(ws.root, name), flag, perm)
	if err != nil {
		return nil, err
	}
	return syncFileCloser{f}, nil
}

// Cancel cancels the set and removes all temporary data
// created in the set.
func (ws *AtomicWriteSet) Cancel() error {
	return os.RemoveAll(ws.root)
}

// Commit moves all created files to the target directory. The
// target directory must not exist and the parent of the target
// directory must exist.
func (ws *AtomicWriteSet) Commit(target string) error {
	return os.Rename(ws.root, target)
}

// String returns the location the set is writing to.
func (ws *AtomicWriteSet) String() string {
	return ws.root
}
