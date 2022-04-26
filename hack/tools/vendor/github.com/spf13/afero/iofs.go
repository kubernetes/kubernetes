// +build go1.16

package afero

import (
	"io"
	"io/fs"
	"os"
	"path"
	"time"
)

// IOFS adopts afero.Fs to stdlib io/fs.FS
type IOFS struct {
	Fs
}

func NewIOFS(fs Fs) IOFS {
	return IOFS{Fs: fs}
}

var (
	_ fs.FS         = IOFS{}
	_ fs.GlobFS     = IOFS{}
	_ fs.ReadDirFS  = IOFS{}
	_ fs.ReadFileFS = IOFS{}
	_ fs.StatFS     = IOFS{}
	_ fs.SubFS      = IOFS{}
)

func (iofs IOFS) Open(name string) (fs.File, error) {
	const op = "open"

	// by convention for fs.FS implementations we should perform this check
	if !fs.ValidPath(name) {
		return nil, iofs.wrapError(op, name, fs.ErrInvalid)
	}

	file, err := iofs.Fs.Open(name)
	if err != nil {
		return nil, iofs.wrapError(op, name, err)
	}

	// file should implement fs.ReadDirFile
	if _, ok := file.(fs.ReadDirFile); !ok {
		file = readDirFile{file}
	}

	return file, nil
}

func (iofs IOFS) Glob(pattern string) ([]string, error) {
	const op = "glob"

	// afero.Glob does not perform this check but it's required for implementations
	if _, err := path.Match(pattern, ""); err != nil {
		return nil, iofs.wrapError(op, pattern, err)
	}

	items, err := Glob(iofs.Fs, pattern)
	if err != nil {
		return nil, iofs.wrapError(op, pattern, err)
	}

	return items, nil
}

func (iofs IOFS) ReadDir(name string) ([]fs.DirEntry, error) {
	items, err := ReadDir(iofs.Fs, name)
	if err != nil {
		return nil, iofs.wrapError("readdir", name, err)
	}

	ret := make([]fs.DirEntry, len(items))
	for i := range items {
		ret[i] = dirEntry{items[i]}
	}

	return ret, nil
}

func (iofs IOFS) ReadFile(name string) ([]byte, error) {
	const op = "readfile"

	if !fs.ValidPath(name) {
		return nil, iofs.wrapError(op, name, fs.ErrInvalid)
	}

	bytes, err := ReadFile(iofs.Fs, name)
	if err != nil {
		return nil, iofs.wrapError(op, name, err)
	}

	return bytes, nil
}

func (iofs IOFS) Sub(dir string) (fs.FS, error) { return IOFS{NewBasePathFs(iofs.Fs, dir)}, nil }

func (IOFS) wrapError(op, path string, err error) error {
	if _, ok := err.(*fs.PathError); ok {
		return err // don't need to wrap again
	}

	return &fs.PathError{
		Op:   op,
		Path: path,
		Err:  err,
	}
}

// dirEntry provides adapter from os.FileInfo to fs.DirEntry
type dirEntry struct {
	fs.FileInfo
}

var _ fs.DirEntry = dirEntry{}

func (d dirEntry) Type() fs.FileMode { return d.FileInfo.Mode().Type() }

func (d dirEntry) Info() (fs.FileInfo, error) { return d.FileInfo, nil }

// readDirFile provides adapter from afero.File to fs.ReadDirFile needed for correct Open
type readDirFile struct {
	File
}

var _ fs.ReadDirFile = readDirFile{}

func (r readDirFile) ReadDir(n int) ([]fs.DirEntry, error) {
	items, err := r.File.Readdir(n)
	if err != nil {
		return nil, err
	}

	ret := make([]fs.DirEntry, len(items))
	for i := range items {
		ret[i] = dirEntry{items[i]}
	}

	return ret, nil
}

// FromIOFS adopts io/fs.FS to use it as afero.Fs
// Note that io/fs.FS is read-only so all mutating methods will return fs.PathError with fs.ErrPermission
// To store modifications you may use afero.CopyOnWriteFs
type FromIOFS struct {
	fs.FS
}

var _ Fs = FromIOFS{}

func (f FromIOFS) Create(name string) (File, error) { return nil, notImplemented("create", name) }

func (f FromIOFS) Mkdir(name string, perm os.FileMode) error { return notImplemented("mkdir", name) }

func (f FromIOFS) MkdirAll(path string, perm os.FileMode) error {
	return notImplemented("mkdirall", path)
}

func (f FromIOFS) Open(name string) (File, error) {
	file, err := f.FS.Open(name)
	if err != nil {
		return nil, err
	}

	return fromIOFSFile{File: file, name: name}, nil
}

func (f FromIOFS) OpenFile(name string, flag int, perm os.FileMode) (File, error) {
	return f.Open(name)
}

func (f FromIOFS) Remove(name string) error {
	return notImplemented("remove", name)
}

func (f FromIOFS) RemoveAll(path string) error {
	return notImplemented("removeall", path)
}

func (f FromIOFS) Rename(oldname, newname string) error {
	return notImplemented("rename", oldname)
}

func (f FromIOFS) Stat(name string) (os.FileInfo, error) { return fs.Stat(f.FS, name) }

func (f FromIOFS) Name() string { return "fromiofs" }

func (f FromIOFS) Chmod(name string, mode os.FileMode) error {
	return notImplemented("chmod", name)
}

func (f FromIOFS) Chown(name string, uid, gid int) error {
	return notImplemented("chown", name)
}

func (f FromIOFS) Chtimes(name string, atime time.Time, mtime time.Time) error {
	return notImplemented("chtimes", name)
}

type fromIOFSFile struct {
	fs.File
	name string
}

func (f fromIOFSFile) ReadAt(p []byte, off int64) (n int, err error) {
	readerAt, ok := f.File.(io.ReaderAt)
	if !ok {
		return -1, notImplemented("readat", f.name)
	}

	return readerAt.ReadAt(p, off)
}

func (f fromIOFSFile) Seek(offset int64, whence int) (int64, error) {
	seeker, ok := f.File.(io.Seeker)
	if !ok {
		return -1, notImplemented("seek", f.name)
	}

	return seeker.Seek(offset, whence)
}

func (f fromIOFSFile) Write(p []byte) (n int, err error) {
	return -1, notImplemented("write", f.name)
}

func (f fromIOFSFile) WriteAt(p []byte, off int64) (n int, err error) {
	return -1, notImplemented("writeat", f.name)
}

func (f fromIOFSFile) Name() string { return f.name }

func (f fromIOFSFile) Readdir(count int) ([]os.FileInfo, error) {
	rdfile, ok := f.File.(fs.ReadDirFile)
	if !ok {
		return nil, notImplemented("readdir", f.name)
	}

	entries, err := rdfile.ReadDir(count)
	if err != nil {
		return nil, err
	}

	ret := make([]os.FileInfo, len(entries))
	for i := range entries {
		ret[i], err = entries[i].Info()

		if err != nil {
			return nil, err
		}
	}

	return ret, nil
}

func (f fromIOFSFile) Readdirnames(n int) ([]string, error) {
	rdfile, ok := f.File.(fs.ReadDirFile)
	if !ok {
		return nil, notImplemented("readdir", f.name)
	}

	entries, err := rdfile.ReadDir(n)
	if err != nil {
		return nil, err
	}

	ret := make([]string, len(entries))
	for i := range entries {
		ret[i] = entries[i].Name()
	}

	return ret, nil
}

func (f fromIOFSFile) Sync() error { return nil }

func (f fromIOFSFile) Truncate(size int64) error {
	return notImplemented("truncate", f.name)
}

func (f fromIOFSFile) WriteString(s string) (ret int, err error) {
	return -1, notImplemented("writestring", f.name)
}

func notImplemented(op, path string) error {
	return &fs.PathError{Op: op, Path: path, Err: fs.ErrPermission}
}
