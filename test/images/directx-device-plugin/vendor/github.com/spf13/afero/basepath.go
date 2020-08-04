package afero

import (
	"errors"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"time"
)

// The BasePathFs restricts all operations to a given path within an Fs.
// The given file name to the operations on this Fs will be prepended with
// the base path before calling the base Fs.
// Any file name (after filepath.Clean()) outside this base path will be
// treated as non existing file.
//
// Note that it does not clean the error messages on return, so you may
// reveal the real path on errors.
type BasePathFs struct {
	source Fs
	path   string
}

func NewBasePathFs(source Fs, path string) Fs {
	return &BasePathFs{source: source, path: path}
}

// on a file outside the base path it returns the given file name and an error,
// else the given file with the base path prepended
func (b *BasePathFs) RealPath(name string) (path string, err error) {
	if err := validateBasePathName(name); err != nil {
		return "", err
	}

	bpath := filepath.Clean(b.path)
	path = filepath.Clean(filepath.Join(bpath, name))
	if !strings.HasPrefix(path, bpath) {
		return name, os.ErrNotExist
	}

	return path, nil
}

func validateBasePathName(name string) error {
	if runtime.GOOS != "windows" {
		// Not much to do here;
		// the virtual file paths all look absolute on *nix.
		return nil
	}

	// On Windows a common mistake would be to provide an absolute OS path
	// We could strip out the base part, but that would not be very portable.
	if filepath.IsAbs(name) {
		return &os.PathError{"realPath", name, errors.New("got a real OS path instead of a virtual")}
	}

	return nil
}

func (b *BasePathFs) Chtimes(name string, atime, mtime time.Time) (err error) {
	if name, err = b.RealPath(name); err != nil {
		return &os.PathError{"chtimes", name, err}
	}
	return b.source.Chtimes(name, atime, mtime)
}

func (b *BasePathFs) Chmod(name string, mode os.FileMode) (err error) {
	if name, err = b.RealPath(name); err != nil {
		return &os.PathError{"chmod", name, err}
	}
	return b.source.Chmod(name, mode)
}

func (b *BasePathFs) Name() string {
	return "BasePathFs"
}

func (b *BasePathFs) Stat(name string) (fi os.FileInfo, err error) {
	if name, err = b.RealPath(name); err != nil {
		return nil, &os.PathError{"stat", name, err}
	}
	return b.source.Stat(name)
}

func (b *BasePathFs) Rename(oldname, newname string) (err error) {
	if oldname, err = b.RealPath(oldname); err != nil {
		return &os.PathError{"rename", oldname, err}
	}
	if newname, err = b.RealPath(newname); err != nil {
		return &os.PathError{"rename", newname, err}
	}
	return b.source.Rename(oldname, newname)
}

func (b *BasePathFs) RemoveAll(name string) (err error) {
	if name, err = b.RealPath(name); err != nil {
		return &os.PathError{"remove_all", name, err}
	}
	return b.source.RemoveAll(name)
}

func (b *BasePathFs) Remove(name string) (err error) {
	if name, err = b.RealPath(name); err != nil {
		return &os.PathError{"remove", name, err}
	}
	return b.source.Remove(name)
}

func (b *BasePathFs) OpenFile(name string, flag int, mode os.FileMode) (f File, err error) {
	if name, err = b.RealPath(name); err != nil {
		return nil, &os.PathError{"openfile", name, err}
	}
	return b.source.OpenFile(name, flag, mode)
}

func (b *BasePathFs) Open(name string) (f File, err error) {
	if name, err = b.RealPath(name); err != nil {
		return nil, &os.PathError{"open", name, err}
	}
	return b.source.Open(name)
}

func (b *BasePathFs) Mkdir(name string, mode os.FileMode) (err error) {
	if name, err = b.RealPath(name); err != nil {
		return &os.PathError{"mkdir", name, err}
	}
	return b.source.Mkdir(name, mode)
}

func (b *BasePathFs) MkdirAll(name string, mode os.FileMode) (err error) {
	if name, err = b.RealPath(name); err != nil {
		return &os.PathError{"mkdir", name, err}
	}
	return b.source.MkdirAll(name, mode)
}

func (b *BasePathFs) Create(name string) (f File, err error) {
	if name, err = b.RealPath(name); err != nil {
		return nil, &os.PathError{"create", name, err}
	}
	return b.source.Create(name)
}

// vim: ts=4 sw=4 noexpandtab nolist syn=go
