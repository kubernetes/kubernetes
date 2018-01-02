package chroot

import (
	"os"
	"path/filepath"
	"strings"

	"gopkg.in/src-d/go-billy.v3"
	"gopkg.in/src-d/go-billy.v3/helper/polyfill"
)

// ChrootHelper is a helper to implement billy.Chroot.
type ChrootHelper struct {
	underlying billy.Filesystem
	base       string
}

// New creates a new filesystem wrapping up the given 'fs'.
// The created filesystem has its base in the given ChrootHelperectory of the
// underlying filesystem.
func New(fs billy.Basic, base string) billy.Filesystem {
	return &ChrootHelper{
		underlying: polyfill.New(fs),
		base:       base,
	}
}

func (fs *ChrootHelper) underlyingPath(filename string) (string, error) {
	if isCrossBoundaries(filename) {
		return "", billy.ErrCrossedBoundary
	}

	return fs.Join(fs.Root(), filename), nil
}

func isCrossBoundaries(path string) bool {
	path = filepath.ToSlash(path)
	path = filepath.Clean(path)

	return strings.HasPrefix(path, "..")
}

func (fs *ChrootHelper) Create(filename string) (billy.File, error) {
	fullpath, err := fs.underlyingPath(filename)
	if err != nil {
		return nil, err
	}

	f, err := fs.underlying.Create(fullpath)
	if err != nil {
		return nil, err
	}

	return newFile(fs, f, filename), nil
}

func (fs *ChrootHelper) Open(filename string) (billy.File, error) {
	fullpath, err := fs.underlyingPath(filename)
	if err != nil {
		return nil, err
	}

	f, err := fs.underlying.Open(fullpath)
	if err != nil {
		return nil, err
	}

	return newFile(fs, f, filename), nil
}

func (fs *ChrootHelper) OpenFile(filename string, flag int, mode os.FileMode) (billy.File, error) {
	fullpath, err := fs.underlyingPath(filename)
	if err != nil {
		return nil, err
	}

	f, err := fs.underlying.OpenFile(fullpath, flag, mode)
	if err != nil {
		return nil, err
	}

	return newFile(fs, f, filename), nil
}

func (fs *ChrootHelper) Stat(filename string) (os.FileInfo, error) {
	fullpath, err := fs.underlyingPath(filename)
	if err != nil {
		return nil, err
	}

	return fs.underlying.Stat(fullpath)
}

func (fs *ChrootHelper) Rename(from, to string) error {
	var err error
	from, err = fs.underlyingPath(from)
	if err != nil {
		return err
	}

	to, err = fs.underlyingPath(to)
	if err != nil {
		return err
	}

	return fs.underlying.Rename(from, to)
}

func (fs *ChrootHelper) Remove(path string) error {
	fullpath, err := fs.underlyingPath(path)
	if err != nil {
		return err
	}

	return fs.underlying.Remove(fullpath)
}

func (fs *ChrootHelper) Join(elem ...string) string {
	return fs.underlying.Join(elem...)
}

func (fs *ChrootHelper) TempFile(dir, prefix string) (billy.File, error) {
	fullpath, err := fs.underlyingPath(dir)
	if err != nil {
		return nil, err
	}

	f, err := fs.underlying.(billy.TempFile).TempFile(fullpath, prefix)
	if err != nil {
		return nil, err
	}

	return newFile(fs, f, fs.Join(dir, filepath.Base(f.Name()))), nil
}

func (fs *ChrootHelper) ReadDir(path string) ([]os.FileInfo, error) {
	fullpath, err := fs.underlyingPath(path)
	if err != nil {
		return nil, err
	}

	return fs.underlying.(billy.Dir).ReadDir(fullpath)
}

func (fs *ChrootHelper) MkdirAll(filename string, perm os.FileMode) error {
	fullpath, err := fs.underlyingPath(filename)
	if err != nil {
		return err
	}

	return fs.underlying.(billy.Dir).MkdirAll(fullpath, perm)
}

func (fs *ChrootHelper) Lstat(filename string) (os.FileInfo, error) {
	fullpath, err := fs.underlyingPath(filename)
	if err != nil {
		return nil, err
	}

	return fs.underlying.(billy.Symlink).Lstat(fullpath)
}

func (fs *ChrootHelper) Symlink(target, link string) error {
	target = filepath.FromSlash(target)

	// only rewrite target if it's already absolute
	if filepath.IsAbs(target) || strings.HasPrefix(target, string(filepath.Separator)) {
		target = fs.Join(fs.Root(), target)
		target = filepath.Clean(filepath.FromSlash(target))
	}

	link, err := fs.underlyingPath(link)
	if err != nil {
		return err
	}

	return fs.underlying.(billy.Symlink).Symlink(target, link)
}

func (fs *ChrootHelper) Readlink(link string) (string, error) {
	fullpath, err := fs.underlyingPath(link)
	if err != nil {
		return "", err
	}

	target, err := fs.underlying.(billy.Symlink).Readlink(fullpath)
	if err != nil {
		return "", err
	}

	if !filepath.IsAbs(target) && !strings.HasPrefix(target, string(filepath.Separator)) {
		return target, nil
	}

	target, err = filepath.Rel(fs.base, target)
	if err != nil {
		return "", err
	}

	return string(os.PathSeparator) + target, nil
}

func (fs *ChrootHelper) Chroot(path string) (billy.Filesystem, error) {
	fullpath, err := fs.underlyingPath(path)
	if err != nil {
		return nil, err
	}

	return New(fs.underlying, fullpath), nil
}

func (fs *ChrootHelper) Root() string {
	return fs.base
}

func (fs *ChrootHelper) Underlying() billy.Basic {
	return fs.underlying
}

type file struct {
	billy.File
	name string
}

func newFile(fs billy.Filesystem, f billy.File, filename string) billy.File {
	filename = fs.Join(fs.Root(), filename)
	filename, _ = filepath.Rel(fs.Root(), filename)

	return &file{
		File: f,
		name: filename,
	}
}

func (f *file) Name() string {
	return f.name
}
