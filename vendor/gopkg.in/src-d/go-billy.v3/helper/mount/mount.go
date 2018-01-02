package mount

import (
	"io"
	"os"
	"path/filepath"
	"strings"

	"fmt"

	"gopkg.in/src-d/go-billy.v3"
	"gopkg.in/src-d/go-billy.v3/helper/polyfill"
)

var separator = string(filepath.Separator)

// Mount is a helper that allows to emulate the behavior of mount in memory.
// Very usufull to create a temporal dir, on filesystem where is a performance
// penalty in doing so.
type Mount struct {
	underlying billy.Filesystem
	source     billy.Filesystem
	mountpoint string
}

// New creates a new filesystem wrapping up 'fs' the intercepts all the calls
// made to `mountpoint` path and redirecting it to `source` filesystem.
func New(fs billy.Basic, mountpoint string, source billy.Basic) *Mount {
	return &Mount{
		underlying: polyfill.New(fs),
		source:     polyfill.New(source),
		mountpoint: cleanPath(mountpoint),
	}
}

func (h *Mount) Create(path string) (billy.File, error) {
	fs, fullpath := h.getBasicAndPath(path)
	if fullpath == "." {
		return nil, os.ErrInvalid
	}

	f, err := fs.Create(fullpath)
	return wrapFile(f, path), err
}

func (h *Mount) Open(path string) (billy.File, error) {
	fs, fullpath := h.getBasicAndPath(path)
	if fullpath == "." {
		return nil, os.ErrInvalid
	}

	f, err := fs.Open(fullpath)
	return wrapFile(f, path), err
}

func (h *Mount) OpenFile(path string, flag int, mode os.FileMode) (billy.File, error) {
	fs, fullpath := h.getBasicAndPath(path)
	if fullpath == "." {
		return nil, os.ErrInvalid
	}

	f, err := fs.OpenFile(fullpath, flag, mode)
	return wrapFile(f, path), err
}

func (h *Mount) Rename(from, to string) error {
	fromInSource := h.isMountpoint(from)
	toInSource := h.isMountpoint(to)

	var fromFS, toFS billy.Basic

	switch {
	case fromInSource && toInSource:
		from = h.mustRelToMountpoint(from)
		to = h.mustRelToMountpoint(to)
		return h.source.Rename(from, to)
	case !fromInSource && !toInSource:
		return h.underlying.Rename(from, to)
	case fromInSource && !toInSource:
		fromFS = h.source
		from = h.mustRelToMountpoint(from)
		toFS = h.underlying
		to = cleanPath(to)
	case !fromInSource && toInSource:
		fromFS = h.underlying
		from = cleanPath(from)
		toFS = h.source
		to = h.mustRelToMountpoint(to)
	}

	if err := copyPath(fromFS, toFS, from, to); err != nil {
		return err
	}

	return fromFS.Remove(from)
}

func (h *Mount) Stat(path string) (os.FileInfo, error) {
	fs, fullpath := h.getBasicAndPath(path)
	return fs.Stat(fullpath)
}

func (h *Mount) Remove(path string) error {
	fs, fullpath := h.getBasicAndPath(path)
	if fullpath == "." {
		return os.ErrInvalid
	}

	return fs.Remove(fullpath)
}

func (h *Mount) ReadDir(path string) ([]os.FileInfo, error) {
	fs, fullpath, err := h.getDirAndPath(path)
	if err != nil {
		return nil, err
	}

	return fs.ReadDir(fullpath)
}

func (h *Mount) MkdirAll(filename string, perm os.FileMode) error {
	fs, fullpath, err := h.getDirAndPath(filename)
	if err != nil {
		return err
	}

	return fs.MkdirAll(fullpath, perm)
}

func (h *Mount) Symlink(target, link string) error {
	fs, fullpath, err := h.getSymlinkAndPath(link)
	if err != nil {
		return err
	}

	resolved := filepath.Join(filepath.Dir(link), target)
	if h.isMountpoint(resolved) != h.isMountpoint(link) {
		return fmt.Errorf("invalid symlink, target is crossing filesystems")
	}

	return fs.Symlink(target, fullpath)
}

func (h *Mount) Join(elem ...string) string {
	return h.underlying.Join(elem...)
}

func (h *Mount) Readlink(link string) (string, error) {
	fs, fullpath, err := h.getSymlinkAndPath(link)
	if err != nil {
		return "", err
	}

	return fs.Readlink(fullpath)
}

func (h *Mount) Lstat(path string) (os.FileInfo, error) {
	fs, fullpath, err := h.getSymlinkAndPath(path)
	if err != nil {
		return nil, err
	}

	return fs.Lstat(fullpath)
}

func (h *Mount) Underlying() billy.Basic {
	return h.underlying
}

func (fs *Mount) getBasicAndPath(path string) (billy.Basic, string) {
	path = cleanPath(path)
	if !fs.isMountpoint(path) {
		return fs.underlying, path
	}

	return fs.source, fs.mustRelToMountpoint(path)
}

func (fs *Mount) getDirAndPath(path string) (billy.Dir, string, error) {
	path = cleanPath(path)
	if !fs.isMountpoint(path) {
		return fs.underlying.(billy.Dir), path, nil
	}

	return fs.source.(billy.Dir), fs.mustRelToMountpoint(path), nil
}

func (fs *Mount) getSymlinkAndPath(path string) (billy.Symlink, string, error) {
	path = cleanPath(path)
	if !fs.isMountpoint(path) {
		return fs.underlying.(billy.Symlink), path, nil
	}

	return fs.source.(billy.Symlink), fs.mustRelToMountpoint(path), nil
}

func (fs *Mount) mustRelToMountpoint(path string) string {
	path = cleanPath(path)
	fullpath, err := filepath.Rel(fs.mountpoint, path)
	if err != nil {
		panic(err)
	}

	return fullpath
}

func (fs *Mount) isMountpoint(path string) bool {
	path = cleanPath(path)
	return strings.HasPrefix(path, fs.mountpoint)
}

func cleanPath(path string) string {
	path = filepath.FromSlash(path)
	rel, err := filepath.Rel(separator, path)
	if err == nil {
		path = rel
	}

	return filepath.Clean(path)
}

// copyPath copies a file across filesystems.
func copyPath(src, dst billy.Basic, srcPath, dstPath string) error {
	dstFile, err := dst.Create(dstPath)
	if err != nil {
		return err
	}

	srcFile, err := src.Open(srcPath)
	if err != nil {
		return nil
	}

	_, err = io.Copy(dstFile, srcFile)
	if err != nil {
		return nil
	}

	err = dstFile.Close()
	if err != nil {
		_ = srcFile.Close()
		return err
	}

	return srcFile.Close()
}

type file struct {
	billy.File
	name string
}

func wrapFile(f billy.File, filename string) billy.File {
	if f == nil {
		return nil
	}

	return &file{
		File: f,
		name: cleanPath(filename),
	}
}

func (f *file) Name() string {
	return f.name
}
