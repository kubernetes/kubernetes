package polyfill

import (
	"os"
	"path/filepath"

	"gopkg.in/src-d/go-billy.v3"
)

// Polyfill is a helper that implements all missing method from billy.Filesystem.
type Polyfill struct {
	billy.Basic
	c capabilities
}

type capabilities struct{ tempfile, dir, symlink, chroot bool }

// New creates a new filesystem wrapping up 'fs' the intercepts all the calls
// made and errors if fs doesn't implement any of the billy interfaces.
func New(fs billy.Basic) billy.Filesystem {
	if original, ok := fs.(billy.Filesystem); ok {
		return original
	}

	h := &Polyfill{Basic: fs}

	_, h.c.tempfile = h.Basic.(billy.TempFile)
	_, h.c.dir = h.Basic.(billy.Dir)
	_, h.c.symlink = h.Basic.(billy.Symlink)
	_, h.c.chroot = h.Basic.(billy.Chroot)
	return h
}

func (h *Polyfill) TempFile(dir, prefix string) (billy.File, error) {
	if !h.c.tempfile {
		return nil, billy.ErrNotSupported
	}

	return h.Basic.(billy.TempFile).TempFile(dir, prefix)
}

func (h *Polyfill) ReadDir(path string) ([]os.FileInfo, error) {
	if !h.c.dir {
		return nil, billy.ErrNotSupported
	}

	return h.Basic.(billy.Dir).ReadDir(path)
}

func (h *Polyfill) MkdirAll(filename string, perm os.FileMode) error {
	if !h.c.dir {
		return billy.ErrNotSupported
	}

	return h.Basic.(billy.Dir).MkdirAll(filename, perm)
}

func (h *Polyfill) Symlink(target, link string) error {
	if !h.c.symlink {
		return billy.ErrNotSupported
	}

	return h.Basic.(billy.Symlink).Symlink(target, link)
}

func (h *Polyfill) Readlink(link string) (string, error) {
	if !h.c.symlink {
		return "", billy.ErrNotSupported
	}

	return h.Basic.(billy.Symlink).Readlink(link)
}

func (h *Polyfill) Lstat(path string) (os.FileInfo, error) {
	if !h.c.symlink {
		return nil, billy.ErrNotSupported
	}

	return h.Basic.(billy.Symlink).Lstat(path)
}

func (h *Polyfill) Chroot(path string) (billy.Filesystem, error) {
	if !h.c.chroot {
		return nil, billy.ErrNotSupported
	}

	return h.Basic.(billy.Chroot).Chroot(path)
}

func (h *Polyfill) Root() string {
	if !h.c.chroot {
		return string(filepath.Separator)
	}

	return h.Basic.(billy.Chroot).Root()
}

func (h *Polyfill) Underlying() billy.Basic {
	return h.Basic
}
