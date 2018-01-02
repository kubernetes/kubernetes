package remotecontext

import (
	"encoding/hex"
	"os"
	"path/filepath"
	"runtime"
	"strings"

	"github.com/docker/docker/builder"
	"github.com/docker/docker/pkg/pools"
	"github.com/pkg/errors"
)

// NewLazySource creates a new LazyContext. LazyContext defines a hashed build
// context based on a root directory. Individual files are hashed first time
// they are asked. It is not safe to call methods of LazyContext concurrently.
func NewLazySource(root string) (builder.Source, error) {
	return &lazySource{
		root: root,
		sums: make(map[string]string),
	}, nil
}

type lazySource struct {
	root string
	sums map[string]string
}

func (c *lazySource) Root() string {
	return c.root
}

func (c *lazySource) Close() error {
	return nil
}

func (c *lazySource) Hash(path string) (string, error) {
	cleanPath, fullPath, err := normalize(path, c.root)
	if err != nil {
		return "", err
	}

	fi, err := os.Lstat(fullPath)
	if err != nil {
		return "", errors.WithStack(err)
	}

	relPath, err := Rel(c.root, fullPath)
	if err != nil {
		return "", errors.WithStack(convertPathError(err, cleanPath))
	}

	sum, ok := c.sums[relPath]
	if !ok {
		sum, err = c.prepareHash(relPath, fi)
		if err != nil {
			return "", err
		}
	}

	return sum, nil
}

func (c *lazySource) prepareHash(relPath string, fi os.FileInfo) (string, error) {
	p := filepath.Join(c.root, relPath)
	h, err := NewFileHash(p, relPath, fi)
	if err != nil {
		return "", errors.Wrapf(err, "failed to create hash for %s", relPath)
	}
	if fi.Mode().IsRegular() && fi.Size() > 0 {
		f, err := os.Open(p)
		if err != nil {
			return "", errors.Wrapf(err, "failed to open %s", relPath)
		}
		defer f.Close()
		if _, err := pools.Copy(h, f); err != nil {
			return "", errors.Wrapf(err, "failed to copy file data for %s", relPath)
		}
	}
	sum := hex.EncodeToString(h.Sum(nil))
	c.sums[relPath] = sum
	return sum, nil
}

// Rel makes a path relative to base path. Same as `filepath.Rel` but can also
// handle UUID paths in windows.
func Rel(basepath, targpath string) (string, error) {
	// filepath.Rel can't handle UUID paths in windows
	if runtime.GOOS == "windows" {
		pfx := basepath + `\`
		if strings.HasPrefix(targpath, pfx) {
			p := strings.TrimPrefix(targpath, pfx)
			if p == "" {
				p = "."
			}
			return p, nil
		}
	}
	return filepath.Rel(basepath, targpath)
}
