// +build !windows

package getter

import (
	"fmt"
	"io"
	"net/url"
	"os"
	"path/filepath"
)

func (g *FileGetter) Get(dst string, u *url.URL) error {
	path := u.Path
	if u.RawPath != "" {
		path = u.RawPath
	}

	// The source path must exist and be a directory to be usable.
	if fi, err := os.Stat(path); err != nil {
		return fmt.Errorf("source path error: %s", err)
	} else if !fi.IsDir() {
		return fmt.Errorf("source path must be a directory")
	}

	fi, err := os.Lstat(dst)
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	// If the destination already exists, it must be a symlink
	if err == nil {
		mode := fi.Mode()
		if mode&os.ModeSymlink == 0 {
			return fmt.Errorf("destination exists and is not a symlink")
		}

		// Remove the destination
		if err := os.Remove(dst); err != nil {
			return err
		}
	}

	// Create all the parent directories
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}

	return os.Symlink(path, dst)
}

func (g *FileGetter) GetFile(dst string, u *url.URL) error {
	path := u.Path
	if u.RawPath != "" {
		path = u.RawPath
	}

	// The source path must exist and be a file to be usable.
	if fi, err := os.Stat(path); err != nil {
		return fmt.Errorf("source path error: %s", err)
	} else if fi.IsDir() {
		return fmt.Errorf("source path must be a file")
	}

	_, err := os.Lstat(dst)
	if err != nil && !os.IsNotExist(err) {
		return err
	}

	// If the destination already exists, it must be a symlink
	if err == nil {
		// Remove the destination
		if err := os.Remove(dst); err != nil {
			return err
		}
	}

	// Create all the parent directories
	if err := os.MkdirAll(filepath.Dir(dst), 0755); err != nil {
		return err
	}

	// If we're not copying, just symlink and we're done
	if !g.Copy {
		return os.Symlink(path, dst)
	}

	// Copy
	srcF, err := os.Open(path)
	if err != nil {
		return err
	}
	defer srcF.Close()

	dstF, err := os.Create(dst)
	if err != nil {
		return err
	}
	defer dstF.Close()

	_, err = io.Copy(dstF, srcF)
	return err
}
