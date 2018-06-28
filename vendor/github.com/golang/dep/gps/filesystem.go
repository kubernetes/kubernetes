// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gps

import (
	"os"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
)

// fsLink represents a symbolic link.
type fsLink struct {
	path string
	to   string

	// circular denotes if evaluating the symlink fails with "too many links" error.
	// This errors means that it's very likely that the symlink has circual refernce.
	circular bool

	// broken denotes that attempting to resolve the link fails, most likely because
	// the destaination doesn't exist.
	broken bool
}

// filesystemState represents the state of a file system.
type filesystemState struct {
	root  string
	dirs  []string
	files []string
	links []fsLink
}

func (s filesystemState) setup() error {
	for _, dir := range s.dirs {
		p := filepath.Join(s.root, dir)

		if err := os.MkdirAll(p, 0777); err != nil {
			return errors.Errorf("os.MkdirAll(%q, 0777) err=%q", p, err)
		}
	}

	for _, file := range s.files {
		p := filepath.Join(s.root, file)

		f, err := os.Create(p)
		if err != nil {
			return errors.Errorf("os.Create(%q) err=%q", p, err)
		}

		if err := f.Close(); err != nil {
			return errors.Errorf("file %q Close() err=%q", p, err)
		}
	}

	for _, link := range s.links {
		p := filepath.Join(s.root, link.path)

		// On Windows, relative symlinks confuse filepath.Walk. So, we'll just sigh
		// and do absolute links, assuming they are relative to the directory of
		// link.path.
		//
		// Reference: https://github.com/golang/go/issues/17540
		//
		// TODO(ibrasho): This was fixed in Go 1.9. Remove this when support for
		// 1.8 is dropped.
		dir := filepath.Dir(p)
		to := ""
		if link.to != "" {
			to = filepath.Join(dir, link.to)
		}

		if err := os.Symlink(to, p); err != nil {
			return errors.Errorf("os.Symlink(%q, %q) err=%q", to, p, err)
		}
	}

	return nil
}

// deriveFilesystemState returns a filesystemState based on the state of
// the filesystem on root.
func deriveFilesystemState(root string) (filesystemState, error) {
	fs := filesystemState{root: root}

	err := filepath.Walk(fs.root, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}

		if path == fs.root {
			return nil
		}

		relPath, err := filepath.Rel(fs.root, path)
		if err != nil {
			return err
		}

		if (info.Mode() & os.ModeSymlink) != 0 {
			l := fsLink{path: relPath}

			l.to, err = filepath.EvalSymlinks(path)
			if err != nil && strings.HasSuffix(err.Error(), "too many links") {
				l.circular = true
			} else if err != nil && os.IsNotExist(err) {
				l.broken = true
			} else if err != nil {
				return err
			}

			fs.links = append(fs.links, l)

			return nil
		}

		if info.IsDir() {
			fs.dirs = append(fs.dirs, relPath)

			return nil
		}

		fs.files = append(fs.files, relPath)

		return nil
	})

	if err != nil {
		return filesystemState{}, err
	}

	return fs, nil
}
