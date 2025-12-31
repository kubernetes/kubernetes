// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package execpath

import (
	"os"
	"path/filepath"
	"strings"
)

func findExecutable(file string) error {
	d, err := os.Stat(file)
	if err != nil {
		return err
	}
	if m := d.Mode(); !m.IsDir() && m&0111 != 0 {
		return nil
	}
	return os.ErrPermission
}

// Look searches for an executable named file, using getenv to look up
// environment variables. If getenv is nil, os.Getenv will be used. If file
// contains a slash, it is tried directly and getenv will not be called.  The
// result may be an absolute path or a path relative to the current directory.
func Look(file string, getenv func(string) string) (string, error) {
	if getenv == nil {
		getenv = os.Getenv
	}

	// skip the path lookup for these prefixes
	skip := []string{"/", "#", "./", "../"}

	for _, p := range skip {
		if strings.HasPrefix(file, p) {
			err := findExecutable(file)
			if err == nil {
				return file, nil
			}
			return "", &Error{file, err}
		}
	}

	path := getenv("path")
	for _, dir := range filepath.SplitList(path) {
		path := filepath.Join(dir, file)
		if err := findExecutable(path); err == nil {
			return path, nil
		}
	}
	return "", &Error{file, ErrNotFound}
}
