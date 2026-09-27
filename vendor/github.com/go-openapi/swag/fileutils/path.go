// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package fileutils

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// GOPATHKey is the name of the environment variable that holds the go search path.
const GOPATHKey = "GOPATH"

// FindInSearchPath finds a package in a list of search paths.
//
// searchPath lists directories separated by the OS path separator,
// in the form accepted by [filepath.SplitList].
// Each directory is probed for a src/pkg subdirectory.
//
// It returns the first match, with symlinks resolved,
// or an empty string when the package is not found in any of the directories.
func FindInSearchPath(searchPath, pkg string) string {
	pathsList := filepath.SplitList(searchPath)
	for _, path := range pathsList {
		if evaluatedPath, err := filepath.EvalSymlinks(filepath.Join(path, "src", pkg)); err == nil {
			if _, err := os.Stat(evaluatedPath); err == nil {
				return evaluatedPath
			}
		}
	}
	return ""
}

// FindInGoSearchPath finds a package in $GOPATH and $GOROOT.
//
// It returns an empty string when the package is not found.
//
// Deprecated: this function is no longer relevant with modern go.
// It uses [runtime.GOROOT] under the hood, which is deprecated as of go1.24.
func FindInGoSearchPath(pkg string) string {
	return FindInSearchPath(FullGoSearchPath(), pkg)
}

// FullGoSearchPath returns the search paths in which a package may be found.
//
// It joins $GOPATH, which defaults to $HOME/go when unset, with [runtime.GOROOT].
// The two are separated by a colon, so the result is not usable on windows.
//
// Deprecated: this function is no longer relevant with modern go.
// It uses [runtime.GOROOT] under the hood, which is deprecated as of go1.24.
func FullGoSearchPath() string {
	allPaths := os.Getenv(GOPATHKey)
	if allPaths == "" {
		allPaths = filepath.Join(os.Getenv("HOME"), "go")
	}
	if allPaths != "" {
		allPaths = strings.Join([]string{allPaths, runtime.GOROOT()}, ":")
	} else {
		allPaths = runtime.GOROOT()
	}
	return allPaths
}
