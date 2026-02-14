// SPDX-FileCopyrightText: Copyright 2015-2025 go-swagger maintainers
// SPDX-License-Identifier: Apache-2.0

package fileutils

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

// GOPATHKey represents the env key for gopath
const GOPATHKey = "GOPATH"

// FindInSearchPath finds a package in a provided lists of paths
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

// FindInGoSearchPath finds a package in the $GOPATH:$GOROOT
//
// Deprecated: this function is no longer relevant with modern go.
// It uses [runtime.GOROOT] under the hood, which is deprecated as of go1.24.
func FindInGoSearchPath(pkg string) string {
	return FindInSearchPath(FullGoSearchPath(), pkg)
}

// FullGoSearchPath gets the search paths for finding packages
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
