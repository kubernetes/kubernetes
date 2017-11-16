// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package swag

import (
	"os"
	"path/filepath"
	"runtime"
	"strings"
)

const (
	// GOPATHKey represents the env key for gopath
	GOPATHKey = "GOPATH"
)

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
func FindInGoSearchPath(pkg string) string {
	return FindInSearchPath(FullGoSearchPath(), pkg)
}

// FullGoSearchPath gets the search paths for finding packages
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
