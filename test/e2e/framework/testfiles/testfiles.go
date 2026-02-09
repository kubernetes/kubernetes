/*
Copyright 2018 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// Package testfiles provides a wrapper around various optional ways
// of retrieving additional files needed during a test run:
// - builtin bindata
// - filesystem access
//
// Because it is a is self-contained package, it can be used by
// test/e2e/framework and test/e2e/manifest without creating
// a circular dependency.
package testfiles

import (
	"embed"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"strings"
)

var filesources []FileSource

// AddFileSource registers another provider for files that may be
// needed at runtime. Should be called during initialization of a test
// binary.
func AddFileSource(filesource FileSource) {
	filesources = append(filesources, filesource)
}

// FileSource implements one way of retrieving test file content.  For
// example, one file source could read from the original source code
// file tree, another from bindata compiled into a test executable.
type FileSource interface {
	// ReadTestFile retrieves the content of a file that gets maintained
	// alongside a test's source code. Files are identified by the
	// relative path inside the repository containing the tests, for
	// example "cluster/gce/upgrade.sh" inside kubernetes/kubernetes.
	//
	// When the file is not found, a nil slice is returned. An error is
	// returned for all fatal errors.
	ReadTestFile(filePath string) ([]byte, error)

	// DescribeFiles returns a multi-line description of which
	// files are available via this source. It is meant to be
	// used as part of the error message when a file cannot be
	// found.
	DescribeFiles() string
}

// Read tries to retrieve the desired file content from
// one of the registered file sources.
func Read(filePath string) ([]byte, error) {
	if len(filesources) == 0 {
		return nil, fmt.Errorf("no file sources registered (yet?), cannot retrieve test file %s", filePath)
	}
	for _, filesource := range filesources {
		data, err := filesource.ReadTestFile(filePath)
		if err != nil {
			return nil, fmt.Errorf("fatal error retrieving test file %s: %w", filePath, err)
		}
		if data != nil {
			return data, nil
		}
	}
	// Here we try to generate an error that points test authors
	// or users in the right direction for resolving the problem.
	err := fmt.Sprintf("Test file %q was not found.\n", filePath)
	for _, filesource := range filesources {
		err += filesource.DescribeFiles()
		err += "\n"
	}
	return nil, errors.New(err)
}

// Exists checks whether a file could be read. Unexpected errors
// are handled by calling the fail function, which then should
// abort the current test.
func Exists(filePath string) (bool, error) {
	for _, filesource := range filesources {
		data, err := filesource.ReadTestFile(filePath)
		if err != nil {
			return false, err
		}
		if data != nil {
			return true, nil
		}
	}
	return false, nil
}

// RootFileSource looks for files relative to a root directory.
type RootFileSource struct {
	Root string
}

// ReadTestFile looks for the file relative to the configured
// root directory. If the path is already absolute, for example
// in a test that has its own method of determining where
// files are, then the path will be used directly.
func (r RootFileSource) ReadTestFile(filePath string) ([]byte, error) {
	var fullPath string
	if path.IsAbs(filePath) {
		fullPath = filePath
	} else {
		fullPath = filepath.Join(r.Root, filePath)
	}
	data, err := os.ReadFile(fullPath)
	if os.IsNotExist(err) {
		// Not an error (yet), some other provider may have the file.
		return nil, nil
	}
	return data, err
}

// DescribeFiles explains that it looks for files inside a certain
// root directory.
func (r RootFileSource) DescribeFiles() string {
	description := fmt.Sprintf("Test files are expected in %q", r.Root)
	if !path.IsAbs(r.Root) {
		// The default in test_context.go is the relative path
		// ../../, which doesn't really help locating the
		// actual location. Therefore we add also the absolute
		// path if necessary.
		abs, err := filepath.Abs(r.Root)
		if err == nil {
			description += fmt.Sprintf(" = %q", abs)
		}
	}
	description += "."
	return description
}

// EmbeddedFileSource handles files stored in a package generated with bindata.
type EmbeddedFileSource struct {
	EmbeddedFS embed.FS
	Root       string
	fileList   []string
}

// ReadTestFile looks for an embedded file with the given path.
func (e EmbeddedFileSource) ReadTestFile(filepath string) ([]byte, error) {
	relativePath := strings.TrimPrefix(filepath, fmt.Sprintf("%s/", e.Root))

	b, err := e.EmbeddedFS.ReadFile(relativePath)
	if err != nil {
		if errors.Is(err, fs.ErrNotExist) {
			return nil, nil
		}
		return nil, err
	}

	return b, nil
}

// DescribeFiles explains that it is looking inside an embedded filesystem
func (e EmbeddedFileSource) DescribeFiles() string {
	var lines []string
	lines = append(lines, "The following files are embedded into the test executable:")

	if len(e.fileList) == 0 {
		e.populateFileList()
	}
	lines = append(lines, e.fileList...)

	return strings.Join(lines, "\n\t")
}

func (e *EmbeddedFileSource) populateFileList() {
	fs.WalkDir(e.EmbeddedFS, ".", func(path string, d fs.DirEntry, err error) error {
		if !d.IsDir() {
			e.fileList = append(e.fileList, filepath.Join(e.Root, path))
		}

		return nil
	})
}
