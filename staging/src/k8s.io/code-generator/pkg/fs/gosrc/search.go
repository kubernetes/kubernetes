/*
Copyright 2023 The Kubernetes Authors.

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

package gosrc

import (
	"bufio"
	"io/fs"
	"os"
	"path"
	"path/filepath"
	"regexp"
	"strings"
)

// Find returns the list of files matching the given matchers. The search is
// performed recursively from the given root directory. The file is only
// considered if it matches all the given matchers.
func Find(root string, matchers ...FileMatcher) ([]string, error) {
	var files []string
	err := filepath.WalkDir(root, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return filepath.SkipDir
		}
		n := path.Base(p)
		if !(d.Type().IsRegular() &&
			strings.HasSuffix(n, ".go") &&
			!strings.HasSuffix(n, "_test.go")) {
			return nil
		}
		for _, m := range matchers {
			if !m.MatchFile(p) {
				return nil
			}
		}
		files = append(files, p)
		return nil

	})
	return files, err
}

// FileEndsWith returns a FileMatcher that matches files with the given suffix.
func FileEndsWith(suffix string) FileMatcher {
	return FileMatcherFunc(func(fp string) bool {
		return strings.HasSuffix(fp, suffix)
	})
}

// FileContains returns a FileMatcher that matches files containing the given
// pattern.
func FileContains(pattern *regexp.Regexp) FileMatcher {
	return FileMatcherFunc(func(fp string) bool {
		return match(pattern, fp)
	})
}

// FileMatcher matches files.
type FileMatcher interface {
	// MatchFile returns true if the given file path matches.
	MatchFile(string) bool
}

// FileMatcherFunc is a function that matches files.
type FileMatcherFunc func(string) bool

// MatchFile implements FileMatcher.
func (f FileMatcherFunc) MatchFile(fp string) bool {
	return f(fp)
}

func match(pattern *regexp.Regexp, fp string) bool {
	// instead of reading the whole file into memory and matching against
	// the whole file contents.
	// The following code is similar to the following shell command:
	//   grep -q -e "$pattern" "$fp"
	// The -q flag is used to suppress the output.
	f, err := os.Open(fp)
	if err != nil {
		return false
	}
	defer func() {
		if err := f.Close(); err != nil {
			panic(err)
		}
	}()
	scanner := bufio.NewScanner(f)
	for scanner.Scan() {
		if pattern.MatchString(scanner.Text()) {
			return true
		}
	}
	return false
}
