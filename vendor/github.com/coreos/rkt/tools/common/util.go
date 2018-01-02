// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
)

// StringSliceWrapper is an implementation of flag.Value interface. It
// is basically a proxy that appends strings to an already existing
// strings slice.
type StringSliceWrapper struct {
	Slice *[]string
}

func (wrapper *StringSliceWrapper) String() string {
	if len(*wrapper.Slice) > 0 {
		return fmt.Sprintf(`["%s"]`, strings.Join(*wrapper.Slice, `" "`))
	}
	return "[]"
}

func (wrapper *StringSliceWrapper) Set(str string) error {
	*wrapper.Slice = append(*wrapper.Slice, str)
	return nil
}

// Warn is just a shorter version of a formatted printing to
// stderr. It appends a newline for you.
func Warn(format string, values ...interface{}) {
	fmt.Fprintf(os.Stderr, fmt.Sprintf("%s%c", format, '\n'), values...)
}

// Die is the same as Warn, but it quits with exit status 1 after
// printing a message.
func Die(format string, values ...interface{}) {
	Warn(format, values...)
	os.Exit(1)
}

// MapFilesToDirectories creates one entry for each file and directory
// passed. Result is a kind of cross-product. For files f1 and f2 and
// directories d1 and d2, the returned list will contain d1/f1 d2/f1
// d1/f2 d2/f2.
//
// The "files" part might be a bit misleading - you can pass a list of
// symlinks or directories there as well.
func MapFilesToDirectories(files, dirs []string) []string {
	mapped := make([]string, 0, len(files)*len(dirs))
	for _, f := range files {
		for _, d := range dirs {
			mapped = append(mapped, filepath.Join(d, f))
		}
	}
	return mapped
}

// MustAbs returns an absolute path. It works like filepath.Abs, but
// panics if it fails.
func MustAbs(dir string) string {
	absDir, err := filepath.Abs(dir)
	if err != nil {
		panic(fmt.Sprintf("Failed to get absolute path of a directory %q: %v\n", dir, err))
	}
	return filepath.Clean(absDir)
}
