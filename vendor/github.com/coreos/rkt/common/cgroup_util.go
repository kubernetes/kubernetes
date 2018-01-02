// Copyright 2014 The rkt Authors
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

//+build linux

package common

// adapted from systemd/src/shared/cgroup-util.c
// TODO this should be moved to go-systemd

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"strings"
)

const (
	unitNameMax = 256
)

var (
	validChars = regexp.MustCompile(`[a-zA-Z0-9:\-_\.\\]+`)
)

// cgEscape implements very minimal escaping for names to be used as file names
// in the cgroup tree: any name which might conflict with a kernel name or is
// prefixed with '_' is prefixed with a '_'. That way, when reading cgroup
// names it is sufficient to remove a single prefixing underscore if there is
// one.
func cgEscape(p string) string {
	needPrefix := false

	switch {
	case strings.HasPrefix(p, "_"):
		fallthrough
	case strings.HasPrefix(p, "."):
		fallthrough
	case p == "notify_on_release":
		fallthrough
	case p == "release_agent":
		fallthrough
	case p == "tasks":
		needPrefix = true
	case strings.Contains(p, "."):
		sp := strings.Split(p, ".")
		if sp[0] == "cgroup" {
			needPrefix = true
		} else {
			n := sp[0]
			if checkHierarchy(n) {
				needPrefix = true
			}
		}
	}

	if needPrefix {
		return "_" + p
	}

	return p
}

func filenameIsValid(p string) bool {
	switch {
	case p == "", p == ".", p == "..", strings.Contains(p, "/"):
		return false
	default:
		return true
	}
}

func checkHierarchy(p string) bool {
	if !filenameIsValid(p) {
		return true
	}

	cc := filepath.Join("/sys/fs/cgroup", p)
	if _, err := os.Stat(cc); os.IsNotExist(err) {
		return false
	}

	return true
}

func cgUnescape(p string) string {
	if p[0] == '_' {
		return p[1:]
	}

	return p
}

func sliceNameIsValid(n string) bool {
	if n == "" {
		return false
	}

	if len(n) >= unitNameMax {
		return false
	}

	if !strings.Contains(n, ".") {
		return false
	}

	if validChars.FindString(n) != n {
		return false
	}

	if strings.Contains(n, "@") {
		return false
	}

	return true
}

// SliceToPath explodes a slice name to its corresponding path in the cgroup
// hierarchy. For example, a slice named "foo-bar-baz.slice" corresponds to the
// path "foo.slice/foo-bar.slice/foo-bar-baz.slice". See systemd.slice(5)
func SliceToPath(unit string) (string, error) {
	if unit == "-.slice" {
		return "", nil
	}

	if !strings.HasSuffix(unit, ".slice") {
		return "", fmt.Errorf("not a slice")
	}

	if !sliceNameIsValid(unit) {
		return "", fmt.Errorf("invalid slice name")
	}

	prefix := unitnameToPrefix(unit)

	// don't allow initial dashes
	if prefix[0] == '-' {
		return "", fmt.Errorf("initial dash")
	}

	prefixParts := strings.Split(prefix, "-")

	var curSlice string
	var slicePath string
	for _, slicePart := range prefixParts {
		if slicePart == "" {
			return "", fmt.Errorf("trailing or double dash")
		}

		if curSlice != "" {
			curSlice = curSlice + "-"
		}
		curSlice = curSlice + slicePart

		curSliceDir := curSlice + ".slice"
		escaped := cgEscape(curSliceDir)

		slicePath = filepath.Join(slicePath, escaped)
	}

	return slicePath, nil
}

func unitnameToPrefix(unit string) string {
	idx := strings.Index(unit, "@")
	if idx == -1 {
		idx = strings.LastIndex(unit, ".")
	}

	return unit[:idx]
}
