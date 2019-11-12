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

package naming

import (
	"fmt"
	"regexp"
	goruntime "runtime"
	"runtime/debug"
	"strconv"
	"strings"
)

// GetNameFromCallsite walks back through the call stack until we find a caller from outside of the ignoredPackages
// it returns back a shortpath/filename:line to aid in identification of this reflector when it starts logging
func GetNameFromCallsite(ignoredPackages ...string) string {
	name := "????"
	const maxStack = 10
	for i := 1; i < maxStack; i++ {
		_, file, line, ok := goruntime.Caller(i)
		if !ok {
			file, line, ok = extractStackCreator()
			if !ok {
				break
			}
			i += maxStack
		}
		if hasPackage(file, append(ignoredPackages, "/runtime/asm_")) {
			continue
		}

		file = trimPackagePrefix(file)
		name = fmt.Sprintf("%s:%d", file, line)
		break
	}
	return name
}

// hasPackage returns true if the file is in one of the ignored packages.
func hasPackage(file string, ignoredPackages []string) bool {
	for _, ignoredPackage := range ignoredPackages {
		if strings.Contains(file, ignoredPackage) {
			return true
		}
	}
	return false
}

// trimPackagePrefix reduces duplicate values off the front of a package name.
func trimPackagePrefix(file string) string {
	if l := strings.LastIndex(file, "/vendor/"); l >= 0 {
		return file[l+len("/vendor/"):]
	}
	if l := strings.LastIndex(file, "/src/"); l >= 0 {
		return file[l+5:]
	}
	if l := strings.LastIndex(file, "/pkg/"); l >= 0 {
		return file[l+1:]
	}
	return file
}

var stackCreator = regexp.MustCompile(`(?m)^created by (.*)\n\s+(.*):(\d+) \+0x[[:xdigit:]]+$`)

// extractStackCreator retrieves the goroutine file and line that launched this stack. Returns false
// if the creator cannot be located.
// TODO: Go does not expose this via runtime https://github.com/golang/go/issues/11440
func extractStackCreator() (string, int, bool) {
	stack := debug.Stack()
	matches := stackCreator.FindStringSubmatch(string(stack))
	if matches == nil || len(matches) != 4 {
		return "", 0, false
	}
	line, err := strconv.Atoi(matches[3])
	if err != nil {
		return "", 0, false
	}
	return matches[2], line, true
}
