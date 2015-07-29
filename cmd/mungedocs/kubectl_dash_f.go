/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package main

import (
	"fmt"
	"os"
	"path"
	"strings"
)

// Looks for lines that have kubectl commands with -f flags and files that
// don't exist.
func checkKubectlFileTargets(file string, markdown []byte) ([]byte, error) {
	inside := false
	lines := splitLines(markdown)
	errors := []string{}
	for i := range lines {
		if strings.HasPrefix(lines[i], "```") {
			inside = !inside
		}
		if inside {
			if err := lookForKubectl(lines, i); err != nil {
				errors = append(errors, err.Error())
			}
		}
	}
	err := error(nil)
	if len(errors) != 0 {
		err = fmt.Errorf("%s", strings.Join(errors, "\n"))
	}
	return markdown, err
}

func lookForKubectl(lines []string, lineNum int) error {
	fields := strings.Fields(lines[lineNum])
	for i := range fields {
		if fields[i] == "kubectl" {
			return gotKubectl(lineNum, fields, i)
		}
	}
	return nil
}

func gotKubectl(line int, fields []string, fieldNum int) error {
	for i := fieldNum + 1; i < len(fields); i++ {
		switch fields[i] {
		case "create", "update", "replace", "delete":
			return gotCommand(line, fields, i)
		}
	}
	return nil
}

func gotCommand(line int, fields []string, fieldNum int) error {
	for i := fieldNum + 1; i < len(fields); i++ {
		if strings.HasPrefix(fields[i], "-f") {
			return gotDashF(line, fields, i)
		}
	}
	return nil
}

func gotDashF(line int, fields []string, fieldNum int) error {
	target := ""
	if fields[fieldNum] == "-f" {
		if fieldNum+1 == len(fields) {
			return fmt.Errorf("ran out of fields after '-f'")
		}
		target = fields[fieldNum+1]
	} else {
		target = fields[fieldNum][2:]
	}
	// Turn dirs into file-like names.
	target = strings.TrimRight(target, "/")

	// Now exclude special-cases

	if target == "-" || target == "FILENAME" {
		// stdin and "FILENAME" are OK
		return nil
	}
	if strings.HasPrefix(target, "http://") || strings.HasPrefix(target, "https://") {
		// URLs are ok
		return nil
	}
	if strings.HasPrefix(target, "./") {
		// Same-dir files are usually created in the same example
		return nil
	}
	if strings.HasPrefix(target, "~/") {
		// Home directory may also be created by the same example
		return nil
	}
	if strings.HasPrefix(target, "/") {
		// Absolute paths tend to be /tmp/* and created in the same example.
		return nil
	}

	// If we got here we expect the file to exist.
	_, err := os.Stat(path.Join(*rootDir, *repoRoot, target))
	if os.IsNotExist(err) {
		return fmt.Errorf("%d: target file %q does not exist", line, target)
	}
	return err
}
