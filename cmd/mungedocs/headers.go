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
	"regexp"
	"strings"
)

var headerRegex = regexp.MustCompile(`^(#+)\s*(.*)$`)
var whitespaceRegex = regexp.MustCompile(`^\s*$`)

func fixHeaderLines(fileBytes []byte) []byte {
	lines := splitLines(fileBytes)
	out := []string{}
	for i := range lines {
		matches := headerRegex.FindStringSubmatch(lines[i])
		if matches == nil {
			out = append(out, lines[i])
			continue
		}
		if i > 0 && !whitespaceRegex.Match([]byte(out[len(out)-1])) {
			out = append(out, "")
		}
		out = append(out, fmt.Sprintf("%s %s", matches[1], matches[2]))
		if i+1 < len(lines) && !whitespaceRegex.Match([]byte(lines[i+1])) {
			out = append(out, "")
		}
	}
	final := strings.Join(out, "\n")
	// Preserve the end of the file.
	if len(fileBytes) > 0 && fileBytes[len(fileBytes)-1] == '\n' {
		final += "\n"
	}
	return []byte(final)
}

// Header lines need whitespace around them and after the #s.
func checkHeaderLines(filePath string, fileBytes []byte) ([]byte, error) {
	fbs, err := splitByPreformatted(fileBytes)
	if err != nil {
		return fileBytes, err
	}
	fbs = append([]fileBlock{{false, []byte{}}}, fbs...)
	fbs = append(fbs, fileBlock{false, []byte{}})

	for i := range fbs {
		block := &fbs[i]
		if block.preformatted {
			continue
		}
		block.data = fixHeaderLines(block.data)
	}
	output := []byte{}
	for _, block := range fbs {
		output = append(output, block.data...)
	}
	return output, nil
}
