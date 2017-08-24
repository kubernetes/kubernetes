/*
Copyright 2015 The Kubernetes Authors.

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
)

var headerRegex = regexp.MustCompile(`^(#+)\s*(.*)$`)

func fixHeaderLine(mlines mungeLines, newlines mungeLines, linenum int) mungeLines {
	var out mungeLines

	mline := mlines[linenum]
	line := mlines[linenum].data

	matches := headerRegex.FindStringSubmatch(line)
	if matches == nil {
		out = append(out, mline)
		return out
	}

	// There must be a blank line before the # (unless first line in file)
	if linenum != 0 {
		newlen := len(newlines)
		if newlines[newlen-1].data != "" {
			out = append(out, blankMungeLine)
		}
	}

	// There must be a space AFTER the ##'s
	newline := fmt.Sprintf("%s %s", matches[1], matches[2])
	newmline := newMungeLine(newline)
	out = append(out, newmline)

	// The next line needs to be a blank line (unless last line in file)
	if len(mlines) > linenum+1 && mlines[linenum+1].data != "" {
		out = append(out, blankMungeLine)
	}
	return out
}

// Header lines need whitespace around them and after the #s.
func updateHeaderLines(filePath string, mlines mungeLines) (mungeLines, error) {
	var out mungeLines
	for i, mline := range mlines {
		if mline.preformatted {
			out = append(out, mline)
			continue
		}
		if !mline.header {
			out = append(out, mline)
			continue
		}
		newLines := fixHeaderLine(mlines, out, i)
		out = append(out, newLines...)
	}
	return out, nil
}
