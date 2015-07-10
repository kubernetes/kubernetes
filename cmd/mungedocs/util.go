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
	"bytes"
	"fmt"
	"strings"
)

// Splits a document up into a slice of lines.
func splitLines(document []byte) []string {
	lines := strings.Split(string(document), "\n")
	// Skip trailing empty string from Split-ing
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

// Replaces the text between matching "beginMark" and "endMark" within the
// document represented by "lines" with "insertThis".
//
// Delimiters should occupy own line.
// Returns copy of document with modifications.
func updateMacroBlock(lines []string, beginMark, endMark, insertThis string) ([]byte, error) {
	var buffer bytes.Buffer
	betweenBeginAndEnd := false
	for _, line := range lines {
		trimmedLine := strings.Trim(line, " \n")
		if trimmedLine == beginMark {
			if betweenBeginAndEnd {
				return nil, fmt.Errorf("found second begin mark while updating macro blocks")
			}
			betweenBeginAndEnd = true
			buffer.WriteString(line)
			buffer.WriteString("\n")
		} else if trimmedLine == endMark {
			if !betweenBeginAndEnd {
				return nil, fmt.Errorf("found end mark without being mark while updating macro blocks")
			}
			buffer.WriteString(insertThis)
			// Extra newline avoids github markdown bug where comment ends up on same line as last bullet.
			buffer.WriteString("\n")
			buffer.WriteString(line)
			buffer.WriteString("\n")
			betweenBeginAndEnd = false
		} else {
			if !betweenBeginAndEnd {
				buffer.WriteString(line)
				buffer.WriteString("\n")
			}
		}
	}
	if betweenBeginAndEnd {
		return nil, fmt.Errorf("never found closing end mark while updating macro blocks")
	}
	return buffer.Bytes(), nil
}

// Tests that a document, represented as a slice of lines, has a line.  Ignores
// leading and trailing space.
func hasLine(lines []string, needle string) bool {
	for _, line := range lines {
		trimmedLine := strings.Trim(line, " \n")
		if trimmedLine == needle {
			return true
		}
	}
	return false
}

// Tests that a document, represented as a slice of lines, has a macro block.
func hasMacroBlock(lines []string, begin string, end string) bool {
	foundBegin := false
	for _, line := range lines {
		trimmedLine := strings.Trim(line, " \n")
		switch {
		case !foundBegin && trimmedLine == begin:
			foundBegin = true
		case foundBegin && trimmedLine == end:
			return true
		}
	}
	return false
}
