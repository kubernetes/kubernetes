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
	"bufio"
	"bytes"
	"fmt"
	"strings"
)

// inserts/updates a table of contents in markdown file.
//
// First, builds a ToC.
// Then, finds <!-- BEGIN GENERATED TOC --> and <!-- END GENERATED TOC -->, and replaces anything between those with
// the ToC, thereby updating any previously inserted ToC.
//
// TODO(erictune): put this in own package with tests
func updateTOC(markdown []byte) ([]byte, error) {
	toc, err := buildTOC(markdown)
	if err != nil {
		return nil, err
	}
	updatedMarkdown, err := updateMacroBlock(markdown, "<!-- BEGIN GENERATED TOC -->", "<!-- END GENERATED TOC -->", string(toc))
	if err != nil {
		return nil, err
	}
	return updatedMarkdown, nil
}

// Replaces the text between matching "beginMark" and "endMark" within "document" with "insertThis".
//
// Delimiters should occupy own line.
// Returns copy of document with modifications.
func updateMacroBlock(document []byte, beginMark, endMark, insertThis string) ([]byte, error) {
	var buffer bytes.Buffer
	lines := strings.Split(string(document), "\n")
	// Skip trailing empty string from Split-ing
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
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

// builds table of contents for markdown file
//
// First scans for all section headers (lines that begin with "#" but not within code quotes)
// and builds a table of contents from those.  Assumes bookmarks for those will be
// like #each-word-in-heading-in-lowercases-with-dashes-instead-of-spaces.
// builds the ToC.
func buildTOC(markdown []byte) ([]byte, error) {
	var buffer bytes.Buffer
	scanner := bufio.NewScanner(bytes.NewReader(markdown))
	for scanner.Scan() {
		line := scanner.Text()
		noSharps := strings.TrimLeft(line, "#")
		numSharps := len(line) - len(noSharps)
		heading := strings.Trim(noSharps, " \n")
		if numSharps > 0 {
			indent := strings.Repeat("  ", numSharps-1)
			bookmark := strings.Replace(strings.ToLower(heading), " ", "-", -1)
			tocLine := fmt.Sprintf("%s- [%s](#%s)\n", indent, heading, bookmark)
			buffer.WriteString(tocLine)
		}
	}
	if err := scanner.Err(); err != nil {
		return []byte{}, err
	}

	return buffer.Bytes(), nil
}
