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
	"regexp"
	"strings"
)

const tocMungeTag = "GENERATED_TOC"

// inserts/updates a table of contents in markdown file.
//
// First, builds a ToC.
// Then, finds the magic macro block tags and replaces anything between those with
// the ToC, thereby updating any previously inserted ToC.
//
// TODO(erictune): put this in own package with tests
func updateTOC(filePath string, markdown []byte) ([]byte, error) {
	toc, err := buildTOC(markdown)
	if err != nil {
		return nil, err
	}
	lines := splitLines(markdown)
	updatedMarkdown, err := updateMacroBlock(lines, beginMungeTag(tocMungeTag), endMungeTag(tocMungeTag), string(toc))
	if err != nil {
		return nil, err
	}
	return updatedMarkdown, nil
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
	inBlockQuotes := false
	for scanner.Scan() {
		line := scanner.Text()
		match, err := regexp.Match("^```", []byte(line))
		if err != nil {
			return nil, err
		}
		if match {
			inBlockQuotes = !inBlockQuotes
			continue
		}
		if inBlockQuotes {
			continue
		}
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
