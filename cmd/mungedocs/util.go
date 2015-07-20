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
	"regexp"
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
func updateMacroBlock(lines []string, token, insertThis string) ([]byte, error) {
	beginMark := beginMungeTag(token)
	endMark := endMungeTag(token)
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

// Add a macro block to the beginning of a set of lines
func prependMacroBlock(token string, lines []string) []string {
	beginMark := beginMungeTag(token)
	endMark := endMungeTag(token)
	return append([]string{beginMark, endMark}, lines...)
}

// Add a macro block to the end of a set of lines
func appendMacroBlock(token string, lines []string) []string {
	beginMark := beginMungeTag(token)
	endMark := endMungeTag(token)
	return append(lines, beginMark, endMark)
}

// Tests that a document, represented as a slice of lines, has a macro block.
func hasMacroBlock(lines []string, token string) bool {
	beginMark := beginMungeTag(token)
	endMark := endMungeTag(token)

	foundBegin := false
	for _, line := range lines {
		trimmedLine := strings.Trim(line, " \n")
		switch {
		case !foundBegin && trimmedLine == beginMark:
			foundBegin = true
		case foundBegin && trimmedLine == endMark:
			return true
		}
	}
	return false
}

// Returns the canonical begin-tag for a given description.  This does not
// include the trailing newline.
func beginMungeTag(desc string) string {
	return fmt.Sprintf("<!-- BEGIN MUNGE: %s -->", desc)
}

// Returns the canonical end-tag for a given description.  This does not
// include the trailing newline.
func endMungeTag(desc string) string {
	return fmt.Sprintf("<!-- END MUNGE: %s -->", desc)
}

// Calls 'replace' for all sections of the document not in ``` / ``` blocks. So
// that you don't have false positives inside those blocks.
func replaceNonPreformatted(input []byte, replace func([]byte) []byte) []byte {
	f := splitByPreformatted(input)
	output := []byte(nil)
	for _, block := range f {
		if block.preformatted {
			output = append(output, block.data...)
		} else {
			output = append(output, replace(block.data)...)
		}
	}
	return output
}

type fileBlock struct {
	preformatted bool
	data         []byte
}

type fileBlocks []fileBlock

var (
	// Finds all preformatted block start/stops.
	preformatRE    = regexp.MustCompile("^\\s*```")
	notPreformatRE = regexp.MustCompile("^\\s*```.*```")
)

func splitByPreformatted(input []byte) fileBlocks {
	f := fileBlocks{}

	cur := []byte(nil)
	preformatted := false
	// SplitAfter keeps the newline, so you don't have to worry about
	// omitting it on the last line or anything. Also, the documentation
	// claims it's unicode safe.
	for _, line := range bytes.SplitAfter(input, []byte("\n")) {
		if !preformatted {
			if preformatRE.Match(line) && !notPreformatRE.Match(line) {
				if len(cur) > 0 {
					f = append(f, fileBlock{false, cur})
				}
				cur = []byte{}
				preformatted = true
			}
			cur = append(cur, line...)
		} else {
			cur = append(cur, line...)
			if preformatRE.Match(line) {
				if len(cur) > 0 {
					f = append(f, fileBlock{true, cur})
				}
				cur = []byte{}
				preformatted = false
			}
		}
	}
	if len(cur) > 0 {
		f = append(f, fileBlock{preformatted, cur})
	}
	return f
}

// As above, but further uses exp to parse the non-preformatted sections.
func replaceNonPreformattedRegexp(input []byte, exp *regexp.Regexp, replace func([]byte) []byte) []byte {
	return replaceNonPreformatted(input, func(in []byte) []byte {
		return exp.ReplaceAllFunc(in, replace)
	})
}
