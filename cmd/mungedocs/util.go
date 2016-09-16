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
	"path"
	"path/filepath"
	"regexp"
	"strings"
	"unicode"
)

// Replaces the text between matching "beginMark" and "endMark" within the
// document represented by "lines" with "insertThis".
//
// Delimiters should occupy own line.
// Returns copy of document with modifications.
func updateMacroBlock(mlines mungeLines, token string, insertThis mungeLines) (mungeLines, error) {
	beginMark := beginMungeTag(token)
	endMark := endMungeTag(token)
	var out mungeLines
	betweenBeginAndEnd := false
	for _, mline := range mlines {
		if mline.preformatted && !betweenBeginAndEnd {
			out = append(out, mline)
			continue
		}
		line := mline.data
		if mline.beginTag && line == beginMark {
			if betweenBeginAndEnd {
				return nil, fmt.Errorf("found second begin mark while updating macro blocks")
			}
			betweenBeginAndEnd = true
			out = append(out, mline)
		} else if mline.endTag && line == endMark {
			if !betweenBeginAndEnd {
				return nil, fmt.Errorf("found end mark without begin mark while updating macro blocks")
			}
			betweenBeginAndEnd = false
			out = append(out, insertThis...)
			out = append(out, mline)
		} else {
			if !betweenBeginAndEnd {
				out = append(out, mline)
			}
		}
	}
	if betweenBeginAndEnd {
		return nil, fmt.Errorf("never found closing end mark while updating macro blocks")
	}
	return out, nil
}

// Tests that a document, represented as a slice of lines, has a line.  Ignores
// leading and trailing space.
func hasLine(lines mungeLines, needle string) bool {
	for _, mline := range lines {
		haystack := strings.TrimSpace(mline.data)
		if haystack == needle {
			return true
		}
	}
	return false
}

func removeMacroBlock(token string, mlines mungeLines) (mungeLines, error) {
	beginMark := beginMungeTag(token)
	endMark := endMungeTag(token)
	var out mungeLines
	betweenBeginAndEnd := false
	for _, mline := range mlines {
		if mline.preformatted {
			out = append(out, mline)
			continue
		}
		line := mline.data
		if mline.beginTag && line == beginMark {
			if betweenBeginAndEnd {
				return nil, fmt.Errorf("found second begin mark while updating macro blocks")
			}
			betweenBeginAndEnd = true
		} else if mline.endTag && line == endMark {
			if !betweenBeginAndEnd {
				return nil, fmt.Errorf("found end mark without begin mark while updating macro blocks")
			}
			betweenBeginAndEnd = false
		} else {
			if !betweenBeginAndEnd {
				out = append(out, mline)
			}
		}
	}
	if betweenBeginAndEnd {
		return nil, fmt.Errorf("never found closing end mark while updating macro blocks")
	}
	return out, nil
}

// Add a macro block to the beginning of a set of lines
func prependMacroBlock(token string, mlines mungeLines) mungeLines {
	beginLine := newMungeLine(beginMungeTag(token))
	endLine := newMungeLine(endMungeTag(token))
	out := mungeLines{beginLine, endLine}
	if len(mlines) > 0 && mlines[0].data != "" {
		out = append(out, blankMungeLine)
	}
	return append(out, mlines...)
}

// Add a macro block to the end of a set of lines
func appendMacroBlock(mlines mungeLines, token string) mungeLines {
	beginLine := newMungeLine(beginMungeTag(token))
	endLine := newMungeLine(endMungeTag(token))
	out := mlines
	if len(mlines) > 0 && mlines[len(mlines)-1].data != "" {
		out = append(out, blankMungeLine)
	}
	return append(out, beginLine, endLine)
}

// Tests that a document, represented as a slice of lines, has a macro block.
func hasMacroBlock(lines mungeLines, token string) bool {
	beginMark := beginMungeTag(token)
	endMark := endMungeTag(token)

	foundBegin := false
	for _, mline := range lines {
		if mline.preformatted {
			continue
		}
		if !mline.beginTag && !mline.endTag {
			continue
		}
		line := mline.data
		switch {
		case !foundBegin && line == beginMark:
			foundBegin = true
		case foundBegin && line == endMark:
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

type mungeLine struct {
	data         string
	preformatted bool
	header       bool
	link         bool
	beginTag     bool
	endTag       bool
}

type mungeLines []mungeLine

func (m1 mungeLines) Equal(m2 mungeLines) bool {
	if len(m1) != len(m2) {
		return false
	}
	for i := range m1 {
		if m1[i].data != m2[i].data {
			return false
		}
	}
	return true
}

func (mlines mungeLines) String() string {
	slice := []string{}
	for _, mline := range mlines {
		slice = append(slice, mline.data)
	}
	s := strings.Join(slice, "\n")
	// We need to tack on an extra newline at the end of the file
	return s + "\n"
}

func (mlines mungeLines) Bytes() []byte {
	return []byte(mlines.String())
}

var (
	// Finds all preformatted block start/stops.
	preformatRE    = regexp.MustCompile("^\\s*```")
	notPreformatRE = regexp.MustCompile("^\\s*```.*```")
	// Is this line a header?
	mlHeaderRE = regexp.MustCompile(`^#`)
	// Is there a link on this line?
	mlLinkRE   = regexp.MustCompile(`\[[^]]*\]\([^)]*\)`)
	beginTagRE = regexp.MustCompile(`<!-- BEGIN MUNGE:`)
	endTagRE   = regexp.MustCompile(`<!-- END MUNGE:`)

	blankMungeLine = newMungeLine("")
)

// Does not set 'preformatted'
func newMungeLine(line string) mungeLine {
	return mungeLine{
		data:     line,
		header:   mlHeaderRE.MatchString(line),
		link:     mlLinkRE.MatchString(line),
		beginTag: beginTagRE.MatchString(line),
		endTag:   endTagRE.MatchString(line),
	}
}

func trimRightSpace(in string) string {
	return strings.TrimRightFunc(in, unicode.IsSpace)
}

// Splits a document up into a slice of lines.
func splitLines(document string) []string {
	lines := strings.Split(document, "\n")
	// Skip trailing empty string from Split-ing
	if len(lines) > 0 && lines[len(lines)-1] == "" {
		lines = lines[:len(lines)-1]
	}
	return lines
}

func getMungeLines(in string) mungeLines {
	var out mungeLines
	preformatted := false

	lines := splitLines(in)
	// We indicate if any given line is inside a preformatted block or
	// outside a preformatted block
	for _, line := range lines {
		if !preformatted {
			if preformatRE.MatchString(line) && !notPreformatRE.MatchString(line) {
				preformatted = true
			}
		} else {
			if preformatRE.MatchString(line) {
				preformatted = false
			}
		}
		ml := newMungeLine(line)
		ml.preformatted = preformatted
		out = append(out, ml)
	}
	return out
}

// filePath is the file we are looking for
// inFile is the file where we found the link. So if we are processing
//    /path/to/repoRoot/docs/admin/README.md and are looking for
//    ../../file.json we can find that location.
// In many cases filePath and processingFile may be the same
func makeRepoRelative(filePath string, processingFile string) (string, error) {
	if filePath, err := filepath.Rel(repoRoot, filePath); err == nil {
		return filePath, nil
	}
	cwd := path.Dir(processingFile)
	return filepath.Rel(repoRoot, path.Join(cwd, filePath))
}

func makeFileRelative(filePath string, processingFile string) (string, error) {
	cwd := path.Dir(processingFile)
	if filePath, err := filepath.Rel(cwd, filePath); err == nil {
		return filePath, nil
	}
	return filepath.Rel(cwd, path.Join(cwd, filePath))
}
