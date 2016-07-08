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
	"strings"
)

const tocMungeTag = "GENERATED_TOC"

var r = regexp.MustCompile("[^A-Za-z0-9-]")

// inserts/updates a table of contents in markdown file.
//
// First, builds a ToC.
// Then, finds the magic macro block tags and replaces anything between those with
// the ToC, thereby updating any previously inserted ToC.
//
// TODO(erictune): put this in own package with tests
func updateTOC(filePath string, mlines mungeLines) (mungeLines, error) {
	toc := buildTOC(mlines)
	updatedMarkdown, err := updateMacroBlock(mlines, tocMungeTag, toc)
	if err != nil {
		return mlines, err
	}
	return updatedMarkdown, nil
}

// builds table of contents for markdown file
//
// First scans for all section headers (lines that begin with "#" but not within code quotes)
// and builds a table of contents from those.  Assumes bookmarks for those will be
// like #each-word-in-heading-in-lowercases-with-dashes-instead-of-spaces.
// builds the ToC.

func buildTOC(mlines mungeLines) mungeLines {
	var out mungeLines
	bookmarks := map[string]int{}

	for _, mline := range mlines {
		if mline.preformatted || !mline.header {
			continue
		}
		// Add a blank line after the munge start tag
		if len(out) == 0 {
			out = append(out, blankMungeLine)
		}
		line := mline.data
		noSharps := strings.TrimLeft(line, "#")
		numSharps := len(line) - len(noSharps)
		heading := strings.Trim(noSharps, " \n")
		if numSharps > 0 {
			indent := strings.Repeat("  ", numSharps-1)
			bookmark := strings.Replace(strings.ToLower(heading), " ", "-", -1)
			// remove symbols (except for -) in bookmarks
			bookmark = r.ReplaceAllString(bookmark, "")
			// Incremental counter for duplicate bookmarks
			next := bookmarks[bookmark]
			bookmarks[bookmark] = next + 1
			if next > 0 {
				bookmark = fmt.Sprintf("%s-%d", bookmark, next)
			}
			tocLine := fmt.Sprintf("%s- [%s](#%s)", indent, heading, bookmark)
			out = append(out, newMungeLine(tocLine))
		}

	}
	// Add a blank line before the munge end tag
	if len(out) != 0 {
		out = append(out, blankMungeLine)
	}
	return out
}
