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
	"io/ioutil"
	"regexp"
	"strings"
)

const exampleToken = "EXAMPLE"

const exampleLineStart = "<!-- BEGIN MUNGE: EXAMPLE"

var exampleMungeTagRE = regexp.MustCompile(beginMungeTag(fmt.Sprintf("%s %s", exampleToken, `(([^ ])*[.]([^.]*))`)))

// syncExamples updates all examples in markdown file.
//
// Finds the magic macro block tags, find the link to the example
// specified in the tags, and replaces anything between those with
// the content of the example, thereby syncing it.
//
// For example,
// <!-- BEGIN MUNGE: EXAMPLE ../../examples/guestbook/frontend-service.yaml -->
//
// ```yaml
// foo:
//    bar:
// ```
//
// [Download example](../../examples/guestbook/frontend-service.yaml?raw=true)
// <!-- END MUNGE: EXAMPLE -->
func syncExamples(filePath string, mlines mungeLines) (mungeLines, error) {
	var err error
	type exampleTag struct {
		token    string
		linkText string
		fileType string
	}
	exampleTags := []exampleTag{}

	// collect all example Tags
	for _, mline := range mlines {
		if mline.preformatted || !mline.beginTag {
			continue
		}
		line := mline.data
		if !strings.HasPrefix(line, exampleLineStart) {
			continue
		}
		match := exampleMungeTagRE.FindStringSubmatch(line)
		if len(match) < 4 {
			err = fmt.Errorf("Found unparsable EXAMPLE munge line %v", line)
			return mlines, err
		}
		tag := exampleTag{
			token:    exampleToken + " " + match[1],
			linkText: match[1],
			fileType: match[3],
		}
		exampleTags = append(exampleTags, tag)
	}
	// update all example Tags
	for _, tag := range exampleTags {
		ft := ""
		if tag.fileType == "json" {
			ft = "json"
		}
		if tag.fileType == "yaml" {
			ft = "yaml"
		}
		example, err := exampleContent(filePath, tag.linkText, ft)
		if err != nil {
			return mlines, err
		}
		mlines, err = updateMacroBlock(mlines, tag.token, example)
		if err != nil {
			return mlines, err
		}
	}
	return mlines, nil
}

// exampleContent retrieves the content of the file at linkPath
func exampleContent(filePath, linkPath, fileType string) (mungeLines, error) {
	repoRel, err := makeRepoRelative(linkPath, filePath)
	if err != nil {
		return nil, err
	}

	fileRel, err := makeFileRelative(linkPath, filePath)
	if err != nil {
		return nil, err
	}

	dat, err := ioutil.ReadFile(repoRel)
	if err != nil {
		return nil, err
	}

	// remove leading and trailing spaces and newlines
	trimmedFileContent := strings.TrimSpace(string(dat))
	content := fmt.Sprintf("\n```%s\n%s\n```\n\n[Download example](%s?raw=true)", fileType, trimmedFileContent, fileRel)
	out := getMungeLines(content)
	return out, nil
}
