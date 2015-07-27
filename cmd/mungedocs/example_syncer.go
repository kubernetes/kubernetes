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
	"io/ioutil"
	"path"
	"regexp"
	"strings"
)

const exampleMungeTag = "EXAMPLE"

// syncExamples updates all examples in markdown file.
//
// Finds the magic macro block tags, find the link to the example
// specified in the tags, and replaces anything between those with
// the content of the example, thereby syncing it.
//
// For example,
// <!-- BEGIN MUNGE: EXAMPLE ../../examples/guestbook/frontend-controller.yaml -->
//
// ```yaml
// foo:
//    bar:
// ```
//
// [Download example](../../examples/guestbook/frontend-controller.yaml)
// <!-- END MUNGE: EXAMPLE -->
func syncExamples(filePath string, markdown []byte) ([]byte, error) {
	// find the example syncer begin tag
	header := beginMungeTag(fmt.Sprintf("%s %s", exampleMungeTag, `(([^ ])*.(yaml|json))`))
	exampleLinkRE := regexp.MustCompile(header)
	lines := splitLines(markdown)
	updatedMarkdown, err := updateExampleMacroBlock(filePath, lines, exampleLinkRE, endMungeTag(exampleMungeTag))
	if err != nil {
		return updatedMarkdown, err
	}
	return updatedMarkdown, nil
}

// exampleContent retrieves the content of the file at linkPath
func exampleContent(filePath, linkPath, fileType string) (content string, err error) {
	realRoot := path.Join(*rootDir, *repoRoot) + "/"
	path := path.Join(realRoot, path.Dir(filePath), linkPath)
	dat, err := ioutil.ReadFile(path)
	if err != nil {
		return content, err
	}
	// remove leading and trailing spaces and newlines
	trimmedFileContent := strings.TrimSpace(string(dat))
	content = fmt.Sprintf("\n```%s\n%s\n```\n\n[Download example](%s)", fileType, trimmedFileContent, linkPath)
	return
}

// updateExampleMacroBlock sync the yaml/json example between begin tag and end tag
func updateExampleMacroBlock(filePath string, lines []string, beginMarkExp *regexp.Regexp, endMark string) ([]byte, error) {
	var buffer bytes.Buffer
	betweenBeginAndEnd := false
	for _, line := range lines {
		trimmedLine := strings.Trim(line, " \n")
		if beginMarkExp.Match([]byte(trimmedLine)) {
			if betweenBeginAndEnd {
				return nil, fmt.Errorf("found second begin mark while updating macro blocks")
			}
			betweenBeginAndEnd = true
			buffer.WriteString(line)
			buffer.WriteString("\n")
			match := beginMarkExp.FindStringSubmatch(line)
			if len(match) < 4 {
				return nil, fmt.Errorf("failed to parse the link in example header")
			}
			// match[0] is the entire expression; [1] is the link text and [3] is the file type (yaml or json).
			linkText := match[1]
			fileType := match[3]
			example, err := exampleContent(filePath, linkText, fileType)
			if err != nil {
				return nil, err
			}
			buffer.WriteString(example)
		} else if trimmedLine == endMark {
			if !betweenBeginAndEnd {
				return nil, fmt.Errorf("found end mark without being mark while updating macro blocks")
			}
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
