/*
Copyright 2023 The Kubernetes Authors.

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

package yh

import (
	"bufio"
	"fmt"
	"io"
	"strings"

	"github.com/fatih/color"
)

var (
	BrightRed        = color.New(color.FgHiRed).SprintfFunc()
	Yellow           = color.New(color.FgYellow).SprintfFunc()
	Blue             = color.New(color.FgBlue).SprintfFunc()
	HiMagenta        = color.New(color.FgHiMagenta).SprintfFunc()
	InvalidLineColor = color.New(color.FgBlack, color.BgHiRed).SprintfFunc()
)

func Highlight(r io.Reader) (string, error) {
	// Service vars
	foundChompingIndicator := false
	indentationSpacesBeforeComment := 0

	// Warm-up the engine
	scanner := bufio.NewScanner(r)

	var buf strings.Builder

	// Get the juice
	for scanner.Scan() {
		if scanner.Text() == "EOF" {
			break
		}

		// Check for errors during Stdin read
		if err := scanner.Err(); err != nil {
			return "", err
		}

		// Drink the juice
		l := yamlLine{raw: scanner.Text()}

		if foundChompingIndicator && (l.indentationSpaces() > indentationSpacesBeforeComment) {
			// Found multiline comment or configmap, not treated as YAML at all
			buf.WriteString(multiline(l))

		} else if l.isKeyValue() {
			// This is a valid YAML key: value line. Key and value are returned in l

			if l.isComment() {
				// This line is a comment
				buf.WriteString(comment(l))
			} else if l.valueIsNumberOrIP() {
				// The value is a number or an IP address x.x.x.x
				buf.WriteString(keyNumberOrIP(l))

			} else if l.valueIsBoolean() {
				// The value is boolean true or false
				buf.WriteString(keyBool(l))

			} else {
				// The is a normal key/value line
				buf.WriteString(keyValue(l))
			}

			if l.valueContainsChompingIndicator() {
				// This line contains a chomping indicator, sign of a possible multiline text coming next

				// Setting flag for next execution
				foundChompingIndicator = true

				// Saving current number of indentation spaces
				indentationSpacesBeforeComment = l.indentationSpaces()

			} else {
				// Resetting multiline flag
				foundChompingIndicator = false
			}

		} else if !l.isEmptyLine() {
			// This is not a YAML key: value line and is not empty

			if l.isUrl() {
				// Value is a URL
				buf.WriteString(url(l))
			} else if l.isComment() {
				// This line is a comment
				buf.WriteString(comment(l))
			} else if l.isElementOfList() {
				// This line is an element of a list
				buf.WriteString(listElement(l))
			} else {
				// This line is not valid YAML
				buf.WriteString(invalidLine(l))
			}

			foundChompingIndicator = false

		} else if l.isEmptyLine() {
			// This is an empty line
			fmt.Println(l.raw)
		}

	}

	return buf.String(), nil
}

func keyValue(l yamlLine) string {
	return fmt.Sprintf("%v: %v\n", BrightRed(l.key), Yellow(l.value))
}

func keyNumberOrIP(l yamlLine) string {
	return fmt.Sprintf("%v: %v\n", BrightRed(l.key), Blue(l.value))
}

func keyBool(l yamlLine) string {
	return fmt.Sprintf("%v: %v\n", BrightRed(l.key), Blue(l.value))
}

func comment(l yamlLine) string {
	return fmt.Sprintf("%v %v\n", HiMagenta(l.key), HiMagenta(l.value))
}

func listElement(l yamlLine) string {
	return fmt.Sprintf("%v\n", Yellow(l.raw))
}

func invalidLine(l yamlLine) string {
	return fmt.Sprintf("%v\n", InvalidLineColor(l.raw))
}

func multiline(l yamlLine) string {
	return fmt.Sprintf("%v\n", HiMagenta(l.raw))
}
func url(l yamlLine) string {
	return fmt.Sprintf("%v\n", Yellow(l.raw))
}
