// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package common

import (
	"fmt"
	"strings"
	"unicode/utf8"

	"golang.org/x/text/width"
)

// Error type which references a location within source and a message.
type Error struct {
	Location Location
	Message  string
}

const (
	dot = "."
	ind = "^"

	// maxSnippetLength is the largest number of characters which can be rendered in an error message snippet.
	maxSnippetLength = 16384
)

var (
	wideDot = width.Widen.String(dot)
	wideInd = width.Widen.String(ind)
)

// ToDisplayString decorates the error message with the source location.
func (e *Error) ToDisplayString(source Source) string {
	var result = fmt.Sprintf("ERROR: %s:%d:%d: %s",
		source.Description(),
		e.Location.Line(),
		e.Location.Column()+1, // add one to the 0-based column for display
		e.Message)
	if snippet, found := source.Snippet(e.Location.Line()); found && len(snippet) <= maxSnippetLength {
		snippet := strings.Replace(snippet, "\t", " ", -1)
		srcLine := "\n | " + snippet
		var bytes = []byte(snippet)
		var indLine = "\n | "
		for i := 0; i < e.Location.Column() && len(bytes) > 0; i++ {
			_, sz := utf8.DecodeRune(bytes)
			bytes = bytes[sz:]
			if sz > 1 {
				indLine += wideDot
			} else {
				indLine += dot
			}
		}
		if _, sz := utf8.DecodeRune(bytes); sz > 1 {
			indLine += wideInd
		} else {
			indLine += ind
		}
		result += srcLine + indLine
	}
	return result
}
