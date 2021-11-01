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
	"strings"
	"unicode/utf8"

	"github.com/google/cel-go/common/runes"

	exprpb "google.golang.org/genproto/googleapis/api/expr/v1alpha1"
)

// Source interface for filter source contents.
type Source interface {
	// Content returns the source content represented as a string.
	// Examples contents are the single file contents, textbox field,
	// or url parameter.
	Content() string

	// Description gives a brief description of the source.
	// Example descriptions are a file name or ui element.
	Description() string

	// LineOffsets gives the character offsets at which lines occur.
	// The zero-th entry should refer to the break between the first
	// and second line, or EOF if there is only one line of source.
	LineOffsets() []int32

	// LocationOffset translates a Location to an offset.
	// Given the line and column of the Location returns the
	// Location's character offset in the Source, and a bool
	// indicating whether the Location was found.
	LocationOffset(location Location) (int32, bool)

	// OffsetLocation translates a character offset to a Location, or
	// false if the conversion was not feasible.
	OffsetLocation(offset int32) (Location, bool)

	// NewLocation takes an input line and column and produces a Location.
	// The default behavior is to treat the line and column as absolute,
	// but concrete derivations may use this method to convert a relative
	// line and column position into an absolute location.
	NewLocation(line, col int) Location

	// Snippet returns a line of content and whether the line was found.
	Snippet(line int) (string, bool)
}

// The sourceImpl type implementation of the Source interface.
type sourceImpl struct {
	runes.Buffer
	description string
	lineOffsets []int32
	idOffsets   map[int64]int32
}

var _ runes.Buffer = &sourceImpl{}

// TODO(jimlarson) "Character offsets" should index the code points
// within the UTF-8 encoded string.  It currently indexes bytes.
// Can be accomplished by using rune[] instead of string for contents.

// NewTextSource creates a new Source from the input text string.
func NewTextSource(text string) Source {
	return NewStringSource(text, "<input>")
}

// NewStringSource creates a new Source from the given contents and description.
func NewStringSource(contents string, description string) Source {
	// Compute line offsets up front as they are referred to frequently.
	lines := strings.Split(contents, "\n")
	offsets := make([]int32, len(lines))
	var offset int32
	for i, line := range lines {
		offset = offset + int32(utf8.RuneCountInString(line)) + 1
		offsets[int32(i)] = offset
	}
	return &sourceImpl{
		Buffer:      runes.NewBuffer(contents),
		description: description,
		lineOffsets: offsets,
		idOffsets:   map[int64]int32{},
	}
}

// NewInfoSource creates a new Source from a SourceInfo.
func NewInfoSource(info *exprpb.SourceInfo) Source {
	return &sourceImpl{
		Buffer:      runes.NewBuffer(""),
		description: info.GetLocation(),
		lineOffsets: info.GetLineOffsets(),
		idOffsets:   info.GetPositions(),
	}
}

// Content implements the Source interface method.
func (s *sourceImpl) Content() string {
	return s.Slice(0, s.Len())
}

// Description implements the Source interface method.
func (s *sourceImpl) Description() string {
	return s.description
}

// LineOffsets implements the Source interface method.
func (s *sourceImpl) LineOffsets() []int32 {
	return s.lineOffsets
}

// LocationOffset implements the Source interface method.
func (s *sourceImpl) LocationOffset(location Location) (int32, bool) {
	if lineOffset, found := s.findLineOffset(location.Line()); found {
		return lineOffset + int32(location.Column()), true
	}
	return -1, false
}

// NewLocation implements the Source interface method.
func (s *sourceImpl) NewLocation(line, col int) Location {
	return NewLocation(line, col)
}

// OffsetLocation implements the Source interface method.
func (s *sourceImpl) OffsetLocation(offset int32) (Location, bool) {
	line, lineOffset := s.findLine(offset)
	return NewLocation(int(line), int(offset-lineOffset)), true
}

// Snippet implements the Source interface method.
func (s *sourceImpl) Snippet(line int) (string, bool) {
	charStart, found := s.findLineOffset(line)
	if !found || s.Len() == 0 {
		return "", false
	}
	charEnd, found := s.findLineOffset(line + 1)
	if found {
		return s.Slice(int(charStart), int(charEnd-1)), true
	}
	return s.Slice(int(charStart), s.Len()), true
}

// findLineOffset returns the offset where the (1-indexed) line begins,
// or false if line doesn't exist.
func (s *sourceImpl) findLineOffset(line int) (int32, bool) {
	if line == 1 {
		return 0, true
	}
	if line > 1 && line <= int(len(s.lineOffsets)) {
		offset := s.lineOffsets[line-2]
		return offset, true
	}
	return -1, false
}

// findLine finds the line that contains the given character offset and
// returns the line number and offset of the beginning of that line.
// Note that the last line is treated as if it contains all offsets
// beyond the end of the actual source.
func (s *sourceImpl) findLine(characterOffset int32) (int32, int32) {
	var line int32 = 1
	for _, lineOffset := range s.lineOffsets {
		if lineOffset > characterOffset {
			break
		} else {
			line++
		}
	}
	if line == 1 {
		return line, 0
	}
	return line, s.lineOffsets[line-2]
}
