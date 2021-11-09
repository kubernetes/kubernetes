// Copyright 2020 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"github.com/google/cel-go/common"
)

// ByteSource converts a byte sequence and location description to a model.Source.
func ByteSource(contents []byte, location string) *Source {
	return StringSource(string(contents), location)
}

// StringSource converts a string and location description to a model.Source.
func StringSource(contents, location string) *Source {
	return &Source{
		Source: common.NewStringSource(contents, location),
	}
}

// Source represents the contents of a single source file.
type Source struct {
	common.Source
}

// Relative produces a RelativeSource object for the content provided at the absolute location
// within the parent Source as indicated by the line and column.
func (src *Source) Relative(content string, line, col int) *RelativeSource {
	return &RelativeSource{
		Source:   src.Source,
		localSrc: common.NewStringSource(content, src.Description()),
		absLoc:   common.NewLocation(line, col),
	}
}

// RelativeSource represents an embedded source element within a larger source.
type RelativeSource struct {
	common.Source
	localSrc common.Source
	absLoc   common.Location
}

// AbsoluteLocation returns the location within the parent Source where the RelativeSource starts.
func (rel *RelativeSource) AbsoluteLocation() common.Location {
	return rel.absLoc
}

// Content returns the embedded source snippet.
func (rel *RelativeSource) Content() string {
	return rel.localSrc.Content()
}

// OffsetLocation returns the absolute location given the relative offset, if found.
func (rel *RelativeSource) OffsetLocation(offset int32) (common.Location, bool) {
	absOffset, found := rel.Source.LocationOffset(rel.absLoc)
	if !found {
		return common.NoLocation, false
	}
	return rel.Source.OffsetLocation(absOffset + offset)
}

// NewLocation creates an absolute common.Location based on a local line, column
// position from a relative source.
func (rel *RelativeSource) NewLocation(line, col int) common.Location {
	localLoc := common.NewLocation(line, col)
	relOffset, found := rel.localSrc.LocationOffset(localLoc)
	if !found {
		return common.NoLocation
	}
	offset, _ := rel.Source.LocationOffset(rel.absLoc)
	absLoc, _ := rel.Source.OffsetLocation(offset + relOffset)
	return absLoc
}

// NewSourceInfo creates SourceInfo metadata from a Source object.
func NewSourceInfo(src common.Source) *SourceInfo {
	return &SourceInfo{
		Comments:    make(map[int64][]*Comment),
		LineOffsets: src.LineOffsets(),
		Description: src.Description(),
		Offsets:     make(map[int64]int32),
	}
}

// SourceInfo contains metadata about the Source such as comments, line positions, and source
// element offsets.
type SourceInfo struct {
	// Comments mapped by source element id to a comment set.
	Comments map[int64][]*Comment

	// LineOffsets contains the list of character offsets where newlines occur in the source.
	LineOffsets []int32

	// Description indicates something about the source, such as its file name.
	Description string

	// Offsets map from source element id to the character offset where the source element starts.
	Offsets map[int64]int32
}

// SourceMetadata enables the lookup for expression source metadata by expression id.
type SourceMetadata interface {
	// CommentsByID returns the set of comments associated with the expression id, if present.
	CommentsByID(int64) ([]*Comment, bool)

	// LocationByID returns the CEL common.Location of the expression id, if present.
	LocationByID(int64) (common.Location, bool)
}

// CommentsByID returns the set of comments by expression id, if present.
func (info *SourceInfo) CommentsByID(id int64) ([]*Comment, bool) {
	comments, found := info.Comments[id]
	return comments, found
}

// LocationByID returns the line and column location of source node by its id.
func (info *SourceInfo) LocationByID(id int64) (common.Location, bool) {
	charOff, found := info.Offsets[id]
	if !found {
		return common.NoLocation, false
	}
	ln, lnOff := info.findLine(charOff)
	return common.NewLocation(int(ln), int(charOff-lnOff)), true
}

func (info *SourceInfo) findLine(characterOffset int32) (int32, int32) {
	var line int32 = 1
	for _, lineOffset := range info.LineOffsets {
		if lineOffset > characterOffset {
			break
		} else {
			line++
		}
	}
	if line == 1 {
		return line, 0
	}
	return line, info.LineOffsets[line-2]
}

// CommentStyle type used to indicate where a comment occurs.
type CommentStyle int

const (
	// HeadComment indicates that the comment is defined in the lines preceding the source element.
	HeadComment CommentStyle = iota + 1

	// LineComment indicates that the comment occurs on the same line after the source element.
	LineComment

	// FootComment indicates that the comment occurs after the source element with at least one
	// blank line before the next source element.
	FootComment
)

// NewHeadComment creates a new HeadComment from the text.
func NewHeadComment(txt string) *Comment {
	return &Comment{Text: txt, Style: HeadComment}
}

// NewLineComment creates a new LineComment from the text.
func NewLineComment(txt string) *Comment {
	return &Comment{Text: txt, Style: LineComment}
}

// NewFootComment creates a new FootComment from the text.
func NewFootComment(txt string) *Comment {
	return &Comment{Text: txt, Style: FootComment}
}

// Comment represents a comment within source.
type Comment struct {
	// Text contains the comment text.
	Text string

	// Style indicates where the comment appears relative to a source element.
	Style CommentStyle
}
