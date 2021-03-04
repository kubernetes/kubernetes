// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package protoreflect

// SourceLocations is a list of source locations.
type SourceLocations interface {
	// Len reports the number of source locations in the proto file.
	Len() int
	// Get returns the ith SourceLocation. It panics if out of bounds.
	Get(int) SourceLocation

	doNotImplement

	// TODO: Add ByPath and ByDescriptor helper methods.
}

// SourceLocation describes a source location and
// corresponds with the google.protobuf.SourceCodeInfo.Location message.
type SourceLocation struct {
	// Path is the path to the declaration from the root file descriptor.
	// The contents of this slice must not be mutated.
	Path SourcePath

	// StartLine and StartColumn are the zero-indexed starting location
	// in the source file for the declaration.
	StartLine, StartColumn int
	// EndLine and EndColumn are the zero-indexed ending location
	// in the source file for the declaration.
	// In the descriptor.proto, the end line may be omitted if it is identical
	// to the start line. Here, it is always populated.
	EndLine, EndColumn int

	// LeadingDetachedComments are the leading detached comments
	// for the declaration. The contents of this slice must not be mutated.
	LeadingDetachedComments []string
	// LeadingComments is the leading attached comment for the declaration.
	LeadingComments string
	// TrailingComments is the trailing attached comment for the declaration.
	TrailingComments string
}

// SourcePath identifies part of a file descriptor for a source location.
// The SourcePath is a sequence of either field numbers or indexes into
// a repeated field that form a path starting from the root file descriptor.
//
// See google.protobuf.SourceCodeInfo.Location.path.
type SourcePath []int32

// TODO: Add SourcePath.String method to pretty-print the path. For example:
//	".message_type[6].nested_type[15].field[3]"
