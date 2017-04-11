// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ianaindex maps names to Encodings as specified by the IANA registry.
// This includes both the MIME and IANA names.
//
// See http://www.iana.org/assignments/character-sets/character-sets.xhtml for
// more details.
package ianaindex

import (
	"golang.org/x/text/encoding"
)

// TODO: allow users to specify their own aliases?
// TODO: allow users to specify their own indexes?
// TODO: allow canonicalizing names

// NOTE: only use these top-level variables if we can get the linker to drop
// the indexes when they are not used. Make them a function or perhaps only
// support MIME otherwise.

var (
	// MIME is an index to map MIME names. It does not support aliases.
	MIME *Index

	// IANA is an index that supports all names and aliases using IANA names as
	// the canonical identifier.
	IANA *Index
)

// Index maps names registered by IANA to Encodings.
type Index struct {
}

// Get returns an Encoding for IANA-registered names. Matching is
// case-insensitive.
func (x *Index) Get(name string) (encoding.Encoding, error) {
	panic("TODO: implement")
}

// Name reports the canonical name of the given Encoding. It will return an
// error if the e is not associated with a known encoding scheme.
func (x *Index) Name(e encoding.Encoding) (string, error) {
	panic("TODO: implement")
}

// TODO: the coverage of this index is rather spotty. Allowing users to set
// encodings would allow:
// - users to increase coverage
// - allow a partially loaded set of encodings in case the user doesn't need to
//   them all.
// - write an OS-specific wrapper for supported encodings and set them.
// The exact definition of Set depends a bit on if and how we want to let users
// write their own Encoding implementations. Also, it is not possible yet to
// only partially load the encodings without doing some refactoring. Until this
// is solved, we might as well not support Set.
// // Set sets the e to be used for the encoding scheme identified by name. Only
// // canonical names may be used. An empty name assigns e to its internally
// // associated encoding scheme.
// func (x *Index) Set(name string, e encoding.Encoding) error {
// 	panic("TODO: implement")
// }
